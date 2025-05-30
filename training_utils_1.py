import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb


def create_features(df_train, df_test=None, target_col='ToPredict'):
    """
    Create features for train and test data simultaneously to ensure consistency
    
    Args:
        df_train: Training dataframe
        df_test: Test dataframe (optional)
        target_col: Name of target column
        
    Returns:
        Tuple of (processed_train_df, processed_test_df) or just processed_train_df if df_test is None
    """
    # Make copies to avoid modifying originals
    df_train = df_train.copy()
    
    # Convert dates and sort
    df_train['Dates'] = pd.to_datetime(df_train['Dates'])
    df_train = df_train.sort_values('Dates').reset_index(drop=True)
    
    # Process test data if provided
    if df_test is not None:
        df_test = df_test.copy()
        df_test['Dates'] = pd.to_datetime(df_test['Dates'])
        
        # Add placeholder for target in test data if it doesn't exist
        if target_col not in df_test.columns:
            df_test[target_col] = np.nan
            
        df_test = df_test.sort_values('Dates').reset_index(drop=True)
    
    # Identify features to drop based on correlation in training data
    feature_cols = [col for col in df_train.columns if col not in ['Dates', target_col]]
    numeric_df = df_train[feature_cols].select_dtypes(include=['number']).dropna()
    
    to_drop = []
    if not numeric_df.empty and numeric_df.shape[1] > 1:
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr().abs()
        
        # Create a mask for the upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than 0.9
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        
        if to_drop:
            print(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
            # Drop these features from both train and test
            df_train = df_train.drop(columns=to_drop)
            if df_test is not None:
                # Only drop columns that exist in test data
                test_to_drop = [col for col in to_drop if col in df_test.columns]
                df_test = df_test.drop(columns=test_to_drop)
    
    # Create lag features for train data
    df_train[f'{target_col}_lag_1'] = df_train[target_col].shift(1)
    df_train[f'{target_col}_lag_2'] = df_train[target_col].shift(2)
    
    # Create rolling statistics for train data
    for window in [261]:
        df_train[f'{target_col}_rolling_mean_{window}'] = df_train[target_col].rolling(window).mean()
        df_train[f'{target_col}_rolling_std_{window}'] = df_train[target_col].rolling(window).std()
    
    # Create date-based features for train data
    df_train['bimonthly'] = ((df_train['Dates'].dt.month - 1) // 2) + 1
    df_train['quarter'] = df_train['Dates'].dt.quarter
    
    # Create Fourier terms for train data
    for period in [91, 61]:  # ~quarterly (91 days), ~bimonthly (61 days)
        for n in range(1, 3):  # Use 2 harmonics for each
            df_train[f'sin_{period}_{n}'] = np.sin(2 * n * np.pi * df_train.index / period)
            df_train[f'cos_{period}_{n}'] = np.cos(2 * n * np.pi * df_train.index / period)
    
    # Add trend feature
    df_train['time_idx'] = np.arange(len(df_train))
    
    # If test data is provided, create the same features
    if df_test is not None:
        # Create lag features for test data
        df_test[f'{target_col}_lag_1'] = df_test[target_col].shift(1)
        df_test[f'{target_col}_lag_2'] = df_test[target_col].shift(2)
        
        # Create rolling statistics for test data
        for window in [261]:
            df_test[f'{target_col}_rolling_mean_{window}'] = df_test[target_col].rolling(window).mean()
            df_test[f'{target_col}_rolling_std_{window}'] = df_test[target_col].rolling(window).std()
        
        # Create date-based features for test data
        df_test['bimonthly'] = ((df_test['Dates'].dt.month - 1) // 2) + 1
        df_test['quarter'] = df_test['Dates'].dt.quarter
        
        # Create Fourier terms for test data
        for period in [91, 61]:
            for n in range(1, 3):
                df_test[f'sin_{period}_{n}'] = np.sin(2 * n * np.pi * df_test.index / period)
                df_test[f'cos_{period}_{n}'] = np.cos(2 * n * np.pi * df_test.index / period)
        
        # Add trend feature
        df_test['time_idx'] = np.arange(len(df_test)) + len(df_train)  # Continue numbering from train
        
        return df_train, df_test
    
    return df_train

def prepare_multioutput_data(df_features, forecast_horizon=5, target_col='ToPredict'):
    """
    Prepare data for multi-output forecasting
    Creates X features and y targets for multiple horizons
    """
    # Remove rows with NaN values from feature creation
    df_clean = df_features.dropna().reset_index(drop=True)
    
    # Select feature columns (exclude target and date)
    feature_cols = [col for col in df_clean.columns 
                   if col not in ['Dates', target_col]]
    
    X_list = []
    y_list = []
    
    # Create training samples
    for i in range(len(df_clean) - forecast_horizon):
        # Features at time t
        X_sample = df_clean[feature_cols].iloc[i].values
        
        # Targets from t+1 to t+forecast_horizon
        y_sample = df_clean[target_col].iloc[i+1:i+1+forecast_horizon].values
        
        # Only use if we have enough future values
        if len(y_sample) == forecast_horizon:
            X_list.append(X_sample)
            y_list.append(y_sample)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y, feature_cols


class MultiOutputTimeSeriesForecaster:
    def __init__(self, forecast_horizon=5, base_model=None):
        """
        Multi-output forecaster for time series that works with pre-processed features
        
        Args:
            forecast_horizon: Number of steps to forecast ahead
            base_model: Base regression model (default: XGBoost)
        """
        self.forecast_horizon = forecast_horizon
        self.base_model = base_model
        
        if self.base_model is None:
            self.base_model = xgb.XGBRegressor(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                random_state=43
            )
        
        self.model = MultiOutputRegressor(self.base_model)
        self.feature_names = None
        
    def fit(self, X, y, feature_names):
        """
        Fit the multi-output model using pre-processed features
        """
        self.feature_names = feature_names
        self.model.fit(X, y)
        return self
    
    def predict_on_test(self, df_train_features, df_test_features, target_col='ToPredict'):
        """
        Generate predictions for test data using pre-processed features
        
        Args:
            df_train_features: Training data with features already created
            df_test_features: Test data with features already created
            target_col: Name of target column
        """
        # Combine train and test data for iterative prediction
        # We need the last part of training data to create lag features for test
        
        # Get the last few rows from training data (enough for max lag)
        max_lag = 30  # Based on your lag features
        train_tail = df_train_features.tail(max_lag).copy()
        
        # Create a combined dataset
        df_test_copy = df_test_features.copy()
        
        # Combine training tail with test data
        combined_data = pd.concat([train_tail, df_test_copy], ignore_index=True)
        combined_data = combined_data.sort_values('Dates').reset_index(drop=True)
        
        # Now we'll predict iteratively, using previous predictions for lag features
        all_predictions = []
        current_data = combined_data.copy()
        
        # Find where test data starts
        test_start_idx = len(train_tail)
        n_test_samples = len(df_test_features)
        
        for i in range(n_test_samples):
            current_idx = test_start_idx + i
            
            # Get features for current prediction (last row)
            if current_idx < len(current_data):
                last_row = current_data.iloc[current_idx]
                
                # Handle missing values in lag features (fill with last known values)
                feature_values = []
                for feat_name in self.feature_names:
                    if feat_name in last_row:
                        val = last_row[feat_name]
                        if pd.isna(val):
                            # For lag features, use the most recent prediction
                            if f'{target_col}_lag' in feat_name:
                                lag_num = int(feat_name.split('_')[-1])
                                if len(all_predictions) >= lag_num:
                                    val = all_predictions[-lag_num]
                                elif len(all_predictions) > 0:
                                    val = all_predictions[-1]
                                else:
                                    val = 0  # fallback
                            else:
                                val = 0  # fallback for other features
                        feature_values.append(val)
                    else:
                        feature_values.append(0)  # fallback
                
                features = np.array(feature_values).reshape(1, -1)
                
                # Make prediction (get first value from multi-output)
                pred = self.model.predict(features)[0][0]
                all_predictions.append(pred)
                
                # Update the combined data with this prediction
                current_data.loc[current_idx, target_col] = pred
        
        return np.array(all_predictions)


