import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import randint, uniform
import lightgbm as lgbm
from sklearn.linear_model import Ridge

from training_utils_1 import MultiOutputTimeSeriesForecaster, create_features, prepare_multioutput_data


# Configuration for hyperparameter tuning
TUNE_HYPERPARAMS = True  # Set to False to skip tuning
N_ITER_SEARCH = 40  # Number of parameter settings sampled in RandomizedSearchCV
CV_SPLITS = 3      # Number of cross-validation splits for tuning

class StackingTimeSeriesForecaster:
    def __init__(self, base_models, meta_model, forecast_horizon=5):
        """
        Stacking ensemble for time series forecasting
        
        Args:
            base_models: List of base models (first level)
            meta_model: Model to combine base model predictions (second level)
            forecast_horizon: Number of steps to forecast ahead
        """
        self.base_models = base_models
        self.meta_model = meta_model    # Meta-model that will learn to combine the predictions
        self.forecast_horizon = forecast_horizon
        
    def fit(self, df_train_features, X_train, y_train, feature_names, target_col='ToPredict'):
        """
        Train the stacking ensemble in two steps:
        1. Train base models using cross-validation to get out-of-fold predictions
        2. Train meta-model on these predictions
        
        Args:
            df_train_features: DataFrame with features for cross-validation
            X_train: Array of training features for final model training
            y_train: Array of target values for final model training
            feature_names: List of feature column names
            target_col: Name of the target column
        """
        print("Training base models...")
        
        # Create time series cross-validation folds
        tscv = TimeSeriesSplit(n_splits=5)
        df_train_features = df_train_features.copy()
        
        # Placeholder for meta-features (predictions from base models)
        meta_features = np.zeros((len(df_train_features), len(self.base_models)))
        
        # For each fold
        for fold, (train_idx, val_idx) in enumerate(tscv.split(df_train_features)):
            print(f"  Processing fold {fold+1}/5 for stacking")
            
            # Split data
            fold_train = df_train_features.iloc[train_idx].copy()
            fold_val = df_train_features.iloc[val_idx].copy()
            
            # Prepare fold training data
            X_fold, y_fold, fold_feature_names = prepare_multioutput_data(
                fold_train, 
                forecast_horizon=self.forecast_horizon, 
                target_col=target_col
            )
            
            # Train each base model on this fold
            for i, model in enumerate(self.base_models):
                model.fit(X_fold, y_fold, fold_feature_names)
                
                # Get predictions for validation fold
                val_predictions = model.predict_on_test(fold_train, fold_val, target_col)
            
                # Calculate and print MSE for this model on this fold
                fold_actual = fold_val[target_col].values
                fold_mse = mean_squared_error(fold_actual, val_predictions)
                print(f"    Model {type(model.base_model).__name__} - MSE: {fold_mse:.6f}")
                
                # Store predictions as meta-features (for the validation indices)
                # Only use the first prediction for each sample as meta-feature
                if len(val_idx) == len(val_predictions):
                    meta_features[val_idx, i] = val_predictions
        
        # Now train the full base models on the entire dataset using the provided parameters
        for model in self.base_models:
            model.fit(X_train, y_train, feature_names)
        
        # Train the meta-model using the out-of-fold predictions
        y_meta = df_train_features[target_col].values
        
        # Filter out rows with NaN meta-features
        valid_indices = ~np.isnan(meta_features).any(axis=1)
        
        self.meta_model.fit(meta_features[valid_indices], y_meta[valid_indices])
        
        print("Stacking ensemble training complete.")
        return self
    
    def predict_on_test(self, df_train_features, df_test_features, target_col='ToPredict'):
        """
        Generate predictions for test data by:
        1. Getting predictions from each base model
        2. Using these as features for the meta-model
        """
        # Get predictions from each base model
        base_predictions = np.zeros((len(df_test_features), len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            preds = model.predict_on_test(df_train_features, df_test_features, target_col)
            base_predictions[:, i] = preds
        
        # Use meta-model to generate final predictions
        final_predictions = self.meta_model.predict(base_predictions)
        
        return final_predictions


def tune_hyperparameters(model_name, base_model_type, X_tune, y_tune):
    """
    Perform hyperparameter tuning using RandomizedSearchCV with pre-processed features
    
    Args:
        model_name: Name of the model for logging
        base_model_type: Base model type to tune
        X_tune: Pre-processed features for tuning
        y_tune: Target values for tuning
        
    Returns:
        Best hyperparameters for the model
    """
    print(f"Tuning hyperparameters for {model_name}...")
    
    param_grid = None
    
    # Define parameter search spaces for different model types
    if model_name == "xgboost":
        param_grid = {
            'learning_rate': uniform(0.001, 0.1),
            'n_estimators': randint(50, 300),
            'max_depth': randint(3, 10),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'min_child_weight': randint(1, 10)
        }
    elif model_name == "random_forest":
        param_grid = {
            'n_estimators': randint(50, 300),
            'max_depth': [15, 20, 25, 30],
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None]
        }
    # elif model_name == "lightgbm":
    #     param_grid = {
    #         'learning_rate': uniform(0.01, 0.1),
    #         'n_estimators': randint(100, 150),
    #         'max_depth': randint(3, 8),
    #         'num_leaves': randint(20, 50),
    #         'subsample': uniform(0.7, 0.3),
    #         'colsample_bytree': uniform(0.7, 0.3),
    #     }
    elif model_name == "ridge":
        param_grid = {
            'alpha': uniform(0.001, 10),
            'fit_intercept': [True, False],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr']
        }
    
    if param_grid is None:
        print(f"No parameter grid defined for {model_name}. Skipping tuning.")
        return base_model_type
    
    # Define the randomized search
    random_search = RandomizedSearchCV(
        estimator=base_model_type,
        param_distributions=param_grid,
        n_iter=N_ITER_SEARCH,
        cv=CV_SPLITS,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )
    
    # Fit the randomized search
    try:
        random_search.fit(X_tune, y_tune)
        
        # Print best parameters and score
        print(f"Best parameters for {model_name}: {random_search.best_params_}")
        print(f"Best MSE for {model_name}: {-random_search.best_score_:.6f}")
        
        # Update the base model with best parameters
        best_params = random_search.best_params_
        best_model = base_model_type.set_params(**best_params)

        return best_model
    
    except Exception as e:
        print(f"Error during hyperparameter tuning for {model_name}: {e}")
        print(f"Using default parameters for {model_name}")
        return base_model_type


def get_base_models(X_tune=None, y_tune=None):
    """
    Define and configure base models
    If TUNE_HYPERPARAMS is True, perform hyperparameter tuning
    
    Returns:
        List of configured MultiOutputTimeSeriesForecaster models
    """
    # Define base model configurations
    base_models_config = [
        # You can uncomment or add more models here
        {
            "name": "xgboost",
            "model": xgb.XGBRegressor(
                # n_estimators=150, 
                # max_depth=4, 
                # learning_rate=0.05, 
                random_state=42,
            ),
            "tune": True
        },
        {
            "name": "random_forest",
            "model": RandomForestRegressor(
                # n_estimators=100, 
                # max_depth=10, 
                random_state=42,
            ),
            "tune": True
        },
        # {
        #     "name": "lightgbm",
        #     "model": lgbm.LGBMRegressor(
        #         # n_estimators=100,
        #         # max_depth=5,
        #         # learning_rate=0.05,
        #         random_state=42,    
        #         min_child_samples=20,         # Add this parameter
        #         min_child_weight=1e-3,        # Add this parameter
        #         min_data_in_leaf=10,          # Add this parameter
        #         # verbose=-1,                   # Reduce verbosity
        #         reg_alpha=0.1,                # Add L1 regularization
        #         reg_lambda=0.1,               # Add L2 regularization
        #         objective="regression",
        #     ),
        #     "tune": True
        # },
    ]

    # Create models with tuned hyperparameters if needed
    forecaster_models = []
    
    for config in base_models_config:
        if TUNE_HYPERPARAMS and config["tune"] and X_tune is not None and y_tune is not None:
            # Tune hyperparameters
            tuned_base_model = tune_hyperparameters(
                config["name"], 
                config["model"],
                X_tune,
                y_tune
            )
        else:
            # Use default hyperparameters
            tuned_base_model = config["model"]
        
        # Create MultiOutputTimeSeriesForecaster with the (possibly tuned) base model
        forecaster = MultiOutputTimeSeriesForecaster(
            forecast_horizon=5,
            base_model=tuned_base_model
        )
        
        forecaster_models.append(forecaster)
    
    return forecaster_models


def get_meta_model(X_tune=None, y_tune=None):
    """
    Define and configure meta model
    If TUNE_HYPERPARAMS is True, perform hyperparameter tuning
    
    Returns:
        Configured meta model
    """
    meta_model_config = {
        "name": "ridge",
        "model": Ridge(alpha=1.0, random_state=44),
        "tune": True
    }
    
    if TUNE_HYPERPARAMS and meta_model_config["tune"] and X_tune is not None and y_tune is not None:
        # Tune hyperparameters
        tuned_meta_model = tune_hyperparameters(
            meta_model_config["name"], 
            meta_model_config["model"],
            X_tune,
            y_tune
        )
    else:
        # Use default hyperparameters
        tuned_meta_model = meta_model_config["model"]
    
    return tuned_meta_model


def main(use_stacking=True):
    # Load raw data
    print("Loading and preprocessing data...")
    df_train = pd.read_csv("data/raw/train.csv")
    df_test = pd.read_csv("data/raw/test.csv")
    
    # Create features once
    print("Creating features for train and test data...")
    df_test['ToPredict'] = np.nan  # Add placeholder for target
    df_train_features, df_test_features = create_features(df_train, df_test)
    
    # Prepare data for training models
    print("Preparing training data for models...")
    X_train, y_train, feature_names = prepare_multioutput_data(
        df_train_features, 
        forecast_horizon=5,
        target_col='ToPredict'
    )
    
    # Create a small sample for hyperparameter tuning
    if TUNE_HYPERPARAMS:
        print("Preparing data for hyperparameter tuning...")
        # Use only a portion of the data for tuning to speed things up
        sample_size = min(1000, len(X_train))
        X_tune = X_train[:sample_size]
        y_tune_temp = y_train[:sample_size]
        # Use only first prediction horizon for tuning
        y_tune = y_tune_temp[:, 0]
    else:
        X_tune, y_tune = None, None
    
    # Get tuned base models
    print("\n=== Configuring base models ===")
    base_models = get_base_models(X_tune, y_tune)
    
    # Get tuned meta model
    print("\n=== Configuring meta model ===")
    meta_model = get_meta_model(X_tune, y_tune)
    
    # Create and train stacking ensemble
    print("\n=== Training stacking ensemble ===")
    stacking_model = StackingTimeSeriesForecaster(
        base_models=base_models, 
        meta_model=meta_model,  
        forecast_horizon=5
    )

    # Make sure to pass all required parameters explicitly
    stacking_model.fit(df_train_features, X_train, y_train, feature_names) 

    # Make predictions on test data using stacking model
    test_predictions = stacking_model.predict_on_test(df_train_features, df_test_features)

    # Create submission dataframe
    submission = pd.DataFrame({
        'Dates': df_test_features['Dates'],
        'ToPredict': test_predictions
    })
    
    submission = submission.rename({"Dates": "ID"}, axis='columns')

    submission.to_csv("submissions/submission_6_3.csv", index=False)
    
    return stacking_model, test_predictions, submission


if __name__ == "__main__":
    start_time = time.time()

    # Set use_stacking=True to use stacking ensemble, False to use single model
    forecaster, predictions, submission = main(use_stacking=True)

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")