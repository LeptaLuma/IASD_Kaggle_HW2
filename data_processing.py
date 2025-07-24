import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("WebAgg")

# === CONFIG ===
# Data paths - modify these according to your setup
DATA_PATH = "data/raw/train.csv"
DATA_PATH_TEST = "data/raw/test.csv"

# Model configuration
TARGET = "ToPredict"
TIME_COLUMN = "Dates"

# Output paths
SCALER_SAVE_PATH = "results/scalers"
FEATURE_SELECTOR_PATH = "results/feature_selector"
PROCESSED_PATH = "results/processed"

# Create output directories
os.makedirs(SCALER_SAVE_PATH, exist_ok=True)
os.makedirs(FEATURE_SELECTOR_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)

def sinusoidal_encoding(df, col, max_val):
    sin_name = f"{col}_sin"
    cos_name = f"{col}_cos"
    df[sin_name] = np.sin(2 * np.pi * df[col] / max_val)
    df[cos_name] = np.cos(2 * np.pi * df[col] / max_val)
    return df

def extract_time_features(df):
    # Basic time components
    df["year"] = df[TIME_COLUMN].dt.year
    df["month"] = df[TIME_COLUMN].dt.month
    df["day"] = df[TIME_COLUMN].dt.day
    df["weekday"] = df[TIME_COLUMN].dt.weekday  # 0-4 for Monday-Friday
    
    # Financial-specific features
    # Quarter (financial reporting periods matter)
    df["quarter"] = df[TIME_COLUMN].dt.quarter
    
    # Week of year (some financial patterns follow weekly cycles)
    df["week"] = df[TIME_COLUMN].dt.isocalendar().week
    
    # Month start/end (often important for financial data)
    df["is_month_start"] = df[TIME_COLUMN].dt.is_month_start.astype(int)
    df["is_month_end"] = df[TIME_COLUMN].dt.is_month_end.astype(int)
    
    # Quarter start/end (earnings seasons)
    df["is_quarter_start"] = df[TIME_COLUMN].dt.is_quarter_start.astype(int)
    df["is_quarter_end"] = df[TIME_COLUMN].dt.is_quarter_end.astype(int)
    
    # Day of financial year (1-260)
    # Approximate calculation assuming continuous trading days
    df["fin_day_of_year"] = df.groupby(df[TIME_COLUMN].dt.year).cumcount() + 1
    
    # --- Cyclical encoding with appropriate periods ---
    # Weekly cycle (5 days)
    df = sinusoidal_encoding(df, "weekday", 5)
    
    # Monthly cycle (approx 21 trading days)
    df = sinusoidal_encoding(df, "day", 21)
    
    # Quarterly cycle (approx 63 trading days)
    df = sinusoidal_encoding(df, "fin_day_of_year", 65)
    
    # Annual cycle (260 trading days)
    df = sinusoidal_encoding(df, "fin_day_of_year", 260)
    
    # Week of year (52 weeks)
    df = sinusoidal_encoding(df, "week", 52)
    
    # Month of year (12 months)
    df = sinusoidal_encoding(df, "month", 12)
    
    # Quarter (4 quarters)
    df = sinusoidal_encoding(df, "quarter", 4)
    
    # Clean up intermediate columns
    df = df.drop(columns=["day", "weekday", "week", "month", "quarter"])
    
    return df

def load_and_process_data(path, is_train=True):
    df = pd.read_csv(path, parse_dates=[TIME_COLUMN])
    df = df.sort_values(TIME_COLUMN).drop_duplicates().reset_index(drop=True)

    df = extract_time_features(df)
    df = add_enhanced_calendar_features(df)  # New function

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in [TARGET]]

    # Add the new feature engineering functions
    df = add_technical_indicators(df, num_cols)
    df = add_volatility_features(df, num_cols, windows=[5, 21, 63])
    df = add_mean_reversion_momentum(df, num_cols, windows=[5, 21, 63])

    # Your existing transformations
    df = log_diff_transform(df, num_cols)
    df = create_lag_features(df, num_cols, lags=[1, 2, 5])
    df = create_rolling_features(df, num_cols, windows=[3, 5])

    # Handle missing and infinite values
    df = replace_inf_values(df)
    df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)

    if is_train and TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' missing in training data.")

    return df

def scale_features_targets_train(df, target_col):
    # Séparer features / target
    Dates = df[TIME_COLUMN]
    X = df.drop(columns=[target_col, TIME_COLUMN])
    y = df[[target_col]]
    
    # One last check for infinities
    X = replace_inf_values(X).fillna(0)
    y = replace_inf_values(y).fillna(0)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Fit sur train uniquement
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y)

    # Sauvegarder scalers
    joblib.dump(feature_scaler, os.path.join(SCALER_SAVE_PATH, "feature_scaler.pkl"))
    joblib.dump(target_scaler, os.path.join(SCALER_SAVE_PATH, "target_scaler.pkl"))
    print(f"Feature scaler and target scaler saved in '{SCALER_SAVE_PATH}'")

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    y_scaled_series = pd.Series(y_scaled.flatten(), name=target_col)

    return X_scaled_df, y_scaled_series, Dates.reset_index(drop=True)

def scale_features_test(df):
    feature_scaler = joblib.load(os.path.join(SCALER_SAVE_PATH, "feature_scaler.pkl"))
    dates = df[TIME_COLUMN]
    df_features = df.drop(columns=[TIME_COLUMN], errors="ignore")
    X_scaled = feature_scaler.transform(df_features)
    X_scaled = pd.DataFrame(X_scaled, columns=df_features.columns)

    return X_scaled, dates.reset_index(drop=True)

def feature_selection(X, y, threshold=0.001):
    model = XGBRegressor(random_state=42, n_estimators=100, tree_method="hist")
    model.fit(X, y)

    importances = model.feature_importances_
    selected_features = X.columns[importances > threshold]

    with open(os.path.join(FEATURE_SELECTOR_PATH, "selected_features.txt"), "w") as f:
        f.writelines([f"{feat}\n" for feat in selected_features])

    # plot_feature_importance(model, X.columns, top_n=30)
    return selected_features

def create_lag_features(df, cols, lags=[1, 3, 7]):
    for col in cols:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df

def create_rolling_features(df, cols, windows=[3, 7]):
    for col in cols:
        for window in windows:
            df[f"{col}_roll_mean_{window}"] = df[col].rolling(window=window).mean()
            df[f"{col}_roll_std_{window}"] = df[col].rolling(window=window).std()
    return df

def log_diff_transform(df, cols):
    for col in cols:
        # Ajout d'une petite constante pour éviter log(0)
        df[f"{col}_log_diff"] = np.log(df[col] + 1e-6).diff()
    return df

def add_technical_indicators(df, cols):
    """Add common technical indicators for financial data."""
    for col in cols:
        # RSI (Relative Strength Index) - 14 day standard period
        delta = df[col].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        # Add epsilon to avoid division by zero
        rs = avg_gain / (avg_loss + 1e-8)
        df[f"{col}_rsi_14"] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        ema12 = df[col].ewm(span=12, adjust=False).mean()
        ema26 = df[col].ewm(span=26, adjust=False).mean()
        df[f"{col}_macd"] = ema12 - ema26
        df[f"{col}_macd_signal"] = df[f"{col}_macd"].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands (20 days is standard)
        rolling_mean = df[col].rolling(window=20).mean()
        rolling_std = df[col].rolling(window=20).std()
        df[f"{col}_bb_upper"] = rolling_mean + (rolling_std * 2)
        df[f"{col}_bb_lower"] = rolling_mean - (rolling_std * 2)
        # Add epsilon to avoid division by zero
        df[f"{col}_bb_width"] = (df[f"{col}_bb_upper"] - df[f"{col}_bb_lower"]) / (rolling_mean + 1e-8)
        
        # Rate of Change - 5, 21 day (week, month)
        df[f"{col}_roc_5"] = df[col].pct_change(periods=5) * 100
        df[f"{col}_roc_21"] = df[col].pct_change(periods=21) * 100
        
    return df

def add_volatility_features(df, cols, windows=[5, 21, 63]):
    """Add various volatility-based features using trading-day windows."""
    for col in cols:
        # Historical volatility (annualized)
        for window in windows:
            # Daily returns
            returns = df[col].pct_change()
            # Rolling volatility (annualized by sqrt(252) for trading days)
            df[f"{col}_volatility_{window}d"] = returns.rolling(window=window).std() * np.sqrt(252)
        
        # Volatility ratios (short-term vs long-term)
        if len(windows) >= 2:
            for i in range(len(windows)-1):
                short_window = windows[i]
                long_window = windows[i+1]
                short_vol = df[f"{col}_volatility_{short_window}d"]
                long_vol = df[f"{col}_volatility_{long_window}d"]
                # Add epsilon to denominator to avoid division by zero
                df[f"{col}_vol_ratio_{short_window}_{long_window}"] = short_vol / (long_vol + 1e-8)
    
    return df

def add_mean_reversion_momentum(df, cols, windows=[5, 21, 63]):
    """Add mean reversion and momentum indicators."""
    for col in cols:
        # Distance from moving averages in standard deviation units
        for window in windows:
            rolling_mean = df[col].rolling(window=window).mean()
            rolling_std = df[col].rolling(window=window).std()
            # Add epsilon to avoid division by zero
            df[f"{col}_zscore_{window}d"] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
        
        # Consecutive movement tracking
        df[f"{col}_direction"] = np.sign(df[col].diff())
        df[f"{col}_streak"] = (df[f"{col}_direction"] != df[f"{col}_direction"].shift(1)).cumsum()
        
        # Fix: Fill NAs before converting to int
        up_condition = ((df[f"{col}_direction"] > 0) & 
                       (df[f"{col}_direction"].shift(1) > 0) & 
                       (df[f"{col}_direction"].shift(2) > 0))
        df[f"{col}_up_days"] = up_condition.fillna(False).astype(int)
        
        down_condition = ((df[f"{col}_direction"] < 0) & 
                         (df[f"{col}_direction"].shift(1) < 0) & 
                         (df[f"{col}_direction"].shift(2) < 0))
        df[f"{col}_down_days"] = down_condition.fillna(False).astype(int)
    
    return df

def add_enhanced_calendar_features(df, time_col='Dates'):
    """Add enhanced trading calendar features."""
    
    # First/last week of month indicators (often have different behavior)
    df['day_of_month'] = df[time_col].dt.day
    df['trading_day_of_month'] = df.groupby([df[time_col].dt.year, 
                                           df[time_col].dt.month]).cumcount() + 1
    df['is_first_week'] = (df['trading_day_of_month'] <= 5).astype(int)
    df['is_last_week'] = (df['trading_day_of_month'] >= 16).astype(int)
    
    # Day of week effects
    df['is_monday'] = (df[time_col].dt.dayofweek == 0).astype(int)
    df['is_friday'] = (df[time_col].dt.dayofweek == 4).astype(int)
    
    # Month-based seasonality
    df['is_january'] = (df[time_col].dt.month == 1).astype(int)
    df['is_december'] = (df[time_col].dt.month == 12).astype(int)
    df['is_quarter_end_month'] = df[time_col].dt.month.isin([3, 6, 9, 12]).astype(int)
    
    # Trading day of year percentile (normalized position in trading year)
    df['trading_day_of_year'] = df.groupby(df[time_col].dt.year).cumcount() + 1
    df['trading_year_percentile'] = df['trading_day_of_year'] / 260
    
    return df

def replace_inf_values(df):
    """Replace infinity values with NaN, which can then be handled by fillna."""
    return df.replace([np.inf, -np.inf], np.nan)

if __name__ == "__main__":
    print("🔄 Loading and processing training data...")
    df_train = load_and_process_data(DATA_PATH)
    print("✅ Train data ready.")

    print("🔄 Scaling features and target...")
    X_train, y_train, dates_train = scale_features_targets_train(df_train, TARGET)

    print("🔍 Selecting top features...")
    selected_feats = feature_selection(X_train, y_train)
    X_train_selected = X_train[selected_feats]
    print(f"The number of selected features: {len(selected_feats)} / {len(X_train.columns)}")

    print("🔄 Loading and processing test data...")
    df_test = load_and_process_data(DATA_PATH_TEST, is_train=False)

    print("🔄 Scaling test features...")
    X_test, dates_test = scale_features_test(df_test)
    X_test_selected = X_test[selected_feats]

    # X_train_selected[TIME_COLUMN] = dates_train
    X_test_selected[TIME_COLUMN] = dates_test

    print("💾 Saving processed datasets...")
    X_train_selected.to_csv(os.path.join(PROCESSED_PATH, "X_scaled.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_PATH, "y_scaled.csv"), index=False)
    X_test_selected.to_csv(os.path.join(PROCESSED_PATH, "X_scaled_test.csv"), index=False)
    print("✅ All done.")
