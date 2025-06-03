import os
from re import X
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBRegressor
import matplotlib

import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("WebAgg")

# === CONFIG ===
DATA_PATH = "/home/brice/Documents/PERSONAL_PROJECT/IASD_Kaggle/IASD_Kaggle_HW2/IASD_Kaggle_HW2/data/raw/train.csv"
DATA_PATH_TEST = "/home/brice/Documents/PERSONAL_PROJECT/IASD_Kaggle/IASD_Kaggle_HW2/IASD_Kaggle_HW2/data/raw/test.csv"
TARGET = "ToPredict"
TIME_COLUMN = "Dates"
SCALER_SAVE_PATH = "results/scalers"
os.makedirs(SCALER_SAVE_PATH, exist_ok=True)

FEATURE_SELECTOR_PATH = "results/feature_selector"
os.makedirs(FEATURE_SELECTOR_PATH, exist_ok=True)
PROCESSED_PATH = "results/processed"


def sinusoidal_encoding(df, col, max_val):
    sin_name = f"{col}_sin"
    cos_name = f"{col}_cos"
    df[sin_name] = np.sin(2 * np.pi * df[col] / max_val)
    df[cos_name] = np.cos(2 * np.pi * df[col] / max_val)
    return df


def extract_time_features(df):
    df["year"] = df[TIME_COLUMN].dt.year
    df["month"] = df[TIME_COLUMN].dt.month
    df["day"] = df[TIME_COLUMN].dt.day
    df["weekday"] = df[TIME_COLUMN].dt.weekday
    df["hour"] = df[TIME_COLUMN].dt.hour
    df["dayofyear"] = df[TIME_COLUMN].dt.dayofyear

    # Encodage cyclique
    df = sinusoidal_encoding(df, "month", 12)
    df = sinusoidal_encoding(df, "weekday", 7)
    df = sinusoidal_encoding(df, "dayofyear", 365)
    df = sinusoidal_encoding(df, "hour", 24)

    # Suppression des colonnes moins pertinentes
    return df.drop(columns=["month", "weekday", "dayofyear", "hour", "day"])


def load_and_process_data(path, is_train=True):
    df = pd.read_csv(path, parse_dates=[TIME_COLUMN])
    df = df.sort_values(TIME_COLUMN).drop_duplicates().reset_index(drop=True)

    df = extract_time_features(df)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in [TARGET]]

    # 1) Stationnarisation par log-diff (si pertinent)
    df = log_diff_transform(df, num_cols)

    # 2) Création des lags sur variables originales (avant diff)
    df = create_lag_features(df, num_cols, lags=[1, 3, 7])

    # 3) Création des statistiques rolling
    df = create_rolling_features(df, num_cols, windows=[3, 7])

    # Après création des nouvelles features, imputation des NaN
    # Pour les lags et rolling, on peut faire un forward fill suivi d'un backward fill
    df = df.fillna(method="ffill").fillna(method="bfill")

    # Optionnel : si certains NaN persistent (ex: au tout début), on peut aussi remplir par 0 ou une autre valeur par défaut
    df = df.fillna(0)

    # On supprime les colonnes constantes
    # nunique = df.nunique()
    # constant_cols = nunique[nunique <= 1].index
    # df = df.drop(columns=constant_cols)

    if is_train and TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' missing in training data.")

    return df


def scale_features_targets_train(df, target_col):
    # Séparer features / target
    Dates = df[TIME_COLUMN]
    X = df.drop(columns=[target_col, TIME_COLUMN])
    y = df[[target_col]]

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


def plot_feature_importance(model, feature_names, top_n=None):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    if top_n:
        indices = indices[:top_n]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(indices)), importances[indices], color="skyblue")
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()


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

    X_train_selected[TIME_COLUMN] = dates_train
    X_test_selected[TIME_COLUMN] = dates_test

    print("💾 Saving processed datasets...")
    X_train_selected.to_csv(os.path.join(PROCESSED_PATH, "X_scaled.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_PATH, "y_scaled.csv"), index=False)
    X_test_selected.to_csv(os.path.join(PROCESSED_PATH, "X_scaled_test.csv"), index=False)
    print("✅ All done.")
