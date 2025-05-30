import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBRegressor
import matplotlib

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


def sinusoidal_encoding(df, col, max_val):
    sin_name = f"{col}_sin"
    cos_name = f"{col}_cos"
    df[sin_name] = np.sin(2 * np.pi * df[col] / max_val)
    df[cos_name] = np.cos(2 * np.pi * df[col] / max_val)
    return df


def load_and_process_data(path):
    # Chargement des données
    df = pd.read_csv(path, parse_dates=[TIME_COLUMN])
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    # --- Nettoyage basique ---
    df = df.drop_duplicates()

    # --- Extraction de composantes temporelles ---
    df["year"] = df[TIME_COLUMN].dt.year
    df["month"] = df[TIME_COLUMN].dt.month
    df["day"] = df[TIME_COLUMN].dt.day
    df["weekday"] = df[TIME_COLUMN].dt.weekday
    df["hour"] = df[TIME_COLUMN].dt.hour
    df["dayofyear"] = df[TIME_COLUMN].dt.dayofyear

    # --- Encodage sinusoïdal cyclique ---
    df = sinusoidal_encoding(df, "month", 12)
    df = sinusoidal_encoding(df, "weekday", 7)
    df = sinusoidal_encoding(df, "dayofyear", 365)  # Approximation annuelle
    df = sinusoidal_encoding(df, "hour", 24)

    # On conserve la colonne "year" (utile parfois) et on supprime les colonnes cycliques classiques + "day" car peu utile
    df = df.drop(columns=["month", "weekday", "dayofyear", "hour", "day"])

    return df


def scale_features_targets_train(df, target_col):
    # Séparer features / target
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

    return X_scaled_df, y_scaled_series


def scale_features_test(df):
    feature_scaler = joblib.load(os.path.join(SCALER_SAVE_PATH, "feature_scaler.pkl"))

    # On enlève la colonne TIME_COLUMN si elle est présente avant transformation
    if TIME_COLUMN in df.columns:
        df_features = df.drop(columns=[TIME_COLUMN])
    else:
        df_features = df.copy()

    X_scaled = feature_scaler.transform(df_features)
    X_scaled_df = pd.DataFrame(X_scaled, columns=df_features.columns)

    # Rajouter la colonne TIME_COLUMN sans la scaler
    if TIME_COLUMN in df.columns:
        X_scaled_df[TIME_COLUMN] = df[TIME_COLUMN].values

    return X_scaled_df


import matplotlib.pyplot as plt
import numpy as np


def plot_feature_importance(model, top_n=None):
    """
    Trace un histogramme des importances des features.

    Args:
        model: modèle XGBoost entraîné (ou tout modèle avec attribut feature_importances_).
        feature_names: liste des noms des features.
        top_n: optionnel, nombre de features à afficher (les plus importantes).

    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # Tri décroissant

    if top_n:
        indices = indices[:top_n]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(indices)), importances[indices], color="skyblue", align="center")
    plt.xticks(range(len(indices)), [i for i in indices], rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.show()


def feature_selection(X, y, threshold=0.0003):
    print("Training XGB model for feature importance...")
    model = XGBRegressor(random_state=42, n_estimators=100, tree_method="hist")
    model.fit(X, y)

    importances = model.feature_importances_
    features = X.columns

    # Sélection des features au-dessus du seuil
    selected_features = features[importances > threshold]
    print(f"Selected {len(selected_features)} features out of {len(features)} based on importance > {threshold}")

    # Sauvegarder la liste des features sélectionnées
    selected_features_path = os.path.join(FEATURE_SELECTOR_PATH, "selected_features.txt")
    with open(selected_features_path, "w") as f:
        for feat in selected_features:
            f.write(f"{feat}\n")
    print(f"Selected features saved to {selected_features_path}")

    plot_feature_importance(model, top_n=100)

    return selected_features


if __name__ == "__main__":
    print("Loading and processing train data...")
    df_processed = load_and_process_data(DATA_PATH)
    print("Train data processed.")

    print("Scaling train features and target...")
    X_scaled, y_scaled = scale_features_targets_train(df_processed, TARGET)
    print("Train data scaled.")

    print("Performing feature selection...")
    selected_feats = feature_selection(X_scaled, y_scaled)
    X_scaled_selected = X_scaled[selected_feats]

    print("Loading and processing test data...")
    df_processed_test = load_and_process_data(DATA_PATH_TEST)
    print("Test data processed.")

    print("Scaling test features...")
    X_scaled_test = scale_features_test(df_processed_test)
    print("Performing feature selection...")
    selected_feats = feature_selection(X_scaled, y_scaled)
    X_scaled_selected = X_scaled[selected_feats]
    print("Test data scaled.")

    # Sauvegarde des données traitées
    os.makedirs("results/processed", exist_ok=True)
    X_scaled.to_csv("results/processed/X_scaled.csv", index=False)
    y_scaled.to_csv("results/processed/y_scaled.csv", index=False)
    X_scaled_test.to_csv("results/processed/X_scaled_test.csv", index=False)
    print("Processed data saved to 'results/processed/'")
