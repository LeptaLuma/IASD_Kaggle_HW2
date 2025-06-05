# train_xgboost_optuna.py

import os
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import optuna
import joblib
import logging
import datetime
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from torch.utils.tensorboard import SummaryWriter
import time

# === CONFIG ===
# DATA_PATH_X = (
#     "/home/paul/Dropbox/OrdinateurPaul/Documents/Scolaire/IASD/S2/Become a Kaggle master/IASD_Kaggle_HW2//results/processed/X_scaled.csv"
# )
# DATA_PATH_Y = (
#     "/home/brice/Documents/PERSONAL_PROJECT/IASD_Kaggle/IASD_Kaggle_HW2/IASD_Kaggle_HW2/results/processed/y_scaled.csv"
# )
DATA_PATH_X = (
    "/home/paul/Dropbox/OrdinateurPaul/Documents/Scolaire/IASD/S2/Become a Kaggle master/IASD_Kaggle_HW2/results/processed/X_scaled.csv"
)
DATA_PATH_Y = (
    "/home/paul/Dropbox/OrdinateurPaul/Documents/Scolaire/IASD/S2/Become a Kaggle master/IASD_Kaggle_HW2/results/processed/y_scaled.csv"
)
# test_path = "/home/brice/Documents/PERSONAL_PROJECT/IASD_Kaggle/IASD_Kaggle_HW2/IASD_Kaggle_HW2/results/processed/X_scaled_test.csv"
test_path = "/home/paul/Dropbox/OrdinateurPaul/Documents/Scolaire/IASD/S2/Become a Kaggle master/IASD_Kaggle_HW2/results/processed/X_scaled_test.csv"
TARGET = "ToPredict"
TIME_COLUMN = "Dates"
N_SPLITS = 5
ID = "11.4"
load_study = False

start_time = time.time()

target_scaler_path = os.path.join("results", "scalers", "target_scaler.pkl")
target_scaler = joblib.load(target_scaler_path)


# === LOGGING SETUP ===
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/train_{ID}.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# === TENSORBOARD SETUP ===
tb_log_dir = f"logs/tensorboard/{ID}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=tb_log_dir)

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH_X)
y_df = pd.read_csv(DATA_PATH_Y)

# === FEATURES ===
features = df.columns.tolist()
X = df[features]
y = y_df[TARGET]

# === CROSS-VALIDATION SETUP ===
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# === OPTUNA STUDY ===
# cat_params = {'iterations': 125, 'depth': 5, 'learning_rate': 0.13399486522400317, 'l2_leaf_reg': 5.621807360933901, 'random_strength': 0.5463101832971233, 'bagging_temperature': 2.869858080198312, "random_seed": 42, "verbose": False}
cat_params = {'iterations': 262, 'depth': 3, 'learning_rate': 0.10897193844903408, 'l2_leaf_reg': 6.0861689401040895, 'random_strength': 1.1914046341614943, 'bagging_temperature': 4.404613169915849}
cat = CatBoostRegressor(**cat_params)

# rf_params = {'n_estimators': 308, 'max_depth': 9, 'min_samples_split': 3, 'min_samples_leaf': 11, 'max_features': 0.7958436250975505, "random_state": 42}
rf_params = {'n_estimators': 296, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 11, 'max_features': 0.6727653863174257}
rf = RandomForestRegressor(**rf_params)

# xgb_params = {'n_estimators': 582, 'max_depth': 13, 'learning_rate': 0.03490976740622871, 'subsample': 0.8209897943101099, 'colsample_bytree': 0.751155885375093, 'gamma': 1.4447501196954968, 'min_child_weight': 9, "random_state": 42, "tree_method": "hist"}
xgb_params = {'n_estimators': 563, 'max_depth': 14, 'learning_rate': 0.012278117143194565, 'subsample': 0.7509328181622928, 'colsample_bytree': 0.6475622787298094, 'gamma': 1.005277295513741, 'min_child_weight': 12}
xgb = XGBRegressor(**xgb_params)


# Add this after loading the existing models but before creating the stacker
ridge_params = {'alpha': 0.896002688408722, 'fit_intercept': False, 'solver': 'cholesky'}
# ridge_params = {'alpha': 1.3838107184105972, 'fit_intercept': False, 'solver': 'svd'}


# Create stacked ensemble with both models
stacker = StackingRegressor(
    estimators=[
        ("xgb", xgb),
        ("cat", cat),
        ("rf", rf),
    ],
    final_estimator=Ridge(**ridge_params),
    n_jobs=-1,
)

stacker.fit(X, y)
logging.info("Final model trained with ensemble of XGBoost, RandomForest and CatBoost.")

# Final model evaluation metrics
mse_scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    preds = stacker.predict(X_val)

    y_val_orig = target_scaler.inverse_transform(y_val.values.reshape(-1, 1)).flatten()
    preds_orig = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_val_orig, preds_orig)

    mse_scores.append(mse)
    logging.info(f"Final fold number {fold} mse: {mse}")
    writer.add_scalar("Final_Fold/MSE", mse, fold)

# Save the model
model_path = os.path.join(os.getcwd(), "results", "models", f"final_xgb_cat_model_{ID}.joblib")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(stacker, model_path)
print(f"Final model saved to {model_path}")

# === PREDICTIONS ===
df_test = pd.read_csv(test_path, parse_dates=[TIME_COLUMN])
X_test = df_test[features]
predictions = stacker.predict(X_test)

predictions_orig = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
submission = pd.DataFrame({"ID": df_test["Dates"], "ToPredict": predictions_orig})

os.makedirs("results/submissions", exist_ok=True)
submission_path = os.path.join("results", "submissions", f"submission_ens_{ID}.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
    
print(f"Total execution time: {time.time() - start_time:.2f} seconds")

writer.close()