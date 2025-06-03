# train_xgboost_optuna.py

import os
from matplotlib import dates
import pandas as pd
import numpy as np
import optuna
import joblib
import logging
import datetime
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from torch.utils.tensorboard import SummaryWriter

# === CONFIG ===
DATA_PATH_X = (
    "/home/brice/Documents/PERSONAL_PROJECT/IASD_Kaggle/IASD_Kaggle_HW2/IASD_Kaggle_HW2/results/processed/X_scaled.csv"
)
DATA_PATH_Y = (
    "/home/brice/Documents/PERSONAL_PROJECT/IASD_Kaggle/IASD_Kaggle_HW2/IASD_Kaggle_HW2/results/processed/y_scaled.csv"
)
test_path = "/home/brice/Documents/PERSONAL_PROJECT/IASD_Kaggle/IASD_Kaggle_HW2/IASD_Kaggle_HW2/results/processed/X_scaled_test.csv"
TARGET = "ToPredict"
TIME_COLUMN = "Dates"
N_SPLITS = 5
ID = "2.0"
load_study = False
n_trials = 100

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
dates_train = df[TIME_COLUMN]

features = df.columns.tolist()
X = df[features]
X.drop(columns=[TIME_COLUMN], inplace=True)
y = y_df[TARGET].drop(columns=[TIME_COLUMN])

# === CROSS-VALIDATION SETUP ===
tscv = TimeSeriesSplit(n_splits=N_SPLITS)


# === OBJECTIVE FUNCTION ===
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42,
        "tree_method": "hist",
    }

    mse_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        preds = model.predict(X_val)
        # Descale y_val et preds
        y_val_orig = target_scaler.inverse_transform(y_val.values.reshape(-1, 1)).flatten()
        preds_orig = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

        mse = mean_squared_error(y_val_orig, preds_orig)

        mse_scores.append(mse)
        writer.add_scalar(f"Fold_{fold}/MSE", mse, trial.number)

    mean_mse = np.mean(mse_scores)
    writer.add_scalar("CV/Mean_MSE", mean_mse, trial.number)
    logging.info(f"Trial {trial.number} - Params: {params} - MSE: {mean_mse:.4f}")

    return mean_mse


# === OPTUNA STUDY ===
study_path = os.path.join(os.getcwd(), "results", "studies", f"study_xgb_{ID}.pkl")
if load_study:
    if os.path.exists(study_path):
        study = joblib.load(study_path)
        print("Loaded existing study.")
    else:
        raise FileNotFoundError(f"Study file not found at {study_path}. Please check the path or create a new study.")
else:
    print("Creating a new study.")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

print("Best trial:")
print(study.best_trial)

# Save the study results
os.makedirs(os.path.dirname(study_path), exist_ok=True)
joblib.dump(study, study_path)
print(f"Study saved to {study_path}")

# === FINAL MODEL TRAINING ===
best_params = study.best_trial.params
best_params.update({"random_state": 42, "tree_method": "hist"})

# Light ensembling with Ridge
xgb = XGBRegressor(**best_params)
stacker = StackingRegressor(estimators=[("xgb", xgb)], final_estimator=Ridge(alpha=1.0))

stacker.fit(X, y)
logging.info("Final model trained with ensembling.")

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
    writer.add_scalar("Final_Fold/MSE", mse, fold)

# Save the model
model_path = os.path.join(os.getcwd(), "results", "models", f"final_xgb_model_{ID}.joblib")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(stacker, model_path)
print(f"Final model saved to {model_path}")

# === PREDICTIONS ===
df_test = pd.read_csv(test_path, parse_dates=[TIME_COLUMN])
dates_test = df_test[TIME_COLUMN]

X_test = df_test[features]
X_test.drop(columns=[TIME_COLUMN], inplace=True, errors="ignore")
predictions = stacker.predict(X_test)

predictions_orig = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
submission = pd.DataFrame({"ID": dates_test, "ToPredict": predictions_orig})

os.makedirs("results/submissions", exist_ok=True)
submission_path = os.path.join("results", "submissions", f"submission_xgb_{ID}.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

writer.close()
