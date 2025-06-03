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
ID = "7.4"
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


# # === OBJECTIVE FUNCTION ===
# def objective(trial):
#     # XGBoost parameters
#     xgb_params = {
#         "n_estimators": trial.suggest_int("xgb_n_estimators", 300, 700),
#         "max_depth": trial.suggest_int("xgb_max_depth", 5, 12),
#         "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.1),
#         "subsample": trial.suggest_float("xgb_subsample", 0.2, 1.0),
#         "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
#         "gamma": trial.suggest_float("xgb_gamma", 0, 5),
#         "min_child_weight": trial.suggest_int("xgb_min_child_weight", 5, 12),
#         "random_state": 42,
#         "tree_method": "hist",
#     }
    
#     # RandomForest parameters
#     rf_params = {
#         "n_estimators": trial.suggest_int("rf_n_estimators", 200, 500),
#         "max_depth": trial.suggest_int("rf_max_depth", 5, 20),
#         "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10),
#         "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
#         "max_features": trial.suggest_float("rf_max_features", 0.1, 1.0),
#         "random_state": 42,
#     }

#     # CatBoost parameters
#     cat_params = {
#         "iterations": trial.suggest_int("cat_iterations", 100, 300),
#         "depth": trial.suggest_int("cat_depth", 4, 10),
#         "learning_rate": trial.suggest_float("cat_learning_rate", 0.05, 0.2),
#         "l2_leaf_reg": trial.suggest_float("cat_l2_leaf_reg", 5, 10),
#         "random_strength": trial.suggest_float("cat_random_strength", 1, 10),
#         "bagging_temperature": trial.suggest_float("cat_bagging_temperature", 5, 10),
#         "random_seed": 42,
#         "verbose": False
#     }
    
#     # Reduce number of folds
#     reduced_tscv = TimeSeriesSplit(n_splits=3)  # Reduce from 5 to 2 folds

#     mse_scores_stacked = []
    
#     for fold, (train_idx, val_idx) in enumerate(reduced_tscv.split(X)):
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#         # Train stacked model directly (skip individual model evaluation)
#         stacker = StackingRegressor(
#             estimators=[
#                 ("xgb", XGBRegressor(**xgb_params)),
#                 # ("rf", RandomForestRegressor(**rf_params)),
#                 ("cat", CatBoostRegressor(**cat_params))
#             ],
#             final_estimator=Ridge(alpha=1.0)
#         )
#         stacker.fit(X_train, y_train)
        
#         # Evaluate only the stacked model
#         preds = stacker.predict(X_val)
#         y_val_orig = target_scaler.inverse_transform(y_val.values.reshape(-1, 1)).flatten()
#         preds_orig = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        
#         mse = mean_squared_error(y_val_orig, preds_orig)
#         mse_scores_stacked.append(mse)
        
#         # Minimal logging
#         logging.info(f"Fold_{fold} mse: {mse}")
#         writer.add_scalar(f"Fold_{fold}/Stacked_MSE", mse, trial.number)

#     mean_mse_stacked = np.mean(mse_scores_stacked)
#     writer.add_scalar("CV/Stacked_Mean_MSE", mean_mse_stacked, trial.number)
#     logging.info(f"Trial {trial.number} - Stacked MSE: {mean_mse_stacked:.4f}")

#     return mean_mse_stacked

# === OPTUNA STUDY ===
study_path_1 = os.path.join(os.getcwd(), "results", "studies", f"study_xgb_cat_6.0.pkl")
study_1 = joblib.load(study_path_1)
print(study_1.best_trial)
os.makedirs(os.path.dirname(study_path_1), exist_ok=True)
joblib.dump(study_1, study_path_1)
print(f"Study saved to {study_path_1}")
best_params_1 = study_1.best_trial.params
xgb_params_1 = {k.replace('xgb_', ''): v for k, v in best_params_1.items() if k.startswith('xgb_')}
cat_params_1 = {k.replace('cat_', ''): v for k, v in best_params_1.items() if k.startswith('cat_')}
xgb_params_1.update({"random_state": 42, "tree_method": "hist"})
cat_params_1.update({"random_seed": 42, "verbose": False})
xgb_1 = XGBRegressor(**xgb_params_1)
cat_1 = CatBoostRegressor(**cat_params_1)


study_path_2 = os.path.join(os.getcwd(), "results", "studies", f"study_xgb_3.0.pkl")
study_2 = joblib.load(study_path_2)
print(study_2.best_trial)
os.makedirs(os.path.dirname(study_path_2), exist_ok=True)
joblib.dump(study_2, study_path_2)
print(f"Study saved to {study_path_2}")
best_params_2 = study_2.best_trial.params
xgb_params_2 = {k.replace('xgb_', ''): v for k, v in best_params_2.items() if k.startswith('xgb_')}
rf_params_2 = {k.replace('rf_', ''): v for k, v in best_params_2.items() if k.startswith('rf_')}
xgb_params_2.update({"random_state": 42, "tree_method": "hist"})
rf_params_2.update({"random_state": 42, "verbose": False})
xgb_2 = XGBRegressor(**xgb_params_2)
rf_2 = RandomForestRegressor(**rf_params_2)


# Add this after loading the existing models but before creating the stacker

# === RIDGE HYPERPARAMETER TUNING ===
def objective(trial):
    # Only tuning Ridge parameters
    ridge_params = {
        "alpha": trial.suggest_float("ridge_alpha", 0.01, 10.0, log=True),
        "fit_intercept": trial.suggest_categorical("ridge_fit_intercept", [True, False]),
        "solver": trial.suggest_categorical("ridge_solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]),
        "random_state": 42
    }
    
    # Use the already loaded models with their best parameters
    stacker = StackingRegressor(
        estimators=[
            ("xgb_1", xgb_1),
            ("cat_1", cat_1),
            ("xgb_2", xgb_2),
            ("rf_2", rf_2),
        ],
        final_estimator=Ridge(**ridge_params),
        n_jobs=-1,
    )
    
    # Use reduced cross-validation for faster tuning
    reduced_tscv = TimeSeriesSplit(n_splits=3)
    mse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(reduced_tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        stacker.fit(X_train, y_train)
        preds = stacker.predict(X_val)
        
        y_val_orig = target_scaler.inverse_transform(y_val.values.reshape(-1, 1)).flatten()
        preds_orig = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        
        mse = mean_squared_error(y_val_orig, preds_orig)
        mse_scores.append(mse)
        
        # Log progress
        writer.add_scalar(f"Ridge_Tuning/Fold_{fold}_MSE", mse, trial.number)
    
    mean_mse = np.mean(mse_scores)
    writer.add_scalar("Ridge_Tuning/Mean_MSE", mean_mse, trial.number)
    logging.info(f"Ridge tuning trial {trial.number} - Mean MSE: {mean_mse:.4f}")
    
    return mean_mse

# Create and run the Ridge tuning study
ridge_study = optuna.create_study(direction="minimize")
# ridge_study.optimize(objective, n_trials=20, n_jobs=-1)  # Adjust n_trials as needed

# Get best Ridge parameters
# best_ridge_params = ridge_study.best_params
# logging.info(f"Best Ridge parameters: {best_ridge_params}")
# print(f"Best Ridge parameters: {best_ridge_params}")

# Save the Ridge study
ridge_study_path = os.path.join(os.getcwd(), "results", "studies", f"study_ridge_{ID}.pkl")
os.makedirs(os.path.dirname(ridge_study_path), exist_ok=True)
joblib.dump(ridge_study, ridge_study_path)
print(f"Ridge study saved to {ridge_study_path}")

# Extract best Ridge parameters
# ridge_params = {k.replace('ridge_', ''): v for k, v in best_ridge_params.items()}
# ridge_params.update({"random_state": 42})

ridge_params = {
    "alpha": 0.0105,
    "fit_intercept": False,
    "solver": "auto",
    "random_state": 42
}
    

# # Now create the final stacker with tuned Ridge
# stacker = StackingRegressor(
#     estimators=[
#         ("xgb_1", xgb_1),
#         ("cat_1", cat_1),
#         ("xgb_2", xgb_2),
#         ("rf_2", rf_2),
#     ],
#     final_estimator=Ridge(**ridge_params),
#     n_jobs=-1,
# )

# Create stacked ensemble with both models
stacker = StackingRegressor(
    estimators=[
        ("xgb_1", xgb_1),
        ("cat_1", cat_1),
        # ("xgb_2", xgb_2),
        ("rf_2", rf_2),
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