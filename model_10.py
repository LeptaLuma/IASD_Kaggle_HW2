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
ID = "10.10"
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

# Add these functions before your main optimization code

def optimize_xgb(n_trials=50):
    def objective_xgb(trial):
        xgb_params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 600),
            "max_depth": trial.suggest_int("max_depth", 8, 14),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1, 4),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 12),
            "random_state": 42,
            "tree_method": "hist",
        }
        
        reduced_tscv = TimeSeriesSplit(n_splits=5)
        mse_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(reduced_tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = XGBRegressor(**xgb_params)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_val)
            y_val_orig = target_scaler.inverse_transform(y_val.values.reshape(-1, 1)).flatten()
            preds_orig = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            
            mse = mean_squared_error(y_val_orig, preds_orig)
            mse_scores.append(mse)
            
        return np.mean(mse_scores)
    
    xgb_study = optuna.create_study(direction="minimize", 
                                   pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    xgb_study.optimize(objective_xgb, n_trials=n_trials, n_jobs=-1)
    return {f"xgb_{k}": v for k, v in xgb_study.best_params.items()}

def optimize_rf(n_trials=50):
    def objective_rf(trial):
        rf_params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 400),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
            "random_state": 42,
        }
        
        reduced_tscv = TimeSeriesSplit(n_splits=5)
        mse_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(reduced_tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = RandomForestRegressor(**rf_params)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_val)
            y_val_orig = target_scaler.inverse_transform(y_val.values.reshape(-1, 1)).flatten()
            preds_orig = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            
            mse = mean_squared_error(y_val_orig, preds_orig)
            mse_scores.append(mse)
            
        return np.mean(mse_scores)
    
    rf_study = optuna.create_study(direction="minimize",
                                  pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    rf_study.optimize(objective_rf, n_trials=n_trials, n_jobs=-1)
    return {f"rf_{k}": v for k, v in rf_study.best_params.items()}

def optimize_cat(n_trials=50):
    def objective_cat(trial):
        cat_params = {
            "iterations": trial.suggest_int("iterations", 100, 300),
            "depth": trial.suggest_int("depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.2),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 5, 8),
            "random_strength": trial.suggest_float("random_strength", 0.5, 8),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 2, 10),
            "random_seed": 42,
            "verbose": False
        }
        
        reduced_tscv = TimeSeriesSplit(n_splits=5)
        mse_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(reduced_tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = CatBoostRegressor(**cat_params)
            model.fit(X_train, y_train, verbose=False)
            
            preds = model.predict(X_val)
            y_val_orig = target_scaler.inverse_transform(y_val.values.reshape(-1, 1)).flatten()
            preds_orig = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            
            mse = mean_squared_error(y_val_orig, preds_orig)
            mse_scores.append(mse)
            
        return np.mean(mse_scores)
    
    cat_study = optuna.create_study(direction="minimize",
                                   pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    cat_study.optimize(objective_cat, n_trials=n_trials, n_jobs=-1)
    return {f"cat_{k}": v for k, v in cat_study.best_params.items()}

# === OPTUNA STUDY ===
study_path = os.path.join(os.getcwd(), "results", "studies", f"study_xgb_cat_6.0.pkl")
# Replace the existing study optimization section

if load_study:
    if os.path.exists(study_path):
        study = joblib.load(study_path)
        print("Loaded existing study.")
    else:
        raise FileNotFoundError(f"Study file not found at {study_path}. Please check the path or create a new study.")
else:
    print("Creating a new study with separate model optimization.")
    
    # First optimize each model individually
    print("Optimizing XGBoost...")
    xgb_best_params = optimize_xgb(n_trials=50)
    print(f"Best XGBoost params: {xgb_best_params}")
    
    print("Optimizing RandomForest...")
    rf_best_params = optimize_rf(n_trials=50)
    print(f"Best RandomForest params: {rf_best_params}")
    
    print("Optimizing CatBoost...")
    cat_best_params = optimize_cat(n_trials=50)
    print(f"Best CatBoost params: {cat_best_params}")
    

# Extract params for each model
xgb_params = {k.replace('xgb_', ''): v for k, v in xgb_best_params.items()}
rf_params = {k.replace('rf_', ''): v for k, v in rf_best_params.items()}
cat_params = {k.replace('cat_', ''): v for k, v in cat_best_params.items()}
ridge_params = {'alpha': 0.896002688408722, 'fit_intercept': False, 'solver': 'cholesky'}
# AUTRE POSSIBILITE : {'ridge_alpha': 1.3838107184105972, 'ridge_fit_intercept': False, 'ridge_solver': 'svd'}

# Add fixed parameters
xgb_params.update({"random_state": 42, "tree_method": "hist"})
rf_params.update({"random_state": 42})
cat_params.update({"random_seed": 42, "verbose": False})

# Create base models with best parameters
xgb = XGBRegressor(**xgb_params)
rf = RandomForestRegressor(**rf_params)
cat = CatBoostRegressor(**cat_params)
ridge = Ridge(**ridge_params)

# Create stacked ensemble with both models
stacker = StackingRegressor(
    estimators=[
        ("xgb", xgb),
        ("rf", rf),
        ("cat", cat)
    ],
    final_estimator=ridge,
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