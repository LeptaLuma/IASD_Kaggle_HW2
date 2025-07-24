"""
Advanced Time Series Forecasting with Ensemble Learning and Hyperparameter Optimization

This script implements a sophisticated machine learning pipeline for time series forecasting using:
- Hyperparameter optimization with Optuna
- Ensemble learning with stacking
- Multiple algorithms: XGBoost, RandomForest, CatBoost
- Proper time series cross-validation
- Experiment tracking with TensorBoard

Code cleaned with the help of an AI
Authors: Paul Malet and Brice Convers (github.com/Worl0r)
"""

import os
import time
import logging
import datetime
from typing import Dict, Any

import pandas as pd
import numpy as np
import optuna
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from torch.utils.tensorboard import SummaryWriter

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths - modify these according to your setup
DATA_PATH_X = "results/processed/X_scaled.csv"
DATA_PATH_Y = "results/processed/y_scaled.csv"
TEST_PATH = "results/processed/X_scaled_test.csv"
TARGET_SCALER_PATH = "results/scalers/target_scaler.pkl"

# Model configuration
TARGET = "ToPredict"
TIME_COLUMN = "Dates"
N_SPLITS = 5
MODEL_ID = "10.10"
LOAD_EXISTING_STUDY = False  # Set to True to load pre-existing optimization results

# ============================================================================
# SETUP
# ============================================================================

start_time = time.time()

# Load target scaler for inverse transformation
target_scaler = joblib.load(TARGET_SCALER_PATH)

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/train_{MODEL_ID}.log", 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Setup TensorBoard for experiment tracking
tb_log_dir = f"logs/tensorboard/{MODEL_ID}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=tb_log_dir)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare training data."""
    df = pd.read_csv(DATA_PATH_X)
    y_df = pd.read_csv(DATA_PATH_Y)
    
    features = df.columns.tolist()
    X = df[features]
    y = y_df[TARGET]
    
    logging.info(f"Loaded data: X shape {X.shape}, y shape {y.shape}")
    return X, y, features

# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================

def optimize_xgboost(X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> Dict[str, Any]:
    """
    Optimize XGBoost hyperparameters using Optuna.
    
    Args:
        X: Feature matrix
        y: Target variable
        n_trials: Number of optimization trials
        
    Returns:
        Dictionary of best parameters with 'xgb_' prefix
    """
    def objective(trial):
        params = {
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
        
        # Use reduced cross-validation for faster optimization
        tscv = TimeSeriesSplit(n_splits=3)
        mse_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_val)
            
            # Transform back to original scale for evaluation
            y_val_orig = target_scaler.inverse_transform(y_val.values.reshape(-1, 1)).flatten()
            preds_orig = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            
            mse = mean_squared_error(y_val_orig, preds_orig)
            mse_scores.append(mse)
            
        return np.mean(mse_scores)
    
    study = optuna.create_study(
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    
    return {f"xgb_{k}": v for k, v in study.best_params.items()}

def optimize_random_forest(X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> Dict[str, Any]:
    """
    Optimize RandomForest hyperparameters using Optuna.
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 400),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
            "random_state": 42,
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        mse_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_val)
            
            y_val_orig = target_scaler.inverse_transform(y_val.values.reshape(-1, 1)).flatten()
            preds_orig = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            
            mse = mean_squared_error(y_val_orig, preds_orig)
            mse_scores.append(mse)
            
        return np.mean(mse_scores)
    
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    
    return {f"rf_{k}": v for k, v in study.best_params.items()}

def optimize_catboost(X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> Dict[str, Any]:
    """
    Optimize CatBoost hyperparameters using Optuna.
    """
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 300),
            "depth": trial.suggest_int("depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.2),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 5, 8),
            "random_strength": trial.suggest_float("random_strength", 0.5, 8),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 2, 10),
            "random_seed": 42,
            "verbose": False
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        mse_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train, verbose=False)
            
            preds = model.predict(X_val)
            
            y_val_orig = target_scaler.inverse_transform(y_val.values.reshape(-1, 1)).flatten()
            preds_orig = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            
            mse = mean_squared_error(y_val_orig, preds_orig)
            mse_scores.append(mse)
            
        return np.mean(mse_scores)
    
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    
    return {f"cat_{k}": v for k, v in study.best_params.items()}

# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

def create_ensemble_model(
        xgb_params: Dict, 
        rf_params: Dict, 
        cat_params: Dict,
        ridge_params: Dict
    ) -> StackingRegressor:
    """
    Create a stacking ensemble with optimized base models.
    
    Args:
        xgb_params: XGBoost parameters
        rf_params: RandomForest parameters  
        cat_params: CatBoost parameters
        ridge_params: Ridge regression parameters for meta-learner
        
    Returns:
        Configured StackingRegressor
    """
    base_models = [
        ("xgb", XGBRegressor(**xgb_params)),
        ("rf", RandomForestRegressor(**rf_params)),
        ("cat", CatBoostRegressor(**cat_params))
    ]
    
    return StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(**ridge_params),
        n_jobs=-1
    )

def evaluate_model(model: StackingRegressor, X: pd.DataFrame, y: pd.Series) -> list:
    """
    Evaluate model using time series cross-validation.
    
    Args:
        model: Trained model to evaluate
        X: Feature matrix
        y: Target variable
        
    Returns:
        List of MSE scores for each fold
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    mse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        preds = model.predict(X_val)
        
        # Transform back to original scale
        y_val_orig = target_scaler.inverse_transform(y_val.values.reshape(-1, 1)).flatten()
        preds_orig = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        
        mse = mean_squared_error(y_val_orig, preds_orig)
        mse_scores.append(mse)
        
        logging.info(f"Fold {fold} MSE: {mse:.6f}")
        writer.add_scalar("CV_Fold/MSE", mse, fold)
    
    return mse_scores

def generate_predictions(model: StackingRegressor, features: list) -> pd.DataFrame:
    """
    Generate predictions on test set and create submission file.
    
    Args:
        model: Trained model
        features: List of feature names
        
    Returns:
        Submission DataFrame
    """
    df_test = pd.read_csv(TEST_PATH, parse_dates=[TIME_COLUMN])
    X_test = df_test[features]
    
    predictions = model.predict(X_test)
    predictions_orig = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    return pd.DataFrame({
        "ID": df_test[TIME_COLUMN], 
        "ToPredict": predictions_orig
    })

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    
    # Load data
    X, y, features = load_data()
    
    # Hyperparameter optimization or load existing study
    study_path = f"results/studies/study_{MODEL_ID}.pkl"
    
    if LOAD_EXISTING_STUDY and os.path.exists(study_path):
        print("Loading existing optimization study...")
        study_results = joblib.load(study_path)
        xgb_best_params = study_results['xgb_params']
        rf_best_params = study_results['rf_params'] 
        cat_best_params = study_results['cat_params']
        
    else:
        print("Starting hyperparameter optimization...")
        
        print("Optimizing XGBoost...")
        xgb_best_params = optimize_xgboost(X, y, n_trials=50)
        print(f"Best XGBoost params: {xgb_best_params}")
        
        print("Optimizing RandomForest...")
        rf_best_params = optimize_random_forest(X, y, n_trials=50)
        print(f"Best RandomForest params: {rf_best_params}")
        
        print("Optimizing CatBoost...")
        cat_best_params = optimize_catboost(X, y, n_trials=50)
        print(f"Best CatBoost params: {cat_best_params}")
        
        # Save optimization results
        os.makedirs("results/studies", exist_ok=True)
        study_results = {
            'xgb_params': xgb_best_params,
            'rf_params': rf_best_params,
            'cat_params': cat_best_params
        }
        joblib.dump(study_results, study_path)
    
    # Extract and prepare parameters for each model
    xgb_params = {k.replace('xgb_', ''): v for k, v in xgb_best_params.items()}
    rf_params = {k.replace('rf_', ''): v for k, v in rf_best_params.items()}
    cat_params = {k.replace('cat_', ''): v for k, v in cat_best_params.items()}
    
    # Add fixed parameters
    xgb_params.update({"random_state": 42, "tree_method": "hist"})
    rf_params.update({"random_state": 42})
    cat_params.update({"random_seed": 42, "verbose": False})
    
    # Ridge parameters for meta-learner (can be optimized separately if needed)
    ridge_params = {'alpha': 0.896002688408722, 'fit_intercept': False, 'solver': 'cholesky'}
    
    # Create and train ensemble model
    print("Training ensemble model...")
    ensemble = create_ensemble_model(xgb_params, rf_params, cat_params, ridge_params)
    ensemble.fit(X, y)
    
    logging.info("Ensemble model training completed")
    
    # Evaluate model performance
    print("Evaluating model performance...")
    mse_scores = evaluate_model(ensemble, X, y)
    
    avg_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    
    print(f"Cross-validation results:")
    print(f"Average MSE: {avg_mse:.6f} (+/- {std_mse:.6f})")
    
    writer.add_scalar("Final/Average_MSE", avg_mse)
    writer.add_scalar("Final/Std_MSE", std_mse)
    
    # Save final model
    model_path = f"results/models/ensemble_model_{MODEL_ID}.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(ensemble, model_path)
    print(f"Model saved to {model_path}")
    
    # Generate predictions
    print("Generating predictions...")
    submission = generate_predictions(ensemble, features)
    
    # Save submission
    os.makedirs("results/submissions", exist_ok=True)
    submission_path = f"results/submissions/submission_{MODEL_ID}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    
    # Cleanup
    execution_time = time.time() - start_time
    print(f"Total execution time: {execution_time:.2f} seconds")
    
    writer.close()
    logging.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()