# IASD Kaggle Competition 2 - Time Series Forecasting

This project implements a machine learning pipeline for financial time series forecasting using ensemble learning and hyperparameter optimization.

## Project Structure

- `data_processing.py` - Data preprocessing and feature engineering pipeline
- `model.py` - Ensemble model training with hyperparameter optimization
- `EDA.ipynb` - Exploratory data analysis notebook

## Features

### Data Processing (`data_processing.py`)
- **Time-based feature engineering**: Extracts cyclical patterns (weekly, monthly, quarterly, annual)
- **Financial indicators**: RSI, MACD, Bollinger Bands, volatility measures
- **Technical features**: Lag features, rolling statistics, log-diff transformations
- **Mean reversion & momentum**: Z-scores, streak detection, directional indicators
- **Feature scaling**: StandardScaler for both features and targets
- **Feature selection**: XGBoost-based importance filtering

### Model Training (`model.py`)
- **Ensemble learning**: Stacking with XGBoost, RandomForest, and CatBoost
- **Hyperparameter optimization**: Optuna-based automated tuning
- **Time series validation**: Proper temporal cross-validation
- **Experiment tracking**: TensorBoard logging and comprehensive metrics
- **Model persistence**: Save/load optimized models and scalers

### Exploratory Data Analysis (`EDA.ipynb`)
- **Trend analysis**: Linear trend detection and moving averages
- **Seasonality**: Periodogram analysis and Fourier decomposition
- **Autocorrelation**: Lag plots and partial autocorrelation analysis
- **Feature correlation**: Heatmaps and multicollinearity detection

## Usage

1. **Data Processing**:
   ```bash
   python data_processing.py
   ```
   - Processes raw train/test data
   - Creates engineered features
   - Saves scaled datasets and scalers

2. **Model Training**:
   ```bash
   python model.py
   ```
   - Optimizes hyperparameters for ensemble models
   - Trains stacking regressor
   - Generates predictions and submission file

3. **Exploratory Analysis**:
   - Open `EDA.ipynb` in Jupyter
   - Run cells to analyze data patterns and relationships

## Key Technical Highlights

- **Feature engineering** with 100+ financial and temporal features
- **Time-aware validation** preventing data leakage
- **Multi-algorithm ensemble** combining gradient boosting and random forests
- **Automated hyperparameter tuning** with pruning for efficiency
- **Comprehensive logging** and experiment tracking

**Authors**: Paul Malet and Brice Convers (github.com/Worl0r)