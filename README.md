# MSDS-451
# MSDS 451 Programming Assignment 1
## Gold Futures (GC=F) Daily Return Direction Prediction

This repository contains my submission for MSDS 451 Financial Engineering Programming Assignment 1.

## Project Summary
This project predicts the direction of daily Gold Futures returns using lagged price, range, volume, and moving-average features derived from Yahoo Finance data. The workflow includes:

- data retrieval with yfinance
- feature engineering with Polars
- feature standardization
- AIC-based feature subset selection using logistic regression
- time-series cross-validation using TimeSeriesSplit
- XGBoost classification
- hyperparameter tuning with RandomizedSearchCV

## Repository Contents
- `gold_futures_pa1.ipynb` - Jupyter Notebook
- `gold_futures_pa1.html` - exported notebook
- `gold_historical_data.csv` - raw downloaded data
- `gold-with-computed-features.csv` - engineered dataset
- `gold_logreturn_distribution.png` - histogram of daily log returns
- `gold_correlation_heatmap.png` - heatmap of selected features
- `gold_roc_curve_insample.png` - in-sample ROC curve
- `gold_confusion_matrix_insample.png` - in-sample confusion matrix
- `gold_roc_curve_oos.png` - out-of-sample ROC curve
- `gold_confusion_matrix_oos.png` - out-of-sample confusion matrix
- `gold_feature_importance.png` - XGBoost feature importance plot
- `report.pdf` - final report

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/CalBoa/MSDS-451.git
