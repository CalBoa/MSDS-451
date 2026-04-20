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
## AI Usage Disclosure
I used AI assistants to help with code debugging, code completion, interpretation of outputs, and drafting documentation. I reviewed,edited, tested, and finalized the submission myself.

## Writeup
### Predicting Gold Futures Daily Return Direction with Lagged Price Features and XGBoost
Problem Description
This project investigates whether lagged technical features derived from daily Gold Futures prices can be used to predict the direction of next-day returns. Gold Futures (GC=F) were selected as the target asset instead of the example asset used in the course materials. Gold is an important financial instrument that often reflects inflation expectations, macroeconomic uncertainty, and investor demand for safe-haven assets. The modeling problem is framed as a binary classification task in which the target variable equals 1 when the daily log return is positive and 0 otherwise.

Data Preparation and Pipeline
Daily historical Gold Futures data were retrieved from Yahoo Finance using the yfinance package for the period from 2000-01-01 through 2025-05-27. The raw data included daily open, high, low, close, and volume observations. Feature engineering was performed in Polars. The engineered features included lagged closing prices (CloseLag1, CloseLag2, CloseLag3), daily trading range variables based on High - Low, intraday variables based on Open - Close, lagged volume variables, and exponential moving averages computed from lagged closing prices. Daily log returns were computed as the natural log of the ratio of current close to lagged close. A binary target variable was then constructed from the sign of the daily log return.

To avoid data leakage, current-day market variables such as open, high, low, close, volume, HML, and OMC were excluded from the predictor matrix. Only lagged versions of these signals were used. In addition, the moving averages were calculated from CloseLag1 rather than the current close. Missing values created by lag operations were removed, and features were standardized using StandardScaler.

Research Design
The study used a time-series research design rather than a random holdout design because financial observations are sequential and not independent. Feature subset selection was performed using logistic regression and the Akaike Information Criterion (AIC). All possible non-empty subsets of the candidate predictor set were evaluated. The subset with the lowest AIC consisted of CloseLag1 and CloseLag3, indicating that recent lagged closing price information provided the most parsimonious explanatory structure among the engineered features.

After selecting the feature subset, the data were evaluated using TimeSeriesSplit with five splits and a 10-observation gap between training and testing windows. This design respects the chronological structure of the time series and reduces contamination across adjacent train and test periods. An initial XGBoost classification model was evaluated under this design, and then hyperparameters were tuned using RandomizedSearchCV.

Programming and Modeling
The modeling workflow was implemented in Python using yfinance, Polars, NumPy, Scikit-Learn, and XGBoost. Logistic regression was used for AIC subset selection, while the final prediction model was an XGBoost classifier. Hyperparameters tuned by randomized search included maximum tree depth, child weight, subsample rate, learning rate, and number of estimators. Performance was summarized using cross-validation accuracy, confusion matrices, ROC curves, ROC AUC, and classification reports.

Results
The AIC feature selection process identified CloseLag1 and CloseLag3 as the lowest-AIC model. This indicates that lagged closing-price levels carried the strongest predictive signal among the engineered variables considered. A correlation heatmap of the selected features and return variable suggested that the lagged price variables were highly related to one another, which is consistent with persistence in financial price levels.

The primary measure of predictive performance was time-series cross-validation accuracy. This metric provides the most appropriate estimate of out-of-sample predictive performance because it evaluates the model on unseen future segments of the series. The tuned XGBoost model produced modest predictive ability, suggesting some signal in lagged gold price behavior but not a highly accurate forecasting system.

For completeness, the final tuned model was also fit on the full sample and evaluated in-sample using a confusion matrix, ROC curve, ROC AUC, and classification report. These diagnostics showed approximately 65% apparent accuracy and better recall for positive-return days than for negative-return days. However, because these metrics were computed on the same data used to fit the model, they likely overstate true predictive performance. A cleaner out-of-sample diagnostic was therefore also produced using the final fold of the time-series split.

Discussion
The project demonstrates that lagged price information in Gold Futures contains some predictive content for next-day return direction, but the signal is limited. The fact that AIC selected only two lagged closing-price variables suggests that the incremental value of more complex engineered features was modest in this setting. The model appeared somewhat better at identifying positive-return days than negative-return days, which may reflect asymmetry in the underlying dynamics or limitations of the selected feature space.

Several limitations should be noted. First, the analysis used only technical features derived from the gold price series itself. Second, the final full-sample fit diagnostics are optimistic and should not be interpreted as true out-of-sample performance. Third, Gold Futures returns may depend on broader market and macroeconomic factors not included here, such as interest rates, inflation expectations, the U.S. dollar, equity volatility, and related commodity prices.

Conclusion
This project developed a complete financial machine learning pipeline for Gold Futures return-direction prediction. The workflow included data acquisition, feature engineering, leakage control, AIC-based subset selection, time-series cross-validation, XGBoost modeling, and hyperparameter tuning. The final results suggest limited but non-zero predictability in daily Gold Futures return direction using lagged price-based features. Future work could improve model performance by incorporating macroeconomic indicators, related markets, or alternative machine learning methods.

References
•	yfinance Documentation: https://ranaroussi.github.io/yfinance/
•	Polars User Guide: https://docs.pola.rs/
•	Scikit-Learn Documentation: https://scikit-learn.org/stable/
•	XGBoost Documentation: https://xgboost.readthedocs.io/en/latest/
