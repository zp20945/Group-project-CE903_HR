Emotion Recognition from PPG & HR Signals
This repository contains the full pipeline for preprocessing, feature extraction, merging, normalization, and machine learning applied to heart rate (HR) and photoplethysmographic (PPG) signals recorded during emotional video stimuli.

Flow Overview
PPG_HR_Longer_Intervals.py
Preprocesses raw CSVs containing PPG and HR data.

Detects and sets headers dynamically

Applies Butterworth filters to remove noise and baseline drift

Interpolates and cleans heart rate signals (using forward fill, backward fill, and linear interpolation)

Normalizes timestamps per stimulus

Splits and merges video intervals and survey data

Standardizes stimulus names and exports clean files

Note: The stimulus intervals are aligned with GSR-based intervals. The aim is to ensure at least 15 seconds per stimulus when possible—sometimes extending the time range by appending part of the corresponding survey.

Time_Frequency_Features_Extraction.py
Extracts classical HRV features in the time and frequency domains from the filtered PPG and HR signals.

PoincarePlos_Features_HR.py
Calculates non-linear HRV features based on Poincaré plots (e.g., SD1, SD2, SD1/SD2 ratio).

merging_all_features.py
Combines all extracted HRV features (time, frequency, and non-linear) into a unified feature matrix per participant and stimulus.

Features_Norm_Grey_Cal_Start.py
Normalizes all features using the Grey Calibration Start as a baseline reference to enable within-subject comparisons.

Features_Arousal_Join.py
Joins subjective arousal ratings (survey data) to the physiological feature matrix for supervised learning.

Correlation_plots_csv.py
Computes and visualizes correlation matrices between HRV features and arousal scores across participants.

HRV_ML.ipynb
Machine learning pipeline for predicting arousal from physiological features.
Includes:

Leave-One-Group-Out cross-validation

GridSearchCV for hyperparameter tuning

Evaluation metrics: RMSE, MAE, R²

Feature importance analysis (e.g., via model coefficients or SHAP)


