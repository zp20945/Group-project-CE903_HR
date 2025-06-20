**Emotion Recognition from PPG & HR Signals**

This repository contains the full pipeline for preprocessing, feature extraction, merging, normalization, and machine learning applied to heart rate (HR) and photoplethysmographic (PPG) signals recorded during emotional video stimuli.

Flow:
-PPG_HR_Longer_Intervals.py
.Preprocesses raw CSVs containing PPG and HR data.
.Detects and sets headers dynamically
.Applies Butterworth filters to remove noise and baseline drift
.Interpolates and cleans heart rate signals. Imputation data is done by forward, backward fill and interpolation.
.Normalizes timestamps
.Splits/merges video intervals and survey stimuli
.Standardizes stimulus names and exports clean files

*The intervals are based in GSR intervals but pretending to have at least 15 seconds per stimuli where possible, that is the reason why some time from the surveys is taken and appened to the stimuli.* 

-Time_Frequency_Features_Extraction.py
Extracts classical HRV features in time and frequency domains from the filtered signals.
PoincarePlos_Features_HR.py
Calculates non-linear HRV metrics using Poincaré plots.

-merging_all_features.py
Merges time-domain, frequency-domain, and non-linear HRV features into a unified feature matrix per participant and video.

-Features_Norm_Grey_Cal_Start.py
Normalizes features using "Grey Calibration Start" as reference, enabling within-subject comparison across stimuli.

-Features_Arousal_Join.py
Adds subjective arousal ratings to the dataset by joining survey responses with corresponding physiological features.

-Correlation_plots_csv.py
Computes and visualizes correlation matrices between HRV features and arousal labels across participants.

-HRV_ML.ipynb
Machine learning pipeline for predicting arousal based on HRV features.
Includes:
Cross-validation using Leave-One-Group-Out
GridSearch for hyperparameter tuning
Model evaluation using RMSE, MAE, R²
Feature importance analysis

