# 🧠 Emotion Recognition from PPG & HR Signals

This repository contains a full pipeline for preprocessing, feature extraction, normalization, and machine learning applied to heart rate (HR) and photoplethysmographic (PPG) signals recorded during emotional video stimuli.

---

## Pipeline Overview

### `1. PPG_HR_Longer_Intervals.py`
Preprocesses raw CSV files containing PPG and HR data:
- Detects and sets headers
- Applies Butterworth high-pass and low-pass filters to remove noise and baseline drift
- Cleans and imputes heart rate signals using forward fill, backward fill, and interpolation
- Normalizes timestamps and adjusts intervals per stimulus
- Splits/merges video intervals and reassigns survey timestamps to target videos
- Standardizes stimulus names and exports clean files

> **Note:** The intervals are based in GSR intervals but pretending to have at least 15 seconds per stimuli where possible, that is the reason why some time from the surveys is taken and appened to the stimuli.
---

### `2. Time_Frequency_Features_Extraction.py`
Extracts classical HRV features in the time and frequency domains from the filtered signals.
---

### `3. PoincarePlos_Features_HR.py`
Plots poincaré plots and Calculates non-linear HRV metrics from Poincaré plots.
---

### `4. merging_all_features.py`
Merges time-domain, frequency-domain, and non-linear features into a single feature matrix per participant and video.

---

### `5. Features_Norm_Grey_Cal_Start.py`
Normalizes all features using the "Grey Calibration Start" baseline as a reference.  
This enables within-subject comparisons across different stimuli.

---

### `6. Features_Arousal_Join.py`
Joins subjective arousal ratings from survey responses with the corresponding physiological feature vectors.

---

### `7. Correlation_plots_csv.py`
Computes and visualizes correlation matrices between HRV features and arousal labels.  
Useful for identifying which physiological signals are most predictive of arousal.

---

### `8. HRV_ML.ipynb`
Machine learning pipeline for arousal prediction based on HRV features:
- Cross-validation using **Leave-One-Group-Out (LOGO)**
- Hyperparameter tuning with **GridSearch**
- Model evaluation using **RMSE**, **MAE**, and **R²**
- Feature importance analysis
---
