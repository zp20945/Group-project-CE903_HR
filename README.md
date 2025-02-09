# PPG Signal Processing and HRV Analysis

This repository contains three Python scripts for processing, filtering, and analyzing Photoplethysmography (PPG) signals. The scripts include data preprocessing, feature extraction, and Poincaré plot generation for Heart Rate Variability (HRV) analysis.

## Project Overview
The three scripts perform the following tasks:
1. **Data Preprocessing & Filtering**: Cleans raw CSV data, applies Butterworth filters, and prepares datasets for further analysis.
2. **HRV Feature Extraction**: Computes HRV time-domain and frequency-domain features from filtered PPG signals.
3. **Poincaré Plot Analysis**: Generates Poincaré plots for HRV analysis and saves the corresponding metrics.

---
##  Data Preprocessing & Filtering (`Filtering_Splitting_Renaming_Participants.py`)
### **Objective**
- Load and preprocess raw PPG signal data from CSV files.
- Extract respondent name.
- Apply high-pass and low-pass Butterworth filters.
- Filter data based on specific SourceStimuliNames.
- Split 6 specific videos according to specific time intervals, creating 6 videos more. 
- Save the processed data into a new CSV file in an specific folder.

##  Data Preprocessing & Filtering (`Filtering_Whole_Videos.py`)
### **Objective**
- Load and preprocess raw PPG signal data from CSV files.
- Extract respondent name.
- Apply high-pass and low-pass Butterworth filters.
- Filter data based on specific SourceStimuliNames.
- Save the processed data into a new CSV file in an specific folder.
  

### **Processing Steps**
- Reads a CSV file and removes unnecessary rows.
- Applies a **high-pass filter** to remove baseline drift.
- Uses a **low-pass Butterworth filter** to smooth the signal.
- Filters data based on predefined video stimuli.
- Converts to seconds timestamp values adn splits videos.
- Saves the filtered datasets.

### **Output**
- Processed CSV file containing cleaned and filtered PPG data.

---
##  HRV Feature Extraction (`Extracting_Features_v2.py`)
### **Objective**
- Extract HRV features from PPG signals.
- Compute time-domain and frequency-domain HRV metrics.
- Save the computed HRV features into a CSV file.

### **Processing Steps**
- Loads the filtered PPG signal data.
- Detects **peaks** in the signal using `find_peaks`.
- Computes **RR intervals** from peak-to-peak differences.
- Extracts time-domain HRV features:
  - Mean RR, Median RR, SDNN, RMSSD, NN50, pNN50.
- Computes **power spectral density (PSD)** using Welch’s method.
- Extracts frequency-domain HRV features:
  - VLF, LF, HF power, LF/HF ratio.
- Computes **Signal-to-Noise Ratio (SNR)**.
- Saves the computed features as `AggregatedFeaturesHRV_38_Videos.csv`.

### **Output**
- CSV file containing HRV metrics for each video stimulus for each participant.

---
## Poincaré Plot Analysis (`PointcarePlot_v3.py`)
### **Objective**
- Generate **Poincaré plots** for visual HRV analysis.
- Compute **SD1**, **SD2**, and **Parasympathetic Index**.
- Save plots and corresponding metrics.

### **Processing Steps**
- Loads filtered PPG data.
- Detects **RR intervals** using peak detection.
- Computes **SD1 and SD2** values from RR intervals.
- Generates **Poincaré plots** using Matplotlib.
- Saves plots and a CSV file containing Poincaré metrics per participants folder and in the main a csv containing parameters of all of them.

### **Output**
- Poincaré plots saved in `Results/PointcarePlots/`.
- CSV file with SD1, SD2, and Parasympathetic Index.
- CSV file with all participants

## Merging and comparission
### **Objective**
-Combine HRV metrics from split videos, whole videos, and arousal ground truth into a single dataset.

Processing Steps
-Load CSV Files:
-HRV metrics from split videos (All_Participants_PoincareMetrics.csv from Participant_Analysis_Intervals).
-HRV metrics from whole videos (All_Participants_PoincareMetrics.csv from Participant_Analysis_Whole).
-Arousal ground truth data (individual_ground_truth_without_hm2.csv).

Expand Stimuli for Whole Videos:
-Some stimuli in the whole-video dataset need to be split into multiple components.
-The script uses a predefined mapping dictionary to create corresponding entries.

Merge Datasets:
-Merge split videos and whole videos datasets using Participant and Stimulus as keys.
-Merge with arousal ground truth data to align physiological metrics with labeled arousal responses.

Select Relevant Features:
-Participant
-Stimulus
-Parasympathetic_SplittedVideos
-Parasympathetic_WholeVideos
-Arousal

Save the Final Merged Dataset:
-The script creates an output folder (Comparission/).
S-aves the merged dataset as final_merged_output.csv for further analysis.

## IBI_plotting
### **Objective**
- Print PPG signal and IBI´s overtime

---
## Dependencies
Ensure the following Python libraries are installed before running the scripts:
```bash
pip install pandas numpy matplotlib scipy
```

---
## Usage
Run each script sequentially to process the PPG signals and extract HRV features:
```bash
filtering_splitting_renaiming_participants.py #Preprocess and filter data
filtering_whole_videos.py  # Preprocess and filter data
Exctracting_features_v2.py  # Extract HRV features
Poincare_plot_v3.py  # Generate Poincaré plots
Merging_Comparing.py # Merging all data and
Plotting_corr_part.py # Correleation graphs
IBI_plotting.py #Generates the HRV plots
```


Links for the resources for the project : 
box : https://essexuniversity.app.box.com/folder/275047382390?tc=collab-folder-invite-treatment-b
GoogleDrive with the essentials of the project: https://drive.google.com/drive/folders/1V_sfkkrW6q8fWkEJMH4I79qfRwFeJv19
-In google drive the folder that i use to make these codes run are : ToTry (to try the codes) & Participants ](to work with all data)


