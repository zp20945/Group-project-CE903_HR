# PPG Signal Processing and HRV Analysis

This repository contains three Python scripts for processing, filtering, and analyzing Photoplethysmography (PPG) signals. The scripts include data preprocessing, feature extraction, and Poincaré plot generation for Heart Rate Variability (HRV) analysis.

## Project Overview
The three scripts perform the following tasks:
1. **Data Preprocessing & Filtering**: Cleans raw CSV data, applies Butterworth filters, and prepares datasets for further analysis.
2. **HRV Feature Extraction**: Computes HRV time-domain and frequency-domain features from filtered PPG signals.
3. **Poincaré Plot Analysis**: Generates Poincaré plots for HRV analysis and saves the corresponding metrics.

---
## 1. Data Preprocessing & Filtering (`FilteringAndSplitting.py`)
### **Objective**
- Load and preprocess raw PPG signal data from CSV files.
- Extract respondent name.
- Apply high-pass and low-pass Butterworth filters.
- Filter data based on specific SourceStimuliNames.
- Split 6 specific videos according to specific time intervals, creating 6 videos more. 
- Save the processed data into a new CSV file.

### **Processing Steps**
- Reads a CSV file and removes unnecessary rows.
- Applies a **high-pass filter** to remove baseline drift.
- Uses a **low-pass Butterworth filter** to smooth the signal.
- Filters data based on predefined video stimuli.
- Converts to seconds timestamp values adn splits videos.
- Saves the filtered dataset as `filtered_ppg_signal_with_all_intervals.csv`.

### **Output**
- Processed CSV file containing cleaned and filtered PPG data.

---
## 2. HRV Feature Extraction (`Extracting_Features.py`)
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
- CSV file containing HRV metrics for each video stimulus.

---
## 3. Poincaré Plot Analysis (`PointcarePlot.py`)
### **Objective**
- Generate **Poincaré plots** for visual HRV analysis.
- Compute **SD1**, **SD2**, and **Parasympathetic Index**.
- Save plots and corresponding metrics.

### **Processing Steps**
- Loads filtered PPG data.
- Detects **RR intervals** using peak detection.
- Computes **SD1 and SD2** values from RR intervals.
- Generates **Poincaré plots** using Matplotlib.
- Saves plots and a CSV file containing Poincaré metrics.

### **Output**
- Poincaré plots saved in `Results/PointcarePlots/`.
- CSV file (`PoincareMetrics.csv`) with SD1, SD2, and Parasympathetic Index.

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
python script_1.py  # Preprocess and filter data
python script_2.py  # Extract HRV features
python script_3.py  # Generate Poincaré plots
```

---
## Folder Structure
```
📂 Project Root
├── 📂 Raw_Participants_Data        # Raw CSV files
├── 📂 Data_Without_Useless_Rows    # Processed data
├── 📂 FilteredData                 # Filtered PPG signals
├── 📂 Features                     # Extracted HRV metrics
├── 📂 Results                      # Poincaré plots & metrics
├── FilteringAndSplitting.py                     # Data preprocessing script
├── Extracting_Features.py                     # HRV feature extraction script
├── PointcarePlot.py                     # Poincaré plot generation script
└── README.md                        # Project documentation
```

Links for the resources for the project : 
box : https://essexuniversity.app.box.com/folder/275047382390?tc=collab-folder-invite-treatment-b
GoogleDrive with the essentials of the project: https://drive.google.com/drive/folders/1V_sfkkrW6q8fWkEJMH4I79qfRwFeJv19



