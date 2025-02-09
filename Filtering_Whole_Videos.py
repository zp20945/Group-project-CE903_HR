import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

## Function to find and set the header and extract participant name
def find_and_set_header(df, start_index=31, expected_header_indicator='Row'):
    header = None
    for i in range(start_index, len(df)):
        row = df.iloc[i]
        respondent_name = df.iloc[1, 1]  # Extract the respondent name

        if row[0] == expected_header_indicator:
            header = row
            df.columns = header  # Set this row as the header
            df = df[i + 1:].reset_index(drop=True)  # Slice from the next row
            break
    return df, header, respondent_name

## Butterworth High-pass filter function
def butter_highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

## Butterworth Low-pass filter function
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# renaming logic to the final dataset
rename_mapping = {
    "HN_1-1": "HN_1",
    "HN_2-1": "HN_2",
    "HN_3-1": "HN_3",
    "HN_4-1": "HN_4",
    "HN_5-1": "HN_5",
    "HN_6-1": "HN_6",
    "HN_7-1": "HN_7",
    "HN_8-1": "HN_8",
    "HP_1-1": "HP_1",
    "HP_2-1": "HP_2",
    "HP_3-1": "HP_3",
    "HP_4-1": "HP_4",
    "HP_5-1": "HP_5",
    "HP_6-1": "HP_6",
    "HP_7-1": "HP_7",
    "HP_8-1": "HP_8",
    "LN_1-1": "LN_1",
    "LN_2-1": "LN_2",
    "LN_3-1": "LN_3",
    "LN_4-1": "LN_4",
    "LN_5-1": "LN_5",
    "LN_6-1": "LN_6",
    "LN_7-1": "LN_7",
    "LN_8-1": "LN_8",
    "LP_1-1": "LP_1",
    "LP_2-1": "LP_2",
    "LP_3-1": "LP_3",
    "LP_4-1": "LP_4",
    "LP_5-1": "LP_5",
    "LP_6-1": "LP_6",
    "LP_7-1": "LP_7",
    "LP_8-1": "LP_8",
}
## Loading and Applying the previous function 
## Paths
input_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Participants"
output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\filterdata_whole_all"

## Ensure the output folder is NEW each time
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  # Delete existing folder and its contents
os.makedirs(output_folder)  # Create a fresh folder

## Process each CSV file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        print(f"Processing file: {filename}")

        # Load the CSV file
        try:
            data = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            print(f"Unicode error in {filename}, trying different encoding...")
            data = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)

        

        
        # Apply function to find header and extract respondent name
        df, header, respondent_name = find_and_set_header(data, 0, 'Row')

# Saving the processed DataFrame to a new CSV file
# df.to_csv(r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Data_Without_Useless_Raws\processed_006_4FoNM_py.csv", index=False)
# print("Processed file saved as 'processed_006_4FoNM_py.csv'")

## The new file to work with 
# file_path = "processed_006_4FoNM_py.csv"
# data = pd.read_csv(file_path)

        df['Internal ADC A13 PPG RAW'] = pd.to_numeric(df['Internal ADC A13 PPG RAW'], errors='coerce')
        


        # If all values become NaN, raising an error before proceeding
        if df['Internal ADC A13 PPG RAW'].isna().all():
            raise ValueError("Error: The 'Internal ADC A13 PPG RAW' column contains no valid numeric data.")

        # Assigning processed DataFrame
        data = df  

        ppg_signal = data['Internal ADC A13 PPG RAW'].astype(float)  # Forces numeric conversion
        fs = 128  # Sampling frequency

        # Handling NaN values
        ppg_signal = ppg_signal.interpolate()

        # Generating time vector
        time = np.arange(len(ppg_signal)) / fs

        # Forward and Backward Propagation
        # Using backward fill as a fallback for any remaining NaN values
        ppg_signal = ppg_signal.bfill()

        # Filling NaN values using forward fill
        ppg_signal = ppg_signal.ffill()


        baseline_cutoff = 0.5  # Cutoff frequency for baseline drift
        ppg_without_baseline = butter_highpass_filter(ppg_signal, baseline_cutoff, fs)


        # Applying low-pass filter
        lowpass_cutoff = 2  # Cutoff frequency in Hz
        filtered_butter = butter_lowpass_filter(ppg_without_baseline, lowpass_cutoff, fs)

        ## For the final result, a new file with the data filtered
        # Adding the Butterworth filtered PPG signal as a new column
        data['Butterworth Filtered PPG Signal'] = filtered_butter
        data['Respondent Name'] = respondent_name

        # Specifying the columns to include in the output
        columns_to_include = ['Respondent Name','Timestamp', 'SourceStimuliName', 'Internal ADC A13 PPG RAW', 'Butterworth Filtered PPG Signal',]

        # Filtering data based on source stimuli names
        source_stimuli_to_include = [
            "HN_1-1", "HN_2-1", "HN_3-1", "HN_4-1", "HN_5-1", "HN_6-1", "HN_7-1", "HN_8-1",
            "HP_1-1", "HP_2-1", "HP_3-1", "HP_4-1", "HP_5-1", "HP_6-1", "HP_7-1", "HP_8-1",
            "LN_1-1", "LN_2-1", "LN_3-1", "LN_4-1", "LN_5-1", "LN_6-1", "LN_7-1", "LN_8-1",
            "LP_1-1", "LP_2-1", "LP_3-1", "LP_4-1", "LP_5-1", "LP_6-1", "LP_7-1", "LP_8-1"
        ]

        filtered_data = data[data['SourceStimuliName'].isin(source_stimuli_to_include)]

        filtered_data['Timestamp'] = pd.to_numeric(filtered_data['Timestamp'], errors='coerce')

        # Resetting timestamps for each video to start from 0 and converting milliseconds to seconds
        filtered_data.loc[:, 'Timestamp'] = filtered_data.groupby('SourceStimuliName')['Timestamp'].transform(
            lambda x: (x - x.min()) / 1000.0
        )

        filtered_data['SourceStimuliName'] = filtered_data['SourceStimuliName'].replace(rename_mapping) 
        filtered_data = filtered_data[columns_to_include]
        final_output_path = os.path.join(output_folder, f"filtered_{filename}")
        filtered_data.to_csv(final_output_path, index=False)

print(f"Final CSVs file with complete videos processed created successfully at: {final_output_path}")
