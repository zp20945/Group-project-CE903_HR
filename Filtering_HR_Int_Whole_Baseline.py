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

# renaming logic to the final dataset
rename_mapping = {
    "HN_1-1": "HN_1",
    "HN_2-1": "HN_2",
    "HN_3-1": "HN_3",
    "HN_4-1": "HN_4",
    "HN_5-1": "HN_5",
    "HN_6-1": "HN_6",
    "HN_7-1": "HN_7",
    "HN_8-1": "HN_9",
    "Baseline_HN_8": "Baseline_HN_9",
    "HP_1-1": "HP_1",
    "HP_2-1": "HP_2",
    "HP_3-1": "HP_3",
    "HP_4-1": "HP_4",
    "HP_5-1": "HP_5",
    "HP_6-1": "HP_6",
    "HP_7-1": "HP_7",
    "HP_8-1": "HP_9",
    "Baseline_HP_8": "Baseline_HP_9",
    "LN_1-1": "LN_1",
    "LN_2-1": "LN_2",
    "LN_3-1": "LN_3",
    "LN_4-1": "LN_4",
    "LN_5-1": "LN_5",
    "LN_6-1": "LN_6",
    "LN_7-1": "LN_7",
    "LN_8-1": "LN_9",
    "Baseline_LN_8": "Baseline_LN_9",
    "LP_1-1": "LP_1",
    "LP_2-1": "LP_2",
    "LP_3-1": "LP_3",
    "LP_4-1": "LP_4",
    "LP_5-1": "LP_5",
    "LP_6-1": "LP_6",
    "LP_7-1": "LP_7",
    "LP_8-1": "LP_9",
    "Baseline_LP_8": "Baseline_LP_9"
}

## Loading and Applying the previous function 
## Paths
input_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Participants"
output_folder = r"c:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\HR_preprocessed_whole_videos"

## Ensuring the output folder is NEW each time
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  # Delete existing folder and its contents
os.makedirs(output_folder)  # Create a fresh folder

## Processing each CSV file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        print(f"Processing file: {filename}")

        # Loading the CSV file
        try:
            data = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            print(f"Unicode error in {filename}, trying different encoding...")
            data = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)

        # Applying function to find header and extract respondent name
        df, header, respondent_name = find_and_set_header(data, 0, 'Row')


        df['Internal ADC A13 PPG RAW'] = pd.to_numeric(df['Internal ADC A13 PPG RAW'], errors='coerce')
        
        # If all values become NaN, raising an error before proceeding
        if df['Heart Rate PPG ALG'].isna().all():
            raise ValueError("Error: The 'Heart Rate PPG ALG' column contains no valid numeric data.")

        # Assigning processed DataFrame
        data = df   

         # Extracting the 'Heart Rate PPG ALG' column and convert it to float
        Heart_Rate = data['Heart Rate PPG ALG'].astype(float)

         # Step 1: Replace invalid values (-1) with NaN to flag them for imputation
        Heart_Rate = Heart_Rate.replace(-1, np.nan)

         
         # Interpolation is suitable for time series data like heart rate, preserving natural trends
        Heart_Rate = Heart_Rate.interpolate(method='linear')

         # Backward fill to handle any NaN left at the beginning (where interpolation can't work)
        Heart_Rate = Heart_Rate.bfill()

         # Forward fill to handle any NaN left at the end
        Heart_Rate = Heart_Rate.ffill()


        ## For the final result, a new file with the data filtered
        # Adding the Butterworth filtered PPG signal as a new column
        data['Heart Rate Signal'] = Heart_Rate
        data['Respondent Name'] = respondent_name

        # Specifying the columns to include in the output
        columns_to_include = ['Respondent Name','Timestamp', 'SourceStimuliName', 'Internal ADC A13 PPG RAW', 'Heart Rate Signal',]

        # Filtering data based on source stimuli names
        source_stimuli_to_include = [
            "HN_1-1", "HN_2-1", "HN_3-1", "HN_4-1", "HN_5-1", "HN_6-1", "HN_7-1", "HN_8-1",
            "HP_1-1", "HP_2-1", "HP_3-1", "HP_4-1", "HP_5-1", "HP_6-1", "HP_7-1", "HP_8-1",
            "LN_1-1", "LN_2-1", "LN_3-1", "LN_4-1", "LN_5-1", "LN_6-1", "LN_7-1", "LN_8-1",
            "LP_1-1", "LP_2-1", "LP_3-1", "LP_4-1", "LP_5-1", "LP_6-1", "LP_7-1", "LP_8-1",

            # Adding Baselines
            "Baseline_HN_1", "Baseline_HN_2", "Baseline_HN_3", "Baseline_HN_4", 
            "Baseline_HN_5", "Baseline_HN_6", "Baseline_HN_7", "Baseline_HN_8",
            "Baseline_HP_1", "Baseline_HP_2", "Baseline_HP_3", "Baseline_HP_4", 
            "Baseline_HP_5", "Baseline_HP_6", "Baseline_HP_7", "Baseline_HP_8",
            "Baseline_LN_1", "Baseline_LN_2", "Baseline_LN_3", "Baseline_LN_4", 
            "Baseline_LN_5", "Baseline_LN_6", "Baseline_LN_7", "Baseline_LN_8",
            "Baseline_LP_1", "Baseline_LP_2", "Baseline_LP_3", "Baseline_LP_4", 
            "Baseline_LP_5", "Baseline_LP_6", "Baseline_LP_7", "Baseline_LP_8",
            "End_Baseline", "MEDITATION AUDIO", "Meditation_audio_E",
        ]

        filtered_data = data[data['SourceStimuliName'].isin(source_stimuli_to_include)]

        filtered_data['Timestamp'] = pd.to_numeric(filtered_data['Timestamp'], errors='coerce')

        # Resetting timestamps for each video to start from 0 and converting milliseconds to seconds
        filtered_data.loc[:, 'Timestamp'] = filtered_data.groupby('SourceStimuliName')['Timestamp'].transform(
            lambda x: (x - x.min()) / 1000.0
        )


        filtered_data['SourceStimuliName'] = filtered_data['SourceStimuliName'].replace(rename_mapping) 
        # Saving the final dataset
        filtered_data = filtered_data[columns_to_include]
        final_output_path = os.path.join(output_folder, f"filtered_{filename}")
        filtered_data.to_csv(final_output_path, index=False)

print(f"Final CSVs files with completed stimuli created successfully at: {final_output_path}")