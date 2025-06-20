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

# Renaming logic to the final dataset
rename_mapping = {
    "HP_1-1": "HP_1_L",
    "HP_3-1": "HP_3_L",
    "HP_7-1": "HP_7_H",
    "HN_2-1": "HN_2_H",
    "HN_3-1": "HN_3_H",
    "LN_7-1": "LN_7_N",
    "HP_1-2": "HP_1_H",
    "HP_3-2": "HP_3_H",
    "HP_7-2": "HP_7_L",
    "HN_2-2": "HN_2_L",
    "HN_3-2": "HN_3_L",
    "LN_7-2": "LN_7_P",
    "HN_1-1": "HN_1",
    "HN_4-1": "HN_4",
    "HN_5-1": "HN_5",
    "HN_6-1": "HN_6",
    "HN_7-1": "HN_7",
    "HN_8-1": "HN_9",
    "Baseline_HN_8": "Baseline_HN_9",
    "HP_2-1": "HP_2",
    "HP_4-1": "HP_4",
    "HP_5-1": "HP_5",
    "HP_6-1": "HP_6",
    "HP_8-1": "HP_9",
    "Baseline_HP_8": "Baseline_HP_9",
    "LN_1-1": "LN_1",
    "LN_2-1": "LN_2",
    "LN_3-1": "LN_3",
    "LN_4-1": "LN_4",
    "LN_5-1": "LN_5",
    "LN_6-1": "LN_6",
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
    "Baseline_LP_8": "Baseline_LP_9",
}

## Paths
input_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Healthy participants"
output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\PPG_HR_Analysis_Longer_Intervals\PPG_HR_Preprocessed_Longer_Intervals"

## Ensuring the output folder is NEW each time
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  # Deleting existing folder and its contents
os.makedirs(output_folder)  # Creating a fresh folder

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
        if df['Internal ADC A13 PPG RAW'].isna().all():
            raise ValueError("Error: The 'Internal ADC A13 PPG RAW' column contains no valid numeric data.")

        # If all values become NaN, raising an error before proceeding
        if df['Heart Rate PPG ALG'].isna().all():
            raise ValueError("Error: The 'Heart Rate PPG ALG' column contains no valid numeric data.")

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

        Heart_Rate = data['Heart Rate PPG ALG'].astype(float)

       # Replacing invalid values (-1) with NaN to flag them for imputation
        Heart_Rate = Heart_Rate.replace(-1, np.nan)

        # Interpolation is suitable for time series data like heart rate, preserving natural trends
        Heart_Rate = Heart_Rate.interpolate(method='linear')

        # Backward fill to handle any NaN left at the beginning (where interpolation can't work)
        Heart_Rate = Heart_Rate.bfill()

        # Forward fill to handle any NaN left at the end
        Heart_Rate = Heart_Rate.ffill()

        ## For the final result, a new file with the data filtered
        # Adding the Butterworth filtered PPG signal and Heart Rate as a new columns
        data['Respondent Name'] = respondent_name
        data['Butterworth Filtered PPG Signal'] = filtered_butter
        data['Heart Rate Signal'] = Heart_Rate

        # Specifying the columns to include in the output
        columns_to_include = ['Respondent Name','Timestamp', 'SourceStimuliName', 'Internal ADC A13 PPG RAW', 'Butterworth Filtered PPG Signal','Heart Rate Signal']

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
            "Grey_cal_mid", "Grey_cal_start", 

            # Adding Surveys
            "survey_HP-1", "survey_HP-3", "survey_HP-5", "survey_HP-6",
            "survey_HP-7", "survey_HN-7", "survey_LP-8",
        ]

        # Some files (2 now) have different names for some stimuli
        fixing_names = {
            "HN_1.1":"HN_1-1", 
            "HN_1.2":"HN_2-1", 
            "HN_1.5":"HN_5-1",
            "survey_HP_1-1":"survey_HP-1", 
            "survey_HP_1":"survey_HP-1", 
            "survey_HP-1.1":"survey_HP-1",
            "survey_HP_2":"survey_HP-2",
            "survey_HP-7.1":"survey_HP-7",
            "survey_HP_7.1":"survey_HP-7",
            "survey_HP_7":"survey_HP-7",
            "survey_HP_8":"survey_HP-8", 
            "survey_HN_2":"survey_HN-2", 
            "survey_HN_3":"survey_HN-3",
            "survey_HN_4":"survey_HN-4", 
            "survey_HN_5":"survey_HN-5",
            "survey_HN_1_7":"survey_HN-7",
            "survey_HN_7":"survey_HN-7",
            "survey_HN-7.1":"survey_HN-7",
            "survey_HN_8":"survey_HN-8", 
            "survey_LP_1_4":"survey_LP-4",
            "survey_LP_5":"survey_LP-5", 
            "survey_LP_1_7":"survey_LP-7",
            "survey_LP_1_7-1":"survey_LP-8", 
            "survey_HP_3":"survey_HP-3",
            "survey_HP-3.1":"survey_HP-3",
            "survey_HP_5":"survey_HP-5",
            "survey_HP-5.1":"survey_HP-5",
            "survey_HP_6": "survey_HP-6",
            "survey_HP-6.1": "survey_HP-6",
            "survey_HP_6.1": "survey_HP-6",
            "survey_LP_8": "survey_LP-8",
            "survey_LP-8.1": "survey_LP-8",
        }

        # Apply fixing names to SourceStimuliName column
        data['SourceStimuliName'] = data['SourceStimuliName'].replace(fixing_names)

        filtered_data = data[data['SourceStimuliName'].isin(source_stimuli_to_include)]

        filtered_data['Timestamp'] = pd.to_numeric(filtered_data['Timestamp'], errors='coerce')

        # Resetting timestamps for each video to start from 0 and converting milliseconds to seconds
        filtered_data.loc[:, 'Timestamp'] = filtered_data.groupby('SourceStimuliName')['Timestamp'].transform(
            lambda x: (x - x.min()) / 1000.0
        )
         # Defining split intervals 
        split_intervals = {
               "HP_1-1": {"original_interval": (23, float('inf')), "merge_intervals": [(0,23)], "new_video_name": "HP_1-2"},
               "HP_3-1": {"original_interval": (0, 30), "merge_intervals": [(30, float('inf'))], "new_video_name": "HP_3-2"},
               "HP_7-1": {"original_interval": (3, float('inf')), "merge_intervals": [(0, 3)], "new_video_name": "HP_7-2"},
               "HN_2-1": {"original_interval": (7, float('inf')), "merge_intervals": [(0, 7)], "new_video_name": "HN_2-2"},
               "HN_3-1": {"original_interval": (20, float('inf')), "merge_intervals": [(0, 20)], "new_video_name": "HN_3-2"},
               "LN_7-1": {"original_interval": (35, float('inf')), "merge_intervals": [(0, 35)], "new_video_name": "LN_7-2"},
        }

        # Initializing an empty DataFrame for the split data
        final_data = pd.DataFrame()

        # Looping through each video in `split_intervals` to process its data
        for video, details in split_intervals.items():
            video_data = filtered_data[filtered_data['SourceStimuliName'] == video]
            if video_data.empty:
                print(f"Warning: No data found for {video}. Skipping.")
                continue

            # Ensures that timestamps for all intervals in the video remain continuous (no resets between intervals).
            current_time = 0

            # Part 1: Keeping the original interval
            original_start, original_end = details["original_interval"]
            part1 = video_data[
                (video_data['Timestamp'] >= original_start) & (video_data['Timestamp'] <= original_end) #Selects rows where Timestamp falls within original_interval.
            ]
            # If the original interval contains data, adjust its timestamps
            # Reseting the index and adjust timestamps so they start from `current_time`
            if not part1.empty:
                part1 = part1.reset_index(drop=True)
                part1['Timestamp'] = current_time + (part1['Timestamp'] - part1['Timestamp'].min()) # Subtracts the smallest timestamp in the interval to start from current_time.
                # Prepares the next interval to continue from the end of this interval.
                current_time = part1['Timestamp'].max() + 0.001  # Incrementing to avoid overlaps

            # Part 2: Merging specified intervals
            merged_data = pd.DataFrame()
            for start, end in details["merge_intervals"]:
                interval_data = video_data[
                    (video_data['Timestamp'] >= start) & (video_data['Timestamp'] <= end)
                ]
                # If the interval contains data, adjust its timestamps
                if not interval_data.empty:
                    # Reseting the index and adjusting timestamps so they start from `current_time`
                    interval_data = interval_data.reset_index(drop=True)
                    interval_data['Timestamp'] = current_time + (interval_data['Timestamp'] - interval_data['Timestamp'].min())
                    # Updating `current_time` to the last timestamp in `interval_data`
                    current_time = interval_data['Timestamp'].max() + 0.001  
                    # Appending the interval data to `merged_data`
                    merged_data = pd.concat([merged_data, interval_data], axis=0)

            # Renaming the merged video
            if not merged_data.empty:
                merged_data['SourceStimuliName'] = details["new_video_name"]

            # Combining part1 and merged_data into split_data
            split_data = pd.concat([part1, merged_data], axis=0)

            # Appending the split data to the final dataset
            final_data = pd.concat([final_data, split_data], axis=0)


        # Processing remaining videos with their specific intervals
        remaining_intervals = {
            "HP_2-1": [(0, float('inf'))],
            "HP_4-1": [(0, float('inf'))],
            "HP_5-1": [(0, float('inf'))],
            "HP_6-1": [(0, float('inf'))],
            "HP_8-1": [(0, float('inf'))],
            "HN_1-1": [(0, 20.0)],
            "HN_4-1": [(0, float('inf'))],
            "HN_5-1": [(0, float('inf'))],
            "HN_6-1": [(0, float('inf'))],
            "HN_7-1": [(0, float('inf'))],
            "HN_8-1": [(0, float('inf'))],
            "LN_1-1": [(0, float('inf'))],
            "LN_2-1": [(0, float('inf'))],
            "LN_3-1": [(0, float('inf'))],
            "LN_4-1": [(0, float('inf'))],
            "LN_5-1": [(0, float('inf'))],
            "LN_6-1": [(0, float('inf'))],
            "LN_8-1": [(0, float('inf'))],
            "LP_1-1": [(0, float('inf'))],
            "LP_2-1": [(0, float('inf'))],
            "LP_3-1": [(0, float('inf'))],
            "LP_4-1": [(0, float('inf'))],
            "LP_5-1": [(0, float('inf'))],
            "LP_6-1": [(0, float('inf'))],
            "LP_7-1": [(0, float('inf'))],
            "LP_8-1": [(0, float('inf'))],
            "survey_HP-7": [(0, 10)],
            "survey_HP-3": [(0, 10)],
            "survey_HP-1": [(0, 6)],
            "survey_HP-5": [(0, 2)],
            "survey_HP-6": [(0, 4)],
            "survey_HN-7": [(0, 5)],
            "survey_LP-8": [(0, 6)],
        }

        # Processing remaining videos with their specific intervals
        for video, intervals in remaining_intervals.items():
            video_data = filtered_data[filtered_data['SourceStimuliName'] == video]
            if video_data.empty:
                print(f"Warning: No data found for {video}. Skipping.")
                continue

            # Initializing a DataFrame to store processed intervals
            merged_data = pd.DataFrame()
            current_time = 0  # Initialize time tracker for continuous timestamps

            # Processing all intervals for the current video
            for start, end in intervals:
                interval_data = video_data[
                    (video_data['Timestamp'] >= start) & (video_data['Timestamp'] <= end)
                ]

                if interval_data.empty:
                    print(f"Warning: No data found for interval {start}-{end} in {video}. Skipping.")
                    continue

                # Resetting timestamps for the interval to ensure continuity
                interval_data = interval_data.reset_index(drop=True)
                interval_data['Timestamp'] = current_time + (interval_data['Timestamp'] - interval_data['Timestamp'].min())

                # Updating current_time for the next interval
                current_time = interval_data['Timestamp'].max() + 0.001  # Add small increment to avoid overlaps

                # Appending interval data to the merged_data DataFrame
                merged_data = pd.concat([merged_data, interval_data], axis=0)

            # Adding the merged data for the current video to the final DataFrame
            final_data = pd.concat([final_data, merged_data], axis=0)
         

      # Adding Baseline stimuli (no splitting required)
        baseline_stimuli = [
            "Baseline_HN_1", "Baseline_HN_2", "Baseline_HN_3", "Baseline_HN_4", 
            "Baseline_HN_5", "Baseline_HN_6", "Baseline_HN_7", "Baseline_HN_8",
            "Baseline_HP_1", "Baseline_HP_2", "Baseline_HP_3", "Baseline_HP_4", 
            "Baseline_HP_5", "Baseline_HP_6", "Baseline_HP_7", "Baseline_HP_8",
            "Baseline_LN_1", "Baseline_LN_2", "Baseline_LN_3", "Baseline_LN_4", 
            "Baseline_LN_5", "Baseline_LN_6", "Baseline_LN_7", "Baseline_LN_8",
            "Baseline_LP_1", "Baseline_LP_2", "Baseline_LP_3", "Baseline_LP_4", 
            "Baseline_LP_5", "Baseline_LP_6", "Baseline_LP_7", "Baseline_LP_8",
            "End_Baseline", "MEDITATION AUDIO", "Meditation_audio_E",
            "Grey_cal_mid", "Grey_cal_start", 
         ]

         # Filter Baseline stimuli from the main dataset
        baseline_data = filtered_data[filtered_data['SourceStimuliName'].isin(baseline_stimuli)]

        # Checking for missing baseline stimuli
        missing_baselines = set(baseline_stimuli) - set(baseline_data['SourceStimuliName'].unique())
        if missing_baselines:
            print(f"Missing baseline stimuli in {filename}: {sorted(missing_baselines)}")

         # Resetting timestamps to start from 0 for each Baseline stimulus
        baseline_data['Timestamp'] = baseline_data.groupby('SourceStimuliName')['Timestamp'].transform(lambda x: x - x.min())

         # Appending Baseline stimuli to final_data
        final_data = pd.concat([final_data, baseline_data], axis=0)

        final_data['SourceStimuliName'] = final_data['SourceStimuliName'].replace(rename_mapping) 

        survey_to_video_map = {
            "survey_HP-7": "HP_7_H",
            "survey_HP-3": "HP_3_H",
            "survey_HP-1": "HP_1_L",
            "survey_HP-5": "HP_5",
            "survey_HP-6": "HP_6",
            "survey_HN-7": "HN_7",
            "survey_LP-8": "LP_9",
        }

        for survey, target_video in survey_to_video_map.items():
            # Extract the survey data
            survey_data = final_data[final_data['SourceStimuliName'] == survey].copy()
            if survey_data.empty:
                print(f"Survey {survey} not found, skipping.")
                continue

            # Remove it from final_data (won't be in output anymore)
            final_data = final_data[final_data['SourceStimuliName'] != survey]

            # Get the latest timestamp of the target video
            max_time = final_data[final_data['SourceStimuliName'] == target_video]['Timestamp'].max()
            if pd.isna(max_time):
                print(f"Target video {target_video} not found or has no timestamp, skipping.")
                continue

            # Reset and shift survey timestamps
            survey_data = survey_data.reset_index(drop=True)
            survey_data['Timestamp'] = max_time + 0.001 + (survey_data['Timestamp'] - survey_data['Timestamp'].min())

            # Rename to target video name
            survey_data['SourceStimuliName'] = target_video

            # Append to final_data
            final_data = pd.concat([final_data, survey_data], axis=0)

        # Saving the final dataset
        final_data['Timestamp'] = final_data.groupby('SourceStimuliName')['Timestamp'].transform(lambda x: x - x.min()) #Adjusting to 0 each stimuli, just to confirm
        final_data = final_data[columns_to_include]
        final_output_path = os.path.join(output_folder, f"filtered_{filename}")
        final_data.to_csv(final_output_path, index=False)

print(f"Final CSVs file with videos splitted in intervals created successfully at: {final_output_path}")


































