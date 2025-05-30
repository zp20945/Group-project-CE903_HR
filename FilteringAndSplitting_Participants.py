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

## Loading and Applying the previous function 
## Paths
input_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Healthy participants\Healthy participants\Participants"
output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\filterdata"

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

        # Defining split intervals 

        split_intervals = {
            "HP_1-1": {"original_interval": (20, 30), "merge_intervals": [(8, 9), (22, 23)], "new_video_name": "HP_1-2"},
            "HP_3-1": {"original_interval": (0, 30), "merge_intervals": [(30, 35)], "new_video_name": "HP_3-2"},
            "HP_7-1": {"original_interval": (3, 8), "merge_intervals": [(0, 3)], "new_video_name": "HP_7-2"},
            "HN_2-1": {"original_interval": (7, 22), "merge_intervals": [(0, 7)], "new_video_name": "HN_2-2"},
            "HN_3-1": {"original_interval": (42, 46), "merge_intervals": [(0, 10)], "new_video_name": "HN_3-2"},
            "LN_7-1": {"original_interval": (60, 78), "merge_intervals": [(0, 10)], "new_video_name": "LN_7-2"},
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
            "HP_2-1": [(7, 22)],
            "HP_4-1": [(10, 17)],
            "HP_5-1": [(5, 10)],
            "HP_6-1": [(2, 8)],
            "HP_8-1": [(1, 38)],
            "HN_1-1": [(0, 20)],
            "HN_4-1": [(6, 10), (17, 20)],  # two intervals
            "HN_5-1": [(10, 20)],
            "HN_6-1": [(10, 20)],
            "HN_7-1": [(0, 8)],
            "HN_8-1": [(13, 46)],
            "LN_1-1": [(0, 15)],
            "LN_2-1": [(0, 17)],
            "LN_3-1": [(0, 10)],
            "LN_4-1": [(0, 10)],
            "LN_5-1": [(10, 17)],
            "LN_6-1": [(0, 10)],
            "LN_8-1": [(0, 5)],
            "LP_1-1": [(14, 16)],
            "LP_2-1": [(1, 5)],
            "LP_3-1": [(2, 15)],
            "LP_4-1": [(14, 21)],
            "LP_5-1": [(6, 14)],
            "LP_6-1": [(1, 9)],
            "LP_7-1": [(9, 30)],
            "LP_8-1": [(3, 9)],
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


        # Saving the final dataset
        final_data['Timestamp'] = final_data.groupby('SourceStimuliName')['Timestamp'].transform(lambda x: x - x.min()) #Adjusting to 0 each stimuli
        final_data = final_data[columns_to_include]
        final_output_path = os.path.join(output_folder, f"filtered_{filename}")
        final_data.to_csv(final_output_path, index=False)

print(f"Final CSVs file with all intervals created successfully at: {final_output_path}")
