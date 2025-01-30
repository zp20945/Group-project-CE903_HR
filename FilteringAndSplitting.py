import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, savgol_filter

## Function to delete he first rows of the file and exctract the participant name
def find_and_set_header(df, start_index=31, expected_header_indicator='Expected Header Indicator'):
    header = None
    for i in range(start_index, len(df)):
        row = df.iloc[i]
        #take the respondent name from the file
        respondent_name = df.iloc[1,1]
        
        if row[0] == expected_header_indicator:  # Check the first cell
            header = row
            df.columns = header  # Set this row as the header
            df = df[i+1:].reset_index(drop=True)  # Slice the DataFrame from the next row
            break
    return df, header, respondent_name

## Loading and Applying the previous function 
# Th CSV file
file_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Raw_Participants_Data\006_4FoNM.csv"
data = pd.read_csv(file_path)

# Applying the function to find and set the header
df, header, respondent_name = find_and_set_header(data, 0, 'Row') 

# Outputing the results
# print("Header row identified and applied:")
# print(header)

# print("\nRespondent name extracted:")
# print(respondent_name)

# Saving the processed DataFrame to a new CSV file
df.to_csv(r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Data_Without_Useless_Raws\processed_006_4FoNM_py.csv", index=False)
print("Processed file saved as 'processed_006_4FoNM_py.csv'")

## The new file to work with 
file_path = "processed_006_4FoNM_py.csv"
data = pd.read_csv(file_path)

ppg_signal = data['Internal ADC A13 PPG RAW']  
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

## Removing Baseline 
# Defining the high-pass Butterworth filter
def butter_highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

# Applying the high-pass filter to remove baseline drift
baseline_cutoff = 0.5  # Cutoff frequency for baseline drift
ppg_without_baseline = butter_highpass_filter(ppg_signal, baseline_cutoff, fs)

## For filtering 
# Defining the low-pass Butterworth filter
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs # Half the sampling rate, used for normalizing the cutoff frequency.
    normal_cutoff = cutoff / nyquist # Half the sampling rate, used for normalizing the cutoff frequency.
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data) # applies the filter using zero-phase filtering (avoids phase distortion).

# Applying low-pass filter
lowpass_cutoff = 2  # Cutoff frequency in Hz
filtered_butter = butter_lowpass_filter(ppg_without_baseline, lowpass_cutoff, fs)

## For the final result, a new file with the data filtered
# Adding the Butterworth filtered PPG signal as a new column
data['Butterworth Filtered PPG Signal'] = filtered_butter
data['Respondent Name'] = respondent_name

# Specifying the columns to include in the output
columns_to_include = ['Respondent Name','Timestamp', 'SourceStimuliName', 'Internal ADC A13 PPG RAW', 'Butterworth Filtered PPG Signal',]

# Filter the rows based on SourceStimuliName
source_stimuli_to_include = [
    "HN_1-1", "HN_2-1", "HN_3-1", "HN_4-1", "HN_5-1", "HN_6-1", "HN_7-1", "HN_8-1",
    "HP_1-1", "HP_2-1", "HP_3-1", "HP_4-1", "HP_5-1", "HP_6-1", "HP_7-1", "HP_8-1",
    "LN_1-1", "LN_2-1", "LN_3-1", "LN_4-1", "LN_5-1", "LN_6-1", "LN_7-1", "LN_8-1",
    "LP_1-1", "LP_2-1", "LP_3-1", "LP_4-1", "LP_5-1", "LP_6-1", "LP_7-1", "LP_8-1"
]

filtered_data = data[data['SourceStimuliName'].isin(source_stimuli_to_include)]

# Reseting timestamps for each video to start from 0 and convert milliseconds to seconds
filtered_data.loc[:, 'Timestamp'] = filtered_data.groupby('SourceStimuliName')['Timestamp'].transform(lambda x: (x - x.min()) / 1000.0)

# Defining split intervals directly in the code
split_intervals = {
    "HP_1-1": {"Start1": 20, "End1": 30, "Start2": 8.22, "End2": 9.23},
    "HP_3-1": {"Start1": 0, "End1": 30, "Start2": 30, "End2": 35},
    "HP_7-1": {"Start1": 3, "End1": 8, "Start2": 0, "End2": 3},
    "HN_2-1": {"Start1": 7, "End1": 22, "Start2": 0, "End2": 7},
    "HN_3-1": {"Start1": 42, "End1": 46, "Start2": 0, "End2": 10},
    "LN_7-1": {"Start1": 60, "End1": 78, "Start2": 0, "End2": 10},
}

# Initializing an empty DataFrame for the split data
split_data = pd.DataFrame()

# Looping through the videos to split based on the dictionary
for video, intervals in split_intervals.items():
    video_data = filtered_data[filtered_data['SourceStimuliName'] == video]

    # Part 1
    part1 = video_data[(video_data['Timestamp'] >= intervals["Start1"]) & (video_data['Timestamp'] <= intervals["End1"])]

    # Part 2
    part2 = video_data[(video_data['Timestamp'] >= intervals["Start2"]) & (video_data['Timestamp'] <= intervals["End2"])]
    part2['SourceStimuliName'] = "-".join(video.split("-")[:-1]) + "-2"

    # Annexing the split parts
    split_data = pd.concat([split_data, part1, part2], axis=0)

# Extracting data for videos that don't need splitting
remaining_videos = filtered_data[~filtered_data['SourceStimuliName'].isin(split_intervals.keys())]

# Combining the split data with the remaining videos
final_data = pd.concat([remaining_videos, split_data], axis=0)

# Saving the final dataset
columns_to_include = ['Respondent Name', 'Timestamp', 'SourceStimuliName', 'Internal ADC A13 PPG RAW', 'Butterworth Filtered PPG Signal']
final_data = final_data[columns_to_include]
final_data.to_csv(r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\FilteredData\filtered_ppg_signal_with_38videos.csv", index=False)

print("Final CSV file with 38 videos created successfully.")