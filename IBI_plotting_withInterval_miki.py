import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import shutil  

# Defining input and main output folders
input_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\filterdata_whole"
main_output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\delimited_graphs"
intervals_file = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\interval.csv"  # Path to the intervals CSV

# Load interval data
intervals_df = pd.read_csv(intervals_file)

# Convert interval_start and interval_end to strings to handle multi-interval cases
intervals_df['interval_start'] = intervals_df['interval_start'].astype(str)
intervals_df['interval_end'] = intervals_df['interval_end'].astype(str)

# Trim the last two characters from stimulus names if they match "-X" pattern
intervals_df['stimuli_base'] = intervals_df['stimuli_names'].apply(lambda x: x[:-2] if '-' in x[-2:] else x)

# Recreating the output folder
if os.path.exists(main_output_folder):
    shutil.rmtree(main_output_folder)  
os.makedirs(main_output_folder, exist_ok=True)  

# Getting all CSV files in the folder
files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

for file in files:
    file_path = os.path.join(input_folder, file)
    data = pd.read_csv(file_path)

    # Extracting participant name
    participant_id = os.path.splitext(file)[0]
    participant_folder = os.path.join(main_output_folder, participant_id)
    os.makedirs(participant_folder, exist_ok=True)

    # Creating subfolders for PPG Peaks and Filtered IBIs
    ppg_folder = os.path.join(participant_folder, "PPG_Peaks")
    ibi_folder = os.path.join(participant_folder, "Filtered_IBIs")
    os.makedirs(ppg_folder, exist_ok=True)
    os.makedirs(ibi_folder, exist_ok=True)

    print(f"Processing participant: {participant_id}")

    # Relevant columns
    time_column = "Timestamp"  
    signal_column = "Butterworth Filtered PPG Signal"  
    stimulus_column = "SourceStimuliName"  

    # Getting unique stimuli
    unique_stimuli = data[stimulus_column].unique()

    for stimulus in unique_stimuli:
        stimulus_base = stimulus[:-2] if '-' in stimulus[-2:] else stimulus  
        print(f"Processing stimulus: {stimulus} (Base: {stimulus_base}) for participant {participant_id}")

        subset = data[data[stimulus_column] == stimulus]

        # Extracting time and signal data for this stimulus
        time = subset[time_column].values
        signal = subset[signal_column].values

        # Detecting peaks in the PPG signal
        sampling_rate = 128  
        distance = int(sampling_rate * 0.5)  
        peaks, _ = find_peaks(signal, distance=distance)

        # Get intervals for the stimulus
        stimulus_intervals = intervals_df[intervals_df['stimuli_base'] == stimulus_base]
        # print(f"Intervals found for {stimulus_base}:")
        # print(stimulus_intervals[['interval_start', 'interval_end']])

        # ---- PPG PLOT ----
        plt.figure(figsize=(12, 6))
        plt.plot(time, signal, label="PPG Signal")
        plt.plot(time[peaks], signal[peaks], "rx", label="Detected Peaks")
        plt.xlabel("Time (s)")
        plt.ylabel("Signal Amplitude")
        plt.title(f"PPG Signal with Detected Peaks over time, Stimulus: {stimulus}\nParticipant: {participant_id}")
        plt.legend()
        plt.grid()

        # Add shading for stimulus intervals (Fixing the 0-start issue)
        if not stimulus_intervals.empty:
            for _, row in stimulus_intervals.iterrows():
                start_times = str(row['interval_start']).split(';')
                end_times = str(row['interval_end']).split(';')

                for start, end in zip(start_times, end_times):
                    try:
                        start = float(start)
                        end = float(end)
                        
                        # Ensure start = 0 gets included
                        if start == 0 or time.min() <= start <= time.max():
                            plt.axvspan(start, end, color='red', alpha=0.3)  
                            # print(f"Shading PPG from {start} to {end}")
                    except ValueError:
                        print(f"Skipping invalid interval: {start}-{end} for stimulus {stimulus}")

        plt.savefig(os.path.join(ppg_folder, f"PPG_peaks_{stimulus}.png"))
        plt.close()

        # ---- IBI CALCULATION ----
        ibi = np.diff(time[peaks])  
        ibi_filtered = ibi[(ibi > 0.3) & (ibi < 2)]  
        ibi_time = time[peaks][1:len(ibi_filtered)+1]  

        # ---- IBI PLOT ----
        plt.figure(figsize=(12, 6))
        plt.plot(ibi_time, ibi_filtered, marker='o', linestyle='-', color='b')
        plt.xlabel("Time (s)")
        plt.ylabel("IBI (s)")
        plt.title(f"Filtered Inter-Beat Interval (IBI) Plot - Stimulus: {stimulus}\nParticipant: {participant_id}")
        plt.grid()

        # Apply shading to IBI plots (Fixing the 0-start issue)
        if not stimulus_intervals.empty:
            for _, row in stimulus_intervals.iterrows():
                start_times = str(row['interval_start']).split(';')
                end_times = str(row['interval_end']).split(';')

                for start, end in zip(start_times, end_times):
                    try:
                        start = float(start)
                        end = float(end)
                        
                        # Ensure start = 0 is included
                        if start == 0 or ibi_time.min() <= start <= ibi_time.max():
                            plt.axvspan(start, end, color='red', alpha=0.3)
                            # print(f"Shading IBI from {start} to {end}")
                    except ValueError:
                        print(f"Skipping invalid interval: {start}-{end} for stimulus {stimulus}")

        plt.savefig(os.path.join(ibi_folder, f"IBI_{stimulus}.png"))
        plt.close()

print("Processing complete. All results saved in", main_output_folder)
