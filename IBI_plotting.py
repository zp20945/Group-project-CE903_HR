import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import shutil  

# Defining input and main output folders
input_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\filterdata_whole"  
main_output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\IBI_results_plots"

# Recreating the output folder
if os.path.exists(main_output_folder):
    shutil.rmtree(main_output_folder)  # Removing the folder and its contents
os.makedirs(main_output_folder, exist_ok=True)  # Recreating the folder

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

    # The relevant columns
    time_column = "Timestamp"  # Replace with your actual timestamp column name
    signal_column = "Butterworth Filtered PPG Signal"  # Replace with your actual signal column name
    stimulus_column = "SourceStimuliName"  # Replace with the stimulus column name

    # Getting unique stimuli
    unique_stimuli = data[stimulus_column].unique()

    for stimulus in unique_stimuli:
        print(f"Processing stimulus: {stimulus} for participant {participant_id}")
        subset = data[data[stimulus_column] == stimulus]

        # Extracting time and signal data for this stimulus
        time = subset[time_column].values
        signal = subset[signal_column].values

        # Detecting peaks in the PPG signal
        sampling_rate = 128  
        distance = int(sampling_rate * 0.5)  # Minimum distance between peaks (~0.5 seconds for 60 bpm)
        peaks, _ = find_peaks(signal, distance=distance)

        # Saving PPG Signal with detected peaks in PPG_Peaks folder
        plt.figure(figsize=(12, 6))
        plt.plot(time, signal, label="PPG Signal")
        plt.plot(time[peaks], signal[peaks], "rx", label="Detected Peaks")
        plt.xlabel("Time (s)")
        plt.ylabel("Signal Amplitude")
        plt.title(f"PPG Signal with Detected Peaks over time, Stimuli: {stimulus}\nParticipant: {participant_id}")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(ppg_folder, f"PPG_peaks_{stimulus}.png"))
        plt.close()

        # Calculating IBIs (time difference between consecutive peaks)
        ibi = np.diff(time[peaks])  # Calculating intervals in seconds

        # Filtering IBIs to remove outliers
        ibi_filtered = ibi[(ibi > 0.3) & (ibi < 2)]  # Filter IBIs between 0.3 and 2 seconds

        # X-axis: Time at which the IBIs occurred
        ibi_time = time[peaks][1:len(ibi_filtered)+1]  # Corresponding times of IBIs (exclude the first peak)

        # Saving filtered IBIs as a plot in Filtered_IBIs folder
        plt.figure(figsize=(12, 6))
        plt.plot(ibi_time, ibi_filtered, marker='o', linestyle='-', color='b')
        plt.xlabel("Time (s)")  # Updated label to reflect time
        plt.ylabel("IBI (s)")
        plt.title(f"Filtered Inter-Beat Interval (IBI) Plot - Stimulus: {stimulus}\nParticipant: {participant_id}")
        plt.grid()
        plt.savefig(os.path.join(ibi_folder, f"IBI_{stimulus}.png"))
        plt.close()
       
print("Processing complete. All results saved in", main_output_folder)
