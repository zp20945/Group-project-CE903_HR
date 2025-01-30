import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.signal import find_peaks

# Function to calculate RR intervals from detected peaks
def calculate_rr_intervals(ppg_signal, timestamps, fs=128):
    peaks, _ = find_peaks(ppg_signal, distance=fs // 2)  # Minimum peak distance = 0.5s
    peak_times = timestamps.iloc[peaks].values  # Extracting the corresponding timestamps
    rr_intervals = np.diff(peak_times)  # Time differences between consecutive peaks
    return rr_intervals

#Function to calculate the SD1 and SD2 and create the PointCarePlot
def process_poincare(stimulus, rr_intervals, file_name, results, output_dir):
    if len(rr_intervals) > 1:
        # Creating pairs of RR intervals
        ibi_pairs = pd.DataFrame({'IBI_n': rr_intervals[:-1], 'IBI_n+1': rr_intervals[1:]})

        # Mean values for center of ellipse
        mean_n = np.mean(ibi_pairs['IBI_n'])
        mean_n1 = np.mean(ibi_pairs['IBI_n+1'])

        # Calculating SD1, SD2, and Parasympathetic Index
        SD1 = np.sqrt(np.sum((ibi_pairs['IBI_n+1'] - mean_n1) ** 2) / (2 * len(ibi_pairs['IBI_n+1'])))
        SD2 = np.sqrt(np.sum((ibi_pairs['IBI_n'] - mean_n) ** 2) / (len(ibi_pairs['IBI_n']) - 1))
        parasympathetic_index = SD1 / SD2 if SD2 != 0 else np.nan

        # Appending the metrics to results
        results.append({
            'File': file_name,
            'Stimulus': stimulus,
            'SD1': SD1,
            'SD2': SD2,
            'Parasympathetic': parasympathetic_index
        })

        # Poincaré plotting
        plt.figure(figsize=(8, 8))
        plt.scatter(ibi_pairs['IBI_n'], ibi_pairs['IBI_n+1'], alpha=0.5, label="IBI Pairs")

        # Adding the Ellipse for SD1 and SD2
        ellipse = Ellipse((mean_n, mean_n1), width=2 * SD2, height=2 * SD1, edgecolor='red', fc='None', lw=2)
        plt.gca().add_patch(ellipse)

        # Format of the plot

        # Lines for SD1 and SD2
        plt.axline((mean_n, mean_n1), slope=-1, color='red', linestyle='--', label='SD1')
        plt.axline((mean_n, mean_n1), slope=1, color='blue', linestyle='--', label='SD2')
        
        # Annotations for SD1 and SD2
        plt.annotate('SD1', (mean_n - SD1 / 2, mean_n1 - SD1 / 2), color='red')
        plt.annotate('SD2', (mean_n + SD2 / 2, mean_n1 + SD2 / 2), color='blue')

        # File name to the plot title
        plt.title(f"Poincaré Plot for Stimulus: {stimulus}\nSD1={SD1:.2f}, SD2={SD2:.2f}, Parasympathetic={parasympathetic_index:.2f}")
        plt.xlabel("IBI_n (ms)")
        plt.ylabel("IBI_n+1 (ms)")
        plt.legend()
        plt.grid(True)

        # Saving the plot created 
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"{file_name}_poincare_plot_{stimulus}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Poincaré plot saved for {stimulus}: {plot_path}")
    else:
        print(f"Not enough data points for stimulus: {stimulus}, skipping.")

# Defining input CSV file path and output directory
input_file = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\filtered_ppg_signal_specific_columns_OnlyVideos.csv"  # Replace with your file path
output_dir = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Results\PointcarePlots"

# Loading the dataset
try:
    data = pd.read_csv(input_file)
    print("File loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {input_file}")
    exit()

# Extracting columns
columns_PPG = ['Timestamp', 'SourceStimuliName', 'Butterworth Filtered PPG Signal']
df_ppg = data[columns_PPG]
unique_stimuli = df_ppg['SourceStimuliName'].unique()
file_name = os.path.splitext(os.path.basename(input_file))[0]

# Results list to store metrics
results = []

# Process each stimulus
for stimulus in unique_stimuli:
    stimulus_data = df_ppg[df_ppg['SourceStimuliName'] == stimulus]
    if stimulus_data.empty:
        continue

    # Calculating RR intervals using "Butterworth Filtered PPG Signal"
    rr_intervals = calculate_rr_intervals(
        ppg_signal=stimulus_data['Butterworth Filtered PPG Signal'],
        timestamps=stimulus_data['Timestamp']
    )
    # Generating the Poincaré plot
    process_poincare(stimulus, rr_intervals, file_name, results, output_dir)

# Saving metrics to a csv file
results_df = pd.DataFrame(results)
os.makedirs(output_dir, exist_ok=True)
results_df.to_csv(os.path.join(output_dir, "PoincareMetrics.csv"), index=False)

print("Processing complete. Poincaré plots and metrics have been saved.")
