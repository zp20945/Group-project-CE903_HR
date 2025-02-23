import pandas as pd
import numpy as np
import os
import shutil  # For deleting the folder
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from matplotlib.patches import Ellipse
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull


# Function to calculate RR intervals from detected peaks
def calculate_rr_intervals(ppg_signal, timestamps, fs=128):
    peaks, _ = find_peaks(ppg_signal, distance=fs // 2)  # Minimum peak distance = 0.5s
    peak_times = timestamps.iloc[peaks].values  # Extracting the corresponding timestamps
    rr_intervals = np.diff(peak_times)  # Time differences between consecutive peaks
    return rr_intervals

# Function to process Poincaré plots and metrics
def process_poincare(stimulus, rr_intervals, file_name, results, participant_output_dir):
    if len(rr_intervals) > 1:
        # Creating pairs of RR intervals
        ibi_n = rr_intervals[:-1]  # RR intervals at n
        ibi_n1 = rr_intervals[1:]  # RR intervals at n+1
        ibi_pairs = pd.DataFrame({'IBI_n': ibi_n, 'IBI_n+1': ibi_n1})

        # Mean values for center of ellipse
        mean_n = np.mean(ibi_pairs['IBI_n'])
        mean_n1 = np.mean(ibi_pairs['IBI_n+1'])

        # Calculating SD1, SD2, and Parasympathetic Index
        SD1 = np.sqrt(np.sum((ibi_pairs['IBI_n+1'] - mean_n1) ** 2) / (2 * len(ibi_pairs['IBI_n+1'])))
        SD2 = np.sqrt(np.sum((ibi_pairs['IBI_n'] - mean_n) ** 2) / (len(ibi_pairs['IBI_n']) - 1))
        parasympathetic_index = SD1 / SD2 if SD2 != 0 else np.nan

        #  NEW FEATURES (F1 - F10) 
        # Poincaré Section Crossing Points (F1)
        num_crossing_points = 0
        angles = np.arange(0, 360, 30)  # Checking sections at 0° to 360° in steps of 30°
        crossing_points = []

        for angle in angles:
            m = np.tan(np.radians(angle))  # Compute slope of section
            b = mean_n1 - m * mean_n  # y-intercept
            
            # Compute intersections
            for i in range(len(ibi_n) - 1):
                x0, x1 = ibi_n[i], ibi_n[i + 1]
                y0, y1 = ibi_n1[i], ibi_n1[i + 1]

                if (y0 - (m * x0 + b)) * (y1 - (m * x1 + b)) < 0:  # Check if it crosses the line
                    num_crossing_points += 1
                    crossing_x = (b - y0 + m * x0) / (m + 1e-10)  # Avoid division by zero
                    crossing_y = m * crossing_x + b
                    crossing_points.append((crossing_x, crossing_y))

        crossing_points = np.array(crossing_points)


        min_area = np.nan
        max_area = np.nan
        mean_area = np.nan
        F5 = np.nan
        F6 = np.nan
        F7 = np.nan
        F8 = np.nan
        F9 = np.nan
        F10 = np.nan

        try:
            if len(crossing_points) > 2:
               
               epsilon = 1e-10  # Small constant to prevent division by zero
               crossing_x_norm = (crossing_points[:, 0] - np.mean(crossing_points[:, 0])) / (np.std(crossing_points[:, 0]) + epsilon)
               crossing_y_norm = (crossing_points[:, 1] - np.mean(crossing_points[:, 1])) / (np.std(crossing_points[:, 1]) + epsilon)

        # Compute Areas (F2, F3, F4)
               min_area = np.min(np.abs(crossing_x_norm * crossing_y_norm)) if np.min(np.abs(crossing_x_norm * crossing_y_norm)) > 0 else np.nan
               max_area = np.max(np.abs(crossing_x_norm * crossing_y_norm)) if np.max(np.abs(crossing_x_norm * crossing_y_norm)) > 0 else np.nan
               mean_area = np.mean(np.abs(crossing_x_norm * crossing_y_norm)) if np.mean(np.abs(crossing_x_norm * crossing_y_norm)) > 0 else np.nan

               # Normalize by the mean before computing standard deviation
               mean_x = np.mean(crossing_points[:, 0]) + 1e-10  # Avoid division by zero
               mean_y = np.mean(crossing_points[:, 1]) + 1e-10

               F5 = np.std(crossing_points[:, 0] / mean_x) if len(crossing_points) > 2 else np.nan
               F6 = np.std(crossing_points[:, 1] / mean_y) if len(crossing_points) > 2 else np.nan


               F7 = skew(crossing_x_norm) if len(crossing_x_norm) > 2 else np.nan
               F8 = skew(crossing_y_norm) if len(crossing_y_norm) > 2 else np.nan
               F9 = kurtosis(crossing_x_norm) if len(crossing_x_norm) > 2 else np.nan
               F10 = kurtosis(crossing_y_norm) if len(crossing_y_norm) > 2 else np.nan


        except Exception as e:
            print(f"Error processing Poincaré metrics: {e}")

        # ---- Append results ----
        results.append({
            'Participant': file_name.split("_")[-1],
            'Stimulus': stimulus,
            'SD1': SD1,
            'SD2': SD2,
            'Parasympathetic': parasympathetic_index,
            'F1_CrossingPoints': num_crossing_points,
            'F2_MinArea': min_area,
            'F3_MaxArea': max_area,
            'F4_MeanArea': mean_area,
            'F5_StdX': F5,
            'F6_StdY': F6,
            'F7_SkewX': F7,
            'F8_SkewY': F8,
            'F9_KurtX': F9,
            'F10_KurtY': F10
        })


        # Poincaré plotting
        plt.figure(figsize=(8, 8))
        plt.scatter(ibi_pairs['IBI_n'], ibi_pairs['IBI_n+1'], alpha=0.5, label="IBI Pairs")

        # Adding the Ellipse for SD1 and SD2
        ellipse = Ellipse((mean_n, mean_n1), width=2 * SD2, height=2 * SD1, edgecolor='red', fc='None', lw=2)
        plt.gca().add_patch(ellipse)

        # Format of the plot
        plt.axline((mean_n, mean_n1), slope=-1, color='red', linestyle='--', label='SD1')
        plt.axline((mean_n, mean_n1), slope=1, color='blue', linestyle='--', label='SD2')
        
        plt.annotate('SD1', (mean_n - SD1 / 2, mean_n1 - SD1 / 2), color='red')
        plt.annotate('SD2', (mean_n + SD2 / 2, mean_n1 + SD2 / 2), color='blue')

        plt.title(f"Poincaré Plot for Stimulus: {stimulus}\nSD1={SD1:.2f}, SD2={SD2:.2f}, Parasympathetic={parasympathetic_index:.2f}")
        plt.xlabel("IBI_n (ms)")
        plt.ylabel("IBI_n+1 (ms)")
        plt.legend()
        plt.grid(True)

        # Saving the plot
        plot_path = os.path.join(participant_output_dir, f"{file_name}_poincare_plot_{stimulus}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Poincaré plot saved for {stimulus}: {plot_path}")
    else:
        print(f"Not enough data points for stimulus: {stimulus}, skipping.")

# Defining input folder containing CSV files and output directory
input_folder = r"C:\\Users\\Salin\\OneDrive\\Documentos\\ESSEX\\DSPROJECT\\filterdata_whole"
output_folder = r"C:\\Users\\Salin\\OneDrive\\Documentos\\ESSEX\\DSPROJECT\\Participant_Analysis_whole_more_st_data"

# Remove the output folder if it exists to start fresh
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

# Path for consolidated results file
all_participants_csv = os.path.join(output_folder, "All_Participants_PoincareMetrics.csv")

# Processing each CSV file
for file in os.listdir(input_folder):
    if file.endswith(".csv"):
        input_file = os.path.join(input_folder, file)
        participant_name = os.path.splitext(file)[0].split("_")[-1]  # Extract part after the last underscore
        participant_output_dir = os.path.join(output_folder, participant_name)
        os.makedirs(participant_output_dir, exist_ok=True)
        
        try:
            data = pd.read_csv(input_file)
            print(f"Processing file: {file}")
        except FileNotFoundError:
            print(f"Error: File not found at {input_file}")
            continue
        
        # Extracting columns
        columns_PPG = ['Timestamp', 'SourceStimuliName', 'Butterworth Filtered PPG Signal']
        df_ppg = data[columns_PPG]
        unique_stimuli = df_ppg['SourceStimuliName'].unique()
        
        # Results list to store metrics
        results = []

        # Process each stimulus
        for stimulus in unique_stimuli:
            stimulus_data = df_ppg[df_ppg['SourceStimuliName'] == stimulus]
            if stimulus_data.empty:
                continue

            # Calculating RR intervals
            rr_intervals = calculate_rr_intervals(
                ppg_signal=stimulus_data['Butterworth Filtered PPG Signal'],
                timestamps=stimulus_data['Timestamp']
            )
            # Generating the Poincaré plot
            process_poincare(stimulus, rr_intervals, participant_name, results, participant_output_dir)

        # Save individual participant's Poincaré metrics
        results_df = pd.DataFrame(results)
        participant_csv_path = os.path.join(participant_output_dir, "PoincareMetrics.csv")
        results_df.to_csv(participant_csv_path, index=False)

        # Save consolidated CSV (overwrite every time)
        if not results_df.empty:
            if not os.path.exists(all_participants_csv):
                # Create the file with headers
                results_df.to_csv(all_participants_csv, index=False, mode='w')
            else:
                # Append without headers
                results_df.to_csv(all_participants_csv, index=False, mode='a', header=False)

        print(f"Processing complete for {participant_name}. Poincaré plots and metrics have been saved.")
        print(f"All participants' data updated in {all_participants_csv}")
