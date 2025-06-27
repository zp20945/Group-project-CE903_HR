import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from matplotlib.patches import Ellipse


input_folder = r"c:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\PPG_HR_Analysis_Longer_Intervals_\PPG_HR_Analysis_Longer_Intervals"
output_folder = r"c:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\PPG_HR_Analysis_Longer_Intervals_\PoincarePlots"
csv_output_path = os.path.join(output_folder, "All_Participants_PoincareMetrics.csv")
fs_ppg = 128  

# Compute RR from PPG
def calculate_rr_intervals_from_ppg(timestamps, ppg_signal, fs=fs_ppg):
    ppg_signal = ppg_signal.astype(float).dropna()
    timestamps = timestamps.astype(float)
    
    if len(ppg_signal) < fs * 5:
        return None  

    peaks, _ = find_peaks(ppg_signal, distance=fs * 0.4)
    if len(peaks) < 3:
        return None

    try:
        peak_times = timestamps.iloc[peaks].values
        rr_intervals = np.diff(peak_times)
        return pd.Series(rr_intervals)
    except:
        return None

# Compute Poincaré features
def process_poincare(stimulus, rr_intervals, file_name, results, participant_output_dir):
    if len(rr_intervals) > 1:
        ibi_n = rr_intervals[:-1].values
        ibi_n1 = rr_intervals[1:].values
        ibi_pairs = pd.DataFrame({'IBI_n': ibi_n, 'IBI_n+1': ibi_n1})

        mean_n = np.mean(ibi_pairs['IBI_n'])
        mean_n1 = np.mean(ibi_pairs['IBI_n+1'])

        SD1 = np.sqrt(np.sum((ibi_pairs['IBI_n+1'] - mean_n1) ** 2) / (2 * len(ibi_pairs['IBI_n+1'])))
        SD2 = np.sqrt(np.sum((ibi_pairs['IBI_n'] - mean_n) ** 2) / (len(ibi_pairs['IBI_n']) - 1))
        parasympathetic_index = SD1 / SD2 if SD2 != 0 else np.nan

        num_crossing_points = 0
        angles = np.arange(0, 360, 30)
        crossing_points = []

        for angle in angles:
            m = np.tan(np.radians(angle))
            b = mean_n1 - m * mean_n

            for i in range(len(ibi_n) - 1):
                x0, x1 = ibi_n[i], ibi_n[i + 1]
                y0, y1 = ibi_n1[i], ibi_n1[i + 1]

                if (y0 - (m * x0 + b)) * (y1 - (m * x1 + b)) < 0:
                    num_crossing_points += 1
                    crossing_x = (b - y0 + m * x0) / (m + 1e-10)
                    crossing_y = m * crossing_x + b
                    crossing_points.append((crossing_x, crossing_y))

        crossing_points = np.array(crossing_points)

        min_area = max_area = mean_area = F5 = F6 = F7 = F8 = F9 = F10 = np.nan

        try:
            if len(crossing_points) > 2:
                epsilon = 1e-10
                crossing_x_norm = (crossing_points[:, 0] - np.mean(crossing_points[:, 0])) / (np.std(crossing_points[:, 0]) + epsilon)
                crossing_y_norm = (crossing_points[:, 1] - np.mean(crossing_points[:, 1])) / (np.std(crossing_points[:, 1]) + epsilon)

                min_area = np.min(np.abs(crossing_x_norm * crossing_y_norm))
                max_area = np.max(np.abs(crossing_x_norm * crossing_y_norm))
                mean_area = np.mean(np.abs(crossing_x_norm * crossing_y_norm))

                mean_x = np.mean(crossing_points[:, 0]) + epsilon
                mean_y = np.mean(crossing_points[:, 1]) + epsilon

                F5 = np.std(crossing_points[:, 0] / mean_x)
                F6 = np.std(crossing_points[:, 1] / mean_y)
                F7 = skew(crossing_x_norm)
                F8 = skew(crossing_y_norm)
                F9 = kurtosis(crossing_x_norm)
                F10 = kurtosis(crossing_y_norm)

        except Exception as e:
            print(f"Error processing Poincaré metrics: {e}")

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

        # Plots
        plt.figure(figsize=(8, 8))
        plt.scatter(ibi_pairs['IBI_n'], ibi_pairs['IBI_n+1'], alpha=0.5, label="IBI Pairs")

        ellipse = Ellipse((mean_n, mean_n1), width=2 * SD2 if SD2 > 1e-6 else 1e-3, height=2 * SD1 if SD1 > 1e-6 else 1e-3,
                          edgecolor='red', fc='None', lw=2)
        plt.gca().add_patch(ellipse)

        plt.axline((mean_n, mean_n1), slope=-1, color='red', linestyle='--', label='SD1')
        plt.axline((mean_n, mean_n1), slope=1, color='blue', linestyle='--', label='SD2')

        plt.annotate('SD1', (mean_n - SD1 / 2, mean_n1 - SD1 / 2), color='red')
        plt.annotate('SD2', (mean_n + SD2 / 2, mean_n1 + SD2 / 2), color='blue')

        plt.title(f"Poincaré Plot for Stimulus: {stimulus}\nSD1={SD1:.2f}, SD2={SD2:.2f}, Parasympathetic={parasympathetic_index:.2f}")
        plt.xlabel("IBI_n (s)")
        plt.ylabel("IBI_n+1 (s)")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(participant_output_dir, f"{file_name}_poincare_plot_{stimulus}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Poincaré plot saved for {stimulus}: {plot_path}")

# Preparing Output Folder 
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

# Main Loop
for file in os.listdir(input_folder):
    if not file.endswith(".csv"):
        continue

    input_file = os.path.join(input_folder, file)
    participant_name = os.path.splitext(file)[0].split("_")[-1]
    participant_output_dir = os.path.join(output_folder, participant_name)
    os.makedirs(participant_output_dir, exist_ok=True)

    try:
        data = pd.read_csv(input_file)
        print(f"Processing file: {file}")
    except FileNotFoundError:
        print(f"Error: File not found at {input_file}")
        continue

    columns_needed = ['Timestamp', 'SourceStimuliName', 'Butterworth Filtered PPG Signal']
    df_ppg = data[columns_needed]
    unique_stimuli = df_ppg['SourceStimuliName'].unique()

    results = []

    for stimulus in unique_stimuli:
        stim_data = df_ppg[df_ppg['SourceStimuliName'] == stimulus]
        if stim_data.empty:
            continue

        rr_intervals = calculate_rr_intervals_from_ppg(
            stim_data['Timestamp'],
            stim_data['Butterworth Filtered PPG Signal']
        )

        if rr_intervals is None or len(rr_intervals) < 3:
            print(f"Skipping {stimulus}: not enough valid RR intervals.")
            continue

        process_poincare(stimulus, rr_intervals, participant_name, results, participant_output_dir)

    # Saving per participant
    results_df = pd.DataFrame(results)
    participant_csv_path = os.path.join(participant_output_dir, "PoincareMetrics.csv")
    results_df.to_csv(participant_csv_path, index=False)

    # Saving consolidated CSV
    if not results_df.empty:
        if not os.path.exists(csv_output_path):
            results_df.to_csv(csv_output_path, index=False, mode='w')
        else:
            results_df.to_csv(csv_output_path, index=False, mode='a', header=False)

    print(f"Finished {participant_name}. Metrics saved.")
