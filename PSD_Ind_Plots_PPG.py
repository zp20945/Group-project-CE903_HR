import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from scipy.signal import find_peaks, welch
from scipy.interpolate import interp1d

input_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\PPG_HR_Analysis_Longer_Intervals_\PPG_HR_Analysis_Longer_Intervals"
output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\PPG_HR_Analysis_Longer_Intervals_\PPG\PSD_Plots"
csv_output_path = os.path.join(output_folder, "true_psd_min_max_frequencies.csv")


if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)


fs_ppg = 128  # Hz
fs_interp = 4.0  # Hz for RR resampling

vlf_band = (0.00, 0.04)
lf_band = (0.04, 0.15)
hf_band = (0.15, 0.5)

min_max_frequencies = []


for file in os.listdir(input_folder):
    if not file.endswith(".csv"):
        continue

    file_path = os.path.join(input_folder, file)
    print(f"\nProcessing: {file}")

    df = pd.read_csv(file_path)
    respondent_name = df['Respondent Name'].iloc[0]
    grouped = df.groupby("SourceStimuliName")

    participant_folder = os.path.join(output_folder, f"Participant_{respondent_name}")
    os.makedirs(participant_folder, exist_ok=True)

    for stimulus_name, group in grouped:
        timestamps = group["Timestamp"].astype(float)
        ppg_signal = group["Butterworth Filtered PPG Signal"].astype(float).dropna()

        # Detecting peaks
        if len(ppg_signal) < fs_ppg * 5:
            continue

        peaks, _ = find_peaks(ppg_signal, distance=fs_ppg * 0.4)
        if len(peaks) < 4:
            continue

        try:
            peak_times = timestamps.iloc[peaks].values
        except:
            continue

        rr_intervals = np.diff(peak_times)
        if len(rr_intervals) < 4 or np.any(rr_intervals <= 0):
            continue

        # Interpolating RR
        cumulative_time = np.cumsum(rr_intervals)
        if cumulative_time[-1] < 5:
            continue

        try:
            uniform_times = np.arange(0, cumulative_time[-1], 1 / fs_interp)
            interp_func = interp1d(cumulative_time, rr_intervals, kind='cubic', fill_value="extrapolate", bounds_error=False)
            rr_uniform = interp_func(uniform_times)
        except:
            continue

        # Welch PSD
        freqs, psd = welch(rr_uniform, fs=fs_interp, nperseg=min(256, len(rr_uniform)))

        # Min/max frequency
        min_freq = freqs[np.argmin(psd)]
        max_freq = freqs[np.argmax(psd)]
        min_max_frequencies.append([respondent_name, stimulus_name, min_freq, max_freq])

        # Plotting
        plt.figure(figsize=(10, 3))
        plt.plot(freqs, psd, color='black', linewidth=1.5, label="True PSD")

        plt.fill_between(freqs, psd, where=(freqs <= vlf_band[1]), color='gray', alpha=0.6, label="VLF")
        plt.fill_between(freqs, psd, where=((freqs > lf_band[0]) & (freqs <= lf_band[1])), color='cyan', alpha=0.6, label="LF")
        plt.fill_between(freqs, psd, where=((freqs > hf_band[0]) & (freqs <= hf_band[1])), color='yellow', alpha=0.6, label="HF")

        for cut in [vlf_band[1], lf_band[1], hf_band[1]]:
            plt.axvline(x=cut, color='red', linestyle='--', linewidth=1)

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD ($s^2$/Hz)")
        plt.title(f"PSD - {stimulus_name} | Participant {respondent_name}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlim(0, 0.6)
        plt.legend()

        # Saving plot
        plot_path = os.path.join(participant_folder, f"True_PSD_{stimulus_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot for {stimulus_name}")

# Saving frequency summary
df_freqs = pd.DataFrame(min_max_frequencies, columns=["Participant", "Stimulus", "Min Frequency (Hz)", "Max Frequency (Hz)"])
df_freqs.to_csv(csv_output_path, index=False)
print(f"\nAll plots saved in: {output_folder}")
print(f"Frequency summary saved in: {csv_output_path}")
