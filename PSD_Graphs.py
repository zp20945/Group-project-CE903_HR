import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from scipy.interpolate import CubicSpline

# Loading the extracted HRV features from CSV
file_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\filterdata_Int_with_Baselines_and_med\features.csv"
df = pd.read_csv(file_path)
output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\filterdata_Int_with_Baselines_and_med\PSD_PLOTS"

# Defining frequency bands
vlf_band = (0.0, 0.04)
lf_band = (0.04, 0.15)
hf_band = (0.15, 0.5)

# Creating a folder for storing plots
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  
os.makedirs(output_folder)

# Data storage for min/max frequency values
min_max_frequencies = []

# Iterating through each video stimulus
for video in df["SourceStimuliName"].unique():
    video_data = df[df["SourceStimuliName"] == video]

    plt.figure(figsize=(10, 4))

    min_freqs = []
    max_freqs = []

    for _, row in video_data.iterrows():
        psd_vlf = row["pVLF"]
        psd_lf = row["pLF"]
        psd_hf = row["pHF"]

        freq_known = np.array([vlf_band[1], lf_band[1], hf_band[1]])  # 0.04, 0.15, 0.5 Hz
        psd_known = np.array([psd_vlf, psd_lf, psd_hf])

        # Generating smooth frequency range
        freq_values = np.linspace(0, 0.6, 200)

        # Using Cubic Spline interpolation
        spline = CubicSpline(freq_known, psd_known, bc_type="natural")
        psd_values = spline(freq_values)
        psd_values = np.clip(psd_values, 0, None)  # Prevent negative values

        # Finding min/max PSD and corresponding frequencies
        min_freq = freq_values[np.argmin(psd_values)]
        max_freq = freq_values[np.argmax(psd_values)]
        min_freqs.append(min_freq)
        max_freqs.append(max_freq)

        # Plotting individual participant PSD
        plt.plot(freq_values, psd_values, linewidth=1, alpha=0.4)

    # Computing overall min/max for this stimulus
    min_frequency = np.min(min_freqs)
    max_frequency = np.max(max_freqs)

    # Storing the results in a list
    min_max_frequencies.append([video, min_frequency, max_frequency])

    # Highlight frequency bands
    plt.axvspan(vlf_band[0], vlf_band[1], color="gray", alpha=0.2, label="VLF")
    plt.axvspan(lf_band[0], lf_band[1], color="cyan", alpha=0.2, label="LF")
    plt.axvspan(hf_band[0], hf_band[1], color="yellow", alpha=0.2, label="HF")

    # Add vertical lines at band limits
    for band in [vlf_band[1], lf_band[1], hf_band[1]]:
        plt.axvline(x=band, color="red", linestyle="--", linewidth=1.5)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD ($s^2/Hz$)")
    plt.title(f"PSD for {video} (All Participants)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0, 0.6)

    # Saving the plot
    save_path = os.path.join(output_folder, f"{video}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# Converting results to DataFrame and save as CSV
csv_output_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\filterdata_Int_with_Baselines_and_med\min_max_frequencies.csv"
df_min_max = pd.DataFrame(min_max_frequencies, columns=["Stimulus", "Min Frequency (Hz)", "Max Frequency (Hz)"])
df_min_max.to_csv(csv_output_path, index=False)

# Printing the extracted min/max frequencies for each stimulus
print(df_min_max)
print(f"\nPSD graphs saved in '{output_folder}' folder.")
print(f"Min/Max frequency values saved in '{csv_output_path}'.")
