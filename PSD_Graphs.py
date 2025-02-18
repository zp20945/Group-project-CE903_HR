import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from scipy.interpolate import CubicSpline  # Using spline for smooth PSD curve

# Loading the extracted HRV features from CSV
file_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Features_Part_Norm\Features_Part_Norm.csv"
df = pd.read_csv(file_path)
output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\PSD_Graphs_Norm"

# Defining frequency bands
vlf_band = (0.0, 0.04)
lf_band = (0.04, 0.15)
hf_band = (0.15, 0.5)

# Creating a folder for storing plots
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  
os.makedirs(output_folder)

# Iterating through each video stimulus
for video in df["SourceStimuliName"].unique():
    video_data = df[df["SourceStimuliName"] == video]

    # Creating a figure
    plt.figure(figsize=(10, 4))

    # Iterating through each participant
    for _, row in video_data.iterrows():
        # Extracting the PSD values for this participant
        psd_vlf = row["pVLF"]
        psd_lf = row["pLF"]
        psd_hf = row["pHF"]

        # Defining known frequency points and their PSD values
        freq_known = np.array([vlf_band[1], lf_band[1], hf_band[1]])  # 0.04, 0.15, 0.5 Hz
        psd_known = np.array([psd_vlf, psd_lf, psd_hf])

        # Generating a smooth frequency axis
        freq_values = np.linspace(0, 0.6, 200)

        # Using Cubic Spline for smooth PSD interpolation
        spline = CubicSpline(freq_known, psd_known, bc_type="natural")
        psd_values = spline(freq_values)

        # Ensuring PSD values don’t go negative
        psd_values = np.clip(psd_values, 0, None)

        # Plotting the individual participant's PSD
        plt.plot(freq_values, psd_values, linewidth=1, alpha=0.4)  # Alpha for transparency

    # Filling areas for different frequency bands (for overall visualization)
    plt.axvspan(vlf_band[0], vlf_band[1], color="gray", alpha=0.2, label="VLF")
    plt.axvspan(lf_band[0], lf_band[1], color="cyan", alpha=0.2, label="LF")
    plt.axvspan(hf_band[0], hf_band[1], color="yellow", alpha=0.2, label="HF")

    # Adding vertical red lines for frequency band separations
    for band in [vlf_band[1], lf_band[1], hf_band[1]]:
        plt.axvline(x=band, color="red", linestyle="--", linewidth=1.5)

    # Formatting
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD ($s^2/Hz$)")
    plt.title(f"PSD for {video} (All Participants)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0, 0.6)

    # Saving the plot
    save_path = os.path.join(output_folder, f"{video}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

print(f"PSD graphs saved in '{output_folder}' folder.")