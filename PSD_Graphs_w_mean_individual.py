import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from scipy.interpolate import CubicSpline  # Using spline for smooth PSD curve

# Loading the extracted HRV features from CSV
file_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\filterdata_Int_with_Baselines_and_med\features.csv"
df = pd.read_csv(file_path)

# Main output folder
main_output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\filterdata_Int_with_Baselines_and_med\PSD_Plots_individual"
csv_output_path = os.path.join(main_output_folder, "min_max_frequencies_per_participant.csv")

# Remove existing output folder and recreate it
if os.path.exists(main_output_folder):
    shutil.rmtree(main_output_folder)
os.makedirs(main_output_folder)

# Defining frequency bands
vlf_band = (0.0, 0.04)
lf_band = (0.04, 0.15)
hf_band = (0.15, 0.5)

# Data storage for min/max frequency values
min_max_frequencies = []

# Iterate through each participant
for participant in df["Participant"].unique():
    participant_data = df[df["Participant"] == participant]

    # Create a folder for this participant
    participant_folder = os.path.join(main_output_folder, f"Participant_{participant}")
    os.makedirs(participant_folder, exist_ok=True)

    # Iterate through each stimulus
    for video in participant_data["SourceStimuliName"].unique():
        stimulus_data = participant_data[participant_data["SourceStimuliName"] == video]

        # Extract PSD values for this participant & stimulus
        psd_vlf = stimulus_data["pVLF"].values[0]
        psd_lf = stimulus_data["pLF"].values[0]
        psd_hf = stimulus_data["pHF"].values[0]

        # Define known frequency points and their PSD values
        freq_known = np.array([vlf_band[1], lf_band[1], hf_band[1]])  # 0.04, 0.15, 0.5 Hz
        psd_known = np.array([psd_vlf, psd_lf, psd_hf])

        # Generate a smooth frequency axis
        freq_values = np.linspace(0, 0.6, 200)

        # Use Cubic Spline for smooth PSD interpolation
        spline = CubicSpline(freq_known, psd_known, bc_type="natural")
        psd_values = spline(freq_values)

        # Ensure PSD values don’t go negative
        psd_values = np.clip(psd_values, 0, None)

        # Find min/max PSD and corresponding frequencies
        min_freq = freq_values[np.argmin(psd_values)]
        max_freq = freq_values[np.argmax(psd_values)]

        # Store min/max frequencies for this participant & stimulus
        min_max_frequencies.append([participant, video, min_freq, max_freq])

        # Create figure for individual participant & stimulus
        plt.figure(figsize=(10, 3))

        # Filling areas for different frequency bands
        plt.fill_between(freq_values, psd_values, where=(freq_values <= vlf_band[1]), color="gray", alpha=0.7, label="VLF")
        plt.fill_between(freq_values, psd_values, where=((freq_values > lf_band[0]) & (freq_values <= lf_band[1])), color="cyan", alpha=0.7, label="LF")
        plt.fill_between(freq_values, psd_values, where=((freq_values > hf_band[0]) & (freq_values <= hf_band[1])), color="yellow", alpha=0.7, label="HF")

        # Plot the smooth PSD curve
        plt.plot(freq_values, psd_values, color="black", linewidth=1.5)

        # Adding vertical red lines for frequency bands
        for band in [vlf_band[1], lf_band[1], hf_band[1]]:
            plt.axvline(x=band, color="red", linestyle="--", linewidth=1.5)

        # Formatting
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD ($s^2/Hz$)")
        plt.title(f"PSD for {video} - Participant {participant}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlim(0, 0.6)

        # Save plot inside the participant folder
        save_path = os.path.join(participant_folder, f"PSD_{video}_Participant_{participant}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

# Convert results to DataFrame and save as CSV
df_min_max = pd.DataFrame(min_max_frequencies, columns=["Participant", "Stimulus", "Min Frequency (Hz)", "Max Frequency (Hz)"])
df_min_max.to_csv(csv_output_path, index=False)

# Print the extracted min/max frequencies for each participant per stimulus
print(df_min_max)
print(f"\nPSD graphs saved in '{main_output_folder}' folder.")
print(f"Min/Max frequency values saved in '{csv_output_path}'.")
