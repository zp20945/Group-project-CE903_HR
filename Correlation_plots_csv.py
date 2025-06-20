import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import re

# Function to normalize a pandas series between -1 and 1
def normalize_series(series):
    min_val = series.min()
    max_val = series.max()
    if min_val == max_val:
        # If constant series, return zeroes to avoid division by zero
        return pd.Series(0, index=series.index)
    return 2 * (series - min_val) / (max_val - min_val) - 1

# Loading the merged dataset
file_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\PPG_HR_Analysis_Longer_Intervals\Grey_Cal\norm_Grey_Cal_Start_residual_arousal.csv"
df = pd.read_csv(file_path)

# Defining the output folder for participant data (plots + CSVs)
output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\PPG_HR_Analysis_Longer_Intervals\Grey_Cal\correlations_norm_Grey_Cal_residual"

# Ensuring a fresh start by deleting and recreating the folder
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

# Identifying feature columns (excluding Participant, Stimulus, and Arousal)
feature_columns = [col for col in df.columns if col not in ["Participant", "Stimulus", "Arousal"]]

# Global storage for all participants' data and correlations
all_participants_data = []
correlation_data = []

# Getting the list of unique participants
participants = df["Participant"].unique()

# Generating plots and CSVs for each participant
for participant in participants:
    participant_df = df[df["Participant"] == participant].copy()  # Filtering data for the participant

    # Normalize arousal and all feature columns between -1 and 1
    participant_df["Arousal"] = normalize_series(participant_df["Arousal"])
    for feature in feature_columns:
        participant_df[feature] = normalize_series(participant_df[feature])

    # Creating a folder for this participant
    participant_folder = os.path.join(output_folder, participant)
    os.makedirs(participant_folder, exist_ok=True)

    # Generating and Save Individual CSV (normalized data)
    participant_data = participant_df[["Participant", "Stimulus"] + feature_columns + ["Arousal"]]
    csv_path = os.path.join(participant_folder, f"{participant}.csv")
    participant_data.to_csv(csv_path, index=False)

    # Storing this data for the global CSV
    all_participants_data.append(participant_data)

    # Computing Feature Correlations (on normalized data)
    correlations = {"Participant": participant}
    for feature in feature_columns:
        if participant_df[feature].nunique() > 1:  # Avoid NaN if feature has constant values
            correlation_value = np.corrcoef(participant_df["Arousal"], participant_df[feature])[0, 1]
            normalized_corr = np.clip(correlation_value, -1, 1)  # Clip correlation between -1 and 1
            correlations[feature] = normalized_corr
        else:
            correlations[feature] = np.nan  # Undefined correlation if no variation

    correlation_data.append(correlations)

    # Generating and Saving Plots (normalized data)
    for feature in feature_columns:
        safe_feature = re.sub(r'[\\/*?:"<>|]', "_", feature)
        plt.figure(figsize=(8, 6))

        # Getting correlation value for labeling
        r_value = correlations[feature]

        # Scattering plot using normalized data
        plt.scatter(participant_df["Arousal"], participant_df[feature],
                    label=f"r = {r_value:.2f}" if not np.isnan(r_value) else "r = N/A", alpha=0.7)

        plt.xlabel("Arousal (Normalized)")
        plt.ylabel(f"{feature} (Normalized)")
        plt.title(f"{feature} vs Arousal for {participant}")
        plt.grid(True)
        plt.legend()

        # Fixing axis boundaries from -1 to 1
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        # Sav the plot
        plot_path = os.path.join(participant_folder, f"{safe_feature}_vs_Arousal.png")
        plt.savefig(plot_path)
        plt.close()

# Saving Global CSV (All Participants Feature Values, normalized)
global_csv_path = os.path.join(output_folder, "all_participants_data.csv")
pd.concat(all_participants_data).to_csv(global_csv_path, index=False)

# Saving Correlation Summary CSV
correlation_csv_path = os.path.join(output_folder, "correlation_summary.csv")
pd.DataFrame(correlation_data).to_csv(correlation_csv_path, index=False)

print(f"All CSVs and plots saved in {output_folder}")
print(f"Global CSV saved at: {global_csv_path}")
print(f"Correlation summary saved at: {correlation_csv_path}")
