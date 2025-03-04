import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

# Loading the merged dataset
file_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Poincare_features_splitted_arousal\merged_file.csv"
df = pd.read_csv(file_path)

# Defining the output folder for participant data (plots + CSVs)
output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Poincare_features_splitted_arousal\Correlation_Plots_CSVs"

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
    participant_df = df[df["Participant"] == participant]  # Filter data for the participant
    
    # Creating a folder for this participant
    participant_folder = os.path.join(output_folder, participant)
    os.makedirs(participant_folder, exist_ok=True)

    # Generating and Save Individual CSV 
    participant_data = participant_df[["Participant", "Stimulus"] + feature_columns + ["Arousal"]]  
    csv_path = os.path.join(participant_folder, f"{participant}.csv")
    participant_data.to_csv(csv_path, index=False)

    # Storing this data for the global CSV
    all_participants_data.append(participant_data)

    # Computing Feature Correlations 
    correlations = {"Participant": participant}
    for feature in feature_columns:
        if participant_df[feature].nunique() > 1:  # Avoiding NaN correlation if feature has constant values
            correlations[feature] = np.corrcoef(participant_df["Arousal"], participant_df[feature])[0, 1]
        else:
            correlations[feature] = np.nan  # If feature has only one value, correlation is undefined

    correlation_data.append(correlations)

    # Generating and Save Plots 
    for feature in feature_columns:
        plt.figure(figsize=(8, 6))

        # Getting correlation value
        r_value = correlations[feature]

        # Scatter plot
        plt.scatter(participant_df["Arousal"], participant_df[feature], label=f"r = {r_value:.2f}" if not np.isnan(r_value) else "r = N/A", alpha=0.7)
        plt.xlabel("Arousal")
        plt.ylabel(feature)
        plt.title(f"{feature} vs Arousal for {participant}")
        plt.grid(True)
        plt.legend()

        # Saving the plot in the participant's folder
        plot_path = os.path.join(participant_folder, f"{feature}_vs_Arousal.png")
        plt.savefig(plot_path)
        plt.close()

# Saving Global CSV (All Participants Feature Values) 
global_csv_path = os.path.join(output_folder, "all_participants_data.csv")
pd.concat(all_participants_data).to_csv(global_csv_path, index=False)

# Saving Correlation Summary CSV 
correlation_csv_path = os.path.join(output_folder, "correlation_summary.csv")
pd.DataFrame(correlation_data).to_csv(correlation_csv_path, index=False)

print(f"All CSVs and plots saved in {output_folder}")
print(f"Global CSV saved at: {global_csv_path}")
print(f"Correlation summary saved at: {correlation_csv_path}")
