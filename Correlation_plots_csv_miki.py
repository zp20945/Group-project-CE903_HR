import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

# Loading the merged dataset
file_path = r"c:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Important_Features\All_Features.csv"
df = pd.read_csv(file_path)

# Defining the output folder for participant data (plots + CSVs)
output_folder = r"c:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Important_Features\Correlation_m"

# Ensuring a fresh start by deleting and recreating the folder
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

# Identifying feature columns (excluding Participant, Stimulus, and Arousal)
feature_columns = [col for col in df.columns if col not in ["Participant", "Stimulus", "Arousal"]]

# Global storage for all participants' data and correlations
all_participants_data = []
correlation_data = []
participant_correlations = []  # For storing per-participant correlations and top 4 features

# Getting the list of unique participants
participants = df["Participant"].unique()

# Calculate the total number of participants
total_participants = len(participants)

# Generating plots and CSVs for each participant
for completed_participants, participant in enumerate(participants, 1):  # Starting from 1
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

    # Storing per-participant correlations for later analysis
    participant_corr = {feature: correlations[feature] for feature in feature_columns}
    participant_corr["Participant"] = participant

    # ✅ NEW: Compute top four features based on absolute correlation for each participant
    abs_correlations = {feature: abs(correlations[feature]) for feature in feature_columns}
    
    # Get top 4 features based on absolute correlation
    top_four_features = sorted(abs_correlations, key=abs_correlations.get, reverse=True)[:8]

    # Add the top four features to the participant's correlation data
    participant_corr["Top_8_Features"] = ', '.join(top_four_features)

    # Adding the participant correlation data to the global list
    participant_correlations.append(participant_corr)

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

    # ✅ NEW: Compute stimulus-wise correlations
    stimulus_corr_list = []

    for index, row in participant_df.iterrows():
        stimulus = row["Stimulus"]
        stimulus_corr = {"Stimulus": stimulus}

        # Computing correlation across all stimuli but storing results per stimulus
        for feature in feature_columns:
            temp_df = participant_df.drop(index)  # Exclude current stimulus for correlation calculation
            if temp_df[feature].nunique() > 1:
                stimulus_corr[feature] = np.corrcoef(temp_df["Arousal"], temp_df[feature])[0, 1]
            else:
                stimulus_corr[feature] = np.nan  # Undefined correlation

        stimulus_corr_list.append(stimulus_corr)

    # Saving the row-wise correlation CSV in the participant's folder
    stimulus_corr_df = pd.DataFrame(stimulus_corr_list)
    stimulus_corr_csv_path = os.path.join(participant_folder, f"{participant}_stimulus_correlation.csv")
    stimulus_corr_df.to_csv(stimulus_corr_csv_path, index=False)

    # ✅ Print progress update for each participant
    print(f"✅ Completed processing for participant {completed_participants} out of {total_participants}: {participant}")

# Saving Global CSV (All Participants Feature Values) 
global_csv_path = os.path.join(output_folder, "all_participants_data.csv")
pd.concat(all_participants_data).to_csv(global_csv_path, index=False)

# Saving Correlation Summary CSV 
correlation_csv_path = os.path.join(output_folder, "correlation_summary.csv")
pd.DataFrame(correlation_data).to_csv(correlation_csv_path, index=False)

# Saving Per-Participant Correlations with Top 4 Features (Participant as first column)
participant_corr_df = pd.DataFrame(participant_correlations)

# Ensure the "Participant" column is the first column
participant_corr_df = participant_corr_df[["Participant"] + [col for col in participant_corr_df.columns if col != "Participant"]]

participant_corr_csv_path = os.path.join(output_folder, "participant_correlations_with_top_4.csv")
participant_corr_df.to_csv(participant_corr_csv_path, index=False)

print("\n🎉 All processing complete!")
print(f"📂 All CSVs and plots saved in: {output_folder}")
print(f"📄 Global CSV saved at: {global_csv_path}")
print(f"📊 Correlation summary saved at: {correlation_csv_path}")
print(f"🔍 Stimulus-wise correlation CSVs saved for each participant.")
print(f"📈 Per-participant correlations (with top 4 features) saved at: {participant_corr_csv_path}")
