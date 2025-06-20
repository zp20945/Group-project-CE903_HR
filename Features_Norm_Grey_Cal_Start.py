import pandas as pd
import numpy as np

# File paths
features_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\PPG_HR_Analysis_Longer_Intervals\merged_features.csv"
output_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\PPG_HR_Analysis_Longer_Intervals\Grey_Cal\norm_Grey_Cal_Start_residual_.csv"
    
 
# Reading the CSV file
df = pd.read_csv(features_path)

# Creating an empty list to store normalized data for all participants
normalized_data = []

# Getting the list of participants
participants = df["Participant"].unique()

# Iterating over each participant
for participant in participants:
    # Filtering the data for the current participant
    participant_df = df[df["Participant"] == participant]

    # Finding the normalization reference row for this participant
    norm_ref_row = participant_df[participant_df["SourceStimuliName"] == "Grey_cal_start"]

    # Checking if the Grey_scal_Start reference exists
    if norm_ref_row.empty:
        print(f"Warning: 'Grey_Scal_Start' reference row not found for Participant {participant}. Skipping normalization for this participant.")
        continue

    # Converting Grey_Scal_Start row to a dictionary for easy access
    norm_reference = norm_ref_row.iloc[0].to_dict()

    # Filtering out baselines, End_baseline, and meditation-related rows
    video_df = participant_df[
        ~participant_df["SourceStimuliName"].str.contains("Baseline|End_baseline|MEDITATION AUDIO|meditation_audio|Grey_Cal", case=False, na=False)
    ]

    # Normalizing each video row using the participant's Grey_scal_start reference
    for index, row in video_df.iterrows():
        # Preserving the Participant and SourceStimuliName columns
        normalized_features = {
            "Participant": row["Participant"],
            "SourceStimuliName": row["SourceStimuliName"]
        }

        # Normalizing the features
        for col in df.columns[2:]:  # Skipping 'Participant' and 'SourceStimuliName'
            norm_value = norm_reference[col]
            video_value = row[col]

            # Ensuring values are numeric
            try:
                norm_value = float(norm_value)
                video_value = float(video_value)

                if norm_value != 0:
                    normalized_features[col] = (video_value - norm_value) / norm_value
                else:
                    normalized_features[col] = np.nan 
                    

            except ValueError:
                print("0")
                normalized_features[col] = 0

        # Appending the normalized row
        normalized_data.append(normalized_features)

# Converting to DataFrame
normalized_df = pd.DataFrame(normalized_data)

# Saving the normalized features to a new CSV
normalized_df.to_csv(output_path, index=False)

print(f"Normalized features for all participants saved to {output_path}")
