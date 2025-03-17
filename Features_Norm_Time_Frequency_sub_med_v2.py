import pandas as pd

# File paths
features_path = r"c:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\HR_preprocessed\features.csv"
output_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\HR_preprocessed\features_norm_med_audio.csv"

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

    # Finding the meditation reference row for this participant
    meditation_ref_row = participant_df[participant_df["SourceStimuliName"] == "Meditation_audio_E"]

    # Checking if the meditation reference exists
    if meditation_ref_row.empty:
        print(f"Warning: 'Meditation_audio_E' reference row not found for Participant {participant}. Skipping normalization for this participant.")
        continue

    # Converting meditation_audio_E row to a dictionary for easy access
    meditation_reference = meditation_ref_row.iloc[0].to_dict()

    # Filtering out baselines, End_baseline, and meditation-related rows
    video_df = participant_df[
        ~participant_df["SourceStimuliName"].str.contains("Baseline|End_baseline|MEDITATION AUDIO|meditation_audio", case=False, na=False)
    ]

    # Normalizing each video row using the participant's meditation_audio_E reference
    for index, row in video_df.iterrows():
        # Preserving the Participant and SourceStimuliName columns
        normalized_features = {
            "Participant": row["Participant"],
            "SourceStimuliName": row["SourceStimuliName"]
        }

        # Normalizing the features
        for col in df.columns[2:]:  # Skipping 'Participant' and 'SourceStimuliName'
            meditation_value = meditation_reference[col]
            video_value = row[col]

            # Ensuring values are numeric
            try:
                meditation_value = float(meditation_value)
                video_value = float(video_value)


                normalized_features[col] = (video_value - meditation_value)

            except ValueError:
                normalized_features[col] = 0

        # Appending the normalized row
        normalized_data.append(normalized_features)

# Converting to DataFrame
normalized_df = pd.DataFrame(normalized_data)

# Saving the normalized features to a new CSV
normalized_df.to_csv(output_path, index=False)

print(f"Normalized features for all participants saved to {output_path}")
