import pandas as pd  

# File paths
features_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\HR_preprocessed\features.csv"
output_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\HR_preprocessed\features_norm_med_audio.csv"

# Read the CSV file
df = pd.read_csv(features_path)

# Find the meditation_audio_E row (reference for normalization)
meditation_ref_row = df[df["SourceStimuliName"] == "Meditation_audio_E"]

# Ensure the meditation reference is found
if meditation_ref_row.empty:
    raise ValueError("Error: 'Meditation_audio_E' reference row not found in the dataset.")

# Convert meditation_audio_E row to a dictionary for easy access
meditation_reference = meditation_ref_row.iloc[0].to_dict()

# Filter out all baselines, End_baseline, and meditation-related rows
video_df = df[
    ~df["SourceStimuliName"].str.contains("Baseline|End_baseline|MEDITATION AUDIO|meditation_audio", case=False, na=False)
]

# Normalize each video row using meditation_audio_E as the reference
normalized_data = []

for index, row in video_df.iterrows():
    # Preserve the Participant column
    normalized_features = {
        "Participant": row["Participant"],  # Include Participant
        "SourceStimuliName": row["SourceStimuliName"]
    }

    for col in df.columns[2:]:  # Skipping 'Participant' and 'SourceStimuliName'
        meditation_value = meditation_reference[col]  # Using meditation_audio_E values as reference
        video_value = row[col]

        # Ensure values are numeric
        try:
            meditation_value = float(meditation_value)
            video_value = float(video_value)

            # Avoiding division by zero
            if pd.notna(meditation_value) and abs(meditation_value) > 1e-6:
                normalized_features[col] = (video_value - meditation_value)
            else:
                normalized_features[col] = 0  # Assign 0 if reference value is zero
        except ValueError:
            normalized_features[col] = 0  # Handle conversion errors

    normalized_data.append(normalized_features)

# Convert to DataFrame
normalized_df = pd.DataFrame(normalized_data)

# Save the normalized features to a new CSV
normalized_df.to_csv(output_path, index=False)

print(f"Normalized features saved to {output_path}")
