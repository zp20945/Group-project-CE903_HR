import os
import pandas as pd  

# File paths
features_path = r"c:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\For_Comparing\Filter_HR\Features_HR.csv"
output_path = r"c:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\For_Comparing\Filter_HR\Features_HR_norm_baselines.csv"

# Read the CSV file
df = pd.read_csv(features_path)

# Identifying baselines (no need to track multiple participants)
baseline_dict = {}
video_data = []

# Loop through each row in the dataset
for index, row in df.iterrows():
    name = row["SourceStimuliName"]

    if "Baseline" in name:
        # Storing baseline features indexed by baseline name
        baseline_dict[name] = row
    else:
        # Storing video data separately
        video_data.append(row)

# Converting video_data to a DataFrame
video_df = pd.DataFrame(video_data)

# Function to find corresponding baseline for each video
def find_baseline(video_name):
    base_name_parts = video_name.split("_")

    # Ensuring at least 2 parts exist
    if len(base_name_parts) < 2:
        print(f"Warning: Unexpected format in video name '{video_name}', skipping baseline matching.")
        return None

    base_name = f"{base_name_parts[0]}_{base_name_parts[1]}"  # Extract base part (e.g., HP_7)

    return baseline_dict.get(f"Baseline_{base_name}", None)

# Normalizing each video's features using its corresponding baseline
normalized_data = []

for index, row in video_df.iterrows():
    video_name = row["SourceStimuliName"]
    baseline_row = find_baseline(video_name)

    if baseline_row is not None:
        # Preserving the Participant column
        normalized_features = {
            "Participant": row["Participant"],  # Include Participant column
            "SourceStimuliName": video_name
        }

        # Normalizing each feature (skip non-numeric columns)
        for col in df.columns[2:]:  # Skipping 'Participant' and 'SourceStimuliName'
            baseline_value = baseline_row[col]
            video_value = row[col]

            # Ensuring values are numeric
            try:
                baseline_value = float(baseline_value)
                video_value = float(video_value)

                # Avoiding division by zero
                if pd.notna(baseline_value) and abs(baseline_value) > 1e-6:
                    normalized_features[col] = (video_value - baseline_value) 
                else:
                    normalized_features[col] = 0  # Assign 0 if baseline value is zero
            except ValueError:
                normalized_features[col] = 0  # Handle conversion errors
        
        normalized_data.append(normalized_features)

# Converting to DataFrame
normalized_df = pd.DataFrame(normalized_data)

# Saving the normalized features to a new CSV
normalized_df.to_csv(output_path, index=False)

print(f"Normalized features saved to {output_path}.")
