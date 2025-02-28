import os
import pandas as pd  
import shutil 

# Loading the extracted HRV features CSV
features_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Features_Baseline_Part_Whole\Features_Baseline_Part.csv"

#output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Participant_Analysis_whole_more_features_with_Baseline"

# Creating a folder for storing plots
#if os.path.exists(output_folder):
#    shutil.rmtree(output_folder)  
#os.makedirs(output_folder)

output_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Features_Baseline_Part_Whole\Features_Baseline_Part_norm.csv"

# Reading the CSV file
df = pd.read_csv(features_path)

# Identifying baselines for each participant
baseline_dict = {}
video_data = []

# Looping through each row in the dataset
for index, row in df.iterrows():
    participant = row["Participant"]
    name = row["SourceStimuliName"]

    if "Baseline" in name:
        # Storing baseline features indexed by Participant & Baseline Name
        baseline_dict[(participant, name)] = row
    else:
        # Storing video data separately
        video_data.append(row)

# Converting video_data to a DataFrame
video_df = pd.DataFrame(video_data)

# Function to find corresponding baseline for each participant and video
def find_baseline(participant, video_name):
    base_name_parts = video_name.split("_")
    base_name = f"{base_name_parts[0]}_{base_name_parts[1]}"  # Extract base part (e.g., HP_7)
    return baseline_dict.get((participant, f"Baseline_{base_name}"), None)

# Normalizing each video's features using its participant-specific baseline
normalized_data = []

for index, row in video_df.iterrows():
    participant = row["Participant"]
    video_name = row["SourceStimuliName"]
    baseline_row = find_baseline(participant, video_name)
    
    if baseline_row is not None:
        normalized_features = {"Participant": participant, "SourceStimuliName": video_name}
        
        # Normalizing each feature (skip non-numeric columns)
        for col in df.columns[2:]:  # Skipping 'Participant' and 'SourceStimuliName'
            baseline_value = baseline_row[col]
            video_value = row[col]
            
            # Avoiding division by zero
            if pd.notna(baseline_value) and abs(baseline_value) > 1e-6: # Avoid NaN and near-zero division
                normalized_features[col] = (video_value - baseline_value) / baseline_value
            else:
                normalized_features[col] = 0  # Assign 0 if baseline value is zero
        
        normalized_data.append(normalized_features)

# Converting to DataFrame
normalized_df = pd.DataFrame(normalized_data)

# Saving the normalized features to a new CSV
normalized_df.to_csv(output_path, index=False)

print(f"Normalized features saved to {output_path}")