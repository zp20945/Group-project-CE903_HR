import pandas as pd
import os
import shutil

# Loading the two CSV files
file1 = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Features_Baseline_Part_Interval\Features_Baseline_Part_norm.csv"  # Replace with actual file path
file2 = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\individual_ground_truth_without_hm2.csv"  # Replace with actual file path

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Renaming columns in df2 to match df1
df1.rename(columns={'SourceStimuliName': 'Stimulus'}, inplace=True)
# Renaming columns in df2 to match df1
df2.rename(columns={'Stimulus_Name': 'Stimulus'}, inplace=True)

# Merging on 'Participant' and 'Stimulus', keeping all data from df1 and adding 'Arousal' from df2
merged_df = df1.merge(df2[['Participant', 'Stimulus', 'Arousal']], on=['Participant', 'Stimulus'], how='left')

# Dropping columns SD1 and SD2
merged_df = merged_df.drop(columns=['SD1', 'SD2'], errors='ignore')

# Defining the output folder
output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Time_Frequency_features_splitted_arousal"

# Ensuring the folder is fresh by deleting and recreating it
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  # Delete folder if it exists
os.makedirs(output_folder)  # Recreate folder

# Saving the merged file
output_path = os.path.join(output_folder, "merged_file.csv")
merged_df.to_csv(output_path, index=False)

print(f"Merged file saved in: {output_path}")
