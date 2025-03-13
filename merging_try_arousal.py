import pandas as pd
import os
import shutil

# Loading the two CSV files
file1 =r"c:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\filterdata_Int_with_Baselines_and_med\features_norm_med_audio.csv"
file2 = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\individual_ground_truth_without_hm2.csv" 

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Renaming columns to match for merging
df1.rename(columns={'SourceStimuliName': 'Stimulus'}, inplace=True)
df2.rename(columns={'Stimulus_Name': 'Stimulus'}, inplace=True)

# Defining patterns to remove
patterns_to_remove = ["Baseline", "MEDITATION AUDIO", "Meditation_audio_E", "End_Baseline"]

# Removing rows where 'Stimulus' starts with any of the defined patterns
df1_filtered = df1[~df1['Stimulus'].astype(str).str.startswith(tuple(patterns_to_remove))]

# Merge only the filtered df1 with df2 on 'Participant' and 'Stimulus'
merged_df = df1_filtered.merge(df2[['Participant', 'Stimulus', 'Arousal']], on=['Participant', 'Stimulus'], how='left')

# Keeping only relevant columns
merged_df = merged_df[['Participant', 'Stimulus', 'SDNN', 'RMSSD', 'Arousal']]

# Defining the output folder
output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\For_Comparing\HR\norm_med_audio"

# Ensuring the folder is fresh by deleting and recreating it
# if os.path.exists(output_folder):
#     shutil.rmtree(output_folder)  # Delete folder if it exists
# os.makedirs(output_folder)  # Recreate folder

# Saving the merged file
output_path = r"c:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\filterdata_Int_with_Baselines_and_med\features_norm_med_audio_arousal.csv"
merged_df.to_csv(output_path, index=False)

print(f"Merged file saved in: {output_path}")
