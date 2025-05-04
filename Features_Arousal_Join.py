import pandas as pd
import os
import shutil

# Loading the two CSV files
file1 = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\PPG_HR_Preprocessed_more_3\merged_features_norm_Med_audio.csv" # Normalized Features
file2 = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\individual_ground_truth_without_hm2.csv" # Ground Truth

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Renaming columns to match for merging
df1.rename(columns={'SourceStimuliName': 'Stimulus'}, inplace=True)
df2.rename(columns={'Stimulus_Name': 'Stimulus'}, inplace=True)

# Merging on 'Participant' and 'Stimulus'
merged_df = df1.merge(df2[['Participant', 'Stimulus', 'Arousal']], on=['Participant', 'Stimulus'], how='left')

# Warning if there are participants without arousal values
missing_arousal = merged_df[merged_df['Arousal'].isna()]['Participant'].unique()
if len(missing_arousal) > 0:
    print("Warning: 'Arousal' value not found for the following participants:")
    for p in missing_arousal:
        print(f"  - Participant {p}")
else:
    print("All participants have 'Arousal' values.")

# Saving the merged file
output_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\PPG_HR_Preprocessed_more_3\merged_features_norm_Med_audio_arousal.csv"
merged_df.to_csv(output_path, index=False)

print(f"\nMerged file saved at: {output_path}")
