import pandas as pd
import numpy as np

# Loading the two CSV files
file1 = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\PPG_HR_Analysis_Longer_Intervals\Grey_Cal\norm_Grey_Cal_Start_residual.csv"# Normalized Features
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

merged_df = merged_df.dropna(subset=['Arousal'])

# Saving the merged file
output_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\PPG_HR_Analysis_Longer_Intervals\Grey_Cal\norm_Grey_Cal_Start_residual_arousal.csv"
merged_df.to_csv(output_path, index=False)

print(f"\nMerged file saved at: {output_path}")
