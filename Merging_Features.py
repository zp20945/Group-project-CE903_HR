import pandas as pd
import os
import shutil 

# Loading the two datasets
file1_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Poincare_features_splitted_arousal\Correlation_Plots_CSVs\all_participants_data.csv"  
file2_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Time_Frequency_features_splitted_arousal\Correlation_Plots_CSVs\all_participants_data.csv"  

df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Ensure the output folder exists and is fresh
output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Important_Features"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  # Delete folder if it exists
os.makedirs(output_folder)  # Recreate folder

# Keeping only one 'Arousal' column (from df1, assuming they are identical)
if 'Arousal' in df1.columns and 'Arousal' in df2.columns:
    df2 = df2.drop(columns=['Arousal'])

# Merging the dataframes on 'Participant' and 'Stimulus'
merged_df = pd.merge(df1, df2, on=['Participant', 'Stimulus'], how='inner')

# Move 'Arousal' column to the end
columns = [col for col in merged_df.columns if col != 'Arousal'] + ['Arousal']
merged_df = merged_df[columns]

# Save the merged file
output_path = os.path.join(output_folder, "All_Features.csv")
merged_df.to_csv(output_path, index=False)

print(f"Merged file saved in: {output_path}")




