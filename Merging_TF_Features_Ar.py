import pandas as pd
import os
import shutil

# Loading the new files
file1 = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Features_Baseline_Part_Interval\Features_Baseline_Part_norm.csv"  # Splitted Videos
file2 = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Features_Baseline_Part_Whole\Features_Baseline_Part_norm.csv"  # Whole Videos
arousal_file = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\individual_ground_truth_without_hm2.csv"  # Ground Truth

# Reading the CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df_arousal = pd.read_csv(arousal_file)

# Renaming stimulus column for consistency
df1.rename(columns={"SourceStimuliName": "Stimulus"}, inplace=True)
df2.rename(columns={"SourceStimuliName": "Stimulus"}, inplace=True)

# Mapping for split stimuli (remains the same)
split_mapping = {
    "HP_1": ["HP_1_L", "HP_1_H"],
    "HP_3": ["HP_3_L", "HP_3_H"],
    "HP_7": ["HP_7_H", "HP_7_L"],
    "HN_2": ["HN_2_H", "HN_2_L"],
    "HN_3": ["HN_3_H", "HN_3_L"],
    "LN_7": ["LN_7_N", "LN_7_P"],
}

# Expanding the smaller dataframe based on the split_mapping
expanded_rows = []
for index, row in df2.iterrows():  # Iterates over each row of df2
    stimulus = row["Stimulus"]
    if stimulus in split_mapping:  # Checks if it's in the map
        for split_stimulus in split_mapping[stimulus]:
            new_row = row.copy()  # Copies the row
            new_row["Stimulus"] = split_stimulus  # Updates stimulus name
            expanded_rows.append(new_row)
    else:
        expanded_rows.append(row)

# Creating a new DataFrame for the expanded rows
df2_expanded = pd.DataFrame(expanded_rows)

# Merging the two initial DataFrames based on 'Participant' and 'Stimulus'
merged_df = pd.merge(
    df1,
    df2_expanded,
    on=["Participant", "Stimulus"],
    how="outer",  # Keeps all rows
    suffixes=("_SplittedVideos", "_WholeVideos"),
)

# Merging the new arousal data with the existing merged DataFrame
final_merged_df = pd.merge(
    merged_df,
    df_arousal,
    left_on=["Participant", "Stimulus"],
    right_on=["Participant", "Stimulus_Name"],
    how="left"
)

# Selecting only relevant columns based on new features
columns_to_keep = [
    "Participant", "Stimulus",
    "Mean RR_SplittedVideos", "Mean RR_WholeVideos",
    "Median RR_SplittedVideos", "Median RR_WholeVideos",
    "SDNN_SplittedVideos", "SDNN_WholeVideos",
    "RMSSD_SplittedVideos", "RMSSD_WholeVideos",
    "NN50_SplittedVideos", "NN50_WholeVideos",
    "pNN50_SplittedVideos", "pNN50_WholeVideos",
    "peakVLF_SplittedVideos", "peakVLF_WholeVideos",
    "peakLF_SplittedVideos", "peakLF_WholeVideos",
    "peakHF_SplittedVideos", "peakHF_WholeVideos",
    "aVLF_SplittedVideos", "aVLF_WholeVideos",
    "aLF_SplittedVideos", "aLF_WholeVideos",
    "aHF_SplittedVideos", "aHF_WholeVideos",
    "aTotal_SplittedVideos", "aTotal_WholeVideos",
    "pVLF_SplittedVideos", "pVLF_WholeVideos",
    "pLF_SplittedVideos", "pLF_WholeVideos",
    "pHF_SplittedVideos", "pHF_WholeVideos",
    "nLF_SplittedVideos", "nLF_WholeVideos",
    "nHF_SplittedVideos", "nHF_WholeVideos",
    "LFHF_SplittedVideos", "LFHF_WholeVideos",
    "SNR_SplittedVideos", "SNR_WholeVideos",
    "Arousal"  # Ground truth variable
]

# Selecting only relevant columns and ensuring consistency
final_merged_df = final_merged_df[columns_to_keep].fillna("")

# Creating the output folder
output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Features_Time_Frequency_Comparission_Splitted_Whole"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  # Deletes the folder and its contents
os.makedirs(output_folder, exist_ok=True)  # Recreates the folder

# Saving the merged DataFrame to a new CSV file inside the folder
output_file = os.path.join(output_folder, "final_merged_output.csv")
final_merged_df.to_csv(output_file, index=False)

print(f"Final merged CSV saved to {output_file}")
