import pandas as pd
import os
import shutil

# Loading the initial two files
file1 = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Participant_Analysis_Intervals\All_Participants_PoincareMetrics.csv"  # Splitted Videos
file2 = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Participant_Analysis_Whole\All_Participants_PoincareMetrics.csv"  # Whole videos
arousal_file = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\individual_ground_truth_without_hm2.csv"  # Ground Truth

# Reading the CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df_arousal = pd.read_csv(arousal_file)

# Mapping for split stimuli
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
for index, row in df2.iterrows():
    stimulus = row["Stimulus"]
    if stimulus in split_mapping:
        for split_stimulus in split_mapping[stimulus]:
            new_row = row.copy()
            new_row["Stimulus"] = split_stimulus
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
    how="outer",
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

# Selecting only relevant columns and ensuring consistency
final_merged_df = final_merged_df[[
    "Participant", "Stimulus", "Parasympathetic_SplittedVideos", "Parasympathetic_WholeVideos", "Arousal"
]].fillna("")

# Creating the output folder
output_folder = r"C:\\Users\\Salin\\OneDrive\\Documentos\\ESSEX\\DSPROJECT\\Comparission"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  # Deletes the folder and its contents
os.makedirs(output_folder, exist_ok=True)  # Recreate the folder

# Saving the merged DataFrame to a new CSV file inside the folder
output_file = os.path.join(output_folder, "final_merged_output.csv")
final_merged_df.to_csv(output_file, index=False)

print(f"Final merged CSV saved to {output_file}")
