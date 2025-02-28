import pandas as pd
import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Loading the merged CSV
merged_file = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Comparission_norm_features\final_merged_output.csv"
df = pd.read_csv(merged_file)

# Features to compare
features_to_compare = [
    "Parasympathetic", "F1_CrossingPoints", "F2_MinArea", "F3_MaxArea",
    "F4_MeanArea", "F5_StdX", "F6_StdY", "F7_SkewX", "F8_SkewY", "F9_KurtX", "F10_KurtY"
]

# Generating feature column names for Splitted and Whole Videos
splitted_features = [f"{feature}_SplittedVideos" for feature in features_to_compare]
whole_features = [f"{feature}_WholeVideos" for feature in features_to_compare]

# Ensuring numeric columns
df[["Arousal"] + splitted_features + whole_features] = \
    df[["Arousal"] + splitted_features + whole_features].apply(pd.to_numeric, errors="coerce")

# Dropping NaN values
df_cleaned = df.dropna(subset=["Arousal"] + splitted_features + whole_features)

# Defining output folder
output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Comparission_norm_features\all_part_Stimuli_Corr"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  # Deletes existing folder
os.makedirs(output_folder, exist_ok=True)  # Creates new folder

# Initialize an empty list to store correlation results
correlation_data = []

# Grouping by Stimulus instead of Participant
grouped = df_cleaned.groupby("Stimulus")

# Generating comparison plots for each stimulus
for stimulus, group in grouped:
    plt.figure(figsize=(10, 6))

    # Dictionary to store correlations for this stimulus
    correlation_row = {"Stimulus": stimulus}

    for feature in features_to_compare:
        feature_splitted = f"{feature}_SplittedVideos"
        feature_whole = f"{feature}_WholeVideos"

        # Computing correlations
        corr_splitted = group["Arousal"].corr(group[feature_splitted])
        corr_whole = group["Arousal"].corr(group[feature_whole])

        # Store in dictionary for CSV output
        correlation_row[feature_splitted] = corr_splitted
        correlation_row[feature_whole] = corr_whole

        # Scatter plots for Splitted and Whole Videos
        sns.regplot(
            x="Arousal", y=feature_splitted, data=group,
            scatter_kws={"alpha": 0.6, "s": 40},
            line_kws={"color": "blue", "linewidth": 1},
            label=f"{feature} Splitted (Corr: {corr_splitted:.2f})"
        )
        sns.regplot(
            x="Arousal", y=feature_whole, data=group,
            scatter_kws={"alpha": 0.6, "s": 40},
            line_kws={"color": "orange", "linewidth": 1},
            label=f"{feature} Whole (Corr: {corr_whole:.2f})"
        )

    # Adding title and legend
    plt.title(f"Feature Comparison for Stimulus: {stimulus}\n")
    plt.xlabel("Arousal")
    plt.ylabel("Feature Values")
    plt.legend(fontsize=8)
    plt.tight_layout()

    # Saving plot
    plot_file = os.path.join(output_folder, f"{stimulus}_comparison.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()

    # Append row to correlation data list
    correlation_data.append(correlation_row)

# Convert correlation data to a DataFrame
correlation_df = pd.DataFrame(correlation_data)

# Define CSV file path
csv_output_path = os.path.join(output_folder, "correlations_results.csv")

# Save DataFrame as CSV
correlation_df.to_csv(csv_output_path, index=False)

print(f"Comparison plots saved in: {output_folder}")
print(f"Correlation results saved in: {csv_output_path}")
