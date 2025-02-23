import pandas as pd
import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the merged CSV
merged_file = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Comparission_std\final_merged_output.csv"
df = pd.read_csv(merged_file)

# Features to compare
features_to_compare = [
    "Parasympathetic", "F1_CrossingPoints", "F2_MinArea", "F3_MaxArea",
    "F4_MeanArea", "F5_StdX", "F6_StdY", "F7_SkewX", "F8_SkewY", "F9_KurtX", "F10_KurtY"
]

# Generate feature column names for Splitted and Whole Videos
splitted_features = [f"{feature}_SplittedVideos" for feature in features_to_compare]
whole_features = [f"{feature}_WholeVideos" for feature in features_to_compare]

# Ensure numeric columns
df[["Arousal"] + splitted_features + whole_features] = \
    df[["Arousal"] + splitted_features + whole_features].apply(pd.to_numeric, errors="coerce")

# Drop NaN values
df_cleaned = df.dropna(subset=["Arousal"] + splitted_features + whole_features)

# Define output folder
output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Comparission_std\Participants_Corr"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  # Deletes existing folder
os.makedirs(output_folder, exist_ok=True)  # Creates new folder

# Group by participant
grouped = df_cleaned.groupby("Participant")

# Generate comparison plots for each participant
for participant, group in grouped:
    plt.figure(figsize=(10, 6))

    # Store correlation values per feature
    correlations_splitted = []
    correlations_whole = []

    for feature in features_to_compare:
        feature_splitted = f"{feature}_SplittedVideos"
        feature_whole = f"{feature}_WholeVideos"

        # Compute correlations
        corr_splitted = group["Arousal"].corr(group[feature_splitted])
        corr_whole = group["Arousal"].corr(group[feature_whole])

        correlations_splitted.append(corr_splitted)
        correlations_whole.append(corr_whole)

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

    # Compute overall correlation average
    avg_corr_splitted = np.nanmean(correlations_splitted)
    avg_corr_whole = np.nanmean(correlations_whole)

    # Add title and legend
    plt.title(f"Feature Comparison for Participant: {participant}\n"
              f"Avg Corr Splitted: {avg_corr_splitted:.2f} | Avg Corr Whole: {avg_corr_whole:.2f}",
              fontsize=14)
    plt.xlabel("Arousal")
    plt.ylabel("Feature Values")
    plt.legend(fontsize=8)
    plt.tight_layout()

    # Save plot
    plot_file = os.path.join(output_folder, f"{participant}_comparison.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()

print(f"Updated comparison plots saved in: {output_folder}")
