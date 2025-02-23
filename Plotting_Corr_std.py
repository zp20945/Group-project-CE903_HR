import pandas as pd
import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt

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
output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Comparission_std\Participants_Corr_Plots"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  # Deletes existing folder
os.makedirs(output_folder, exist_ok=True)  # Creates new folder

# Group by participant
grouped = df_cleaned.groupby("Participant")

# Generate simplified comparison plots
for participant, group in grouped:
    for feature in features_to_compare:
        plt.figure(figsize=(8, 6))

        feature_splitted = f"{feature}_SplittedVideos"
        feature_whole = f"{feature}_WholeVideos"

        # Compute correlations
        corr_splitted = group["Arousal"].corr(group[feature_splitted])
        corr_whole = group["Arousal"].corr(group[feature_whole])

        # Scatter plots for Splitted and Whole Videos
        sns.regplot(
            x="Arousal", y=feature_splitted, data=group,
            scatter_kws={"alpha": 0.6, "s": 50},
            line_kws={"color": "blue", "linewidth": 2},
            label=f"Splitted Videos (Corr: {corr_splitted:.2f})"
        )
        sns.regplot(
            x="Arousal", y=feature_whole, data=group,
            scatter_kws={"alpha": 0.6, "s": 50},
            line_kws={"color": "orange", "linewidth": 2},
            label=f"Whole Videos (Corr: {corr_whole:.2f})"
        )

        # Customize plot
        plt.title(f"Participant: {participant} - {feature}", fontsize=14)
        plt.xlabel("Arousal")
        plt.ylabel(feature)
        plt.legend()
        plt.tight_layout()

        # Save plot
        plot_file = os.path.join(output_folder, f"{participant}_{feature}_comparison.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()

print(f"Simplified feature comparison plots saved in: {output_folder}")
