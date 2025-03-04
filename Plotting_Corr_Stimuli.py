import pandas as pd
import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the merged CSV
merged_file = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Comparission\final_merged_output.csv"
df = pd.read_csv(merged_file)

# Ensuring numeric columns for plotting
df[["Parasympathetic_SplittedVideos", "Parasympathetic_WholeVideos", "Arousal"]] = df[
    ["Parasympathetic_SplittedVideos", "Parasympathetic_WholeVideos", "Arousal"]
].apply(pd.to_numeric, errors="coerce")

# Dropping rows with NaN values in the relevant columns
df_cleaned = df.dropna(subset=["Parasympathetic_SplittedVideos", "Parasympathetic_WholeVideos", "Arousal"])

# Defining the output folder
output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Comparission\Video_Plots"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  # Deletes the folder and its contents
os.makedirs(output_folder, exist_ok=True)  # Recreate the folder

# Preparing a file to save correlation results
correlation_file = os.path.join(output_folder, "video_correlations.txt")
with open(correlation_file, "w") as f:
    f.write("Correlation Results by Stimulus (Video):\n")
    f.write("=" * 50 + "\n")

# Grouping by Stimulus (video)
grouped = df_cleaned.groupby("Stimulus")

# Generating combined scatter plots for each video
for stimulus, group in grouped:
    # Calculating correlations for the current stimulus
    corr_splitted = group["Arousal"].corr(group["Parasympathetic_SplittedVideos"])
    corr_whole = group["Arousal"].corr(group["Parasympathetic_WholeVideos"])

    # Saving correlation results to the file
    with open(correlation_file, "a") as f:
        f.write(f"Stimulus: {stimulus}\n")
        f.write(f"  Correlation (Splitted Videos): {corr_splitted:.4f}\n")
        f.write(f"  Correlation (Whole Videos): {corr_whole:.4f}\n")
        f.write("-" * 50 + "\n")

    # Creating a single plot for this stimulus
    plt.figure(figsize=(8, 6))

    # Scattering plot for Splitted Videos
    sns.regplot(
        x="Arousal",
        y="Parasympathetic_SplittedVideos",
        data=group,
        scatter_kws={"alpha": 0.6, "s": 50},
        line_kws={"color": "blue", "linewidth": 2},
        label=f"Splitted Videos (Corr: {corr_splitted:.2f})"
    )

    # Scattering plot for Whole Videos
    sns.regplot(
        x="Arousal",
        y="Parasympathetic_WholeVideos",
        data=group,
        scatter_kws={"alpha": 0.6, "s": 50},
        line_kws={"color": "orange", "linewidth": 2},
        label=f"Whole Videos (Corr: {corr_whole:.2f})"
    )

    # Customizing the plot
    plt.title(f"Stimulus: {stimulus}", fontsize=14)
    plt.xlabel("Arousal")
    plt.ylabel("Parasympathetic")
    plt.legend()
    plt.tight_layout()

    # Saving the plot for the current stimulus
    plot_file = os.path.join(output_folder, f"{stimulus}_combined_plot.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()

print(f"Combined scatter plots and correlations saved in: {output_folder}")
