import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the merged CSV
merged_file = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Comparission\final_merged_output.csv"
df = pd.read_csv(merged_file)

# Ensuring numeric columns for analysis and plotting
df[["Parasympathetic_SplittedVideos", "Parasympathetic_WholeVideos", "Arousal"]] = df[
    ["Parasympathetic_SplittedVideos", "Parasympathetic_WholeVideos", "Arousal"]
].apply(pd.to_numeric, errors="coerce")

# Dropping rows with NaN values in the relevant columns
correlation_df = df.dropna(subset=["Parasympathetic_SplittedVideos", "Parasympathetic_WholeVideos", "Arousal"])

# Calculating correlations
correlation_splitted = correlation_df["Arousal"].corr(correlation_df["Parasympathetic_SplittedVideos"])
correlation_whole = correlation_df["Arousal"].corr(correlation_df["Parasympathetic_WholeVideos"])

# Printing the correlation results
print(f"Correlation between Arousal and Parasympathetic (Splitted Videos): {correlation_splitted:.4f}")
print(f"Correlation between Arousal and Parasympathetic (Whole Videos): {correlation_whole:.4f}")

# Creating an output folder to save results
output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\Comparission\Plotting_Corr_All"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Saving the correlation results to a text file
correlation_file = os.path.join(output_folder, "correlation_results.txt")
with open(correlation_file, "w") as f:
    f.write(f"Correlation between Arousal and Parasympathetic (Splitted Videos): {correlation_splitted:.4f}\n")
    f.write(f"Correlation between Arousal and Parasympathetic (Whole Videos): {correlation_whole:.4f}\n")

# Plotting both datasets on a single plot
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))

# Scattering plot for Splitted Videos
sns.regplot(
    x="Arousal",
    y="Parasympathetic_SplittedVideos",
    data=correlation_df,
    scatter_kws={"alpha": 0.6, "s": 50},
    line_kws={"color": "blue", "linewidth": 2},
    label=f"Splitted Videos (Corr: {correlation_splitted:.2f})"
)

# Scattering plot for Whole Videos
sns.regplot(
    x="Arousal",
    y="Parasympathetic_WholeVideos",
    data=correlation_df,
    scatter_kws={"alpha": 0.6, "s": 50},
    line_kws={"color": "orange", "linewidth": 2},
    label=f"Whole Videos (Corr: {correlation_whole:.2f})"
)

# Customizing the plot
plt.title("Correlation Between Arousal and Parasympathetic", fontsize=14)
plt.xlabel("Arousal")
plt.ylabel("Parasympathetic")
plt.legend()
plt.tight_layout()

# Saving the plot as a single image
plot_file = os.path.join(output_folder, "combined_correlation_plot.png")
plt.savefig(plot_file, dpi=300)
plt.close()

# Informing the user
print(f"Correlation results saved to: {correlation_file}")
print(f"Correlation plot saved to: {plot_file}")
