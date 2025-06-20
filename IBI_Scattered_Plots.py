import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import shutil 

# PATHS
participants_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\PPG_HR_Preprocessed_Videos_Plus_Surveys"
interval_file = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\interval.csv"
output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\IBI_ScatterPlots"

# Loading interval data
interval_df = pd.read_csv(interval_file)


## Ensuring the output folder is NEW each time
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  # Deleting existing folder and its contents
os.makedirs(output_folder)  # Creating a fresh folder

# Cleaning stimulus names for matching
def clean_stimulus_name(name):
    return ''.join(e.lower() for e in str(name) if e.isalnum())

# Normalizing to [0, 1]
def normalize_series(series):
    min_val = series.min()
    max_val = series.max()
    if min_val == max_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)

# Main function
def plot_ibi_scatter_normalized(participant_file):
    df = pd.read_csv(participant_file)
    participant_name = df['Respondent Name'].iloc[0]
    stimuli_names = df['SourceStimuliName'].unique()

    for stimulus in stimuli_names:
        stimulus_df = df[df['SourceStimuliName'] == stimulus].copy()
        stimulus_df = stimulus_df.sort_values(by='Timestamp')

        # Cleaning and compute IBI
        stimulus_df['Heart Rate Signal'] = pd.to_numeric(stimulus_df['Heart Rate Signal'], errors='coerce')
        stimulus_df = stimulus_df.dropna(subset=['Heart Rate Signal'])
        stimulus_df = stimulus_df[stimulus_df['Heart Rate Signal'] > 0]
        stimulus_df['IBI'] = 60 / stimulus_df['Heart Rate Signal']
        stimulus_df['IBI_norm'] = normalize_series(stimulus_df['IBI'])

        # Output path
        stimulus_folder = os.path.join(output_folder, stimulus)
        os.makedirs(stimulus_folder, exist_ok=True)

        # Getting intervals
        stimulus_clean = clean_stimulus_name(stimulus)
        intervals = interval_df[interval_df['stimuli_names'].apply(lambda x: stimulus_clean in clean_stimulus_name(x))]

        # Starting plot
        plt.figure(figsize=(15, 6))
        plt.scatter(
            stimulus_df['Timestamp'],
            stimulus_df['IBI_norm'],
            color='darkgreen',
            s=25,
            alpha=0.6,
            marker='o',
            edgecolors='none'
        )
        plt.title(f'Normalized IBI Scatter Plot - {stimulus} - {participant_name}')
        plt.xlabel('Timestamp (s)')
        plt.ylabel('IBI (normalized)')
        plt.ylim([0, 1])
        plt.grid(True, which='major', axis='y', linestyle='--', alpha=0.3)

        import matplotlib.ticker as ticker
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(3))
        plt.xticks(rotation=45)

        # Interval shading and optional labels
        for _, row in intervals.iterrows():
            starts = str(row['interval_start']).split(';')
            ends = str(row['interval_end']).split(';')
            label = str(row['stimuli_renamed'])

            # Determine interval color
            color = 'red' if label.endswith('H') else 'blue' if label.endswith('L') else 'gray'

            for start, end in zip(starts, ends):
                try:
                    start = float(start.strip())
                    end = float(end.strip())

                    plt.axvspan(start, end, color=color, alpha=0.2)
                    plt.axvline(x=start, color='gray', linestyle='--', linewidth=1)
                    plt.axvline(x=end, color='gray', linestyle='--', linewidth=1)

                    # Only plotting label if it's not NaN or 'nan'
                    if label.lower() != 'nan':
                        plt.text(
                            (start + end) / 2,
                            0.93,  # slightly below top
                            label,
                            fontsize=8,
                            ha='center',
                            va='bottom',
                            rotation=45
                        )
                except ValueError:
                    print(f"Skipping invalid interval: {start} - {end}")

        # Saving figure
        plot_filename = f"{stimulus.replace(' ', '_')}_{participant_name}_IBI_scatter_cleaned.png"
        plt.tight_layout()
        plt.savefig(os.path.join(stimulus_folder, plot_filename))
        plt.close()

# Main loop
participant_files = glob.glob(os.path.join(participants_folder, '*.csv'))

for file in participant_files:
    print(f"Processing {file}...")
    plot_ibi_scatter_normalized(file)

print("All cleaned and normalized IBI scatter plots have been saved.")