import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# PATHS
participants_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\HR_preprocessed_whole_videos"
interval_file = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\interval.csv"
output_folder = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\HR_plotted"      

# LOADING INTERVALS
interval_df = pd.read_csv(interval_file)

# CREATING OUTPUT DIRECTORY
os.makedirs(output_folder, exist_ok=True)

# FUNCTION TO PLOT 
def plot_participant(participant_file):
    # Reading participant CSV
    df = pd.read_csv(participant_file)
    
    participant_name = df['Respondent Name'].iloc[0]
    stimuli_names = df['SourceStimuliName'].unique()

    for stimulus in stimuli_names:
        stimulus_df = df[df['SourceStimuliName'] == stimulus]
        
        # Creating folder for the stimulus if not exist
        stimulus_folder = os.path.join(output_folder, stimulus)
        os.makedirs(stimulus_folder, exist_ok=True)
        
        plt.figure(figsize=(15, 6))
        
        # Plotting heart rate signal vs time
        plt.plot(stimulus_df['Timestamp'], stimulus_df['Heart Rate Signal'], color='black', label='Heart Rate')
        
        # Filter intervals that belong to this stimulus
        # Cleaned stimulus name, to match intervals
        stimulus_clean = stimulus.split()[0]
        intervals = interval_df[interval_df['stimuli_names'].str.contains(stimulus_clean, case=False, na=False)]
        
        # Mark intervals with lines and optionally with colors
        for _, row in intervals.iterrows():
            starts = str(row['interval_start']).split(';')
            ends = str(row['interval_end']).split(';')
            label = str(row['stimuli_renamed'])
            
            # Determine color only if ends with H or L
            if label.endswith('H'):
                color = 'red'
            elif label.endswith('L'):
                color = 'blue'
            else:
                color = None  # no color, but we still plot the lines
            
            for start, end in zip(starts, ends):
                try:
                    start = float(start.strip())
                    end = float(end.strip())

                    # Plot vertical dashed lines for boundaries
                    plt.axvline(x=start, color='gray', linestyle='--', linewidth=1)
                    plt.axvline(x=end, color='gray', linestyle='--', linewidth=1)

                    # Fill colored region if H or L
                    if color:
                        plt.axvspan(start, end, color=color, alpha=0.3)
                        plt.text((start + end) / 2, max(stimulus_df['Heart Rate Signal']),
                                 label, fontsize=8, ha='center', va='bottom')
                except ValueError:
                    print(f"Skipping invalid interval: {start} - {end}")
        
        plt.title(f'Heart Rate Signal - {stimulus} - {participant_name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Heart Rate Signal')
        plt.legend()
        
        # Saving plot to the stimulus folder
        plot_filename = f"{stimulus.replace(' ', '_')}_{participant_name}.png"
        plt.savefig(os.path.join(stimulus_folder, plot_filename))
        plt.close()

# MAIN LOOP FOR PARTICIPANTS 
participant_files = glob.glob(os.path.join(participants_folder, '*.csv'))

for file in participant_files:
    print(f"Processing {file}...")
    plot_participant(file)

print("All plots are generated and saved!")
