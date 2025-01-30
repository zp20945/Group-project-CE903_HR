import pandas as pd  
import numpy as np  
from scipy.signal import find_peaks, welch  
import matplotlib.pyplot as plt  

# Loading the updated CSV file
file_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\filtered_ppg_signal_specific_columns_OnlyVideos.csv"  # Specifying the input file path

# Reading the CSV file into a pandas DataFrame 
try:
    data = pd.read_csv(file_path, encoding='utf-8')  
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")  
    exit()
except Exception as e:
    print(f"Error reading file: {e}")  
    exit()

# Initializing an empty list to store the features for each video type
results = []
respondent_name = data['Respondent Name'].iloc[0] # As only one is being analyzed

# Defining a function to calculate the power within a specific frequency band
def band_power(freqs, psd, band):
    band_freqs = (freqs >= band[0]) & (freqs <= band[1])  # Filtering the frequencies that lie within the band
    return np.sum(psd[band_freqs])  # Summing the power spectral density values within the band

# Grouping the dataset by the video type (SourceStimuliName column)
grouped_data = data.groupby('SourceStimuliName')  # Grouping the data by video type
for video_name, group in grouped_data:  # Iterating over each video type
    print(f"Processing video: {video_name}")  # Displaying the current video type being processed
    
    # Extracting columns for the current video type
    timestamps = group['Timestamp']  # Extracting the 'Timestamp' column
    ppg_signal = group['Butterworth Filtered PPG Signal']  # Extracting the filtered PPG signal
    raw_ppg_signal = group['Internal ADC A13 PPG RAW']  # Extracting the raw PPG signal
    
    
    
    # Converting timestamps to seconds if they are in milliseconds
    if timestamps.max() > 1e6:  
        timestamps = timestamps / 1000  

    fs = 128  # Setting the sampling frequency to 128 Hz
    distance = fs // 2  # Defining the minimum distance between peaks as half a second

    # STEP 1: Detecting peaks in the PPG signal
    peaks, _ = find_peaks(ppg_signal, distance=distance)  # Identifying peaks in the filtered PPG signal
    peaks = np.array(peaks)  # Ensuring peaks are stored as a NumPy array
    peak_times = timestamps.iloc[peaks]  # Extracting the times corresponding to the detected peaks

    # STEP 2: Calculating RR intervals (time differences between consecutive peaks)
    rr_intervals = np.diff(peak_times)  # Computing the differences between consecutive peak times
    if len(rr_intervals) < 2:  # Checking if there are enough RR intervals for analysis
        print(f"Not enough RR intervals for {video_name}, skipping.")  # Skipping analysis for this video type if insufficient data
        continue

    # STEP 3: Calculating time-domain HRV features
    mean_rr = np.mean(rr_intervals)  
    median_rr = np.median(rr_intervals)  
    sdnn = np.std(rr_intervals)  
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))  # Computing the root mean square of successive differences
    nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 0.05)  # Counting RR intervals differing by more than 50ms
    pnn50 = (nn50 / len(rr_intervals)) * 100  # Calculating the percentage of NN50 intervals over the total

    # STEP 4: Calculating frequency-domain HRV features using Welch's method
    freqs, psd = welch(rr_intervals, fs=1/np.mean(rr_intervals), nperseg=len(rr_intervals)//2)  # Estimating power spectral density
    vlf_band, lf_band, hf_band = (0.0, 0.04), (0.04, 0.15), (0.15, 0.5)  # Defining frequency bands

    # Calculating power within each frequency band
    a_vlf = band_power(freqs, psd, vlf_band)  # Power in the VLF band
    a_lf = band_power(freqs, psd, lf_band)  # Power in the LF band
    a_hf = band_power(freqs, psd, hf_band)  # Power in the HF band
    a_total = a_vlf + a_lf + a_hf  # Total power across all bands

    # Calculating relative power and ratios
    p_vlf, p_lf, p_hf = a_vlf / a_total if a_total > 0 else 0, a_lf / a_total if a_total > 0 else 0, a_hf / a_total if a_total > 0 else 0
    nlf = a_lf / (a_lf + a_hf) if (a_lf + a_hf) > 0 else 0  # Normalized LF power
    nhf = a_hf / (a_lf + a_hf) if (a_lf + a_hf) > 0 else 0  # Normalized HF power
    lf_hf_ratio = a_lf / a_hf if a_hf > 0 else 0  # LF/HF ratio

    # Identifying peak power within each frequency band
    peak_vlf = max(psd[(freqs >= vlf_band[0]) & (freqs <= vlf_band[1])], default=0)  # Peak VLF power
    peak_lf = max(psd[(freqs >= lf_band[0]) & (freqs <= lf_band[1])], default=0)  # Peak LF power
    peak_hf = max(psd[(freqs >= hf_band[0]) & (freqs <= hf_band[1])], default=0)  # Peak HF power

    # STEP 5: Calculating SNR (Signal-to-Noise Ratio)
    noise = raw_ppg_signal - ppg_signal  # Estimating noise as the difference between raw and filtered signals
    signal_power = np.sum(ppg_signal ** 2)  # Calculating the power of the filtered signal
    noise_power = np.sum(noise ** 2)  # Calculating the power of the noise
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')  # Computing SNR in dB

    # Saving the features for the current video type
    results.append({
        "Participant": respondent_name ,  # In this case, only one participant is being analyzed
        "SourceStimuliName": video_name,  # Assigning the video type
        "Mean RR": mean_rr,  # Mean RR interval
        "Median RR": median_rr,  # Median RR interval
        "SDNN": sdnn,  # Standard deviation of RR intervals
        "RMSSD": rmssd,  # RMSSD
        "NN50": nn50,  # NN50 count
        "pNN50": pnn50,  # pNN50 percentage
        "peakVLF": peak_vlf,  # Peak VLF power
        "peakLF": peak_lf,  # Peak LF power
        "peakHF": peak_hf,  # Peak HF power
        "aVLF": a_vlf,  # Total VLF power
        "aLF": a_lf,  # Total LF power
        "aHF": a_hf,  # Total HF power
        "aTotal": a_total,  # Total power across all bands
        "pVLF": p_vlf,  # Relative VLF power
        "pLF": p_lf,  # Relative LF power
        "pHF": p_hf,  # Relative HF power
        "nLF": nlf,  # Normalized LF power
        "nHF": nhf,  # Normalized HF power
        "LFHF": lf_hf_ratio,  # LF/HF ratio
        "SNR": snr  # Signal-to-Noise Ratio
    })

# Creating a DataFrame from the results list
final_df = pd.DataFrame(results)  # Creating a DataFrame to store features for all video types

# Saving the aggregated features to a CSV file
output_path = r'C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\AggregatedFeaturesHRV_.csv'  
final_df.to_csv(output_path, index=False)  

print(f"Aggregated features saved to {output_path}")  
