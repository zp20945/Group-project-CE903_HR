import pandas as pd  
import numpy as np  
import os
from scipy.signal import welch, find_peaks
from scipy.interpolate import interp1d

# Input and outputs paths
input_dir = r"c:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\PPG_HR_Analysis_Longer_Intervals_\PPG_HR_Analysis_Longer_Intervals"
output_csv = r"c:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\PPG_HR_Analysis_Longer_Intervals_\Time_Frequency_Features.csv"

# Function to calculate band power
def band_power(freqs, psd, band):
    band_freqs = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(band_freqs):
        return 0
    return np.trapezoid(psd[band_freqs], freqs[band_freqs])

results = []

# Looping through files
for file_name in os.listdir(input_dir):
    if not file_name.endswith(".csv"):
        continue
    
    file_path = os.path.join(input_dir, file_name)
    print(f"\nProcessing file: {file_name}")
    
    try:
        data = pd.read_csv(file_path, encoding='utf-8')  
    except Exception as e:
        print(f"Error reading file: {e}")  
        continue

    respondent_name = data['Respondent Name'].iloc[0]
    grouped_data = data.groupby('SourceStimuliName')

    for video_name, group in grouped_data:
        timestamps = group['Timestamp'].astype(float)
        ppg_signal = group['Butterworth Filtered PPG Signal'].astype(float).dropna()

        # Sampling rate of the PPG signal 
        fs_ppg = 128  # Hz

        # Peak detection on PPG signal
        if len(ppg_signal) < fs_ppg * 5:
            print(f"Not enough PPG data for {video_name}, skipping.")
            continue

        peaks, _ = find_peaks(ppg_signal, distance=fs_ppg * 0.4)  # ~150 bpm max

        if len(peaks) < 3:
            print(f"Not enough peaks for {video_name}, skipping.")
            continue

        try:
            peak_times = timestamps.iloc[peaks].values
        except Exception as e:
            print(f"Timestamp alignment error for {video_name}: {e}")
            continue

        rr_intervals = np.diff(peak_times)  # in seconds

        if len(rr_intervals) < 2 or np.any(np.isnan(rr_intervals)) or np.any(rr_intervals <= 0):
            print(f"Invalid or insufficient RR intervals for {video_name}, skipping.")
            continue

        # Time domain features
        mean_rr = np.mean(rr_intervals)
        median_rr = np.median(rr_intervals)
        sdnn = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
        nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 0.05)
        pnn50 = (nn50 / len(rr_intervals)) * 100

        # Cumulative time for RR intervals
        cumulative_time = np.cumsum(rr_intervals)

        # Checking duration and number of intervals
        if len(rr_intervals) < 4 or cumulative_time[-1] < 5:
            print(f"Not enough RR duration or count for interpolation in {video_name}, skipping.")
            continue

        # Interpolation to uniform time axis
        fs_interp = 4.0  # Hz
        uniform_times = np.arange(0, cumulative_time[-1], 1 / fs_interp)

        try:
            interp_func = interp1d(cumulative_time, rr_intervals, kind='cubic', fill_value="extrapolate", bounds_error=False)
            uniform_rr = interp_func(uniform_times)
        except Exception as e:
            print(f"Interpolation error in {video_name}: {e}")
            continue

        # Welch PSD
        freqs, psd = welch(uniform_rr, fs=fs_interp, nperseg=min(256, len(uniform_rr)))

        # Frequency bands
        vlf_band = (0.00, 0.04)
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.5)

        # Band powers
        a_vlf = band_power(freqs, psd, vlf_band)
        a_lf = band_power(freqs, psd, lf_band)
        a_hf = band_power(freqs, psd, hf_band)
        a_total = a_vlf + a_lf + a_hf

        # Relative powers and ratios
        p_vlf = a_vlf / a_total if a_total > 0 else 0
        p_lf = a_lf / a_total if a_total > 0 else 0
        p_hf = a_hf / a_total if a_total > 0 else 0
        lf_hf_ratio = a_lf / a_hf if a_hf > 0 else 0
        nlf = a_lf / (a_lf + a_hf) if (a_lf + a_hf) > 0 else 0
        nhf = a_hf / (a_lf + a_hf) if (a_lf + a_hf) > 0 else 0

        # Saving results
        results.append({
            "Participant": respondent_name,
            "SourceStimuliName": video_name,
            "Mean RR": mean_rr,
            "Median RR": median_rr,
            "SDNN": sdnn,
            "RMSSD": rmssd,
            "NN50": nn50,
            "pNN50": pnn50,
            "aVLF": a_vlf,
            "aLF": a_lf,
            "aHF": a_hf,
            "aTotal": a_total,
            "pVLF": p_vlf,
            "pLF": p_lf,
            "pHF": p_hf,
            "LF/HF": lf_hf_ratio,
            "nLF": nlf,
            "nHF": nhf
        })

# Saving to CSV
final_df = pd.DataFrame(results)
if not final_df.empty:
    final_df.to_csv(output_csv, index=False)
    print(f"\nHRV features saved to: {output_csv}")
else:
    print("\nNo HRV features were extracted from any files.")
