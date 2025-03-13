import pandas as pd  
import numpy as np  
from scipy.signal import welch  
from scipy.interpolate import interp1d
import os

# Input and outputs paths
input_dir = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\HR_preprocessed"
output_csv = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\DSPROJECT\HR_preprocessed\features.csv"

# Function of band power 
def band_power(freqs, psd, band):
    band_freqs = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(band_freqs):
        return 0
    return np.trapezoid(psd[band_freqs], freqs[band_freqs])


results = []

# looping trhough files
for file_name in os.listdir(input_dir):
    if not file_name.endswith(".csv"):
        continue
    
    file_path = os.path.join(input_dir, file_name)
    print(f"\nProcessing file: {file_name}")
    
    try:
        data = pd.read_csv(file_path, encoding='utf-8')  
    except FileNotFoundError:
        print(f"File not found at {file_path}")  
        continue
    except Exception as e:
        print(f"Error reading file: {e}")  
        continue

    respondent_name = data['Respondent Name'].iloc[0]
    
    grouped_data = data.groupby('SourceStimuliName')

    for video_name, group in grouped_data:
        

        timestamps = group['Timestamp']
        heart_rate = group['Heart Rate Signal'].astype(float)

        # Removing invalid data if existent
        heart_rate[heart_rate <= 0] = np.nan
        heart_rate = heart_rate.dropna()

        if len(heart_rate) < 3:
            print(f"Not enough heart rate data for {video_name}, skipping.")
            continue

        # rr intervals calculation
        rr_intervals = 60.0 / heart_rate
        rr_intervals = rr_intervals.dropna()

        if len(rr_intervals) < 2:
            print(f"Not enough RR intervals for {video_name}, skipping.")
            continue

        # Time domain features
        mean_rr = np.mean(rr_intervals)
        median_rr = np.median(rr_intervals)
        sdnn = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
        nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 0.05)
        pnn50 = (nn50 / len(rr_intervals)) * 100

        # Frequency domain features
        # Cumulative time vector from RR intervals
        cumulative_time = np.cumsum(rr_intervals)

        if cumulative_time.empty or cumulative_time.iloc[-1] == 0:
            print(f"Empty cumulative time for {video_name}, skipping.")
            continue

        # Interpolating RR intervals to uniform sampling
        fs_interp = 4  # Hz
        uniform_times = np.arange(0, cumulative_time.iloc[-1], 1 / fs_interp)

        interp_func = interp1d(cumulative_time, rr_intervals, kind='cubic', fill_value="extrapolate")
        uniform_rr = interp_func(uniform_times)

        # Welch PSD on interpolated RR series
        freqs, psd = welch(uniform_rr, fs=fs_interp, nperseg=min(256, len(uniform_rr)))

        # Frequency bands (Hz)
        vlf_band = (0.00, 0.04)
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.5)

        # Abs powers
        a_vlf = band_power(freqs, psd, vlf_band)
        a_lf = band_power(freqs, psd, lf_band)
        a_hf = band_power(freqs, psd, hf_band)
        a_total = a_vlf + a_lf + a_hf

        # Relative  powers and ratios
        p_vlf = a_vlf / a_total if a_total > 0 else 0
        p_lf = a_lf / a_total if a_total > 0 else 0
        p_hf = a_hf / a_total if a_total > 0 else 0
        lf_hf_ratio = a_lf / a_hf if a_hf > 0 else 0
        nlf = a_lf / (a_lf + a_hf) if (a_lf + a_hf) > 0 else 0
        nhf = a_hf / (a_lf + a_hf) if (a_lf + a_hf) > 0 else 0

        # Appending results
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

# Exporting and saving
final_df = pd.DataFrame(results)

if not final_df.empty:
    final_df.to_csv(output_csv, index=False)
    print(f"\nHR-based HRV features for ALL files saved to:\n{output_csv}")
else:
    print("\n No HRV features were extracted from any files. Check your data!")
