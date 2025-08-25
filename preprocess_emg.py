import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def preprocess_emg(signal, sampling_rate, low_cutoff=10, high_cutoff=450):
    """
    Preprocesses EMG signal following the steps from the provided MATLAB code:
    1. Bandpass filter (10-450 Hz)
    2. 60 Hz notch filter
    3. DC removal (0.1 Hz high-pass filter)

    Parameters:
        signal (1D np.ndarray): Raw EMG signal
        sampling_rate (int): Sampling frequency in Hz
        low_cutoff (float): Low cutoff for bandpass filter (Hz)
        high_cutoff (float): High cutoff for bandpass filter (Hz)

    Returns:
        processed_signal (1D np.ndarray): Preprocessed EMG signal
    """
    signal = np.asarray(signal).flatten()
    nyq = 0.5 * sampling_rate

    # --- 1. Bandpass filter ---
    low = low_cutoff / nyq
    high = high_cutoff / nyq
    b_band, a_band = butter(4, [low, high], btype="band")
    filtered_signal = filtfilt(b_band, a_band, signal)

    # --- 2. 60 Hz notch filter ---
    wo = 60 / nyq  # Normalized frequency
    q = 35  # The MATLAB code calculates Q from the bandwidth (bw = wo/35)
    b_notch, a_notch = iirnotch(wo, q)
    notch_filtered_signal = filtfilt(b_notch, a_notch, filtered_signal)
    
    # --- 3. DC removal (high-pass filter with 0.1Hz cutoff) ---
    highpass_cutoff = 0.1
    b_dc, a_dc = butter(1, highpass_cutoff / nyq, btype='high')
    processed_signal = filtfilt(b_dc, a_dc, notch_filtered_signal)

    return processed_signal