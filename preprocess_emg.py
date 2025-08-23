import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def preprocess_emg(signal, fs, lowcut=10, highcut=450, notch_freq=50.0, q=30.0):
    """
    Preprocess EMG signal with:
      1. Notch filter to remove powerline noise (default 50 Hz),
      2. Bandpass filter (default 10–450 Hz).

    Parameters:
        signal (1D np.ndarray): Raw EMG signal
        fs (int): Sampling frequency (Hz)
        lowcut (float): Low cutoff for bandpass filter (Hz)
        highcut (float): High cutoff for bandpass filter (Hz)
        notch_freq (float): Powerline frequency to remove (Hz)
        q (float): Quality factor for notch filter

    Returns:
        filtered (1D np.ndarray): Preprocessed EMG signal
    """
    signal = np.asarray(signal).flatten()

    # --- Notch filter (powerline noise removal) ---
    b_notch, a_notch = iirnotch(w0=notch_freq, Q=q, fs=fs)
    signal = filtfilt(b_notch, a_notch, signal)

    # --- Bandpass filter (retain EMG band) ---
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b_band, a_band = butter(4, [low, high], btype="band")
    filtered = filtfilt(b_band, a_band, signal)

    return filtered
