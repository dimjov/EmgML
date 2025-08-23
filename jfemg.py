import numpy as np

def jfemg(feature_name, signal):
    """
    Compute standard EMG features from the input signal.

    Supported features:
        'mav'   : Mean Absolute Value
        'wl'    : Waveform Length
        'emav'  : Enhanced Mean Absolute Value
        'ewl'   : Enhanced Waveform Length
        'rms'   : Root Mean Square
        'zc'    : Zero Crossing (with threshold)
        'ssc'   : Slope Sign Change (with threshold)

    Parameters:
        feature_name (str): Name of feature
        signal (1D np.ndarray): Input EMG segment

    Returns:
        float: Feature value
    """
    signal = np.asarray(signal).flatten()

    if feature_name == "mav":  # Mean Absolute Value
        return np.mean(np.abs(signal))

    elif feature_name == "wl":  # Waveform Length
        return np.sum(np.abs(np.diff(signal)))

    elif feature_name == "emav":  # Enhanced Mean Absolute Value
        N = len(signal)
        if N == 0:
            return 0.0
        w = np.ones(N)
        w[:int(0.25 * N)] = 0.5
        w[-int(0.25 * N):] = 0.5
        return np.sum(w * np.abs(signal)) / N

    elif feature_name == "ewl":  # Enhanced Waveform Length
        N = len(signal)
        if N < 2:
            return 0.0
        diff_sig = np.abs(np.diff(signal))
        w = np.ones(N - 1)
        w[:int(0.25 * (N - 1))] = 0.5
        w[-int(0.25 * (N - 1)):] = 0.5
        return np.sum(w * diff_sig)

    elif feature_name == "rms":  # Root Mean Square
        return np.sqrt(np.mean(signal ** 2))

    elif feature_name == "zc":  # Zero Crossing
        threshold = 1e-3
        return np.sum(((signal[:-1] * signal[1:]) < 0) &
                      (np.abs(signal[:-1] - signal[1:]) >= threshold))

    elif feature_name == "ssc":  # Slope Sign Change
        threshold = 1e-3
        return np.sum(((signal[1:-1] - signal[:-2]) * (signal[1:-1] - signal[2:]) > 0) &
                      (np.abs(signal[1:-1] - signal[:-2]) >= threshold) &
                      (np.abs(signal[1:-1] - signal[2:]) >= threshold))

    else:
        raise ValueError(f"Unknown feature name: {feature_name}")
