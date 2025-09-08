# B_windowing.py
# Capstone Part II (Option 2): Windowing for deep learning

import os
import numpy as np
from preprocess_emg import preprocess_emg
from globals import *

# Sampling frequency (from dataset)
FS = 2048  

# Window parameters (200 ms with 50% overlap)
WINDOW_SIZE = int(round(0.2 * FS))   # ~410 samples
STEP_SIZE = WINDOW_SIZE // 2         # ~205 samples

def segment_signal(signal, window_size, step):
    """
    Splits a multi-channel signal into overlapping windows.
    
    Args:
        signal (ndarray): shape (n_samples, n_channels)
        window_size (int): number of samples per window
        step (int): step size between windows
    
    Returns:
        ndarray: shape (n_windows, window_size, n_channels)
    """
    n_samples = signal.shape[0]
    windows = []
    for start in range(0, n_samples - window_size + 1, step):
        end = start + window_size
        windows.append(signal[start:end, :])
    return np.stack(windows, axis=0)


def main():
    input_root = "Converted_data"
    output_root = "Windowed_data"
    os.makedirs(output_root, exist_ok=True)

    X_list, y_list, subj_list = [], [], []

    for sessionNum in range(1, NUM_OF_SESSIONS+1):
        sessionFolder = os.path.join(input_root, f"Session{sessionNum}_Converted")

        for participantNum in range(1, NUM_OF_PARTICIPANTS+1):
            participantFile = f"session{sessionNum}_participant{participantNum}.npz"
            filePath = os.path.join(sessionFolder, participantFile)

            if not os.path.exists(filePath):
                print(f"Missing {filePath}, skipping...")
                continue

            data = np.load(filePath, allow_pickle=True)
            forearmData = data["forearmData"].item()
            wristData = data["wristData"].item()

            for (trialNum, gestureNum), trialData in forearmData.items():
                if trialData is None:
                    print(f"TrialData recording is missing, skipping...")
                    continue

                # --- Choose channels: here we concatenate forearm + wrist ---
                trialF = forearmData.get((trialNum, gestureNum))
                trialW = wristData.get((trialNum, gestureNum))
                if trialF is None or trialW is None:
                    print(f"trialF or trialW recording is missing, skipping...")
                    continue

                trial = np.hstack([trialF, trialW])  # shape (n_samples, n_channels_total)

                # --- Preprocess each channel ---
                trial_pre = np.array([
                    preprocess_emg(trial[:, ch], FS) for ch in range(trial.shape[1])
                ]).T  # shape (n_samples, n_channels)

                # --- Segment into windows ---
                windows = segment_signal(trial_pre, WINDOW_SIZE, STEP_SIZE)

                # --- Store with labels ---
                X_list.append(windows.astype(np.float32))
                y_list.append(np.full(windows.shape[0], gestureNum, dtype=np.int32))
                subj_list.append(np.full(windows.shape[0], participantNum, dtype=np.int32))

            print(f"Windowed P{participantNum} S{sessionNum}")

    # --- Concatenate all participants ---
    X = np.concatenate(X_list, axis=0)   # (N, win_size, n_channels)
    y = np.concatenate(y_list, axis=0)   # (N,)
    subj = np.concatenate(subj_list, axis=0)

    # --- Save ---
    save_path = os.path.join(output_root, "windowed_dataset.npz")
    np.savez_compressed(save_path, X=X, y=y, subj=subj)
    print(f"Saved windowed dataset: {save_path}")
    print(f"Shape X: {X.shape}, y: {y.shape}")


if __name__ == "__main__":
    main()
