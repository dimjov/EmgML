# B_feature_extraction.py
# Capstone Part II: Feature Extraction (all 3 sessions, .npz input/output)

import os
import numpy as np
from preprocess_emg import preprocess_emg
from jfemg import jfemg
import pywt

def main():
    """
    Main function to perform feature extraction on EMG data.
    It loads the pre-processed .npz files, applies a preprocessing filter,
    and then extracts features using Discrete Wavelet Transform (DWT).
    The final feature vectors are saved to a new .npz file.
    """
    mainDataFolder = "./Converted_data"

    fs = 2048  # Sampling frequency
    noOfSessions = 3
    noOfParticipants = 43
    # Selected gestures from the paper: Little Finger Extension, Index Finger Extension,
    # Thumb Finger Extension, Hand Open, Hand Close.
    gestures = [8, 9, 10, 15, 16]

    # Storage containers for all participants' data
    completeSet_forearm = []
    completeSet_wrist = []

    # ----------------- Load all sessions' data -----------------
    for sessionNum in range(1, noOfSessions + 1):
        sessionFolder = os.path.join(mainDataFolder, f"Session{sessionNum}_Converted")
        print(f"Processing session {sessionNum} from {sessionFolder}")

        for participantNum in range(1, noOfParticipants + 1):
            participantFile = f"session{sessionNum}_participant{participantNum}.npz"
            filePath = os.path.join(sessionFolder, participantFile)

            if not os.path.exists(filePath):
                print(f"Missing file {filePath}, skipping")
                continue

            data = np.load(filePath, allow_pickle=True)
            forearmData, wristData = data["forearmData"].item(), data["wristData"].item()

            subj_forearm, subj_wrist = [], []
            for g in gestures:
                # Use trial 1 for consistency with the paper's methodology
                trialF = forearmData.get((1, g))
                trialW = wristData.get((1, g))

                subj_forearm.append(trialF)
                subj_wrist.append(trialW)

            completeSet_forearm.append(subj_forearm)
            completeSet_wrist.append(subj_wrist)

            print(f"Loaded: Session {sessionNum} Participant {participantNum}")

    # ----------------- Preprocessing -----------------
    # Filters and rectifies the raw EMG signals
    preprocess_forearm = []
    preprocess_wrist = []

    for subj_idx in range(len(completeSet_forearm)):
        subjF, subjW = [], []
        for g_idx in range(len(gestures)):
            # Handle potential NaN values by converting them to zero
            oneF = np.nan_to_num(completeSet_forearm[subj_idx][g_idx].T)
            oneW = np.nan_to_num(completeSet_wrist[subj_idx][g_idx].T)

            # Apply the preprocessing function to each channel
            # This is an array of shape (num_channels, num_samples)
            preF = np.array([preprocess_emg(ch, fs, 10, 450) for ch in oneF])
            preW = np.array([preprocess_emg(ch, fs, 10, 450) for ch in oneW])

            subjF.append(preF.T)
            subjW.append(preW.T)

        preprocess_forearm.append(subjF)
        preprocess_wrist.append(subjW)

        print(f"Preprocessed participant {subj_idx+1}/{len(completeSet_forearm)}")

    # ----------------- Feature extraction -----------------
    # Extracts features using Discrete Wavelet Transform
    FV_forearm = []
    FV_wrist = []

    for subj_idx in range(len(preprocess_forearm)):
        featF, featW = [], []
        for g_idx in range(len(gestures)):
            f11, f12 = [], []

            oneF = preprocess_forearm[subj_idx][g_idx]
            oneW = preprocess_wrist[subj_idx][g_idx]

            # Forearm channels
            for iChannelNum in range(oneF.shape[1]):
                # Decompose signal using DWT with bior3.3 wavelet at level 4
                coeffs = pywt.wavedec(oneF[:, iChannelNum], "bior3.3", level=4)
                # Apply jfemg functions to each set of coefficients
                features = [[jfemg(fn, coeff) for fn in ["mav", "wl", "emav", "ewl", "rms", "zc", "ssc"]] for coeff in coeffs]
                f11.append(features)
            featF.append(np.array(f11))

            # Wrist channels
            for jChannelNum in range(oneW.shape[1]):
                # Decompose signal using DWT with bior3.3 wavelet at level 4
                coeffs = pywt.wavedec(oneW[:, jChannelNum], "bior3.3", level=4)
                # Apply jfemg functions to each set of coefficients
                features = [[jfemg(fn, coeff) for fn in ["mav", "wl", "emav", "ewl", "rms", "zc", "ssc"]] for coeff in coeffs]
                f12.append(features)
            featW.append(np.array(f12))

        FV_forearm.append(featF)
        FV_wrist.append(featW)

        print(f"Extracted features for participant {subj_idx+1}")

    # Save combined features for all sessions
    np.savez_compressed("Feature_vector_allSessions.npz", FV_forearm=FV_forearm, FV_wrist=FV_wrist)
    print("Feature vectors saved for all sessions as .npz")

if __name__ == "__main__":
    main()
