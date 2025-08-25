# B_feature_extraction.py
# Capstone Part II: Feature Extraction (all 3 sessions, all 7 trials, .npz input/output)

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
    mainDataFolder = r".\Converted_data"

    fs = 2048  # Sampling frequency
    noOfSessions = 3
    noOfParticipants = 43
    noOfTrials = 7
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
                # Loop through all 7 trials for each gesture
                trialF_list, trialW_list = [], []
                for trialNum in range(1, noOfTrials + 1):
                    trialF = forearmData.get((trialNum, g))
                    trialW = wristData.get((trialNum, g))
                    
                    if trialF is not None and trialW is not None:
                        trialF_list.append(trialF)
                        trialW_list.append(trialW)
                    else:
                        print(f"Missing data for Trial {trialNum}, Gesture {g}, Participant {participantNum}, skipping this trial.")
                        # Append None or an empty array to maintain structure if needed
                        # For now, we just skip it, which might cause length mismatches later. 
                        # A better approach would be to fill with NaN or zeros.
                        pass
                
                subj_forearm.append(trialF_list)
                subj_wrist.append(trialW_list)

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
            trial_preF, trial_preW = [], []
            for trial_idx in range(len(completeSet_forearm[subj_idx][g_idx])):
                oneF = completeSet_forearm[subj_idx][g_idx][trial_idx].T
                oneW = completeSet_wrist[subj_idx][g_idx][trial_idx].T

                if oneF.size == 0 or oneW.size == 0:
                    print("empty trial")
                    continue  # Skip empty trials

                # Handle potential NaN values by converting them to zero
                oneF = np.nan_to_num(oneF)
                oneW = np.nan_to_num(oneW)

                # Apply the preprocessing function to each channel
                preF = np.array([preprocess_emg(ch, fs, 10, 450) for ch in oneF])
                preW = np.array([preprocess_emg(ch, fs, 10, 450) for ch in oneW])

                trial_preF.append(preF.T)
                trial_preW.append(preW.T)
            
            subjF.append(trial_preF)
            subjW.append(trial_preW)

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
            trial_featF, trial_featW = [], []
            for trial_idx in range(len(preprocess_forearm[subj_idx][g_idx])):
                f11, f12 = [], []

                oneF = preprocess_forearm[subj_idx][g_idx][trial_idx]
                oneW = preprocess_wrist[subj_idx][g_idx][trial_idx]

                # Forearm channels
                for iChannelNum in range(oneF.shape[1]):
                    coeffs = pywt.wavedec(oneF[:, iChannelNum], "bior3.3", level=4)
                    features = [[jfemg(fn, coeff) for fn in ["mav", "wl", "emav", "ewl", "rms", "zc", "ssc"]] for coeff in coeffs]
                    f11.append(features)
                trial_featF.append(np.array(f11))

                # Wrist channels
                for jChannelNum in range(oneW.shape[1]):
                    coeffs = pywt.wavedec(oneW[:, jChannelNum], "bior3.3", level=4)
                    features = [[jfemg(fn, coeff) for fn in ["mav", "wl", "emav", "ewl", "rms", "zc", "ssc"]] for coeff in coeffs]
                    f12.append(features)
                trial_featW.append(np.array(f12))

            featF.append(trial_featF)
            featW.append(trial_featW)

        FV_forearm.append(featF)
        FV_wrist.append(featW)

        print(f"Extracted features for participant {subj_idx+1}")

    # Save combined features for all sessions
    np.savez_compressed("Feature_vector_allSessions.npz", FV_forearm=FV_forearm, FV_wrist=FV_wrist)
    print("Feature vectors saved for all sessions as .npz")

if __name__ == "__main__":
    main()