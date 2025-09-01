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

    FV_forearm = []
    FV_wrist = []

    # ----------------- Loop through participants -----------------
    for participantNum in range(1, noOfParticipants + 1):
        participant_forearm_all_sessions = []
        participant_wrist_all_sessions = []

        print(f"Processing Participant {participantNum}...")

        # ----------------- Load all sessions for this participant -----------------
        for sessionNum in range(1, noOfSessions + 1):
            sessionFolder = os.path.join(mainDataFolder, f"Session{sessionNum}_Converted")
            participantFile = f"session{sessionNum}_participant{participantNum}.npz"
            filePath = os.path.join(sessionFolder, participantFile)

            if not os.path.exists(filePath):
                print(f"Missing file {filePath}, skipping")
                continue

            data = np.load(filePath, allow_pickle=True)
            forearmData, wristData = data["forearmData"].item(), data["wristData"].item()

            subj_forearm, subj_wrist = [], []
            for g in gestures:
                trialF_list, trialW_list = [], []
                for trialNum in range(1, noOfTrials + 1):
                    trialF = forearmData.get((trialNum, g))
                    trialW = wristData.get((trialNum, g))
                    if trialF is not None and trialW is not None:
                        trialF_list.append(trialF)
                        trialW_list.append(trialW)
                subj_forearm.append(trialF_list)
                subj_wrist.append(trialW_list)

            participant_forearm_all_sessions.append(subj_forearm)
            participant_wrist_all_sessions.append(subj_wrist)

        # ----------------- Combine all sessions for this participant -----------------
        # Result: gestures × (trials × sessions) × channels × levels × features
        combined_forearm = []
        combined_wrist = []

        for g_idx in range(len(gestures)):
            # Gather trials across sessions
            trialsF = []
            trialsW = []
            for session_dataF, session_dataW in zip(participant_forearm_all_sessions, participant_wrist_all_sessions):
                trialsF.extend(session_dataF[g_idx])
                trialsW.extend(session_dataW[g_idx])
            combined_forearm.append(trialsF)
            combined_wrist.append(trialsW)

        # ----------------- Preprocessing -----------------
        preprocessedF, preprocessedW = [], []
        for g_idx in range(len(gestures)):
            trial_preF, trial_preW = [], []
            for trial_idx in range(len(combined_forearm[g_idx])):
                oneF = np.nan_to_num(combined_forearm[g_idx][trial_idx].T)
                oneW = np.nan_to_num(combined_wrist[g_idx][trial_idx].T)
                preF = np.array([preprocess_emg(ch, fs, 10, 450) for ch in oneF]).T
                preW = np.array([preprocess_emg(ch, fs, 10, 450) for ch in oneW]).T
                trial_preF.append(preF)
                trial_preW.append(preW)
            preprocessedF.append(trial_preF)
            preprocessedW.append(trial_preW)

        # ----------------- Feature Extraction -----------------
        featF, featW = [], []
        for g_idx in range(len(gestures)):
            trial_featF, trial_featW = [], []
            for trial_idx in range(len(preprocessedF[g_idx])):
                f11, f12 = [], []

                oneF = preprocessedF[g_idx][trial_idx]
                oneW = preprocessedW[g_idx][trial_idx]

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

        print(f"Extracted features for Participant {participantNum}")

    # ----------------- Save participant-level features -----------------
    np.savez_compressed("Feature_vector_allParticipants.npz", FV_forearm=FV_forearm, FV_wrist=FV_wrist)
    print("Feature vectors saved for all participants as .npz")

if __name__ == "__main__":
    main()