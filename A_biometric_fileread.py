# A_biometric_fileread.py
# Capstone Part I: Preparation of dataset

import os
import numpy as np
import wfdb  # MIT-BIH compatible reader
from globals import *  # import project constants

def main():
    # Dataset path (update if needed)
    mainFolder = r".\\Grabmyo-1.0.2"

    # Output folder
    converted_root = "Converted_data"
    os.makedirs(converted_root, exist_ok=True)

    count = 0
    for sessionNum in range(1, NUM_OF_SESSIONS+1):
        sessionFolder = f"Session{sessionNum}_Converted"
        sessionPath = os.path.join(converted_root, sessionFolder)
        os.makedirs(sessionPath, exist_ok=True)

        for participantNum in range(1, NUM_OF_PARTICIPANTS+1):
            participantFile = f"session{sessionNum}_participant{participantNum}"
            forearmData = {}
            wristData = {}

            for gestureNum in range(1, NUM_OF_GESTURES+2):  # +1 for resting position (17 total)
                for trialNum in range(1, NUM_OF_TRIALS+1):
                    filename = f"session{sessionNum}_participant{participantNum}_gesture{gestureNum}_trial{trialNum}"
                    filepath = os.path.join(mainFolder, f"Session{sessionNum}", participantFile, filename)

                    try:
                        record = wfdb.rdrecord(filepath)
                        data_emg = record.p_signal
                    except Exception as e:
                        print(f"Error reading {filepath}: {e}")
                        continue

                    # Forearm channels [1-16], Wrist channels [18-23] + [26-31]
                    forearm_channels = list(range(0, 16))
                    wrist_channels = list(range(17, 23)) + list(range(25, 31))

                    forearmData[(trialNum, gestureNum)] = data_emg[:, forearm_channels]
                    wristData[(trialNum, gestureNum)] = data_emg[:, wrist_channels]

            # Save data for participant as NPZ
            save_path = os.path.join(sessionPath, f"{participantFile}.npz")
            np.savez_compressed(save_path, forearmData=forearmData, wristData=wristData)

            count += 1
            print(f"Converted: {count} of {NUM_OF_PARTICIPANTS * NUM_OF_SESSIONS} participants")

    print("Participants file conversion completed (saved as .npz).")

if __name__ == "__main__":
    main()