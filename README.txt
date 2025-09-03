====================================================================
BUILDING A MACHINE LEARNING ALGORITHM FOR GENERAL PREDICTION OF 
HAND GESTURES
By: Dimitar Jovanovski
====================================================================

1. INTRODUCTION
The code in this repository uses PhysioNet's GrabMyo 1.0.2 dataset 
for feature extraction and building and training machine learning 
models for general prediction of hand movements/gestures.

2. DATASET STRUCTURE
The dataset used in this code is the PhysioNet GrabMyo 1.0.2 dataset 
which provides EMG signals recordings of 43 participants over 3 
different days while performing multiple hand gestures. The signals 
have been acquired at a sampling rate of 2048 Hz using the EMGUSB2+ 
device (OT Bioelletronica, Italy). EMG signals have been collected 
from 16 locations (channels) on the forearm and 12 locations on the wrist. 
Link to dataset: https://www.physionet.org/content/grabmyo/1.0.2/

3. DATA PROCESSING PIPELINE
To prepare the signals for classification, several preprocessing and 
feature engineering steps were applied:
- Average referencing across channels to reduce noise.  
- Filtering (band-pass 10–450 Hz, notch filter at 60 Hz, high-pass 0.1 Hz) 
  to remove interference and DC drift.  
- Discrete Wavelet Transform (Biorthogonal 3.3, level 4) for signal decomposition.  
- Feature extraction using MAV, WL, ZC, SSC, RMS, EWLs, and EMAV.  
- Dimensionality reduction with Linear Discriminant Analysis (LDA).  

These steps transform noisy raw EMG signals into compact, informative 
feature vectors suitable for machine learning.

4. MACHINE LEARNING MODELS & RESULTS
Three classical ML algorithms were tested: Naïve Bayes, K-Nearest Neighbor 
(KNN), and Support Vector Machine (SVM). The dataset was split into 
training (70%) and testing (30%) sets, and models were evaluated using 
accuracy, precision, recall, and F1-score. 
Results show consistent performance across models with overall accuracy 
in the range of 78–81%. Among them, LDA-SVM achieved the best performance, 
confirming its robustness for EMG-based gesture recognition.
