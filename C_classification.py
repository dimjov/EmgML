# C_classification.py (with Train-Validation-Test Split)
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Suppress all future warnings to reduce clutter
warnings.simplefilter(action='ignore', category=FutureWarning)

def prepare_data(forearm_features, wrist_features, gestures):
    """
    Step 1: Flattens the feature vectors and creates a corresponding label vector.

    Parameters:
        forearm_features (np.ndarray): The multi-dimensional forearm feature vectors.
        wrist_features (np.ndarray): The multi-dimensional wrist feature vectors.
        gestures (list): A list of gesture labels.

    Returns:
        tuple: A tuple containing the flattened feature matrix (X) and label vector (y).
    """
    X = []
    y = []
    
    # Iterate through the nested data structure to flatten each trial's features
    for participant_idx in range(len(forearm_features)):
        for gesture_idx in range(len(forearm_features[participant_idx])):
            for trial_idx in range(len(forearm_features[participant_idx][gesture_idx])):
                # Flatten the features for a single trial from both sensors
                forearm_flat = forearm_features[participant_idx][gesture_idx][trial_idx].flatten()
                wrist_flat = wrist_features[participant_idx][gesture_idx][trial_idx].flatten()
                
                # Combine forearm and wrist features into a single vector
                combined_features = np.concatenate((forearm_flat, wrist_flat))
                X.append(combined_features)
                
                # Assign the corresponding gesture label
                y.append(gestures[gesture_idx])
    
    return np.array(X), np.array(y)

def evaluate_model(y_test, y_pred, class_order):
    # Calculate the confusion matrix
    cm = confusion_matrix(
        y_test, 
        y_pred, 
        labels=class_order
    )

    TP = np.diag(cm)                    # True positives (TP)
    FP = np.sum(cm, axis=0) - TP        # False Positives (FP)
    FN = np.sum(cm, axis=1) - TP        # False Negatives (FN)
    TN = np.sum(cm) - (TP + FP + FN)    # True Negatives (TN)

    # Print TP, FP, FN, TN for each gesture/class
    for i, cls in enumerate(class_order):
        print(f"\nClass {cls}:")
        print(f"    TP = {TP[i]}")
        print(f"    FP = {FP[i]}")
        print(f"    FN = {FN[i]}")
        print(f"    TN = {TN[i]}")
    
    # Print the confusion matrix
    print(cm)

    # === Build evaluation table ===
    metrics = []
    for i, cls in enumerate(class_order):
        precision = TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) > 0 else 0
        recall = TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else 0
        accuracy = (TP[i] + TN[i]) / (TP[i] + FP[i] + FN[i] + TN[i])
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        metrics.append([
            TP[i], FP[i], FN[i], TN[i],
            precision, recall, accuracy, f1
        ])

    # Macro averages
    macro = np.mean(metrics, axis=0)

    # Create DataFrame
    df = pd.DataFrame(metrics, 
                      index=[f"Class {cls}" for cls in class_order],
                      columns=["True Positive", "False Positive", "False Negative", "True Negative",
                               "Precision", "Sensitivity", "Accuracy", "F-measure"])

    # Add macro average row
    df.loc["Macro Average"] = macro

    print("\n============ Evaluation Table ============")
    print(df.to_string(float_format="%.5f"))

    return

def SVM_classifier(X_train_reduced, X_test_reduced, y_train):
    # Initialize the SVM classifier
    svm_model = SVC(kernel='linear', random_state=42)
    
    # Train the SVM model on the reduced training data
    svm_model.fit(X_train_reduced, y_train)
    
    # Make predictions on the reduced test data
    y_pred = svm_model.predict(X_test_reduced)

    return y_pred

def main():
    """
    Main function to orchestrate the entire classification process.
    """
    # Step 1: Load the pre-processed data
    print("============ Step 1: Loading and preparing data ============")
    data = np.load("Feature_vector_allSessions.npz", allow_pickle=True)
    forearm_fv = data["FV_forearm"]
    wrist_fv = data["FV_wrist"]

    # Chosen gestures from B_feature_extraction.py
    gestures = [8, 9, 10, 15, 16]
    X, y = prepare_data(forearm_fv, wrist_fv, gestures)
    
    print(f"Shape of feature matrix X: {X.shape}")
    print(f"Shape of label vector y: {y.shape}")
    

    # Step 2: 70/30 Train-Test Split
    print("============ Step 2: 70/30 Train-Test Split ============")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    class_order = sorted(np.unique(y))
    
    print(f"Train set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Perform LDA for dimensionality reduction
    # The number of components is set to the number of classes - 1.
    lda = LinearDiscriminantAnalysis(n_components=len(gestures) - 1)
    X_train_reduced = lda.fit_transform(X_train_scaled, y_train)
    X_test_reduced = lda.transform(X_test_scaled)
    
    print(f"Original feature space dimension: {X_train.shape[1]}")
    print(f"Reduced feature space dimension (after LDA): {X_train_reduced.shape[1]}")
    
    # Build SVM classifier, train it and make predictions
    y_pred = SVM_classifier(X_train_reduced, X_test_reduced, y_train)
    
    # Evaluate the model (performance)
    evaluate_model(y_test, y_pred, class_order)

    # global counts
    # TP_total = np.sum(TP)
    # FP_total = np.sum(FP)
    # FN_total = np.sum(FN)
    # TN_total = np.sum(TN)

    # print("\nOverall counts:")
    # print(f"TP = {TP_total}, FP = {FP_total}, FN = {FN_total}, TN = {TN_total}")


    # cm = svm_cm.astype("float") / svm_cm.sum(axis=1)[:, np.newaxis]

    # fig, ax = plt.subplots(figsize=(8, 6))
    # sns.heatmap(cm, annot=svm_cm, fmt="d", cmap="Blues", 
    #             xticklabels=class_order, yticklabels=class_order, cbar=False, ax=ax)

    # # Highlight TP, FP, FN, TN
    # for i in range(len(class_order)):
    #     for j in range(len(class_order)):
    #         value = svm_cm[i, j]
    #         if i == j and value > 0:  
    #             ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="green", lw=3))  # TP
    #         elif i != j and value > 0:
    #             if i < j:
    #                 ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="red", lw=2))  # FP
    #             else:
    #                 ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="orange", lw=2))  # FN

    # ax.set_xlabel("Predicted gesture")
    # ax.set_ylabel("True gesture")
    # ax.set_title("LDA-SVM model confusion matrix")

    # plt.show()

    

if __name__ == "__main__":
    main()
