# C_classification.py (with Train-Validation-Test Split)
import os
import joblib
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
    Summarize features per trial:
    - Average across wavelet levels
    - Average across channels
    - Concatenate forearm + wrist features
    Returns fixed-length feature vectors.
    """
    X, y = [], []
    n_gestures = len(gestures)

    for participant_idx in range(len(forearm_features)):
        for g_idx, gesture in enumerate(gestures):
            n_trials = forearm_features[participant_idx][g_idx].shape[0]

            for trial_idx in range(n_trials):
                # Forearm features: (channels, levels, features)
                f_trial = forearm_features[participant_idx][g_idx][trial_idx]
                # Wrist features: same structure
                w_trial = wrist_features[participant_idx][g_idx][trial_idx]

                # Average over channels and levels → (features,)
                f_summary = np.mean(f_trial, axis=(0, 1))
                w_summary = np.mean(w_trial, axis=(0, 1))

                # Concatenate to form trial-level feature vector
                combined_features = np.concatenate((f_summary, w_summary))

                X.append(combined_features)
                y.append(gesture)

    X = np.array(X)
    y = np.array(y)
    return X, y

def evaluate_model(y_test, y_pred, class_order, model_name):
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

    # === Save evaluation table to Excel ===
    table_path = os.path.join("./models", f"{model_name}_evaluation.xlsx")
    df.to_excel(table_path, float_format="%.5f")
    print(f"Evaluation table saved to {table_path}")

    # === Save confusion matrix plot ===
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_order, yticklabels=class_order, cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")

    plot_path = os.path.join("./plots", f"{model_name}_confusion_matrix.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix plot saved to {plot_path}")

    return df

def SVM_classifier(X_train_reduced, X_test_reduced, y_train, model_name="lda-svm_5gesture_model.pkl"):
    # Initialize the SVM classifier
    svm_model = SVC(kernel='linear', random_state=42)
    
    # Train the SVM model on the reduced training data
    svm_model.fit(X_train_reduced, y_train)

    # Save the trained model
    model_path = os.path.join("./models", model_name)
    joblib.dump(svm_model, model_path)
    print(f"LDA-SVM model saved to {model_path}")
    
    # Make predictions on the reduced test data
    y_pred = svm_model.predict(X_test_reduced)

    return y_pred

def NaiveBayes_classifier(X_train_reduced, X_test_reduced, y_train, model_name="lda-naivebayes_5gesture_model.pkl"):
    # Initialize the Naive Bayes classifier
    nb_model = GaussianNB()

    # Train the Naive Bayes model
    nb_model.fit(X_train_reduced, y_train)

    # Save the trained model
    model_path = os.path.join("./models", model_name)
    joblib.dump(nb_model, model_path)
    print(f"LDA-Naive Bayes model saved to {model_path}")

    # Make predictions on the reduced test data
    y_pred = nb_model.predict(X_test_reduced)

    return y_pred

def KNN_classifier(X_train_reduced, X_test_reduced, y_train, model_name="lda-knn_5gesture_model.pkl", n_neighbors=5):
    # Initialize the KNN classifier
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train the KNN model
    knn_model.fit(X_train_reduced, y_train)

    # Save the trained model
    model_path = os.path.join("./models", model_name)
    joblib.dump(knn_model, model_path)
    print(f"LDA-KNN model saved to {model_path}")

    # Make predictions on the reduced test data
    y_pred = knn_model.predict(X_test_reduced)

    return y_pred

def main():
    # Load pre-processed data
    print("============ Step 1: Loading participant-level features ============")
    data = np.load("Feature_vector_allParticipants.npz", allow_pickle=True)
    forearm_fv = data["FV_forearm"]
    wrist_fv = data["FV_wrist"]

    gestures = [8, 9, 10, 15, 16]  # selected gestures

    # ----------------- Step 2: Participant-level 70/30 split -----------------
    participants = np.arange(len(forearm_fv))  # 0..42
    train_subj, test_subj = train_test_split(participants, test_size=0.3, random_state=42)

    X_train, y_train, X_test, y_test = [], [], [], []

    for idx, (f_feat, w_feat) in enumerate(zip(forearm_fv, wrist_fv)):
        X_subj, y_subj = prepare_data([f_feat], [w_feat], gestures)  # treat each participant as one entry
        if idx in train_subj:
            X_train.append(X_subj)
            y_train.append(y_subj)
        else:
            X_test.append(X_subj)
            y_test.append(y_subj)
    
    # Flatten lists into arrays
    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)
    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)

    print(f"Train set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # ----------------- Step 3: Scale features -----------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----------------- Step 4: LDA for dimensionality reduction -----------------
    lda = LinearDiscriminantAnalysis(n_components=len(gestures) - 1)
    X_train_reduced = lda.fit_transform(X_train_scaled, y_train)
    X_test_reduced = lda.transform(X_test_scaled)

    print(f"Original feature space dimension: {X_train.shape[1]}")
    print(f"Reduced feature space dimension (after LDA): {X_train_reduced.shape[1]}")

    # Create directories for models and plots
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./plots", exist_ok=True)
    class_order = sorted(np.unique(y_train))

    # SVM
    y_pred = SVM_classifier(X_train_reduced, X_test_reduced, y_train)
    evaluate_model(y_test, y_pred, class_order, "LDA-SVM")

    # Naive Bayes
    y_pred = NaiveBayes_classifier(X_train_reduced, X_test_reduced, y_train)
    evaluate_model(y_test, y_pred, class_order, "LDA-Naive Bayes")

    # KNN
    y_pred = KNN_classifier(X_train_reduced, X_test_reduced, y_train)
    evaluate_model(y_test, y_pred, class_order, "LDA-KNN")
    

if __name__ == "__main__":
    main()
