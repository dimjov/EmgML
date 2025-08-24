# C_classification.py
# Capstone Part III: Machine Learning Classification
# - 5-fold stratified cross-validation
# - LDA for dimensionality reduction (fit INSIDE each fold)
# - ECOC wrappers (One-vs-One) to mirror MATLAB fitcecoc
# - Reports per-fold accuracy and mean/std; also aggregate confusion matrix

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import warnings

# Suppress all future warnings to reduce clutter
warnings.simplefilter(action='ignore', category=FutureWarning)

def stats_of_measure(cm, gesture_names):
    """
    Calculates and prints classification metrics from a confusion matrix.
    This function mimics the output of the MATLAB helper function.

    Inputs:
    cm (numpy.ndarray): The confusion matrix.
    gesture_names (list): List of class names.

    Returns:
    dict: A dictionary containing calculated metrics.
    """
    n_classes = cm.shape[0]
    stats = {}

    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    # Per-class metrics
    sensitivity = np.nan_to_num(TP / (TP + FN))
    specificity = np.nan_to_num(TN / (TN + FP))
    precision = np.nan_to_num(TP / (TP + FP))
    f1_score = np.nan_to_num(2 * (precision * sensitivity) / (precision + sensitivity))

    # Overall and Macro-averaged metrics
    overall_accuracy = np.sum(TP) / np.sum(cm)
    macro_precision = np.mean(precision)
    macro_sensitivity = np.mean(sensitivity)
    macro_specificity = np.mean(specificity)
    macro_f1_score = np.mean(f1_score)

    stats = {
        'Overall Accuracy': overall_accuracy,
        'Macro Precision': macro_precision,
        'Macro Sensitivity': macro_sensitivity,
        'Macro Specificity': macro_specificity,
        'Macro F1-Score': macro_f1_score,
        'Per-Class Precision': precision,
        'Per-Class Sensitivity': sensitivity,
        'Per-Class Specificity': specificity,
        'Per-Class F1-Score': f1_score
    }

    # Print a formatted table
    print("\nPer-Class Metrics:")
    df = pd.DataFrame({
        'Class': gesture_names,
        'Precision': precision,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1-Score': f1_score
    }).round(4)
    print(df.to_string(index=False))
    print("-" * 65)
    print(f"Macro Average Precision: {macro_precision:.4f}")
    print(f"Macro Average Sensitivity: {macro_sensitivity:.4f}")
    print(f"Macro Average Specificity: {macro_specificity:.4f}")
    print(f"Macro Average F1-Score: {macro_f1_score:.4f}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}\n")

    return stats


def cv_eval_model(X, y, model_name, base_estimator, use_lda_dimred=False):
    """
    Performs 5-fold stratified cross-validation and evaluates a model.

    Inputs:
    X (numpy.ndarray): The feature matrix.
    y (numpy.ndarray): The class labels.
    model_name (str): Name of the model for printing.
    base_estimator (sklearn.base.BaseEstimator): The classifier to evaluate.
    use_lda_dimred (bool): Flag to perform LDA dimensionality reduction.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    accuracies = []
    all_true_labels = []
    all_predicted_labels = []

    # Get unique class labels for LDA
    class_order = sorted(np.unique(y))
    # Removed the n_components setting here, as it will be handled by the pipeline
    # The LDA within the OneVsOneClassifier will automatically determine the number
    # of components as 1 for each binary classification task.

    print(f"\n--- {model_name} Results ---")
    
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if use_lda_dimred:
            # Create a pipeline with StandardScaler, LDA, and the classifier
            # OneVsOneClassifier wraps the entire pipeline
            pipeline = OneVsOneClassifier(make_pipeline(
                StandardScaler(),
                LinearDiscriminantAnalysis(), # Removed n_components
                base_estimator
            ))
        else:
            # Create a pipeline with StandardScaler and the classifier only
            pipeline = OneVsOneClassifier(make_pipeline(
                StandardScaler(),
                base_estimator
            ))
        
        # Fit the pipeline and make predictions
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Calculate accuracy for this fold
        fold_accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(fold_accuracy)
        print(f"Fold {fold+1}: Accuracy = {fold_accuracy:.4f}")
        
        # Append results for aggregate confusion matrix
        all_true_labels.extend(y_test)
        all_predicted_labels.extend(y_pred)
        
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    # Generate aggregate confusion matrix and stats
    cm = confusion_matrix(all_true_labels, all_predicted_labels, labels=class_order)
    print("\nAggregate Confusion Matrix:")
    print(cm)
    stats_of_measure(cm, class_order)

def main():
    """
    Main function to load features, reshape them, and run all classification models.
    """
    # Load the feature vectors from the .npz file
    try:
        data = np.load("Feature_vector_allSessions.npz", allow_pickle=True)
        FV_forearm = data["FV_forearm"]
        FV_wrist = data["FV_wrist"]
    except FileNotFoundError:
        print("Error: 'Feature_vector_allSessions.npz' not found.")
        print("Please run B_feature_extraction.py first to generate the required file.")
        return
    except KeyError:
        print("Error: 'FV_forearm' or 'FV_wrist' keys not found in the .npz file.")
        print("Please check the output of B_feature_extraction.py.")
        return

    # These are the 5 gestures chosen for classification, matching B_feature_extraction.py
    gestures = [8, 9, 10, 15, 16]
    motion_names = [
        "Lateral Prehension", "Thumb Adduction", "Thumb and Little Finger Opposition",
        "Thumb and Index Finger Opposition", "Thumb and Index Finger Extension", "Thumb and Little Finger Extension",
        "Index and Middle Finger Extension", "Little Finger Extension", "Index Finger Extension",
        "Thumb Finger Extension", "Wrist Extension", "Wrist Flexion",
        "Forearm Supination", "Forearm Pronation", "Hand Open",
        "Hand Close", "Rest"
    ]
    selected_motion_names = [motion_names[g - 1] for g in gestures]

    print("Reshaping feature vectors for ML models...")
    
    X = []
    y = []
    
    # Reshape the loaded data into a flat feature matrix (X) and label vector (y)
    # The structure from B_feature_extraction is a list of lists:
    # FV_forearm[participant_idx][gesture_idx]
    for p_idx, (p_forearm_data, p_wrist_data) in enumerate(zip(FV_forearm, FV_wrist)):
        for g_idx, (g_forearm_features, g_wrist_features) in enumerate(zip(p_forearm_data, p_wrist_data)):
            # Flatten the 3D feature arrays into 1D vectors
            forearm_vector = g_forearm_features.flatten()
            wrist_vector = g_wrist_features.flatten()
            
            # Concatenate the vectors
            combined_feature_vector = np.concatenate((forearm_vector, wrist_vector))
            
            X.append(combined_feature_vector)
            y.append(selected_motion_names[g_idx])
            
    X = np.array(X)
    y = np.array(y)

    print(f"Total samples prepared for ML: {X.shape[0]}")
    print(f"Shape of predictors (X): {X.shape}")
    print(f"Shape of labels (y): {y.shape}")

    # Define the classifiers
    dt_base = DecisionTreeClassifier(random_state=42)
    knn_base = KNeighborsClassifier(n_neighbors=5)
    nb_base = GaussianNB()
    svm_base = SVC(kernel="linear", random_state=42, probability=True)

    # Perform cross-validated evaluation for each model
    cv_eval_model(X, y, "Decision Tree", dt_base, use_lda_dimred=True)
    cv_eval_model(X, y, "KNN", knn_base, use_lda_dimred=True)
    cv_eval_model(X, y, "Naive Bayes", nb_base, use_lda_dimred=True)
    cv_eval_model(X, y, "SVM", svm_base, use_lda_dimred=True)
    
if __name__ == "__main__":
    main()