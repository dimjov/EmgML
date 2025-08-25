import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import warnings

# Suppress all future warnings to reduce clutter
warnings.simplefilter(action='ignore', category=FutureWarning)

def stats_of_measure(cm, gesture_names, model_name):
    """
    Calculates and prints classification metrics from a confusion matrix.
    This function mimics the output format of the provided table.

    Inputs:
    cm (numpy.ndarray): The confusion matrix.
    gesture_names (list): List of class names.
    model_name (str): The name of the model for the table title.

    Returns:
    dict: A dictionary containing calculated metrics.
    """
    n_classes = cm.shape[0]
    
    # Calculate True Positive, False Positive, False Negative, True Negative
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm.sum()) - (TP + FP + FN)

    # Per-class metrics
    precision = np.nan_to_num(TP / (TP + FP))
    sensitivity = np.nan_to_num(TP / (TP + FN))
    specificity = np.nan_to_num(TN / (TN + FP))
    accuracy = np.nan_to_num((TP + TN) / (TP + TN + FP + FN))
    f1_score = np.nan_to_num(2 * (precision * sensitivity) / (precision + sensitivity))

    # Macro-averaged metrics
    macro_precision = np.mean(precision)
    macro_sensitivity = np.mean(sensitivity)
    macro_accuracy = np.mean(accuracy)
    macro_f1_score = np.mean(f1_score)

    # Create and print the table
    print("\n" + "=" * 80)
    print(f"{'':<20}{model_name} (Three Sessions){'':>25}")
    print("-" * 80)
    
    table_data = {
        'Evaluation Metrics': ['True Positive', 'False Positive', 'False Negative', 'True Negative', 'Precision', 'Sensitivity', 'Accuracy', 'F-measure'],
    }
    
    # Use a DataFrame to simplify printing and formatting
    df = pd.DataFrame({
        'Evaluation Metrics': ['True Positive', 'False Positive', 'False Negative', 'True Negative',
                               'Precision', 'Sensitivity', 'Accuracy', 'F-measure'],
    })

    # Add per-class data
    for i, gesture in enumerate(gesture_names):
        df[gesture] = [
            int(TP[i]), int(FP[i]), int(FN[i]), int(TN[i]),
            f"{precision[i]:.5f}", f"{sensitivity[i]:.5f}", f"{accuracy[i]:.5f}", f"{f1_score[i]:.5f}"
        ]
    
    # Add Macro Average column
    df['Macro Average'] = [
        int(np.mean(TP)), int(np.mean(FP)), int(np.mean(FN)), int(np.mean(TN)),
        f"{macro_precision:.5f}", f"{macro_sensitivity:.5f}", f"{macro_accuracy:.5f}", f"{macro_f1_score:.5f}"
    ]

    # Print the DataFrame without index and with a clean header
    print(df.to_string(index=False))

    print("=" * 80)

    stats = {
        'Overall Accuracy': np.sum(TP) / np.sum(cm),
        'Macro Precision': macro_precision,
        'Macro Sensitivity': macro_sensitivity,
        'Macro F1-Score': macro_f1_score,
        'Per-Class Precision': precision,
        'Per-Class Sensitivity': sensitivity,
        'Per-Class Accuracy': accuracy,
        'Per-Class F1-Score': f1_score,
    }
    return stats


def cv_eval_model(X, y, model_name, base_estimator, use_lda_dimred=True):
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

    class_order = sorted(np.unique(y))
    
    print(f"\n--- {model_name} Results ---")
    
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Create a pipeline with StandardScaler, LDA (if requested), and the classifier
        pipeline_steps = [StandardScaler()]
        if use_lda_dimred:
            # LDA automatically determines the correct number of components for
            # each binary subproblem in the OneVsOneClassifier wrapper.
            pipeline_steps.append(LinearDiscriminantAnalysis())
        
        pipeline_steps.append(base_estimator)
        
        pipeline = OneVsOneClassifier(make_pipeline(*pipeline_steps))
        
        # Fit the pipeline and make predictions
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        fold_accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(fold_accuracy)
        print(f"Fold {fold+1}: Accuracy = {fold_accuracy:.4f}")
        
        all_true_labels.extend(y_test)
        all_predicted_labels.extend(y_pred)
        
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    # Generate aggregate confusion matrix and stats
    cm = confusion_matrix(all_true_labels, all_predicted_labels, labels=class_order)
    stats = stats_of_measure(cm, class_order, model_name)
    
    print("-" * 65)
    print(f"Mean Fold Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
    print(f"Overall Accuracy: {stats['Overall Accuracy']:.4f}")
    print("-" * 65)
    
def main():
    """
    Main function to load features, reshape them, and run all classification models.
    """
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
    
    for p_idx, (p_forearm_data, p_wrist_data) in enumerate(zip(FV_forearm, FV_wrist)):
        for g_idx, (g_forearm_features, g_wrist_features) in enumerate(zip(p_forearm_data, p_wrist_data)):
            # The data structure now includes trials. We need to loop through them.
            for trial_data_forearm, trial_data_wrist in zip(g_forearm_features, g_wrist_features):
                forearm_vector = trial_data_forearm.flatten()
                wrist_vector = trial_data_wrist.flatten()
                
                combined_feature_vector = np.concatenate((forearm_vector, wrist_vector))
                
                X.append(combined_feature_vector)
                y.append(selected_motion_names[g_idx])
                
    X = np.array(X)
    y = np.array(y)
    
    print(f"Total samples prepared for ML: {X.shape[0]}")
    print(f"Shape of predictors (X): {X.shape}")
    print(f"Shape of labels (y): {y.shape}")
    
    # Define the classifiers
    knn_base = KNeighborsClassifier(n_neighbors=5)
    nb_base = GaussianNB()
    svm_base = SVC(kernel="linear", random_state=42, probability=False)

    # The MATLAB code's LDA is not wrapped in fitcecoc. 
    # The LDA is a standalone step for dimensionality reduction.
    # We will replicate this by training the LDA model separately.
    
    # Run the classification models
    # The LDA n_components is set to the number of classes - 1
    # This is a standard practice and reflects what MATLAB's LDA does by default
    cv_eval_model(X, y, "KNN", knn_base, use_lda_dimred=True)
    cv_eval_model(X, y, "Naive Bayes", nb_base, use_lda_dimred=True)
    cv_eval_model(X, y, "SVM", svm_base, use_lda_dimred=True)
    
if __name__ == "__main__":
    main()