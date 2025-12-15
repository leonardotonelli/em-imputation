import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import miceforest as mf
from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def mode_imputation_labels(data, label_column_index, true_proportions):
    """
    Impute missing categorical labels using mode imputation.
    
    Parameters:
    -----------
    data : array-like
        Full dataset with missing values (features + label column)
    label_column_index : int
        Index of the label column in the dataset
    true_proportions : dict or array-like
        True class mixing proportions. If dict: {class_label: proportion}
        If array: proportions in order of sorted unique classes
    
    Returns:
    --------
    proportion_error : float
        L2 norm of difference between true and imputed class proportions
    imputed_data : array-like
        Complete dataset after label imputation
    """
    start = time()
    data = np.array(data)
    labels = data[:, label_column_index]
    
    imputer = SimpleImputer(strategy='most_frequent')
    # Reshape to 2D array as required by SimpleImputer
    labels_reshaped = labels.reshape(-1, 1)
    labels_imputed = imputer.fit_transform(labels_reshaped).flatten()
    
    # Compute class proportions from imputed labels
    unique_classes, counts = np.unique(labels_imputed, return_counts=True)
    imputed_proportions = counts / len(labels_imputed)
    
    # Convert true_proportions to array format
    if isinstance(true_proportions, dict):
        true_prop_array = np.array([true_proportions.get(c, 0) for c in unique_classes])
    else:
        true_prop_array = np.array(true_proportions)
    
    # Ensure same length (in case of missing classes in imputed data)
    if len(true_prop_array) > len(imputed_proportions):
        # Pad imputed proportions with zeros for missing classes
        imputed_proportions = np.pad(imputed_proportions, 
                                     (0, len(true_prop_array) - len(imputed_proportions)))
    elif len(imputed_proportions) > len(true_prop_array):
        # Pad true proportions with zeros
        true_prop_array = np.pad(true_prop_array, 
                                 (0, len(imputed_proportions) - len(true_prop_array)))
    
    # Compute L2 norm of difference
    proportion_error = np.linalg.norm(true_prop_array - imputed_proportions)
    
    # Create imputed dataset
    imputed_data = data.copy()
    imputed_data[:, label_column_index] = labels_imputed
    end = time()
    return proportion_error, end - start


def knn_imputation_labels(data, label_column_index, true_proportions, k=5):
    """
    Impute missing categorical labels using KNN Classifier based on features.
    
    Parameters:
    -----------
    data : array-like
        Full dataset with missing values (features + label column)
    label_column_index : int
        Index of the label column in the dataset
    true_proportions : dict or array-like
        True class mixing proportions. If dict: {class_label: proportion}
        If array: proportions in order of sorted unique classes
    k : int
        Number of neighbors for KNN classification
    
    Returns:
    --------
    proportion_error : float
        L2 norm of difference between true and imputed class proportions
    execution_time : float
        Time taken for imputation
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.impute import KNNImputer
    import numpy as np
    from time import time
    
    start = time()
    data = np.array(data)
    
    # Separate features and labels
    features = np.delete(data, label_column_index, axis=1)
    labels = data[:, label_column_index]
    
    # Identify rows with missing labels
    missing_mask = np.isnan(labels)
    has_label_mask = ~missing_mask
    
    # If there are missing values in features, impute them first
    if np.isnan(features).any():
        feature_imputer = KNNImputer(n_neighbors=k)
        features = feature_imputer.fit_transform(features)
    
    # Train KNN classifier on labeled data
    if has_label_mask.sum() > 0:  # Check if we have any labeled data
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(features[has_label_mask], labels[has_label_mask])
        
        # Predict missing labels
        if missing_mask.sum() > 0:
            labels[missing_mask] = knn.predict(features[missing_mask])
    
    # Compute class proportions from imputed labels
    unique_classes, counts = np.unique(labels, return_counts=True)
    imputed_proportions = counts / len(labels)
    
    # Convert true_proportions to array format
    if isinstance(true_proportions, dict):
        true_prop_array = np.array([true_proportions.get(c, 0) for c in unique_classes])
    else:
        true_prop_array = np.array(true_proportions)
    
    # Ensure same length (in case of missing classes in imputed data)
    if len(true_prop_array) > len(imputed_proportions):
        # Pad imputed proportions with zeros for missing classes
        imputed_proportions = np.pad(imputed_proportions, 
                                     (0, len(true_prop_array) - len(imputed_proportions)))
    elif len(imputed_proportions) > len(true_prop_array):
        # Pad true proportions with zeros
        true_prop_array = np.pad(true_prop_array, 
                                 (0, len(imputed_proportions) - len(true_prop_array)))
    
    # Compute L2 norm of difference
    proportion_error = np.linalg.norm(true_prop_array - imputed_proportions)
    end = time()
    
    return proportion_error, end - start


def select_k_cv(data, label_col, k_values=None, n_folds=10):
    """
    Select optimal k for KNN classification using cross-validation.
    Only uses observations with non-missing labels for CV.
    
    Parameters:
    -----------
    data : array-like or DataFrame
        Complete dataset (features + label column)
    label_col : int or str
        Index or name of the label column
    k_values : list or None
        List of k values to test. If None, defaults to [1, 3, 5, 7, 9, 11, 15, 20]
    n_folds : int
        Number of folds for cross-validation (default: 10)
    
    Returns:
    --------
    best_k : int
        Optimal number of neighbors that minimizes misclassification error
    results : dict
        Dictionary with k values as keys and average misclassification error as values
    """
    if k_values is None:
        k_values = [1, 3, 5, 7, 9, 11, 15, 20]
    
    # Convert to numpy array if needed
    if hasattr(data, 'values'):
        data_array = data.values
        if isinstance(label_col, str):
            label_col = data.columns.get_loc(label_col)
    else:
        data_array = np.array(data)
    
    # Get observations with non-missing labels
    mask_complete = ~np.isnan(data_array[:, label_col])
    complete_data = data_array[mask_complete]
    
    # Separate features and labels
    X = np.delete(complete_data, label_col, axis=1)
    y = complete_data[:, label_col]
    
    # Store results for each k
    results = {}
    
    for k in k_values:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_errors = []
        
        for train_idx, val_idx in kf.split(X):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit KNN classifier on training data
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            knn_classifier.fit(X_train, y_train)
            
            # Predict on validation set
            y_pred = knn_classifier.predict(X_val)
            
            # Compute misclassification error
            misclass_error = np.mean(y_pred != y_val)
            fold_errors.append(misclass_error)
        
        # Average misclassification error across folds
        avg_error = np.mean(fold_errors)
        results[k] = avg_error
    
    # Select best k (minimum average misclassification error)
    best_k = min(results, key=results.get)
    
    return best_k

def rf_imputation_labels(data, label_column_index, true_proportions, n_estimators=100, random_state=42):
    """
    Impute missing categorical labels using Random Forest Classifier based on features.
    
    Parameters:
    -----------
    data : array-like
        Full dataset with missing values (features + label column)
    label_column_index : int
        Index of the label column in the dataset
    true_proportions : dict or array-like
        True class mixing proportions. If dict: {class_label: proportion}
        If array: proportions in order of sorted unique classes
    n_estimators : int
        Number of trees in the random forest (default: 100)
    random_state : int
        Random state for reproducibility (default: 42)
    
    Returns:
    --------
    proportion_error : float
        L2 norm of difference between true and imputed class proportions
    execution_time : float
        Time taken for imputation
    """
    
    start = time()
    data = np.array(data)
    
    # Separate features and labels
    features = np.delete(data, label_column_index, axis=1)
    labels = data[:, label_column_index]
    
    # Identify rows with missing labels
    missing_mask = np.isnan(labels)
    has_label_mask = ~missing_mask
    
    # If there are missing values in features, impute them first (simple mean/median)
    if np.isnan(features).any():
        feature_imputer = SimpleImputer(strategy='mean')
        features = feature_imputer.fit_transform(features)
    
    # Train Random Forest classifier on labeled data
    if has_label_mask.sum() > 0 and missing_mask.sum() > 0:
        rf = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        rf.fit(features[has_label_mask], labels[has_label_mask])
        
        # Predict missing labels
        labels[missing_mask] = rf.predict(features[missing_mask])
    
    # Compute class proportions from imputed labels
    unique_classes, counts = np.unique(labels[~np.isnan(labels)], return_counts=True)
    imputed_proportions = counts / len(labels[~np.isnan(labels)])
    
    # Convert true_proportions to array format
    if isinstance(true_proportions, dict):
        true_prop_array = np.array([true_proportions.get(c, 0) for c in unique_classes])
    else:
        true_prop_array = np.array(true_proportions)
    
    # Ensure same length (in case of missing classes in imputed data)
    if len(true_prop_array) > len(imputed_proportions):
        imputed_proportions = np.pad(imputed_proportions, 
                                     (0, len(true_prop_array) - len(imputed_proportions)))
    elif len(imputed_proportions) > len(true_prop_array):
        true_prop_array = np.pad(true_prop_array, 
                                 (0, len(imputed_proportions) - len(true_prop_array)))
    
    # Compute L2 norm of difference
    proportion_error = np.linalg.norm(true_prop_array - imputed_proportions)
    end = time()
    
    return proportion_error, end - start
    

if __name__ == "__main__":
    # True class proportions (example for 3 classes)
    true_proportions = [0.3, 0.4, 0.3]
    
    # Load dataset with missing labels
    df = pd.read_csv("tests\\MAR_missing_20pct.csv", skiprows=0)
    data_array = df.to_numpy()
    
    # Specify which column contains the labels (e.g., last column)
    label_column_index = -1
    
    # Optional: Select best k using cross-validation for KNN
    # Note: You would need to adapt select_k_cv for label imputation as well
    # best_k = select_k_cv_labels(data_array, label_column_index, true_proportions, k_values=None, n_folds=10)
    # For now, use a default k value
    best_k = select_k_cv(data_array, label_column_index, k_values=None, n_folds=10)
    print(f"Using k={best_k} for KNN imputation")
    
    # Perform different imputation methods
    mode_err, mode_time = mode_imputation_labels(data_array, label_column_index, true_proportions)
    knn_err, knn_time = knn_imputation_labels(data_array, label_column_index, true_proportions, k=best_k)
    mice_err, mice_time = rf_imputation_labels(data_array, label_column_index, true_proportions, iterations=5)
    
    # Print results
    print("\n=== Label Imputation Results ===")
    print(f"Mode Imputation - Proportion Error: {mode_err:.6f}")
    print(f"KNN Imputation (k={best_k}) - Proportion Error: {knn_err:.6f}")
    print(f"MICE Imputation - Proportion Error: {mice_err:.6f}")
    
    # # Optional: Display imputed class distributions
    # print("\n=== Imputed Class Distributions ===")
    # for method_name in [("Mode"), ("KNN"), ("MICE")]:
    #     labels = imputed_data[:, label_column_index]
    #     unique, counts = np.unique(labels, return_counts=True)
    #     proportions = counts / len(labels)
    #     print(f"\n{method_name}:")
    #     for cls, prop in zip(unique, proportions):
    #         print(f"  Class {cls}: {prop:.4f}")
    
    # print("\n=== True Proportions ===")
    # if isinstance(true_proportions, dict):
    #     for cls, prop in true_proportions.items():
    #         print(f"  Class {cls}: {prop:.4f}")
    # else:
    #     for i, prop in enumerate(true_proportions):
    #         print(f"  Class {i}: {prop:.4f}")