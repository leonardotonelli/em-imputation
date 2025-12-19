import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from utils.synthetic_GMM.EM_GMM import em_semi_supervised, e_step_semi_supervised
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import pingouin as pg
from scipy.spatial.distance import mahalanobis

def test_gmm_normality_assumptions(X, labels, alpha=0.05):
    """
    Performs Henze-Zirkler MVN test and generates Q-Q plots for each GMM cluster.
    """
    unique_clusters = np.unique(labels)
    mvn_results = []

    print("--- Multivariate Normality Diagnostic ---")
    
    for cluster_id in unique_clusters:
        # Extract data for this specific cluster
        cluster_data = X[labels == cluster_id]
        n_samples, n_dims = cluster_data.shape
        
        print(f"\nAnalyzing Cluster {cluster_id} (n={n_samples})")
        
        # 1. Statistical Test: Henze-Zirkler
        # Note: HZ test requires at least more samples than dimensions
        if n_samples > n_dims + 1:
            hz_result = pg.multivariate_normality(cluster_data, alpha=alpha)
            is_normal = hz_result.normal
            p_val = hz_result.pval
        else:
            is_normal = False
            p_val = np.nan
            print(f"Warning: Cluster {cluster_id} has too few samples for HZ test.")

        mvn_results.append({
            'Cluster': cluster_id,
            'Samples': n_samples,
            'HZ_P-Value': p_val,
            'Is_Normal': is_normal
        })

        # 2. Visual Check: Mahalanobis Distance Q-Q Plot
        # Calculate Mahalanobis Distance for each point in the cluster
        mu = np.mean(cluster_data, axis=0)
        try:
            cov = np.cov(cluster_data, rowvar=False)
            inv_cov = np.linalg.inv(cov)
            
            # Compute squared Mahalanobis distances
            d2 = []
            for i in range(n_samples):
                diff = cluster_data[i] - mu
                d2.append(diff.dot(inv_cov).dot(diff))
            
            d2 = np.sort(d2)
            # Theoretical Quantiles for Chi-Square distribution (df = n_dims)
            theoretical_quantiles = stats.chi2.ppf(np.linspace(0.01, 0.99, n_samples), df=n_dims)

            # Plotting
            plt.figure(figsize=(6, 5))
            plt.scatter(theoretical_quantiles, d2, alpha=0.6, color='teal', label='Data Points')
            
            # Add 45-degree reference line
            max_val = max(max(d2), max(theoretical_quantiles))
            plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Theoretical MVN')
            
            plt.title(f"Cluster {cluster_id}: Chi-Square Q-Q Plot\n(HZ p-val: {p_val:.4f})")
            plt.xlabel("Theoretical Chi-Square Quantiles")
            plt.ylabel("Observed Mahalanobis Distances ($D^2$)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        except np.linalg.LinAlgError:
            print(f"Error: Covariance matrix for Cluster {cluster_id} is singular.")

    return pd.DataFrame(mvn_results)


def _mask_labels(y, missing_pct, random_state=None):
    rng = np.random.RandomState(random_state)
    maskable = ~np.isnan(y)
    idx = np.where(maskable)[0]
    n_mask = int(np.round(len(idx) * missing_pct))
    masked_idx = rng.choice(idx, size=n_mask, replace=False)
    y_masked = y.copy().astype(float)
    y_masked[masked_idx] = np.nan
    return y_masked, masked_idx


def _mode_impute(labels_masked):
    """
    Returns: (imputed_labels, pi_estimated)
    """
    labels = labels_masked.copy()
    mask = np.isnan(labels)
    if mask.sum() == 0:
        pi_est = np.mean(labels == 1)
        return labels, pi_est
    
    vals, counts = np.unique(labels[~mask], return_counts=True)
    if len(vals) == 0:
        labels[mask] = 0
        return labels, 0.0
    
    mode = vals[np.argmax(counts)]
    labels[mask] = mode
    
    # Pi estimated: proportion of class 1 in the observed (non-missing) data
    pi_est = np.sum(labels[~mask] == 1) / (~mask).sum()
    
    return labels, pi_est


def _knn_impute(X, labels_masked, k=5):
    """
    Returns: (imputed_labels, pi_estimated)
    """
    labels = labels_masked.copy()
    mask = np.isnan(labels)
    if mask.sum() == 0:
        pi_est = np.mean(labels == 1)
        return labels, pi_est
    
    if (~mask).sum() == 0:
        return _mode_impute(labels)
    
    knn = KNeighborsClassifier(n_neighbors=min(k, (~mask).sum()))
    knn.fit(X[~mask], labels[~mask])
    labels[mask] = knn.predict(X[mask])
    
    # Pi estimated: use the predicted probabilities from KNN
    unique_classes, counts = np.unique(labels, return_counts=True)
    imputed_proportions = counts / len(labels)
    
    return labels, imputed_proportions[1]


def _rf_impute(X, labels_masked, n_estimators=100, random_state=42):
    """
    Returns: (imputed_labels, pi_estimated)
    """
    labels = labels_masked.copy()
    mask = np.isnan(labels)
    if mask.sum() == 0:
        pi_est = np.mean(labels == 1)
        return labels, pi_est
    
    if (~mask).sum() == 0:
        return _mode_impute(labels)
    
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X[~mask], labels[~mask])
    labels[mask] = rf.predict(X[mask])
    
    unique_classes, counts = np.unique(labels[~np.isnan(labels)], return_counts=True)
    imputed_proportions = counts / len(labels[~np.isnan(labels)])

    return labels, imputed_proportions[1]


def _em_impute(X, labels_masked, n_components=2, max_iter=200, tol=1e-4, verbose=False):
    """
    Returns: (imputed_labels, pi_estimated)
    """
    y = labels_masked.astype(float)

    # Run EM
    pi_est, mu, Sigma, _ = em_semi_supervised(
        X, y,
        n_components=n_components,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose
    )

    # Responsibilities and hard assignments
    r = e_step_semi_supervised(X, y, pi_est, mu, Sigma)
    assign = np.argmax(r, axis=1)

    # Map clusters -> class labels via majority vote on labeled data
    mapping = {}
    labeled_idx = ~np.isnan(labels_masked)

    for c in range(n_components):
        members = np.where(assign == c)[0]
        labeled_members = members[labeled_idx[members]]

        if len(labeled_members) == 0:
            mapping[c] = 0
        else:
            vals, counts = np.unique(
                labels_masked[labeled_members].astype(int),
                return_counts=True
            )
            mapping[c] = vals[np.argmax(counts)]

    # Imputed labels
    pred = np.array([mapping[c] for c in assign])

    return pred, pi_est[1]



def evaluate_imputers(X_pca, y_experts, y_groundtruth, n_components=2):
    """
    Imputes missing values in y_experts using various methods (Mode, KNN, RF, EM),
    evaluates the results against y_groundtruth, and computes the difference 
    in class proportions (pi) using each algorithm's internal estimate."""
    
    # Ensure ground truth is integer type
    y_true = y_groundtruth.astype(int)

    # Calculate True Class Proportions (pi_true)
    n_total = len(y_true)
    pi_true_1 = np.sum(y_true == 1) / n_total
    
    results = []

    # Define imputation methods
    imputation_methods = {
        'Mode Imputer': lambda: _mode_impute(y_experts),
        'KNN Imputer (k=5)': lambda: _knn_impute(X_pca, y_experts, k=5),
        'Random Forest Imputer': lambda: _rf_impute(X_pca, y_experts, n_estimators=100, random_state=42),
        'EM-GMM Imputer': lambda: _em_impute(X_pca, y_experts, n_components=n_components, max_iter=200, tol=1e-4, verbose=False)
    }

    # Loop through imputation methods
    for name, impute_func in imputation_methods.items():
        # Perform imputation and get algorithm's pi estimate
        y_pred, pi_est_1 = impute_func()
        y_pred = y_pred.astype(int)
        
        # Calculate Metrics
        acc = accuracy_score(y_true, y_pred)
        
        # Calculate recall for both classes
        rec_class_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        rec_class_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        # Calculate precision and F1
        prec_class_1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1_class_1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        # Calculate confusion matrix for additional insight
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Calculate the difference in proportions for Class 1 (Est - True)
        # Using the algorithm's internal estimate, not empirical count
        pi_diff_class_1 = abs(pi_est_1 - pi_true_1)
        
        # Empirical pi from predictions (for comparison)
        pi_empirical = np.sum(y_pred == 1) / n_total
        
        results.append({
            'Imputer': name,
            'Accuracy': acc,
            'Recall_Class_0': rec_class_0,
            'Recall_Class_1': rec_class_1,
            'Precision_Class_1': prec_class_1,
            'F1_Class_1': f1_class_1,
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn,
            'Pi_Estimated': pi_est_1,
            'Pi_Empirical': pi_empirical,
            'Pi_Diff_Class_1': pi_diff_class_1
        })

    # Return the Results DataFrame
    results_df = pd.DataFrame(results).set_index('Imputer')
    return results_df


def visualize_evaluation_results(results_df, save_dir='evaluation_plots'):
    """
    Visualizes the imputation evaluation results (Accuracy, Recall, 
    and Pi Difference) using bar plots and saves them to a directory.

    Args:
        results_df (pd.DataFrame): DataFrame containing 'Accuracy', 'Recall', 
                                   and 'Pi_Diff_Class_1' columns.
        save_dir (str): Directory where the plots will be saved.
    """

    # Prepare the Save Directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    # Define the metrics and their plot parameters
    metrics = {
        'Accuracy': {
            'title': 'Imputation Method Accuracy vs. Ground Truth',
            'ylabel': 'Accuracy Score',
            'color': 'skyblue',
            'ylim': (0, 1.05)
        },
        'Recall_Class_1': {
            'title': 'Imputation Method Recall (Class 1) vs. Ground Truth',
            'ylabel': 'Recall Score (Class 1)',
            'color': 'lightcoral',
            'ylim': (0, 1.05)
        },
        'Precision_Class_1': {
            'title': 'Imputation Method Precision (Class 1) vs. Ground Truth',
            'ylabel': 'Precision Score (Class 1)',
            'color': 'lightyellow',
            'ylim': (0, 1.05)
        },
        'F1_Class_1': {
            'title': 'Imputation Method F1 Score (Class 1) vs. Ground Truth',
            'ylabel': 'F1 Score (Class 1)',
            'color': 'lightblue',
            'ylim': (0, 1.05)
        },
        'Pi_Diff_Class_1': {
            'title': 'Distributional Bias',
            'ylabel': 'p',
            'color': 'lightgreen',
            'ylim': (0, 1.05)
        }
    }

    # Create and Save Plots
    print("Generating and saving plots...")
    
    for col, params in metrics.items():
        if col not in results_df.columns:
            print(f"Skipping plot for '{col}' - column not found in results.")
            continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar plot
        x_pos = np.arange(len(results_df.index))
        bars = ax.bar(x_pos, results_df[col], color=params['color'], alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(results_df.iterrows()):
            value = row[col]
            ax.text(i, value + (0.02 if value >= 0 else -0.02), 
                   f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top',
                   fontsize=10, fontweight='bold')
        
        ax.set_title(params['title'], fontsize=16, fontweight='bold')
        ax.set_xlabel('Imputation Method', fontsize=12)
        ax.set_ylabel(params['ylabel'], fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(results_df.index, rotation=15, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Set y-axis limits if specified
        if params['ylim'] is not None:
            ax.set_ylim(params['ylim'])
        
        # For the Pi Difference plot, add a reference line at zero
        if col == 'Pi_Diff_Class_1':
            ax.axhline(0, color='red', linestyle='--', linewidth=2, 
                      label=r'Zero Bias ($\Delta\pi=0$)', alpha=0.8)
            ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        # Save the figure
        filename = f"{col}_bar_plot.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    

    # Create and Save Plots
    print("Generating and saving plots...")
    
    for col, params in metrics.items():
        if col not in results_df.columns:
            print(f"Skipping plot for '{col}' - column not found in results.")
            continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar plot
        x_pos = np.arange(len(results_df.index))
        bars = ax.bar(x_pos, results_df[col], color=params['color'], alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(results_df.iterrows()):
            value = row[col]
            ax.text(i, value + (0.02 if value >= 0 else -0.02), 
                   f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top',
                   fontsize=10, fontweight='bold')
        
        ax.set_title(params['title'], fontsize=16, fontweight='bold')
        ax.set_xlabel('Imputation Method', fontsize=12)
        ax.set_ylabel(params['ylabel'], fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(results_df.index, rotation=15, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Set y-axis limits if specified
        if params['ylim'] is not None:
            ax.set_ylim(params['ylim'])
        
        # For the Pi Difference plot, add a reference line at zero
        if col == 'Pi_Diff_Class_1':
            ax.axhline(0, color='red', linestyle='--', linewidth=2, 
                      label=r'Zero Bias ($\Delta\pi=0$)', alpha=0.8)
            ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        # Save the figure
        filename = f"{col}_bar_plot.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    print(f"\nAll plots saved to '{save_dir}/' directory.")