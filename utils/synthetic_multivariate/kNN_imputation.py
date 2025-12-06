import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import os

def impute_knn_and_estimate_params(X_missing, n_neighbors=5):
    knn = KNNImputer(n_neighbors=n_neighbors, metric='nan_euclidean')
    X_imputed = knn.fit_transform(X_missing)
    
    mu_hat = np.mean(X_imputed, axis=0)
    sigma_hat = np.cov(X_imputed, rowvar=False)
    
    return mu_hat, sigma_hat, X_imputed

def get_matrix_diff_norm(A, B):
    return np.linalg.norm(A - B, ord='fro') / A.shape[0]

# ------ EXAMPLE USAGE -------

def main():
    print("### kNN Algorithm Test (Loading from CSV) ###")
    
    true_mu = np.array([50, 100, 25, 75])
    true_sigma = np.array([
        [10,  5,  2,  3],
        [ 5, 20,  4,  6],
        [ 2,  4, 15,  1],
        [ 3,  6,  1, 12]
    ])
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "tests", "MAR_missing_30pct.csv")
    
    print(f"Attempting to load data from: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    try:
        X_missing = pd.read_csv(csv_path).values
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    n_samples, n_features = X_missing.shape
    print(f"Data Loaded: {n_samples} samples, {n_features} features")
    print(f"Missing Values: {np.isnan(X_missing).sum()}")

    est_mu, est_sigma, _ = impute_knn_and_estimate_params(X_missing, n_neighbors=5)

    np.set_printoptions(precision=3, suppress=True)

    print("\n" + "="*40)
    print("RESULTS COMPARISON")
    print("="*40)

    print("\n1. Mean Vector (Mu):")
    print(f"True:      {true_mu}")
    print(f"Estimated: {est_mu}")
    
    if len(true_mu) == len(est_mu):
        print(f"-> L2 Error: {np.linalg.norm(true_mu - est_mu):.4f}")
    else:
        print(f"Warning: Dimension mismatch")

    print("\n2. Covariance Matrix (Sigma):")
    print("True Matrix:")
    print(true_sigma)
    print("\nEstimated Matrix (kNN):")
    print(est_sigma)
    
    if true_sigma.shape == est_sigma.shape:
        cov_error = get_matrix_diff_norm(true_sigma, est_sigma)
        print(f"\n-> Matrix Error (Normalized Frobenius): {cov_error:.4f}")

        if np.trace(est_sigma) < np.trace(true_sigma):
            print("\nNote: Estimated trace is lower than true trace (smoothing effect).")
    else:
        print(f"Warning: Dimension mismatch")

if __name__ == "__main__":
    main()