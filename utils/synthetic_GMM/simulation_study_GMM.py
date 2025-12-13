import numpy as np
import pandas as pd
import os
from pathlib import Path
from time import time

try:
    from .EM_GMM import em_semi_supervised
    from .data_generation_GMM import generate_gmm_data, inject_class_missingness
    from .imputations import *
except ImportError:
    from EM_GMM import em_semi_supervised
    from data_generation_GMM import generate_gmm_data, inject_class_missingness
    from imputations import *

def simulation_study_gmm(
    result_path,
    gmm_configs,
    n_samples_to_test,
    percentages_to_test,
    data_path=None,
    max_iter=200,
    tol=1e-5,
    random_state=42
):
    """
    Comprehensive simulation study for EM algorithm on GMM with missing class labels.
    
    Parameters:
    -----------
    result_path : str
        Directory path to save results
    data_path : str
        Directory path to save datasets
    gmm_configs : list of dict
        List of GMM configurations, each dict contains:
        - 'n_components': int
        - 'means': list of arrays
        - 'cov_matrices': list of arrays
        - 'weights': list of floats
    n_samples_to_test : list of int
        List of sample sizes to test
    percentages_to_test : list of float
        List of missingness percentages for class variable (0 to 1)
    max_iter : int
        Maximum iterations for EM algorithm
    tol : float
        Convergence tolerance for EM algorithm
    random_state : int
        Random seed for reproducibility
    """
    
    Path(result_path).mkdir(parents=True, exist_ok=True)
    if data_path is not None:
        Path(data_path).mkdir(parents=True, exist_ok=True)
    
    results_list = []
    mechanisms = ["MCAR", "MAR", "MNAR"]
    simulation_id = 0
    
    print("=" * 80)
    print("STARTING GMM SIMULATION STUDY")
    print("=" * 80)
    
    for config_idx, config in enumerate(gmm_configs):
        n_components = config['n_components']
        means = config['means']
        cov_matrices = config['cov_matrices']
        weights = config['weights']
        
        for n_samples in n_samples_to_test:
            
            print(f"\n{'='*60}")
            print(f"Generating GMM data: Config={config_idx}, K={n_components}, N={n_samples}")
            print(f"{'='*60}")
            
            # Generate complete GMM data
            data, class_labels = generate_gmm_data(
                n_samples=n_samples,
                n_components=n_components,
                means=means,
                cov_matrices=cov_matrices,
                weights=weights,
                random_state=random_state + simulation_id
            )
            
            # Save complete dataset
            df_complete = data.copy()
            df_complete['class'] = class_labels
            if data_path is not None:
                complete_filename = f"complete_config{config_idx}_n{n_samples}.csv"
                df_complete.to_csv(os.path.join(data_path, complete_filename), index=False)
            
            # Test each mechanism
            for mechanism in mechanisms:
                
                percentages = percentages_to_test
                
                for miss_pct in percentages:
                    
                    simulation_id += 1
                    
                    miss_pct_str = f"{miss_pct*100:.0f}%"
                    print(f"\n[Simulation {simulation_id}] Config={config_idx}, "
                          f"N={n_samples}, Mechanism={mechanism}, Miss={miss_pct_str}")
                    
                    # Inject missingness in class variable
                    df_missing = inject_class_missingness(
                        data=data,
                        class_labels=class_labels,
                        mechanism=mechanism,
                        missingness_percentage=miss_pct,
                        random_state=random_state + simulation_id
                    )
                    
                    # Save missing dataset
                    if data_path is not None:   
                        missing_filename = (f"missing_{mechanism}_config{config_idx}_"
                                            f"n{n_samples}_miss{int(miss_pct*100)}.csv")
                        df_missing.to_csv(os.path.join(data_path, missing_filename), index=False)
                    
                    # Prepare data for EM (features only, no class column)
                    feature_cols = [col for col in df_missing.columns if col != 'class']
                    data_obs = df_missing[feature_cols].values
                    
                    # Calculate actual missingness in class variable
                    class_col = df_missing['class'].values
                    actual_miss_pct = np.sum(np.isnan(class_col)) / len(class_col)
                    print(f"  Actual class missingness: {actual_miss_pct*100:.2f}%")
                    
                    # Run EM algorithm
                    X, y = data_obs, df_missing['class'].values
                    start_time = time()
                    pi_est, mu_est, Sigma_est, num_iterations = em_semi_supervised(
                        X,
                        y,
                        n_components=n_components,
                        max_iter=max_iter,
                        tol=tol,
                    )
                    em_time = time() - start_time
                    
                    # Calculate errors (need to match components due to label switching)
                    pi_error = np.linalg.norm(np.array(weights) - np.array(pi_est))
                    
                    print(f"  Converged in {num_iterations} iterations ({em_time:.2f}s)")
                    print(f"  Pi Error: {pi_error:.4f}")

                    # Imputation methods for class labels
                    mode_prop_err = np.nan
                    knn_prop_err = np.nan
                    rf_prop_err = np.nan
                    best_k = None
                    data_array = df_missing[feature_cols + ['class']].to_numpy()

                    # Select k for KNN using available labeled observations
                    try:
                        best_k = select_k_cv(data_array, label_col=-1, k_values=None, n_folds=10)
                    except Exception:
                        best_k = 10

                    mode_prop_err, mode_time = mode_imputation_labels(data_array, label_column_index=-1, true_proportions=weights)
                    knn_prop_err, knn_time = knn_imputation_labels(data_array, label_column_index=-1, true_proportions=weights, k=best_k)
                    rf_prop_err, rf_time = rf_imputation_labels(data_array, label_column_index=-1, true_proportions=weights)

                    print(f"  Mode prop error: {mode_prop_err:.4f}, KNN prop error (k={best_k}): {knn_prop_err:.4f}, RF prop error: {rf_prop_err if not np.isnan(rf_prop_err) else 'NA'}")

                    # Store results
                    result = {
                        'simulation_id': simulation_id,
                        'config_idx': config_idx,
                        'n_components': n_components,
                        'n_samples': n_samples,
                        'missingness_pct': miss_pct,
                        'actual_missingness_pct': actual_miss_pct,
                        'mechanism': mechanism,
                        'num_iterations': num_iterations,
                        'convergence_time': em_time,
                        'pi_error': pi_error,
                        'true_pi': str(weights),
                        'estimated_pi': str(pi_est.tolist()),
                        'true_means': str([m.tolist() for m in means]),
                        'estimated_means': str([m.tolist() for m in mu_est]),
                        'mode_imputation_prop_error': mode_prop_err,
                        'knn_imputation_prop_error': knn_prop_err,
                        'knn_imputation_k': int(best_k) if best_k is not None else None,
                        'rf_imputation_prop_error': rf_prop_err,
                        'mode_imputation_time': mode_time,
                        'knn_imputation_time': knn_time,
                        'rf_imputation_time': rf_time
                    }

                    results_list.append(result)
    
    # Save results
    results_df = pd.DataFrame(results_list)
    results_filename = os.path.join(result_path, "simulation_results_gmm.csv")
    results_df.to_csv(results_filename, index=False)
    
    print("\n" + "=" * 80)
    print(f"GMM SIMULATION STUDY COMPLETE")
    print(f"Total simulations: {simulation_id}")
    print(f"Results saved to: {results_filename}")
    print("=" * 80)
    
    # Summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"\nAverage iterations: {results_df['num_iterations'].mean():.1f}")
    print(f"Average Pi error: {results_df['pi_error'].mean():.4f}")
    
    print("\n=== Error by Mechanism ===")
    for mech in mechanisms:
        mech_df = results_df[results_df['mechanism'] == mech]
        if len(mech_df) > 0:
            print(f"{mech}: Pi={mech_df['pi_error'].mean():.4f}")
    
    return results_df


if __name__ == "__main__":
    
    # Define GMM configurations to test
    gmm_configs = [
        {
            'n_components': 3,
            'means': [
                np.array([0, 0]),
                np.array([5, 5]),
                np.array([0, 5])
            ],
            'cov_matrices': [
                np.array([[1.0, 0.3], [0.3, 1.0]]),
                np.array([[1.5, -0.5], [-0.5, 1.5]]),
                np.array([[1.0, 0.0], [0.0, 1.0]])
            ],
            'weights': [0.3, 0.4, 0.3]
        },
        # {
        #     'n_components': 2,
        #     'means': [
        #         np.array([0, 0]),
        #         np.array([4, 4])
        #     ],
        #     'cov_matrices': [
        #         np.array([[1.0, 0.0], [0.0, 1.0]]),
        #         np.array([[1.0, 0.0], [0.0, 1.0]])
        #     ],
        #     'weights': [0.5, 0.5]
        # }
    ]
    
    n_samples_to_test = [500, 1000]
    percentages_to_test = [0.1, 0.3, 0.5]
    
    # Run simulation study
    results = simulation_study_gmm(
        result_path="tests",
        data_path=None,
        gmm_configs=gmm_configs,
        n_samples_to_test=n_samples_to_test,
        percentages_to_test=percentages_to_test,
        max_iter=200,
        tol=1e-5,
        random_state=42
    )
    
    print("\nSimulation study completed successfully!")