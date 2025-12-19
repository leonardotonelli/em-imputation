import numpy as np
import pandas as pd
import os
from pathlib import Path
from time import time
from tqdm import tqdm

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
    means_to_test,
    cov_matrices_to_test,
    weights_to_test,
    n_samples_to_test,
    percentages_to_test,
    data_path=None,
    max_iter=200,
    tol=1e-5,
    random_state=42
):
    """
    Comprehensive simulation study for EM algorithm on GMM with missing class labels.
    """
    
    Path(result_path).mkdir(parents=True, exist_ok=True)
    if data_path is not None:
        Path(data_path).mkdir(parents=True, exist_ok=True)
    
    results_list = []
    mechanisms = ["MCAR", "MAR", "MNAR"]
    simulation_id = 0

    # Calculate total number of simulations
    total_simulations = (len(means_to_test)* len(weights_to_test)*len(cov_matrices_to_test) * len(n_samples_to_test) * 
                        len(percentages_to_test) * len(mechanisms))
    
    print("=" * 80)
    print("STARTING GMM SIMULATION STUDY")
    print(f"Total simulations to run: {total_simulations}")
    print("=" * 80)
    
    # Main progress bar for overall simulation
    with tqdm(total=total_simulations, desc="Simulations", unit="sim") as pbar:
        
        for mean_idx, mean in enumerate(means_to_test):
            for cov_idx, cov in enumerate(cov_matrices_to_test):
                for weight_idx, weight in enumerate(weights_to_test):
                    n_components = len(weight)
                    means = mean
                    cov_matrices = cov
                    weights = weight
                    config_idx = int(f"{mean_idx}{cov_idx}{weight_idx}")

                    for n_samples in n_samples_to_test:
                        
                        # Generate complete GMM data (silent)
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
                            complete_filepath = os.path.join(data_path, f"datasets_complete\\{n_samples}_samples")
                            Path(complete_filepath).mkdir(parents=True, exist_ok=True)
                            df_complete.to_csv(
                                os.path.join(complete_filepath, f"mean{mean_idx}_cov{cov_idx}.csv"), 
                                index=False
                            )
                        # Test each mechanism
                        for mechanism in mechanisms:
                            
                            percentages = percentages_to_test
                            
                            for miss_pct in percentages:
                                
                                simulation_id += 1
                                
                                # Update progress bar with current simulation info
                                pbar.set_postfix({
                                    'Config': config_idx,
                                    'K': n_components,
                                    'N': n_samples,
                                    'Miss%': int(miss_pct*100),
                                    'Mech': mechanism
                                })
                                
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
                                    missing_filepath = os.path.join(data_path, f"datasets_missingness\\{mechanism}\\{int(miss_pct*100)}%_missing\\{n_samples}_samples")
                                    Path(missing_filepath).mkdir(parents=True, exist_ok=True)
                                    df_missing.to_csv(
                                        os.path.join(missing_filepath, f"mean{mean_idx}_cov{cov_idx}.csv"), 
                                        index=False
                                    )
                                
                                # Prepare data for EM (features only, no class column)
                                feature_cols = [col for col in df_missing.columns if col != 'class']
                                data_obs = df_missing[feature_cols].values
                                
                                # Calculate actual missingness in class variable
                                class_col = df_missing['class'].values
                                actual_miss_pct = np.sum(np.isnan(class_col)) / len(class_col)
                                
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

                                # Store results
                                result = {
                                    'simulation_id': simulation_id,
                                    'config_idx': config_idx,
                                    'mean_idx': mean_idx,
                                    'cov_idx': cov_idx,
                                    'weight_idx': weight_idx,
                                    'n_components': n_components,
                                    'n_samples': n_samples,
                                    'missingness_pct': miss_pct,
                                    'actual_missingness_pct': actual_miss_pct,
                                    'mechanism': mechanism,
                                    'time_per_iteration': em_time/num_iterations, #EM
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
                                
                                # Update progress bar
                                pbar.update(1)
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving results...")
    results_df = pd.DataFrame(results_list)
    results_filename = os.path.join(result_path, "simulation_results_gmm.csv")
    results_df.to_csv(results_filename, index=False)
    
    print("=" * 80)
    print(f"GMM SIMULATION STUDY COMPLETE")
    print(f"Total simulations: {simulation_id}")
    print(f"Results saved to: {results_filename}")
    print("=" * 80)
    
    # Summary statistics
    print("\n=== EM SUMMARY STATISTICS ===")
    print(f"Average Pi error: {results_df['pi_error'].mean():.4f}")
    
    print("\n=== EM Error by Mechanism ===")
    for mech in mechanisms:
        mech_df = results_df[results_df['mechanism'] == mech]
        if len(mech_df) > 0:
            print(f"{mech}: Pi={mech_df['pi_error'].mean():.4f}")
    
    return results_df


if __name__ == "__main__":
    
    DIM = 20
    MEANS_GMM = [
        [np.zeros(DIM), np.ones(DIM)*10, np.ones(DIM)*20],
    ]
    COVS_GMM = [
        [np.eye(DIM), np.eye(DIM), np.eye(DIM)],           # Setting 1
        [2*np.eye(DIM), 2*np.eye(DIM), 2*np.eye(DIM)],   # Setting 2
        [0.5*np.eye(DIM), 0.5*np.eye(DIM), 0.5*np.eye(DIM)],  # Setting 3
    ]
    WEIGHTS_GMM = [
        [0.33, 0.33, 0.34],
    ]   
    N_SAMPLES_GMM = [100, 200]
    PERCENTAGES_CLASS_MISSINGNESS = [0.1, 0.3, 0.5]

    # Run GMM simulation study (if not in results folder)
    results_gmm = simulation_study_gmm(
        result_path="tests",
        data_path="tests",
        means_to_test=MEANS_GMM,
        cov_matrices_to_test=COVS_GMM,
        weights_to_test=WEIGHTS_GMM,
        n_samples_to_test=N_SAMPLES_GMM,
        percentages_to_test=PERCENTAGES_CLASS_MISSINGNESS,
        max_iter=200,
        tol=1e-5,
        random_state=42
    )
    
    print("\nSimulation study completed successfully!")