import numpy as np
import pandas as pd
import os
from pathlib import Path
import time
from tqdm import tqdm

# adjust import if running as main module
try:
    from EM import em_multivariate_gaussian
    from data_generation import generate_multivariate_gaussian, inject_missingness
    from imputations import mean_imputation, median_imputation, mode_imputation, knn_imputation, mice_imputation
except ImportError:
    from utils.synthetic_multivariate.EM import em_multivariate_gaussian
    from utils.synthetic_multivariate.data_generation import generate_multivariate_gaussian, inject_missingness
    from utils.synthetic_multivariate.imputations import mean_imputation, median_imputation, mode_imputation, knn_imputation, mice_imputation


def simulation_study_multivariate(
    result_path,
    means_to_test,
    cov_to_test,
    n_samples_to_test,
    percentages_to_test,
    data_path=None,
    max_iter=200,
    tol=1e-5,
    random_state=42
):
    """
    Comprehensive simulation study for EM algorithm with missing data.
    Uses generate_multivariate_gaussian() and inject_missingness() to create data,
    and em_multivariate_gaussian() to fit the model.
    """
    
    # Create output directory
    Path(result_path).mkdir(parents=True, exist_ok=True)
    
    # Results storage
    results_list = []
    
    # Mechanisms to test
    mechanisms = ["MCAR", "MAR", "MNAR"]
    
    # Counter for simulation runs
    simulation_id = 0
    
    # Calculate total number of simulations
    total_simulations = (len(means_to_test) * len(cov_to_test) * 
                        len(n_samples_to_test) * len(percentages_to_test) * 
                        len(mechanisms))
    
    print("=" * 80)
    print("STARTING SIMULATION STUDY")
    print(f"Total simulations to run: {total_simulations}")
    print("=" * 80)
    
    # Main progress bar for overall simulation
    with tqdm(total=total_simulations, desc="Simulations", unit="sim") as pbar:
        
        # Iterate through all combinations
        for mean_idx, means in enumerate(means_to_test):
            for cov_idx, cov_matrix in enumerate(cov_to_test):
                for i, n_samples in enumerate(n_samples_to_test):
                    
                    # Convert to appropriate types
                    means_array = np.array(means) if not isinstance(means, np.ndarray) else means
                    p = len(means_array)
                    column_names = [f'var_{i+1}' for i in range(p)]
                    
                    # Generate complete data (silent)
                    
                    df_complete = generate_multivariate_gaussian(
                        n_samples=n_samples,
                        means=means_array.tolist(),
                        cov_matrix=cov_matrix,
                        column_names=column_names,
                        random_state=random_state + simulation_id
                    )
                    
                    # Save complete dataset
                    if data_path is not None:
                        complete_filepath = os.path.join(data_path, f"datasets_complete\\{n_samples}_samples")
                        Path(complete_filepath).mkdir(parents=True, exist_ok=True)
                        df_complete.to_csv(
                            os.path.join(complete_filepath, f"mean{mean_idx}_cov{cov_idx}.csv"), 
                            index=False
                        )
                    
                    # Iterate through missingness percentages
                    for miss_pct in percentages_to_test:
                        
                        k = 10  # KNN parameter
                        
                        # Iterate through mechanisms
                        for mechanism in mechanisms:
                            
                            simulation_id += 1
                            
                            # Update progress bar with current simulation info
                            pbar.set_postfix({
                                'Mean': mean_idx,
                                'Cov': cov_idx,
                                'N': n_samples,
                                'Miss%': int(miss_pct*100),
                                'Mech': mechanism
                            })
                            
                            # Inject missingness
                            df_missing_list = inject_missingness(
                                data=df_complete,
                                missingness_percentages=[miss_pct],
                                target_column_percentage=0.4,
                                mechanism=mechanism,
                                random_state=random_state + simulation_id
                            )
                            
                            df_missing = df_missing_list[0]
                            
                            # Save missing dataset
                            if data_path is not None:
                                missing_filepath = os.path.join(data_path, f"datasets_missingness\\{mechanism}\\{int(miss_pct*100)}%_missing\\{n_samples}_samples")
                                Path(missing_filepath).mkdir(parents=True, exist_ok=True)
                                df_missing.to_csv(
                                    os.path.join(missing_filepath, f"mean{mean_idx}_cov{cov_idx}.csv"), 
                                    index=False
                                )
                            
                            # Convert to numpy array
                            data_obs = df_missing.values
                            
                            # Calculate actual missingness
                            actual_miss_pct = np.sum(np.isnan(data_obs)) / data_obs.size
                            
                            # Run EM algorithm
                            start_time = time.time()
                            mu_estimated, Sigma_estimated, num_iterations = \
                                em_multivariate_gaussian(
                                    data_obs, max_iter=max_iter, tol=tol
                                )
                            em_time = time.time() - start_time
                            
                            # Calculate errors
                            mu_error = np.linalg.norm(mu_estimated - means_array)
                            Sigma_error = (np.linalg.norm(Sigma_estimated - cov_matrix) 
                                         / len(cov_matrix)**2)
                            
                            # Run imputation methods
                            mean_err, mean_cov_err, mean_time = mean_imputation(
                                data_obs, means_array, cov_matrix
                            )
                            
                            # Median imputation
                            median_err, median_cov_err, median_time = median_imputation(
                                data_obs, means_array, cov_matrix
                            )
                            
                            # Mode imputation
                            mode_err, mode_cov_err, mode_time = mode_imputation(
                                data_obs, means_array, cov_matrix
                            )
                            
                            # KNN imputation
                            knn_err, knn_cov_err, knn_time = knn_imputation(
                                data_obs, means_array, cov_matrix, k=k
                            )
                            
                            # MICE imputation
                            mice_err, mice_cov_err, mice_time = mice_imputation(
                                data_obs, means_array, cov_matrix, iterations=5
                            )
                            
                            # Store results
                            result = {
                                'simulation_id': simulation_id,
                                'mean_idx': mean_idx,
                                'cov_idx': cov_idx,
                                'n_samples': n_samples,
                                'missingness_pct': miss_pct,
                                'actual_missingness_pct': actual_miss_pct,
                                'mechanism': mechanism,
                                'time_per_iteration': em_time/num_iterations, #EM
                                'convergence_time': em_time, #EM
                                'mu_error': mu_error, #EM
                                'sigma_error': Sigma_error, #EM
                                'true_mu': str(means_array.tolist()),
                                'estimated_mu': str(mu_estimated.tolist()), #EM
                                'true_sigma': str(cov_matrix.tolist()),
                                'estimated_sigma': str(Sigma_estimated.tolist()),
                                'mean_imputation_error': mean_err,
                                'mean_imputation_cov_error': mean_cov_err,  
                                'median_imputation_error': median_err,
                                'median_imputation_cov_error': median_cov_err,
                                'mode_imputation_error': mode_err,
                                'mode_imputation_cov_error': mode_cov_err,
                                'knn_imputation_error': knn_err,
                                'knn_imputation_cov_error': knn_cov_err,
                                'mice_imputation_error': mice_err,
                                'mice_imputation_cov_error': mice_cov_err,
                                'mean_imputation_time': mean_time,
                                'median_imputation_time': median_time,
                                'mode_imputation_time': mode_time,
                                'knn_imputation_time': knn_time,
                                'mice_imputation_time': mice_time
                            }
                            
                            results_list.append(result)
                            
                            # Update progress bar
                            pbar.update(1)
    
    # Save results to CSV
    print("\n" + "=" * 80)
    print("Saving results...")
    results_df = pd.DataFrame(results_list)
    results_filename = os.path.join(result_path, "simulation_results.csv")
    results_df.to_csv(results_filename, index=False)
    
    print("=" * 80)
    print(f"SIMULATION STUDY COMPLETE")
    print(f"Total simulations: {simulation_id}")
    print(f"Results saved to: {results_filename}")
    print("=" * 80)
    
    # Print EM summary statistics
    print("\n=== EM SUMMARY STATISTICS ===")
    print(f"Average Mu error: {results_df['mu_error'].mean():.4f}")
    print(f"Average Sigma error: {results_df['sigma_error'].mean():.4f}")
    
    print("\n=== EM Error by Mechanism ===")
    for mech in mechanisms:
        mech_df = results_df[results_df['mechanism'] == mech]
        print(f"{mech}: Mu Error = {mech_df['mu_error'].mean():.4f}, "
              f"Sigma Error = {mech_df['sigma_error'].mean():.4f}")
    
    return results_df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Define test parameters
    means_to_test = [
        [50, 100, 25, 75],
        # [10, 20, 30, 40],
    ]
    
    cov_to_test = [
        np.array([
            [10,  5,  2,  3],
            [ 5, 20,  4,  6],
            [ 2,  4, 15,  1],
            [ 3,  6,  1, 12]
        ]),
        # np.array([
        #     [5, 1, 0, 0],
        #     [1, 5, 1, 0],
        #     [0, 1, 5, 1],
        #     [0, 0, 1, 5]
        # ]),
    ]
    
    n_samples_to_test = [500, 1000]
    percentages_to_test = [0.1, 0.2, 0.3]
    
    # Run simulation study
    results = simulation_study_multivariate(
        result_path="tests",
        data_path="tests",
        means_to_test=means_to_test,
        cov_to_test=cov_to_test,
        n_samples_to_test=n_samples_to_test,
        percentages_to_test=percentages_to_test,
        max_iter=200,
        tol=1e-5,
        random_state=42
    )
    print("Results saved.")