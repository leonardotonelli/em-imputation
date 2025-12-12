import numpy as np
import pandas as pd
import os
from pathlib import Path
import time
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
    
    Parameters:
    -----------
    path : str
        Directory path to save datasets and results
    means_to_test : list of arrays/lists
        List of mean vectors to test
    cov_to_test : list of arrays
        List of covariance matrices to test
    n_samples_to_test : list of int
        List of sample sizes to test
    percentages_to_test : list of float
        List of missingness percentages to test (0 to 1)
    max_iter : int
        Maximum iterations for EM algorithm
    tol : float
        Convergence tolerance for EM algorithm
    random_state : int
        Random seed for reproducibility
    """
    
    # Create output directory
    Path(result_path).mkdir(parents=True, exist_ok=True)
    # Path(data_path).mkdir(parents=True, exist_ok=True)
    
    # Results storage
    results_list = []
    
    # Mechanisms to test
    mechanisms = ["MCAR", "MAR", "MNAR"]
    
    # Counter for simulation runs
    simulation_id = 0
    
    print("=" * 80)
    print("STARTING SIMULATION STUDY")
    print("=" * 80)
    
    # Iterate through all combinations
    for mean_idx, means in enumerate(means_to_test):
        for cov_idx, cov_matrix in enumerate(cov_to_test):
            for i, n_samples in enumerate(n_samples_to_test):
                
                # Convert to appropriate types
                means_array = np.array(means) if not isinstance(means, np.ndarray) else means
                p = len(means_array)
                column_names = [f'var_{i+1}' for i in range(p)]
                
                print(f"\n{'='*60}")
                print(f"Generating data: Mean={mean_idx}, Cov={cov_idx}, N={n_samples}")
                print(f"{'='*60}")
                
                # Use generate_multivariate_gaussian function
                df_complete = generate_multivariate_gaussian(
                    n_samples=n_samples,
                    means=means_array.tolist(),
                    cov_matrix=cov_matrix,
                    column_names=column_names,
                    random_state=random_state + simulation_id
                )
                
                # Save complete dataset
                if data_path is not None:
                    complete_filename = (f"complete_mean{mean_idx}_cov{cov_idx}_"
                                    f"n{n_samples}.csv")
                    df_complete.to_csv(
                        os.path.join(data_path, complete_filename), 
                        index=False
                    )
                

                # Use inject_missingness function for all mechanisms
                for miss_pct in percentages_to_test:
                    
                    print(f"\nInjecting {miss_pct*100:.0f}% missingness...")

                    k = 10 # to see
                    
                    # Use inject_missingness to generate missing data for all mechanisms at once
                    for mechanism in mechanisms:
                        
                        simulation_id += 1
                        
                        print(f"\n[Simulation {simulation_id}] Mean={mean_idx}, "
                              f"Cov={cov_idx}, N={n_samples}, Miss={miss_pct*100:.0f}%, "
                              f"Mech={mechanism}")
                        
                        # Use inject_missingness function
                        df_missing_list = inject_missingness(
                            data=df_complete,
                            missingness_percentages=[miss_pct],
                            target_column_percentage=0.5,
                            mechanism=mechanism,
                            random_state=random_state + simulation_id
                        )
                        
                        df_missing = df_missing_list[0]
                        
                        # Save missing dataset
                        if data_path is not None:
                            missing_filename = (f"missing_{mechanism}_mean{mean_idx}_"
                                          f"cov{cov_idx}_n{n_samples}_"
                                          f"miss{int(miss_pct*100)}.csv")
                            df_missing.to_csv(
                                os.path.join(data_path, missing_filename), 
                                index=False
                            )
                            
                        # Convert to numpy array
                        data_obs = df_missing.values
                        
                        # Calculate actual missingness
                        actual_miss_pct = np.sum(np.isnan(data_obs)) / data_obs.size
                        
                        print(f"  Actual missingness: {actual_miss_pct*100:.2f}%")
                        
                        # Run EM algorithm using em_multivariate_gaussian function
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
                        
                        print(f"  Converged in {num_iterations} iterations "
                              f"({em_time:.2f}s)")
                        print(f"  Mu Error: {mu_error:.4f}, "
                              f"Sigma Error: {Sigma_error:.4f}")
                        
                        # mean imputation error
                        mean_err, mean_cov_err, mean_time = mean_imputation(data_obs, means_array, cov_matrix)

                        # median imputation error
                        median_err, median_cov_err, median_time = median_imputation(data_obs, means_array, cov_matrix)

                        # mode imputation error
                        mode_err, mode_cov_err, mode_time = mode_imputation(data_obs, means_array, cov_matrix)

                        # knn imputation error
                        knn_err, knn_cov_err, knn_time = knn_imputation(data_obs, means_array, cov_matrix, k=k)

                        # mice imputation error
                        mice_err, mice_cov_err, mice_time = mice_imputation(data_obs, means_array, cov_matrix, iterations=5)  

                        # Store results
                        result = {
                            'simulation_id': simulation_id,
                            'mean_idx': mean_idx,
                            'cov_idx': cov_idx,
                            'n_samples': n_samples,
                            'missingness_pct': miss_pct,
                            'actual_missingness_pct': actual_miss_pct,
                            'mechanism': mechanism,
                            'num_iterations': num_iterations,
                            'convergence_time': em_time,
                            'mu_error': mu_error,
                            'sigma_error': Sigma_error,
                            'true_mu': str(means_array.tolist()),
                            'estimated_mu': str(mu_estimated.tolist()),
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
    
    # Save results to CSV
    results_df = pd.DataFrame(results_list)
    results_filename = os.path.join(result_path, "simulation_results.csv")
    results_df.to_csv(results_filename, index=False)
    
    print("\n" + "=" * 80)
    print(f"SIMULATION STUDY COMPLETE")
    print(f"Total simulations: {simulation_id}")
    print(f"Results saved to: {results_filename}")
    print("=" * 80)
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"\nAverage iterations: {results_df['num_iterations'].mean():.1f}")
    print(f"Average Mu error: {results_df['mu_error'].mean():.4f}")
    print(f"Average Sigma error: {results_df['sigma_error'].mean():.4f}")
    
    print("\n=== Error by Mechanism ===")
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
        [10, 20, 30, 40],
    ]
    
    cov_to_test = [
        np.array([
            [10,  5,  2,  3],
            [ 5, 20,  4,  6],
            [ 2,  4, 15,  1],
            [ 3,  6,  1, 12]
        ]),
        np.array([
            [5, 1, 0, 0],
            [1, 5, 1, 0],
            [0, 1, 5, 1],
            [0, 0, 1, 5]
        ]),
    ]
    
    n_samples_to_test = [500, 1000]
    percentages_to_test = [0.1, 0.2, 0.3]
    
    # Run simulation study
    results = simulation_study_multivariate(
        result_path="tests",
        data_path = None,
        means_to_test=means_to_test,
        cov_to_test=cov_to_test,
        n_samples_to_test=n_samples_to_test,
        percentages_to_test=percentages_to_test,
        max_iter=200,
        tol=1e-5,
        random_state=42
    )
    print("Results saved.")