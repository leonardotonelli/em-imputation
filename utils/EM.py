import numpy as np
import pandas as pd

# Support function

def initialize_parameters(data_obs): # see how to initialize better #TODO
    """
    Initialize mean (mu_hat) and covariance (Sigma_hat) using data statistics,
    ignoring NaN values.
    """
    mu_hat = np.nanmean(data_obs, axis=0)
    
    # Simple imputation: fill NaN with column mean for initial covariance estimate
    data_imputed = data_obs.copy()
    for j in range(data_obs.shape[1]):
        col_mean = mu_hat[j]
        data_imputed[np.isnan(data_imputed[:, j]), j] = col_mean
        
    Sigma_hat = np.cov(data_imputed, rowvar=False) 
    
    # Ensure Sigma is invertible
    if np.linalg.det(Sigma_hat) < 1e-6:
        p = data_obs.shape[1]
        Sigma_hat += np.eye(p) * 1e-4
        
    return mu_hat, Sigma_hat


# E Step
def e_step(data_obs, mu_l_minus_1, Sigma_l_minus_1):
    """
    E-step: Compute conditional expectation (X_hat_n) and conditional covariance (C_n).
    """
    N = len(data_obs)
    p = len(mu_l_minus_1)
    X_hat_n = np.zeros((N, p))
    C_n = [np.zeros((p, p)) for _ in range(N)]
    
    for n in range(N):
        x_obs_n = data_obs[n]
        
        is_observed = ~np.isnan(x_obs_n)
        obs_idx = np.where(is_observed)[0]
        mis_idx = np.where(~is_observed)[0]
        
        # No missing data: use observation directly
        if len(mis_idx) == 0:
            X_hat_n[n] = x_obs_n
            C_n[n] = np.zeros((p, p)) 
            continue
            
        x_obs_val = x_obs_n[obs_idx]
        mu_obs = mu_l_minus_1[obs_idx]
        mu_mis = mu_l_minus_1[mis_idx]
        
        Sigma_obs_obs = Sigma_l_minus_1[np.ix_(obs_idx, obs_idx)]
        Sigma_mis_obs = Sigma_l_minus_1[np.ix_(mis_idx, obs_idx)]
        Sigma_mis_mis = Sigma_l_minus_1[np.ix_(mis_idx, mis_idx)]
        
        # Conditional mean: E[x_mis | x_obs]
        try:
            Sigma_obs_obs_inv = np.linalg.inv(Sigma_obs_obs)
        except np.linalg.LinAlgError:
            jitter = np.eye(len(obs_idx)) * 1e-6
            Sigma_obs_obs_inv = np.linalg.inv(Sigma_obs_obs + jitter)
        
        correction_term = Sigma_mis_obs @ Sigma_obs_obs_inv @ (x_obs_val - mu_obs)
        x_mis_hat = mu_mis + correction_term
        
        x_hat_n = np.zeros(p)
        x_hat_n[obs_idx] = x_obs_val
        x_hat_n[mis_idx] = x_mis_hat
        X_hat_n[n] = x_hat_n
        
        # Conditional covariance: Cov[x_mis | x_obs]
        C_mis_mis = Sigma_mis_mis - Sigma_mis_obs @ Sigma_obs_obs_inv @ Sigma_mis_obs.T
        
        C_mat_n = np.zeros((p, p))
        C_mat_n[np.ix_(mis_idx, mis_idx)] = C_mis_mis
        C_n[n] = C_mat_n
        
    return X_hat_n, C_n

# M Step
def m_step(X_hat_n, C_n, N):
    """
    M-step: Update parameters (mu_hat, Sigma_hat) by maximizing Q-function.
    """
    # Update mean
    mu_hat_l = np.mean(X_hat_n, axis=0)
    
    # Update covariance
    p = len(mu_hat_l)
    Sigma_hat_l = np.zeros((p, p))
    
    for n in range(N):
        x_n = X_hat_n[n]
        C_mat_n = C_n[n]
        
        diff = x_n - mu_hat_l
        term1 = np.outer(diff, diff) 
        term2 = C_mat_n
        
        Sigma_hat_l += (term1 + term2)
        
    Sigma_hat_l /= N
    
    return mu_hat_l, Sigma_hat_l


# EM for multivariate Gaussian 
def em_multivariate_gaussian(data_obs, max_iter=100, tol=1e-4):
    """
    Complete EM algorithm for estimating mu and Sigma of a multivariate Gaussian
    with missing data.
    """
    N = len(data_obs)
    
    mu_hat, Sigma_hat = initialize_parameters(data_obs)
    
    print(f"Initialization mu: {mu_hat}")
    
    for i in range(max_iter):
        mu_old, Sigma_old = mu_hat.copy(), Sigma_hat.copy()

        # E-step
        X_hat_n, C_n = e_step(data_obs, mu_hat, Sigma_hat)
        
        # M-step
        mu_hat, Sigma_hat = m_step(X_hat_n, C_n, N)
        
        # Check convergence
        mu_diff = np.linalg.norm(mu_hat - mu_old)
        Sigma_diff = np.linalg.norm(Sigma_hat - Sigma_old) / len(Sigma_hat)**2
        
        if mu_diff < tol and Sigma_diff < tol:
            print(f"Converged after {i+1} iterations.")
            break
            
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}: Delta Mu = {mu_diff:.5f}, Delta Sigma = {Sigma_diff:.5f}")

    return mu_hat, Sigma_hat, i+1


if __name__ == "__main__":
    
    # Practical example
    print("### EM Algorithm Example for Missing Data ###")

    # Real parameters
    mu_real = [50, 100, 25, 75]
    Sigma_real = np.array([
            [10,  5,  2,  3],
            [ 5, 20,  4,  6],
            [ 2,  4, 15,  1],
            [ 3,  6,  1, 12]
        ])

    data_obs = np.array(pd.read_csv("tests\\MAR_missing_30pct.csv", skiprows=0))

    print(f"\nObserved data dimensions: {data_obs.shape}")
    print(f"Missing values percentage: {np.sum(np.isnan(data_obs)) / data_obs.size * 100:.2f}%")

    print("\n--- Real Parameters ---")
    print(f"Real Mu: \n{mu_real}")
    print(f"Real Sigma: \n{Sigma_real}")

    # Run EM algorithm
    mu_estimated, Sigma_estimated, num_iterations = em_multivariate_gaussian(data_obs, max_iter=200, tol=1e-5)

    # Compare results
    print("\n--- EM Estimation Results ---")
    print("Estimated Mu:")
    print(mu_estimated)

    print("\nEstimated Sigma:")
    print(Sigma_estimated)

    # Calculate error (Euclidean distance)
    mu_error = np.linalg.norm(mu_estimated - mu_real)
    Sigma_error = np.linalg.norm(Sigma_estimated - Sigma_real) / len(Sigma_estimated)**2

    print(f"\nERROR: Distance from Real Mu: {mu_error:.4f}")
    print(f"ERROR: Distance from Real Sigma: {Sigma_error:.4f}")