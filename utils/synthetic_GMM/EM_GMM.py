import numpy as np
import pandas as pd


def initialize_gmm_parameters(data_obs, n_components, random_state=42):
    """
    Initialize GMM parameters using k-means-style initialization on available data.
    """
    np.random.seed(random_state)
    
    N, p = data_obs.shape
    
    # Simple imputation for initialization: fill NaN with column mean
    data_imputed = data_obs.copy()
    col_means = np.nanmean(data_obs, axis=0)
    for j in range(p):
        data_imputed[np.isnan(data_imputed[:, j]), j] = col_means[j]
    
    # Random initialization of responsibilities
    responsibilities = np.random.dirichlet(np.ones(n_components), size=N)
    
    # Initialize mixing coefficients
    pi = np.ones(n_components) / n_components
    
    # Initialize means using weighted average
    mu = []
    for k in range(n_components):
        weighted_mean = np.average(data_imputed, axis=0, weights=responsibilities[:, k])
        mu.append(weighted_mean)
    mu = np.array(mu)
    
    # Initialize covariances
    Sigma = []
    for k in range(n_components):
        cov = np.cov(data_imputed, rowvar=False)
        if np.linalg.det(cov) < 1e-6:
            cov += np.eye(p) * 1e-4
        Sigma.append(cov)
    
    return pi, mu, Sigma


def gaussian_pdf(x, mu, Sigma):
    """
    Compute multivariate Gaussian PDF, handling NaN values.
    """
    p = len(mu)
    
    is_observed = ~np.isnan(x)
    obs_idx = np.where(is_observed)[0]
    
    if len(obs_idx) == 0:
        return 1.0  # Uniform when no observations
    
    x_obs = x[obs_idx]
    mu_obs = mu[obs_idx]
    Sigma_obs = Sigma[np.ix_(obs_idx, obs_idx)]
    
    try:
        Sigma_inv = np.linalg.inv(Sigma_obs)
        det_Sigma = np.linalg.det(Sigma_obs)
    except np.linalg.LinAlgError:
        Sigma_obs += np.eye(len(obs_idx)) * 1e-6
        Sigma_inv = np.linalg.inv(Sigma_obs)
        det_Sigma = np.linalg.det(Sigma_obs)
    
    d = len(obs_idx)
    diff = x_obs - mu_obs
    
    exponent = -0.5 * diff.T @ Sigma_inv @ diff
    coefficient = 1.0 / np.sqrt((2 * np.pi) ** d * det_Sigma)
    
    return coefficient * np.exp(exponent)


def e_step_gmm(data_obs, pi, mu, Sigma):
    """
    E-step for GMM with missing data:
    - Compute responsibilities (gamma)
    - Compute conditional expectations (X_hat) and covariances (C) for each component
    """
    N, p = data_obs.shape
    K = len(pi)
    
    # Responsibilities
    gamma = np.zeros((N, K))
    
    # Conditional statistics for each observation and component
    X_hat = np.zeros((N, K, p))
    C = [[np.zeros((p, p)) for _ in range(K)] for _ in range(N)]
    
    for n in range(N):
        x_obs_n = data_obs[n]
        
        # Compute responsibilities
        likelihoods = np.array([pi[k] * gaussian_pdf(x_obs_n, mu[k], Sigma[k]) for k in range(K)])
        total_likelihood = np.sum(likelihoods)
        
        if total_likelihood > 0:
            gamma[n] = likelihoods / total_likelihood
        else:
            gamma[n] = np.ones(K) / K
        
        # Compute conditional expectations for each component
        is_observed = ~np.isnan(x_obs_n)
        obs_idx = np.where(is_observed)[0]
        mis_idx = np.where(~is_observed)[0]
        
        for k in range(K):
            if len(mis_idx) == 0:
                # No missing data
                X_hat[n, k] = x_obs_n
                C[n][k] = np.zeros((p, p))
            else:
                # Compute conditional mean and covariance
                x_obs_val = x_obs_n[obs_idx]
                mu_obs = mu[k][obs_idx]
                mu_mis = mu[k][mis_idx]
                
                Sigma_obs_obs = Sigma[k][np.ix_(obs_idx, obs_idx)]
                Sigma_mis_obs = Sigma[k][np.ix_(mis_idx, obs_idx)]
                Sigma_mis_mis = Sigma[k][np.ix_(mis_idx, mis_idx)]
                
                try:
                    Sigma_obs_obs_inv = np.linalg.inv(Sigma_obs_obs)
                except np.linalg.LinAlgError:
                    Sigma_obs_obs_inv = np.linalg.inv(Sigma_obs_obs + np.eye(len(obs_idx)) * 1e-6)
                
                # Conditional mean
                x_mis_hat = mu_mis + Sigma_mis_obs @ Sigma_obs_obs_inv @ (x_obs_val - mu_obs)
                
                x_hat_n = np.zeros(p)
                x_hat_n[obs_idx] = x_obs_val
                x_hat_n[mis_idx] = x_mis_hat
                X_hat[n, k] = x_hat_n
                
                # Conditional covariance
                C_mis_mis = Sigma_mis_mis - Sigma_mis_obs @ Sigma_obs_obs_inv @ Sigma_mis_obs.T
                C_mat = np.zeros((p, p))
                C_mat[np.ix_(mis_idx, mis_idx)] = C_mis_mis
                C[n][k] = C_mat
    
    return gamma, X_hat, C


def m_step_gmm(gamma, X_hat, C, N):
    """
    M-step for GMM: Update pi, mu, and Sigma.
    """
    K = gamma.shape[1]
    p = X_hat.shape[2]
    
    N_k = np.sum(gamma, axis=0)
    
    # Update mixing coefficients
    pi = N_k / N
    
    # Update means
    mu = np.zeros((K, p))
    for k in range(K):
        mu[k] = np.sum(gamma[:, k:k+1] * X_hat[:, k, :], axis=0) / N_k[k]
    
    # Update covariances
    Sigma = []
    for k in range(K):
        Sigma_k = np.zeros((p, p))
        for n in range(N):
            diff = X_hat[n, k] - mu[k]
            Sigma_k += gamma[n, k] * (np.outer(diff, diff) + C[n][k])
        Sigma_k /= N_k[k]
        
        # Ensure invertibility
        if np.linalg.det(Sigma_k) < 1e-6:
            Sigma_k += np.eye(p) * 1e-4
        
        Sigma.append(Sigma_k)
    
    return pi, mu, Sigma


def compute_log_likelihood(data_obs, pi, mu, Sigma):
    """
    Compute log-likelihood of observed data.
    """
    N = len(data_obs)
    K = len(pi)
    
    log_likelihood = 0.0
    for n in range(N):
        x_obs_n = data_obs[n]
        likelihood = sum(pi[k] * gaussian_pdf(x_obs_n, mu[k], Sigma[k]) for k in range(K))
        if likelihood > 0:
            log_likelihood += np.log(likelihood)
    
    return log_likelihood


def em_gmm(data_obs, n_components, max_iter=100, tol=1e-4, random_state=42):
    """
    EM algorithm for Gaussian Mixture Model with missing data.
    """
    N, p = data_obs.shape
    
    # Initialize parameters
    pi, mu, Sigma = initialize_gmm_parameters(data_obs, n_components, random_state)
    
    print(f"Initialization:")
    print(f"  Mixing coefficients: {pi}")
    print(f"  Number of components: {n_components}")
    
    log_likelihood_old = compute_log_likelihood(data_obs, pi, mu, Sigma)
    
    for iteration in range(max_iter):
        # E-step
        gamma, X_hat, C = e_step_gmm(data_obs, pi, mu, Sigma)
        
        # M-step
        pi, mu, Sigma = m_step_gmm(gamma, X_hat, C, N)
        
        # Compute log-likelihood
        log_likelihood = compute_log_likelihood(data_obs, pi, mu, Sigma)
        
        # Check convergence
        ll_change = abs(log_likelihood - log_likelihood_old)
        
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}: Log-likelihood = {log_likelihood:.4f}, Change = {ll_change:.6f}")
        
        if ll_change < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break
        
        log_likelihood_old = log_likelihood
    
    return pi, mu, Sigma, gamma, iteration + 1


if __name__ == "__main__":
    print("### EM Algorithm for GMM with Missing Data ###\n")
    
    # True parameters (3-component GMM)
    n_components_true = 3
    pi_true = [0.3, 0.4, 0.3]
    mu_true = [
        np.array([0, 0]),
        np.array([5, 5]),
        np.array([0, 5])
    ]
    Sigma_true = [
        np.array([[1.0, 0.3], [0.3, 1.0]]),
        np.array([[1.5, -0.5], [-0.5, 1.5]]),
        np.array([[1.0, 0.0], [0.0, 1.0]])
    ]
    
    # Load data with missing values
    data_obs = pd.read_csv("synthetic_gmm/MAR_missing_30pct.csv").values
    
    print(f"Observed data dimensions: {data_obs.shape}")
    print(f"Missing values percentage: {np.sum(np.isnan(data_obs)) / data_obs.size * 100:.2f}%\n")
    
    print("--- True Parameters ---")
    print(f"Mixing coefficients: {pi_true}")
    for k in range(n_components_true):
        print(f"Component {k}: mu = {mu_true[k]}, Sigma =\n{Sigma_true[k]}")
    
    # Run EM algorithm
    print("\n--- Running EM Algorithm ---")
    pi_est, mu_est, Sigma_est, gamma_est, num_iter = em_gmm(
        data_obs, 
        n_components=n_components_true,
        max_iter=200, 
        tol=1e-5,
        random_state=42
    )
    
    print("\n--- Estimated Parameters ---")
    print(f"Mixing coefficients: {pi_est}")
    for k in range(n_components_true):
        print(f"Component {k}:")
        print(f"  mu = {mu_est[k]}")
        print(f"  Sigma =\n{Sigma_est[k]}")