import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

def initialize_parameters(X, y, n_components, random_state=42):
    """
    Initialize parameters using labeled data to guide the starting point.
    If a class has no labeled data, we initialize it randomly.
    """
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    
    # Initialize arrays
    pi = np.ones(n_components) / n_components
    mu = np.zeros((n_components, n_features))
    Sigma = []
    
    # Helper: Global mean/cov for fallback
    global_mean = np.mean(X, axis=0)
    global_cov = np.cov(X, rowvar=False)
    
    for c in range(n_components):
        # Filter for labeled data of this class
        # We assume -1 indicates "unlabeled"
        X_c = X[y == c]
        
        if len(X_c) > 1:
            # If we have labels, use them to initialize mean/cov (Supervised init)
            mu[c] = np.mean(X_c, axis=0)
            cov_c = np.cov(X_c, rowvar=False) + np.eye(n_features) * 1e-6
            Sigma.append(cov_c)
            # Update mixing coef based on counts (optional, but helpful)
            pi[c] = len(X_c) / len(y[y != -1])
        else:
            # Fallback: Random initialization if no labels for this class
            mu[c] = global_mean + np.random.randn(n_features) * 0.1
            Sigma.append(global_cov + np.eye(n_features) * 1e-6)
            
    # Normalize pi just in case
    pi = pi / np.sum(pi)
    
    return pi, mu, Sigma

def e_step_semi_supervised(X, y, pi, mu, Sigma):
    """
    E-Step: Estimate responsibilities (r).
    """
    N, K = len(X), len(pi)
    responsibilities = np.zeros((N, K))
    
    # 1. Calculate unnormalized posteriors for ALL points first
    for k in range(K):
        # We use a small reg to prevent numerical errors
        try:
            responsibilities[:, k] = pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=Sigma[k], allow_singular=True)
        except np.linalg.LinAlgError:
            # Fallback for singular matrices
            responsibilities[:, k] = pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=Sigma[k] + np.eye(X.shape[1])*1e-5)

    # Normalize to get probabilities 
    row_sums = responsibilities.sum(axis=1)[:, np.newaxis]
    # Avoid division by zero
    row_sums[row_sums == 0] = 1e-10
    responsibilities = responsibilities / row_sums
    
    # 2. OVERRIDE with Known Labels (The Semi-Supervised constraint)
    # For labeled data (not NaN), we do not guess; we enforce the truth.
    for i in range(N):
        if not np.isnan(y[i]):
            # Create a one-hot vector for the true class
            true_class = int(y[i])
            responsibilities[i, :] = 0.0
            responsibilities[i, true_class] = 1.0
            
    return responsibilities

def m_step_semi_supervised(X, responsibilities):
    """
    M-Step: Update parameters maximize expected log-likelihood.
    """
    N, D = X.shape
    K = responsibilities.shape[1]
    
    # 1. Effective number of points in each cluster (N_c)
    # Sum of responsibilities for class c 
    N_c = responsibilities.sum(axis=0)
    
    # 2. Update Priors (pi)
    # count / total_samples 
    pi_new = N_c / N
    
    mu_new = np.zeros((K, D))
    Sigma_new = []
    
    for k in range(K):
        # Avoid division by zero if a cluster dies out
        total_weight = N_c[k] if N_c[k] > 1e-10 else 1.0
        
        # 3. Update Means (mu)
        # Weighted sum of x / sum of weights 
        # Note: For labeled data, weight is 1.0; for unlabeled, it is r_c.
        mu_k = np.sum(responsibilities[:, k].reshape(-1, 1) * X, axis=0) / total_weight
        mu_new[k] = mu_k
        
        # 4. Update Covariances (Sigma)
        # Weighted sum of (x-mu)(x-mu)^T 
        diff = X - mu_k
        # Efficient weighted covariance calculation:
        sigma_k = np.dot((responsibilities[:, k].reshape(-1, 1) * diff).T, diff) / total_weight
        
        # Regularization to ensure invertibility
        sigma_k += np.eye(D) * 1e-5
        Sigma_new.append(sigma_k)
        
    return pi_new, mu_new, Sigma_new

def compute_log_likelihood(X, pi, mu, Sigma):
    """
    Compute the observed data log-likelihood to check convergence.
    """
    N, K = len(X), len(pi)
    likelihoods = np.zeros((N, K))
    for k in range(K):
        likelihoods[:, k] = pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=Sigma[k], allow_singular=True)
    
    # Sum over components (marginalizing out z)
    total_likelihood = np.sum(likelihoods, axis=1)
    
    # Avoid log(0)
    total_likelihood[total_likelihood < 1e-10] = 1e-10
    return np.sum(np.log(total_likelihood))

def em_semi_supervised(X, y, n_components, max_iter=100, tol=1e-4, verbose=False):
    """
    Main Loop for Semi-Supervised EM.
    """
    if verbose:
        print("--- Starting Semi-Supervised EM ---")
        print(f"Total samples: {len(X)}")
        print(f"Labeled samples: {np.sum(~np.isnan(y))}")
        print(f"Unlabeled samples: {np.sum(np.isnan(y))}\n")
    
    # Initialization
    pi, mu, Sigma = initialize_parameters(X, y, n_components)
    
    log_likelihood_old = -np.inf
    
    for iteration in range(max_iter):
        # E-STEP
        responsibilities = e_step_semi_supervised(X, y, pi, mu, Sigma)
        
        # M-STEP
        pi, mu, Sigma = m_step_semi_supervised(X, responsibilities)
        
        # Check Convergence
        log_likelihood = compute_log_likelihood(X, pi, mu, Sigma)
        change = np.abs(log_likelihood - log_likelihood_old)
        
        if verbose:
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"Iteration {iteration+1}: Log-Likelihood = {log_likelihood:.4f}")
                
        if change < tol:
            if verbose:
                print(f"Converged at iteration {iteration+1}")
            num_iterations = iteration + 1
            return pi, mu, Sigma, num_iterations
            break
            
        log_likelihood_old = log_likelihood
    num_iterations = max_iter
    return pi, mu, Sigma, num_iterations

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Generate Synthetic Data
    np.random.seed(0)
    # Cluster 0
    X0 = np.random.randn(50, 2) + np.array([0, 0])
    # Cluster 1
    X1 = np.random.randn(50, 2) + np.array([5, 5])
    # Cluster 2
    X2 = np.random.randn(50, 2) + np.array([0, 5])
    
    X = np.vstack([X0, X1, X2])
    # True labels
    y_true = np.array([0]*50 + [1]*50 + [2]*50)
    
    # 2. Create Semi-Supervised Scenario (Mask 80% of labels)
    y_semi = y_true.copy()
    mask = np.random.rand(len(y_semi)) < 1  # 80% missing
    y_semi[mask] = -1  # -1 denotes unlabeled
    
    # 3. Run Algorithm
    pi_est, mu_est, Sigma_est = em_semi_supervised(X, y_semi, n_components=3)
    
    
    print("\n--- Final Results ---")
    print(f"Estimated Means:\n{mu_est}")
    
    # Simple accuracy check on the UNLABELED portion
    unlabeled_idx = (y_semi == -1)
    print(f"Accuracy on unlabeled data: {acc*100:.2f}%")