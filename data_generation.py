import numpy as np
import pandas as pd
import os

def generate_gmm_data(n_samples=1000, n_features=4, n_components=3):
    """
    Generates data from a Gaussian Mixture Model.
    """
    data = []
    # Create random centers and covariances for the components
    for _ in range(n_components):
        mean = np.random.uniform(-5, 5, n_features)
        # Create a positive semi-definite covariance matrix
        A = np.random.rand(n_features, n_features)
        cov = np.dot(A, A.transpose()) 
        
        # Generate samples for this component
        component_data = np.random.multivariate_normal(mean, cov, n_samples // n_components)
        data.append(component_data)
        
    return np.vstack(data)

def create_missing_datasets(mechanisms, percentages, output_dir="gmm_synthetic_data"):
    """
    Generates datasets with specific missingness patterns and saves them as CSV.
    
    Args:
        mechanisms (list): List of strings ['MCAR', 'MAR', 'MNAR']
        percentages (list): List of floats [0.1, 0.2] (representing 10%, 20%)
    """
    # 1. Generate complete GMM Data
    X_complete = generate_gmm_data(n_samples=1000, n_features=4)
    df_base = pd.DataFrame(X_complete, columns=[f'Feature_{i}' for i in range(4)])
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Iterate through requirements
    for mech in mechanisms:
        for p in percentages:
            df = df_base.copy()
            n_rows, n_cols = df.shape
            
            # --- Missingness Logic ---
            if mech == 'MCAR':
                # Missing Completely At Random: Uniform random mask
                mask = np.random.rand(n_rows, n_cols) < p
                df[mask] = np.nan

            elif mech == 'MAR':
                # Missing At Random: Missingness in Col 1 depends on observed Col 0
                # We sort by Col 0 and remove the top p% of Col 1
                cutoff = df['Feature_0'].quantile(1 - p)
                mask = df['Feature_0'] > cutoff
                df.loc[mask, 'Feature_1'] = np.nan 
                # (Applied to specific column to maintain mathematical definition of MAR)

            elif mech == 'MNAR':
                # Missing Not At Random: Missingness in Col 1 depends on Col 1 itself
                # We sort by Col 1 and remove the top p%
                cutoff = df['Feature_1'].quantile(1 - p)
                mask = df['Feature_1'] > cutoff
                df.loc[mask, 'Feature_1'] = np.nan

            # 3. Save to CSV
            filename = f"{output_dir}/gmm_data_{mech}_{int(p*100)}pct.csv"
            df.to_csv(filename, index=False)
            print(f"Saved: {filename}")

# --- Execution ---
missingness_patterns = ['MCAR', 'MAR', 'MNAR']
missing_percentages = [0.10, 0.30, 0.50]

create_missing_datasets(missingness_patterns, missing_percentages)