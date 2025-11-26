import numpy as np
import pandas as pd
import os
from sklearn.datasets import load_iris

def apply_missingness_iris(mechanisms, percentages, output_dir="iris_missing_data"):
    """
    Loads the Iris dataset and injects missingness (MCAR, MAR, MNAR) 
    into a specific target column.
    """
    # 1. Load Real Data
    # The Iris dataset has 3 distinct clusters (species), fitting a GMM structure.
    raw_data = load_iris()
    df_base = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
    
    # We will inject missingness into 'sepal width (cm)'
    target_col = 'sepal width (cm)'
    # We will use 'sepal length (cm)' as the observed covariate for MAR
    covariate_col = 'sepal length (cm)'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Base Dataset loaded. Shape: {df_base.shape}")
    print(f"Injecting missingness into column: '{target_col}'\n")

    # 2. Iterate and Generate
    for mech in mechanisms:
        for p in percentages:
            df = df_base.copy()
            
            if mech == 'MCAR':
                # Randomly remove p% of values in the target column
                # No correlation to observed or unobserved data
                mask = np.random.rand(len(df)) < p
                df.loc[mask, target_col] = np.nan

            elif mech == 'MAR':
                # Missingness depends on OBSERVED data (covariate_col)
                # Logic: Flowers with the largest Sepal Length are missing their Sepal Width
                cutoff = df[covariate_col].quantile(1 - p)
                mask = df[covariate_col] > cutoff
                df.loc[mask, target_col] = np.nan

            elif mech == 'MNAR':
                # Missingness depends on the MISSING value itself (target_col)
                # Logic: Flowers with the largest Sepal Width are less likely to have it recorded
                cutoff = df[target_col].quantile(1 - p)
                mask = df[target_col] > cutoff
                df.loc[mask, target_col] = np.nan

            # 3. Save
            filename = f"{output_dir}/iris_{mech}_{int(p*100)}pct.csv"
            df.to_csv(filename, index=False)
            print(f"Saved: {filename}")

# --- Execution ---
missing_mechanisms = ['MCAR', 'MAR', 'MNAR']
missing_rates = [0.10, 0.30, 0.50]

apply_missingness_iris(missing_mechanisms, missing_rates)