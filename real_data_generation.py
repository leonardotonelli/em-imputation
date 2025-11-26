import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_wine
import os

# --- CORE LOGIC ---
def inject_missingness(df, mech, p, target_col, covariate_col):
    """
    Helper function to apply the missingness mask based on the mechanism.
    """
    df_out = df.copy()
    n = len(df)
    
    if mech == 'MCAR':
        # Random removal
        mask = np.random.rand(n) < p
        df_out.loc[mask, target_col] = np.nan
        
    elif mech == 'MAR':
        # Missingness on target depends on covariate_col
        # Logic: Remove target if covariate is in the top p percentile
        cutoff = df_out[covariate_col].quantile(1 - p)
        mask = df_out[covariate_col] > cutoff
        df_out.loc[mask, target_col] = np.nan
        
    elif mech == 'MNAR':
        # Missingness depends on target_col itself
        # Logic: Remove target if target is in the top p percentile
        cutoff = df_out[target_col].quantile(1 - p)
        mask = df_out[target_col] > cutoff
        df_out.loc[mask, target_col] = np.nan
        
    return df_out

def save_variants(df_base, dataset_name, mechs, percs, target, covariate):
    """
    Iterates through patterns and percentages and saves files.
    """
    output_dir = f"data_{dataset_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save Full Data (Ground Truth)
    df_base.to_csv(f"{output_dir}/{dataset_name}_complete.csv", index=False)
    
    for m in mechs:
        for p in percs:
            df_miss = inject_missingness(df_base, m, p, target, covariate)
            fname = f"{output_dir}/{dataset_name}_{m}_{int(p*100)}pct.csv"
            df_miss.to_csv(fname, index=False)
            print(f"[{dataset_name}] Saved: {fname}")

# --- DATASET 1: PALMER PENGUINS ---
def generate_penguins(mechs, percs):
    # Load
    try:
        df = sns.load_dataset('penguins')
    except:
        df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv")
    
    # Preprocess: Keep continuous vars, Drop existing NaNs
    cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    df = df[cols].dropna().reset_index(drop=True)
    
    # Logic: Missing Bill Length based on Flipper Length (Correlation ~0.65)
    save_variants(df, "penguins", mechs, percs, 
                  target='bill_length_mm', 
                  covariate='flipper_length_mm')

# --- DATASET 2: OLD FAITHFUL GEYSER ---
def generate_geyser(mechs, percs):
    # Load
    try:
        df = sns.load_dataset('geyser')
    except:
        df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/geyser.csv")
        
    # Preprocess: The dataset is usually clean, but we ensure it
    df = df[['duration', 'waiting']].dropna().reset_index(drop=True)
    
    # Logic: Missing Duration (eruption time) based on Waiting time (Correlation ~0.90)
    save_variants(df, "geyser", mechs, percs, 
                  target='duration', 
                  covariate='waiting')

# --- DATASET 3: UCI WINE ---
def generate_wine(mechs, percs):
    # Load
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Logic: Missing 'magnesium' based on 'alcohol'
    # We choose magnesium as it is a continuous chemical property 
    # fitting the GMM assumption.
    save_variants(df, "wine", mechs, percs, 
                  target='magnesium', 
                  covariate='alcohol')

# --- EXECUTION ---
if __name__ == "__main__":
    patterns = ['MCAR', 'MAR', 'MNAR']
    percentages = [0.10, 0.30] # 10% and 30%

    print("--- Generating Penguins Data ---")
    generate_penguins(patterns, percentages)
    
    print("\n--- Generating Geyser Data ---")
    generate_geyser(patterns, percentages)
    
    print("\n--- Generating Wine Data ---")
    generate_wine(patterns, percentages)