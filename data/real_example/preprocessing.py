import pandas as pd
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_data_binary(data_file='data.csv', gt_file='ground_truth.xlsx'):
    """
    Loads and processes the dataset for Binary Classification.
    
    Classes:
        0: Benign (Hyperplasic)
        1: Malign (Adenoma + Serrated)
        
    Returns:
        X (numpy array): Feature matrix (n_samples, 1396).
        y_groundtruth (numpy array): Binary ground truth labels.
        y_experts (numpy array): Binary expert labels (unanimous agreement only, NaN otherwise).
    """
    
    # ---------------------------------------------------------
    # 1. Process Features (X)
    # ---------------------------------------------------------
    # Read raw data. 
    # Structure: Rows=Features, Cols=Samples. First 3 rows are metadata.
    df_raw = pd.read_csv(data_file, header=None)
    
    # Transpose to (Samples x Features)
    df_t = df_raw.T
    
    # Extract metadata from first 3 columns (originally rows)
    # Col 0: Lesion Name, Col 1: Type (Integer), Col 2: Light (1=WL, 2=NBI)
    num_features = df_t.shape[1] - 3
    feature_cols = [f'feat_{i}' for i in range(num_features)]
    df_t.columns = ['lesion_name', 'lesion_type_id', 'light_type'] + feature_cols
    
    # Clean strings and types
    df_t['lesion_name'] = df_t['lesion_name'].astype(str).str.strip()
    df_t['light_type'] = pd.to_numeric(df_t['light_type'], errors='coerce')
    
    # Split WL and NBI
    df_wl = df_t[df_t['light_type'] == 1].set_index('lesion_name')[feature_cols]
    df_nbi = df_t[df_t['light_type'] == 2].set_index('lesion_name')[feature_cols]
    
    # Concatenate features: [WL_features, NBI_features] -> 1396 features total
    X_df = df_wl.join(df_nbi, lsuffix='_wl', rsuffix='_nbi', how='inner')
    X_df = X_df.astype(float)
    
    # ---------------------------------------------------------
    # 2. Process Labels (y)
    # ---------------------------------------------------------
    # Read Ground Truth file. Header is at row index 2 (line 3).
    df_gt = pd.read_excel(gt_file, header=2)
    
    # Keep relevant columns
    cols_needed = ['LESION', 'GROUND TRUTH', 'EXPERT 1', 'EXPERT 2', 'EXPERT 3', 'EXPERT 4']
    df_gt = df_gt[cols_needed].copy()
    
    # Drop empty rows
    df_gt = df_gt.dropna(subset=['LESION'])
    df_gt['LESION'] = df_gt['LESION'].astype(str).str.strip()
    
    # Define Binary Mapping: Benign (0) vs Malign (1)
    # Benign: Hyperplasic + Serrated
    # Malign: Adenoma
    binary_map = {
        'hyperplasic': 0,  # Benign
        'serrated': 1,      # Malign
        'adenoma': 1        # Malign
    }
    
    def map_label(val):
        val = str(val).lower().strip()
        return binary_map.get(val, np.nan)
        
    # Map Ground Truth
    df_gt['y_true'] = df_gt['GROUND TRUTH'].apply(map_label)
    
    # Map Experts
    expert_cols = ['EXPERT 1', 'EXPERT 2', 'EXPERT 3', 'EXPERT 4']
    expert_votes = df_gt[expert_cols].map(map_label)
    
    # Function to check unanimous agreement
    def get_unanimous_vote(row):
        # Filter out NaNs if any
        votes = [v for v in row if not np.isnan(v)]
        
        # If we don't have all 4 votes, return NaN
        if len(votes) != 4:
            return np.nan
        
        # Check if all votes are identical
        if len(set(votes)) == 1:
            return votes[0]
        else:
            return np.nan  # No unanimous agreement

    df_gt['y_experts'] = expert_votes.apply(get_unanimous_vote, axis=1)
    
    # Set index to align with X
    df_gt.set_index('LESION', inplace=True)
    
    # ---------------------------------------------------------
    # 3. Final Merge
    # ---------------------------------------------------------
    # Inner join features and labels
    final_df = X_df.join(df_gt[['y_true', 'y_experts']], how='inner')
    
    # Separate into arrays
    X = final_df.drop(columns=['y_true', 'y_experts']).values
    y_groundtruth = final_df['y_true'].values.astype(int)
    
    # Keep y_experts as float to preserve NaN values
    y_experts = final_df['y_experts'].values
    
    return X, y_groundtruth, y_experts


if __name__ == "__main__":
    X, y_groundtruth, y_experts = load_data_binary()

    # 1. Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Apply PCA (Retaining 95% variance)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)