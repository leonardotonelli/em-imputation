import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 1. ROBUST PATH SETUP ---
# Get the absolute path of the folder containing THIS script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the root directory to Python path to ensure imports work
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Construct absolute paths for data files
# This solves the FileNotFoundError regardless of where you run the script from
DATA_FILE = os.path.join(BASE_DIR, 'data', 'real_example', 'data.csv')
GT_FILE = os.path.join(BASE_DIR, 'data', 'real_example', 'ground_truth.xlsx')

# Import the loader from your package
try:
    from data.real_example.preprocessing import load_data_binary
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    print("Ensure this script is located in the root project folder 'project-luca-leo-vale_stat'")
    sys.exit(1)

# --- 2. LOAD DATA ---
print(f"Loading data from:\n -> {DATA_FILE}")

if not os.path.exists(DATA_FILE):
    print(f"ERROR: File {DATA_FILE} does not exist. Check the path.")
    sys.exit(1)

X, y_groundtruth, y_experts = load_data_binary(
    data_file=DATA_FILE, 
    gt_file=GT_FILE
)

# --- 3. PROCESSING: STANDARD SCALING + PCA ---
print("Running Standardization and PCA...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 2 dimensions for visualization and testing
pca = PCA(n_components=2)
X_pca_2d = pca.fit_transform(X_scaled)

# Create a DataFrame for plotting
df_vis = pd.DataFrame(X_pca_2d, columns=['PC1', 'PC2'])
df_vis['Label'] = y_groundtruth
df_vis['Label'] = df_vis['Label'].map({0: 'Benign', 1: 'Malignant'})

# --- 4. GENERATE PLOT ---
print("Generating inspection plot...")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_vis, x='PC1', y='PC2', hue='Label', style='Label', s=100, alpha=0.7)
plt.title("Visual Inspection of Gaussian Assumption (PCA Space)", fontsize=14, fontweight='bold')
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Ground Truth')

# Save the plot to the 'tests' folder (using absolute path)
OUTPUT_DIR = os.path.join(BASE_DIR, 'tests\\3_d')
os.makedirs(OUTPUT_DIR, exist_ok=True)
save_path = os.path.join(OUTPUT_DIR, "normality_check_pca.png")

plt.savefig(save_path, dpi=300)
print(f"Plot successfully saved to: {save_path}")
# plt.show() # Uncomment if you want to see the plot on screen

# --- 5. SHAPIRO-WILK STATISTICAL TEST ---
print("\n=== Shapiro-Wilk Test for Multivariate Normality ===")
print("(H0: The data follows a Normal distribution. If p-value < 0.05, we reject H0)")

for cls, name in zip([0, 1], ['Benign', 'Malignant']):
    print(f"\nClass: {name}")
    X_cls = X_pca_2d[y_groundtruth == cls]
    
    for i in range(2):
        stat, p = stats.shapiro(X_cls[:, i])
        result = "NOT Normal (Non-Gaussian)" if p < 0.05 else "Normal (Gaussian)"
        print(f"  PC{i+1}: W={stat:.4f}, p-value={p:.4e} -> {result}")

print("\nAnalysis complete.")