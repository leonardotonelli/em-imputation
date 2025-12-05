import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os
import glob 

# -----------------------------------------------
# 1. Real Parameters
# -----------------------------------------------

means = np.array([50, 100, 25, 75])
cov_matrix = np.array([
        [10,  5,  2,  3],
        [ 5, 20,  4,  6],
        [ 2,  4, 15,  1],
        [ 3,  6,  1, 12]
    ])

# -----------------------------------------------
# 2. Imputation Functions
# -----------------------------------------------

# Mean Imputation
def mean_imputation(data, means, cov_matrix):
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(data)
    mean_computed = np.mean(X_imputed, axis = 0)
    cov_computed= np.cov(X_imputed, rowvar = False)
    mean_error = np.linalg.norm(means - mean_computed)
    cov_error = np.linalg.norm(cov_matrix - cov_computed)
    return mean_error, cov_error
    
# Median Imputation
def median_imputation(data, means, cov_matrix): 
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(data)
    mean_computed = np.mean(X_imputed, axis = 0)
    cov_computed = np.cov(X_imputed, rowvar = False)
    mean_error = np.linalg.norm(means - mean_computed)
    cov_error = np.linalg.norm(cov_matrix - cov_computed)
    return mean_error, cov_error

# Mode Imputation
def mode_imputation(data, means, cov_matrix): 
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = imputer.fit_transform(data)
    mean_computed = np.mean(X_imputed, axis = 0)
    cov_computed = np.cov(X_imputed, rowvar = False)
    mean_error = np.linalg.norm(means - mean_computed)
    cov_error = np.linalg.norm(cov_matrix - cov_computed)
    return mean_error, cov_error

# -----------------------------------------------
# 3. Execution
# -----------------------------------------------
directory_path = "tests"
file_pattern = os.path.join(directory_path, "*.csv")
csv_files = glob.glob(file_pattern)

results_dict = {}

for file_path in csv_files:
    file_name = os.path.basename(file_path)

    try:
        df = pd.read_csv(file_path, skiprows=0)
        data_array = df.to_numpy()
        
        mean_err, mean_cov_err = mean_imputation(data_array, means, cov_matrix)
        median_err, median_cov_err = median_imputation(data_array, means, cov_matrix)
        mode_err, mode_cov_err = mode_imputation(data_array, means, cov_matrix)
        
        results_dict[file_name] = {
            'Mean_Mean_Error': mean_err,
            'Mean_Cov_Error': mean_cov_err,
            'Median_Mean_Error': median_err,
            'Median_Cov_Error': median_cov_err,
            'Mode_Mean_Error': mode_err,
            'Mode_Cov_Error': mode_cov_err
        }
        
    except Exception as e:
        print(f"Error during elaboration of {file_name}: {e}")


# -----------------------------------------------
# 4. Saving results
# -----------------------------------------------
output_folder = 'Imputation results'
output_filename = 'mean_median_mode_imputation_results.csv'
output_file_path = os.path.join(output_folder, output_filename)

try:
    os.makedirs(output_folder, exist_ok=True)
except Exception as e:
    print(f"\nError during folder creation: {e}")
    output_file_path = output_filename

final_df = pd.DataFrame.from_dict(results_dict, orient='index')

tuples = [
    ('Mean', 'Mean Error'), ('Mean', 'Cov Error'),
    ('Median', 'Mean Error'), ('Median', 'Cov Error'),
    ('Mode', 'Mean Error'), ('Mode', 'Cov Error')
]
final_df.columns = pd.MultiIndex.from_tuples(tuples, names=['Imputation Method', 'Error Type'])
final_df = final_df.reset_index().rename(columns={'index': 'File'})

final_df.to_csv(output_file_path, index=False)

