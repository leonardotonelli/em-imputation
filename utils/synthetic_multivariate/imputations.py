import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import KFold
import miceforest as mf
from time import time


# Mean Imputation
def mean_imputation(data, means, cov_matrix):
    start = time()
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(data)
    mean_computed = np.mean(X_imputed, axis = 0)
    cov_computed= np.cov(X_imputed, rowvar = False)
    mean_error = np.linalg.norm(means - mean_computed)
    cov_error = np.linalg.norm(cov_matrix - cov_computed)
    cov_error = cov_error/len(cov_matrix)**2
    end = time()
    return mean_error, cov_error, end - start
    
# Median Imputation
def median_imputation(data, means, cov_matrix): 
    start = time()
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(data)
    mean_computed = np.mean(X_imputed, axis = 0)
    cov_computed = np.cov(X_imputed, rowvar = False)
    mean_error = np.linalg.norm(means - mean_computed)
    cov_error = np.linalg.norm(cov_matrix - cov_computed)
    cov_error = cov_error/len(cov_matrix)**2
    end = time()
    return mean_error, cov_error, end - start

# Mode Imputation
def mode_imputation(data, means, cov_matrix): 
    start = time()
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = imputer.fit_transform(data)
    mean_computed = np.mean(X_imputed, axis = 0)
    cov_computed = np.cov(X_imputed, rowvar = False)
    mean_error = np.linalg.norm(means - mean_computed)
    cov_error = np.linalg.norm(cov_matrix - cov_computed)
    cov_error = cov_error/len(cov_matrix)**2
    end = time()
    return mean_error, cov_error, end - start


def knn_imputation(data, means, cov_matrix, k=5):
    """
    Impute missing values using KNN and compute errors.
    """
    start = time()
    imputer = KNNImputer(n_neighbors=k)
    X_imputed = imputer.fit_transform(data)
    mean_computed = np.mean(X_imputed, axis=0)
    cov_computed = np.cov(X_imputed, rowvar=False)
    mean_error = np.linalg.norm(means - mean_computed)
    cov_error = np.linalg.norm(cov_matrix - cov_computed)
    cov_error = cov_error/len(cov_matrix)**2
    end = time()
    return mean_error, cov_error, end - start


def mice_imputation(data, means, cov_matrix, iterations=5):
    """
    Impute missing values using MICE (Multivariate Imputation by Chained Equations)
    via miceforest and compute errors.
    """
    start = time()
    # Convert to pandas DataFrame if numpy array
    import pandas as pd
    if isinstance(data, np.ndarray):
        # Create DataFrame with string column names
        data_df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(data.shape[1])])
    else:
        data_df = data.copy()
        # Ensure column names are strings
        data_df.columns = [str(col) for col in data_df.columns]
    
    # Create MICE kernel
    kernel = mf.ImputationKernel(
        data_df, 
        save_all_iterations_data=False,
        random_state=42
    )
    
    # Run the MICE algorithm
    kernel.mice(iterations=iterations, verbose=False)
    
    # Get the completed dataset
    X_imputed = kernel.complete_data()
    
    # Convert to numpy array if it's a DataFrame
    if hasattr(X_imputed, 'values'):
        X_imputed = X_imputed.values
    
    mean_computed = np.mean(X_imputed, axis=0)
    cov_computed = np.cov(X_imputed, rowvar=False)
    mean_error = np.linalg.norm(means - mean_computed)
    cov_error = np.linalg.norm(cov_matrix - cov_computed)
    cov_error = cov_error/len(cov_matrix)**2
    end = time()
    return mean_error, cov_error, end - start



if __name__ == "__main__":

    means = np.array([50, 100, 25, 75])
    cov_matrix = np.array([
            [10,  5,  2,  3],
            [ 5, 20,  4,  6],
            [ 2,  4, 15,  1],
            [ 3,  6,  1, 12]
        ])
    
    df = pd.read_csv("synthetic_multivariate\\tests\\MNAR_missing_30pct.csv", skiprows=0)
    data_array = df.to_numpy()
    mean_err, mean_cov_err, mean_time = mean_imputation(data_array, means, cov_matrix)
    median_err, median_cov_err, median_time = median_imputation(data_array, means, cov_matrix)
    mode_err, mode_cov_err, mode_time = mode_imputation(data_array, means, cov_matrix)
    knn_err, knn_cov_err, knn_time = knn_imputation(data_array, means, cov_matrix, n_neighbors=10)
    mice_err, mice_cov_err, mice_time = mice_imputation(data_array, means, cov_matrix, iterations=5)

