import numpy as np
from utils.synthetic_multivariate.simulation_study import simulation_study_multivariate
from utils.synthetic_multivariate.visualizations import create_full_report
from utils.synthetic_GMM.simulation_study_GMM import simulation_study_gmm
from utils.synthetic_GMM.visualizations_GMM import create_full_report_gmm
import pandas as pd

### SYNTHETIC MULTIVARIATE GAUSSIAN DATASET SIMULATION STUDY ###

# Define simulation parameters
MEANS = [
    [50, 100, 25, 75],
    [10, 20, 30, 40],
]
COVARIANCES = [
    np.array([
        [10,  5,  2,  3],
        [ 5, 20,  4,  6],
        [ 2,  4, 15,  1],
        [ 3,  6,  1, 12]
    ]),
    np.array([
        [5, 1, 0, 0],
        [1, 5, 1, 0],
        [0, 1, 5, 1],
        [0, 0, 1, 5]
    ]),
]
N_SAMPLES = [500, 1000]
PERCENTAGES_MISSINGNESS = [0.1, 0.2, 0.3]

# # Run simulation study (if not in results folder)
# results = simulation_study_multivariate(
#     result_path="results\\synthetic_multivariate",
#     data_path="data\\synthetic_multivariate",
#     means_to_test=MEANS,
#     cov_to_test=COVARIANCES,
#     n_samples_to_test=N_SAMPLES,
#     percentages_to_test=PERCENTAGES_MISSINGNESS,
#     max_iter=200,
#     tol=1e-5,
#     random_state=42
# )

# If the results folder already exists, load the results
results = pd.read_csv("results\\synthetic_multivariate\\simulation_results.csv")

# Visualize results
create_full_report(results, output_folder='plots\\synthetic_multivariate')


### SYNTHETIC GMM DATASET SIMULATION STUDY ###

# Define GMM simulation parameters
GMM_CONFIGS = [
    {
        'n_components': 3,
        'means': [
            np.array([0, 0]),
            np.array([5, 5]),
            np.array([0, 5])
        ],
        'cov_matrices': [
            np.array([[1.0, 0.3], [0.3, 1.0]]),
            np.array([[1.5, -0.5], [-0.5, 1.5]]),
            np.array([[1.0, 0.0], [0.0, 1.0]])
        ],
        'weights': [0.3, 0.4, 0.3]
    },
    {
        'n_components': 2,
        'means': [
            np.array([0, 0]),
            np.array([4, 4])
        ],
        'cov_matrices': [
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0]])
        ],
        'weights': [0.5, 0.5]
    }
]

N_SAMPLES_GMM = [500, 1000]
PERCENTAGES_CLASS_MISSINGNESS = [0.1, 0.3, 0.5]

# Run GMM simulation study (if not in results folder)
results_gmm = simulation_study_gmm(
    result_path="results\\synthetic_gmm",
    data_path="data\\synthetic_gmm",
    gmm_configs=GMM_CONFIGS,
    n_samples_to_test=N_SAMPLES_GMM,
    percentages_to_test=PERCENTAGES_CLASS_MISSINGNESS,
    max_iter=200,
    tol=1e-5,
    random_state=42
)

# If the results folder already exists, load the results
# results_gmm = pd.read_csv("results\\synthetic_gmm\\simulation_results_gmm.csv")

# Visualize GMM results
create_full_report_gmm(results_gmm, output_folder='plots\\synthetic_gmm')

print("\n" + "="*80)
print("ALL SIMULATION STUDIES COMPLETED")
print("="*80)
print("\nMultivariate Gaussian results: plots\\synthetic_multivariate")
print("GMM results: plots\\synthetic_gmm")