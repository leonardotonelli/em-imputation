import numpy as np
import pandas as pd
from utils.synthetic_multivariate.simulation_study import simulation_study_multivariate
from utils.synthetic_multivariate.visualizations import create_full_report
from utils.synthetic_GMM.simulation_study_GMM import simulation_study_gmm
from utils.synthetic_GMM.visualizations_GMM import create_full_report_gmm


### SYNTHETIC MULTIVARIATE GAUSSIAN DATASET SIMULATION STUDY ###

# Define dimension and mean and various covariance matrices for MVN
DIM = 5
eye = np.eye(DIM)
zeros = [0] * DIM

cov_weak = np.full((DIM, DIM), 0.3)
np.fill_diagonal(cov_weak, 1.0)

cov_strong = np.full((DIM, DIM), 0.9)
np.fill_diagonal(cov_strong, 1.0)

cov_block = np.eye(DIM)
cov_block[0:2, 0:2] = 0.8
np.fill_diagonal(cov_block, 1.0)
cov_block[2:5, 2:5] = 0.8
np.fill_diagonal(cov_block, 1.0)

# Define parameter for Cigar-shaped covariance
cov_cigar = np.full((DIM, DIM), 0.9)
np.fill_diagonal(cov_cigar, 1.0)
cov_tight = np.eye(DIM) * 0.1

# Define simulation parameters
MEANS = [zeros]
COVARIANCES = [
    eye,
    cov_weak,
    cov_strong,
    cov_block
]

N_SAMPLES = np.arange(100, 5000, 200)
PERCENTAGES_MISSINGNESS = np.arange(0.1, 0.31, 0.05)

# Run simulation study (if not in results folder) UNCOMMENT IF YOU WANT TO RUN THE SIMULATION AGAIN, it takes a while
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

# # If the results folder already exists, load the results
results = pd.read_csv("results\\synthetic_multivariate\\simulation_results.csv")

# Visualize results
create_full_report(results, output_folder='plots\\synthetic_multivariate')




### SYNTHETIC GMM DATASET SIMULATION STUDY ###
DIM = 5
MEANS_GMM = [
    [np.zeros(DIM), np.ones(DIM)*10, np.ones(DIM)*20],
    [np.zeros(DIM), np.ones(DIM)*2, np.ones(DIM)*4]
]
COVS_GMM = [
    [eye, eye, eye],
    [eye, cov_cigar, cov_tight]
]
WEIGHTS_GMM = [
    [0.33, 0.33, 0.34],
    [0.85, 0.10, 0.05],
    [0.20, 0.20, 0.60]
]   
N_SAMPLES_GMM = np.arange(100, 5000, 200)
PERCENTAGES_CLASS_MISSINGNESS = np.arange(0.1, 0.71, 0.1)

# # Run GMM simulation study (if not in results folder) UNCOMMENT IF YOU WANT TO RUN THE SIMULATION AGAIN, it takes a while
# results_gmm = simulation_study_gmm(
#     result_path="results\\synthetic_gmm_alternative",
#     data_path="data\\synthetic_gmm_alternative",
#     means_to_test=MEANS_GMM,
#     cov_matrices_to_test=COVS_GMM,
#     weights_to_test=WEIGHTS_GMM,
#     n_samples_to_test=N_SAMPLES_GMM,
#     percentages_to_test=PERCENTAGES_CLASS_MISSINGNESS,
#     max_iter=200,
#     tol=1e-5,
#     random_state=42
# )

# If the results folder already exists, load the results
results_gmm = pd.read_csv("results\\synthetic_gmm\\simulation_results_gmm.csv")

# Visualize GMM results
create_full_report_gmm(results_gmm, output_folder='plots\\synthetic_gmm_alternative')

print("\n" + "="*80)
print("ALL SIMULATION STUDIES COMPLETED")
print("="*80)
print("\nMultivariate Gaussian results: plots\\synthetic_gmm_alternative")
print("GMM results: plots\\synthetic_gmm")