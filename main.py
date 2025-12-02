import numpy as np
from utils.simulation_study import simulation_study_multivariate
from utils.visualizations import create_full_report
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

# Run simulation study (if not in results folder)
# results = simulation_study_multivariate(
#     results_path="results\\synthetic_multivariate",
#     data_path = "data\\synthetic_multivariate",
#     means_to_test=MEANS,
#     cov_to_test=COVARIANCES,
#     n_samples_to_test=N_SAMPLES,
#     percentages_to_test=PERCENTAGES_MISSINGNESS,
#     max_iter=200,
#     tol=1e-5,
#     random_state=42
# )

# if the results folder already exists, load the results
results = pd.read_csv("results\\synthetic_multivariate\\simulation_results.csv")

# visualize results
create_full_report(results, output_folder='plots\\synthetic_multivariate')