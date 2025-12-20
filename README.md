## 1. Environment Setup and Installation

- Python: use **Python 3.10 or 3.11**.
- Create an isolated environment and install pinned dependencies (example using venv):

```bash
# From repository root
python -m venv .venv
# Windows
.venv\Scripts\activate
# POSIX
# source .venv/bin/activate
pip install -r requirements.txt
```

- The repository contains `requirements.txt` with pinned versions; use `pip freeze > requirements.freeze.txt` to capture an exact environment for publication or review.

- Randomness in experiments is controlled via the `random_state` parameter available in the simulation and utility functions.


## 2. Running the Project

- `simulation_run.py` is provided to generate figures and reproduce plots without re-running the full simulation experiments; by default it imports and processes previously saved CSV results located in `results/` and writes plots to `plots/`.

- To re-run the full simulations, open `simulation_run.py` and **uncomment** the calls to `simulation_study_multivariate` and `simulation_study_gmm`. Simulations and other parameter configurations can be changed directly in `simulation_run.py`. Please be careful to change the output directories to avoid overwriting our simulation results.

- Caution: running full simulations with a reasonably large set of parameter combinations (many sample sizes, missingness rates, covariance settings, etc.) requires substantial compute time and frequently takes several hours.


## 3. Directory Structure

```text
project-luca-leo-vale_stat/
├── README.md
├── requirements.txt
├── simulation_run.py
├── writeup.tex
├── real_example.ipynb
├── additional_visualizations.ipynb
├── data/
│   ├── real_example/
│   │   ├── data.csv
│   │   ├── preprocessing.py
│   │   └── processed_data_multiclass.npz
│   ├── synthetic_multivariate/
│   │   ├── datasets_complete/
│   │   └── datasets_missingness/
│   ├── synthetic_GMM/
│   │   ├── datasets_complete/
│   │   └── datasets_missingness/
│   └── synthetic_gmm_alternative/
├── results/
│   ├── synthetic_multivariate/
│   │   └── simulation_results.csv
│   ├── synthetic_gmm/
│   │   └── simulation_results_gmm.csv
│   └── real_example/
│       ├── 10_d/
│       └── 2_d/
├── plots/
│   ├── synthetic_multivariate/
│   └── synthetic_gmm/
└── utils/
    ├── synthetic_multivariate/
    │   ├── data_generation.py
    │   ├── EM.py
    │   ├── imputations.py
    │   ├── simulation_study.py
    │   ├── visualizations.py
    │   └── tests/
    ├── synthetic_GMM/
    │   ├── data_generation_GMM.py
    │   ├── EM_GMM.py
    │   ├── imputations.py
    │   ├── simulation_study_GMM.py
    │   ├── visualizations_GMM.py
    │   └── tests/
    └── real_example/
        ├── evaluation.py
        └── tests/
```

The tree above reflects the repository layout and is consistent with the descriptive entries in this section.

## 4. Directory Detailed Description

Below is a concise, complete description of the repository layout and the purpose of each file and directory.

Top-level files
- `README.md` — this document (environment, running instructions, directory map, notebooks, and testing behavior).
- `requirements.txt` — pinned Python package versions required to run the code.
- `simulation_run.py` — primary script to load results, generate plots, and (optionally) run full experiments by calling `simulation_study_multivariate` and `simulation_study_gmm`.
- `real_example.ipynb` — notebook executing the real-world example (see Section 4).
- `additional_visualizations.ipynb` — supplementary visualization notebook (see Section 4).
- `report.tex` — LaTeX source of the project report.
- `report.pdf` — pdf project report.

Data
- `data/` — root data directory.
  - `real_example/`
    - `data.csv` — raw clinical dataset used in the real example.
    - `ground_truth.xlsx` (if present) — ground truth labels used for evaluation in the notebook.
    - `preprocessing.py` — loader and preprocessing functions for the real dataset (e.g., `load_data_binary`).
    - `processed_data_multiclass.npz` — preprocessed representation saved for reproducibility.
  - `synthetic_multivariate/` and `synthetic_gmm_alternative/` and `synthetic_GMM/` — folders that store generated datasets used in simulation studies. Each of these contains:
    - `datasets_complete/` — complete generated datasets organized by sample-size subfolders (e.g., `100_samples`, `300_samples`, ...).
    - `datasets_missingness/` — datasets with induced missingness, organized by mechanism (`MCAR`, `MAR`, `MNAR`) and by missingness percentage subfolders.
    - `synthetic_multivariate/` and other patterned subfolders mirror the same structure for different experimental conditions.

Results and plots
- `results/` — per-experiment CSV summaries produced by simulation routines.
  - `synthetic_multivariate/simulation_results.csv` — aggregated MVN experiment results.
  - `synthetic_gmm/simulation_results_gmm.csv` — aggregated GMM experiment results.
  - `real_example/` — notebook outputs for the real example (e.g., `10d` and `2d` metrics CSVs).
- `plots/` — figure outputs produced from the `results/` CSV files; organized by study (`synthetic_multivariate`, `synthetic_gmm`, `synthetic_multivariate` missingness mechanisms, etc.).

Utilities and core code
- `utils/` — contains implementation modules and small test fixtures.
  - `utils/synthetic_multivariate/`
    - `data_generation.py` — functions to generate multivariate normal datasets with specified means and covariances.
    - `EM.py` — EM algorithm implementation for multivariate Gaussian data with missing entries.
    - `imputations.py` — wrapper functions that perform imputations (mean, median, KNN, MICE, etc.) and return evaluation metrics.
    - `simulation_study.py` — high-level orchestration of MVN simulation experiments (`simulation_study_multivariate`).
    - `visualizations.py` — functions to create summary plots and reports for MVN results.
    - `tests/` — test fixtures and small reference CSVs used by the module tests.
  - `utils/synthetic_GMM/`
    - `data_generation_GMM.py` — functions to generate GMM datasets and to inject class-label missingness.
    - `EM_GMM.py` — EM implementation for semi-supervised GMM fitting.
    - `imputations.py` — simple label imputation algorithms (mode, KNN, random forest-based) and helpers.
    - `simulation_study_GMM.py` — high-level orchestration of GMM experiments (`simulation_study_gmm`).
    - `visualizations_GMM.py` — plotting and reporting utilities for GMM experiments.
    - `tests/` — test fixtures and example cases for GMM utilities.
  - `utils/real_example/`
    - `evaluation.py` — evaluation helpers used by `real_example.ipynb` (e.g., `_em_impute`, `evaluate_imputers`, `test_gmm_normality_assumptions`).
    - `tests/` — small fixtures used to validate evaluation routines.


## 5. Notebooks Description

- `additional_visualizations.ipynb`: used to create the supplementary visualizations that are included in the report; it provides flexible plotting utilities for independent exploration of simulation results.

- `real_example.ipynb`: runs the real-world example described in the report and reproduces the figures and metrics referenced in that section.


## 6. Utils and Testing Behavior

- Each module file inside `utils/` (except "real_example/evaluation.py) can be executed independently as a small test runner. When executed, a module runs tests of its most relevant functions and writes the test outputs into a `tests/` or `test_outputs/` folder located alongside that module.

- Some unit/functional tests depend on previously generated data or aggregated results (for example, visualizers often expect a `results` CSV to be present). Therefore the execution order of utils tests is important: the first test(s) in a test sequence should generate or download any required datasets so that subsequent tests can operate on those artifacts.

- For reproducible grading and review, run the modules that generate synthetic data first, then run dependent module tests and visualizers in the order they state in their docstrings or the module headers.

---
