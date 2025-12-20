import os
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path


def inject_missingness(
    data: pd.DataFrame,
    missingness_percentages: List[float],
    target_column_percentage: float,
    mechanism: str,
    random_state: int = None
) -> List[pd.DataFrame]:
    if random_state is not None:
        np.random.seed(random_state)
    
    if not 0 < target_column_percentage <= 1:
        raise ValueError("target_column_percentage must be between 0 and 1")
    
    if mechanism not in ["MCAR", "MAR", "MNAR"]:
        raise ValueError("mechanism must be 'MCAR', 'MAR', or 'MNAR'")
    
    n_target_cols = max(1, int(len(data.columns) * target_column_percentage))
    target_cols = np.random.choice(data.columns, size=n_target_cols, replace=False)
    observed_cols = [col for col in data.columns if col not in target_cols]
    
    results = []
    
    for miss_pct in missingness_percentages:
        if not 0 <= miss_pct <= 1:
            raise ValueError(f"Missingness percentage {miss_pct} must be between 0 and 1")
        
        n_total_cols = len(data.columns)

        while miss_pct * n_total_cols / len(target_cols) > 1:
            n_target_cols += 1
            target_cols = np.random.choice(data.columns, size=n_target_cols, replace=False)
            observed_cols = [col for col in data.columns if col not in target_cols]

        per_column_miss_pct = miss_pct * n_total_cols / len(target_cols)
        
        df_missing = data.copy()
        
        if mechanism == "MCAR":
            df_missing = _apply_mcar(df_missing, target_cols, per_column_miss_pct)
        elif mechanism == "MAR":
            if len(observed_cols) == 0:
                raise ValueError("MAR requires at least one observed column")
            df_missing = _apply_mar(df_missing, target_cols, observed_cols, per_column_miss_pct)
        elif mechanism == "MNAR":
            df_missing = _apply_mnar(df_missing, target_cols, per_column_miss_pct)
        
        results.append(df_missing)
    
    return results


def _apply_mcar(df: pd.DataFrame, target_cols: np.ndarray, per_column_miss_pct: float) -> pd.DataFrame:
    n_rows = len(df)
    
    for col in target_cols:
        n_missing = int(n_rows * per_column_miss_pct)
        missing_indices = np.random.choice(n_rows, size=n_missing, replace=False)
        df.loc[missing_indices, col] = np.nan
    
    return df


def _apply_mar(df: pd.DataFrame, target_cols: np.ndarray, observed_cols: List[str], 
               per_column_miss_pct: float) -> pd.DataFrame:
    for col in target_cols:
        cond_col = np.random.choice(observed_cols)
        values = df[cond_col].values
        quantile = np.quantile(values, q=per_column_miss_pct)
        missing_indices = df[cond_col] <= quantile
        df.loc[missing_indices, col] = np.nan
    
    return df


def _apply_mnar(df: pd.DataFrame, target_cols: np.ndarray, per_column_miss_pct: float) -> pd.DataFrame:
    for col in target_cols:
        values = df[col].values
        quantile = np.quantile(values, q=per_column_miss_pct)
        missing_indices = df[col] <= quantile
        df.loc[missing_indices, col] = np.nan
    
    return df


def generate_missing_data_files(
    data: pd.DataFrame,
    missingness_percentages: List[float],
    target_column_percentage: float = 0.5,
    output_folder: str = "synthetic_multivariate",
    random_state: int = 42
) -> None:
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    mechanisms = ["MCAR", "MAR", "MNAR"]
    
    for mechanism in mechanisms:
        results = inject_missingness(
            data=data,
            missingness_percentages=missingness_percentages,
            target_column_percentage=target_column_percentage,
            mechanism=mechanism,
            random_state=random_state
        )
        
        for pct, df_missing in zip(missingness_percentages, results):
            filename = f"{mechanism}_missing_{int(pct*100)}pct.csv"
            filepath = os.path.join(output_folder, filename)
            df_missing.to_csv(filepath, index=False)
            print(f"Saved: {filepath}")


def generate_multivariate_gaussian(
    n_samples: int,
    means: List[float],
    cov_matrix: np.ndarray,
    column_names: List[str] = None,
    random_state: int = 42
) -> pd.DataFrame:
    np.random.seed(random_state)
    
    if len(means) != cov_matrix.shape[0] or len(means) != cov_matrix.shape[1]:
        raise ValueError("Length of means must match dimensions of covariance matrix")
    
    data = np.random.multivariate_normal(means, cov_matrix, size=n_samples)
    
    if column_names is None:
        column_names = [f'var_{i+1}' for i in range(len(means))]
    
    return pd.DataFrame(data, columns=column_names)


if __name__ == "__main__":
    means = [50, 100, 25, 75]
    cov_matrix = np.array([
        [10,  5,  2,  3],
        [ 5, 20,  4,  6],
        [ 2,  4, 15,  1],
        [ 3,  6,  1, 12]
    ])
    column_names = ['age', 'income', 'score', 'rating']
    
    df = generate_multivariate_gaussian(
        n_samples=1000,
        means=means,
        cov_matrix=cov_matrix,
        column_names=column_names,
        random_state=42
    )
    
    generate_missing_data_files(
        data=df,
        missingness_percentages=[0.1, 0.2, 0.3],
        target_column_percentage=0.5,
        output_folder="utils\\synthetic_multivariate\\tests\\datasets_generated",
        random_state=42
    )