import os
import numpy as np
import pandas as pd
from typing import List, Tuple
from pathlib import Path


def generate_gmm_data(
    n_samples: int,
    n_components: int,
    means: List[np.ndarray],
    cov_matrices: List[np.ndarray],
    weights: List[float],
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate data from a Gaussian Mixture Model."""
    np.random.seed(random_state)
    
    if len(means) != n_components or len(cov_matrices) != n_components or len(weights) != n_components:
        raise ValueError("Number of means, covariances, and weights must match n_components")
    
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("Weights must sum to 1")
    
    # Sample component assignments
    component_assignments = np.random.choice(n_components, size=n_samples, p=weights)
    
    # Generate data points
    n_features = len(means[0])
    data = np.zeros((n_samples, n_features))
    
    for i in range(n_samples):
        component = component_assignments[i]
        data[i] = np.random.multivariate_normal(means[component], cov_matrices[component])
    
    df = pd.DataFrame(data, columns=[f'x{i+1}' for i in range(n_features)])
    
    return df, component_assignments


def inject_class_missingness(
    data: pd.DataFrame,
    class_labels: np.ndarray,
    mechanism: str,
    missingness_percentage: float,
    random_state: int = None
) -> pd.DataFrame:
    """Inject missingness into class variable only."""
    if random_state is not None:
        np.random.seed(random_state)
    
    if mechanism not in ["MCAR", "MAR", "MNAR", "LATENT"]:
        raise ValueError("mechanism must be 'MCAR', 'MAR', 'MNAR', or 'LATENT'")
    
    if not 0 <= missingness_percentage <= 1:
        raise ValueError("missingness_percentage must be between 0 and 1")
    
    df_result = data.copy()
    df_result['class'] = class_labels.copy()
    
    if mechanism == "LATENT":
        # Fully latent - no class variable included
        df_result["class"] = np.nan
    elif mechanism == "MCAR":
        df_result = _apply_mcar_class(df_result, missingness_percentage)
    elif mechanism == "MAR":
        df_result = _apply_mar_class(df_result, missingness_percentage)
    elif mechanism == "MNAR":
        df_result = _apply_mnar_class(df_result, missingness_percentage)
    
    return df_result


def _apply_mcar_class(df: pd.DataFrame, miss_pct: float) -> pd.DataFrame:
    """MCAR: Random missingness in class variable."""
    n_rows = len(df)
    n_missing = int(n_rows * miss_pct)
    missing_indices = np.random.choice(n_rows, size=n_missing, replace=False)
    df.loc[missing_indices, 'class'] = np.nan
    return df


def _apply_mar_class(df: pd.DataFrame, miss_pct: float) -> pd.DataFrame:
    """MAR: Missingness depends on observed features."""
    feature_cols = [col for col in df.columns if col != 'class']
    cond_col = np.random.choice(feature_cols)
    
    values = df[cond_col].values
    quantile = np.quantile(values, q=miss_pct)
    missing_indices = df[cond_col] <= quantile
    df.loc[missing_indices, 'class'] = np.nan
    
    return df


def _apply_mnar_class(df: pd.DataFrame, miss_pct: float) -> pd.DataFrame:
    """MNAR: Missingness depends on class variable itself."""
    # Make certain classes more likely to be missing
    class_probs = df['class'].value_counts(normalize=True).sort_index()
    
    # Classes with lower frequency are more likely to be missing
    missing_probs = 1 - class_probs.values
    missing_probs = missing_probs / missing_probs.sum()
    
    n_missing = int(len(df) * miss_pct)
    
    # Sample classes to make missing (weighted by inverse frequency)
    classes_to_hide = np.random.choice(
        class_probs.index, 
        size=n_missing, 
        p=missing_probs,
        replace=True
    )
    
    # For each class, randomly select observations to make missing
    for cls in np.unique(classes_to_hide):
        class_indices = df[df['class'] == cls].index
        n_to_hide = np.sum(classes_to_hide == cls)
        n_to_hide = min(n_to_hide, len(class_indices))
        
        if n_to_hide > 0:
            hide_indices = np.random.choice(class_indices, size=n_to_hide, replace=False)
            df.loc[hide_indices, 'class'] = np.nan
    
    return df


def generate_gmm_missing_files(
    n_samples: int,
    n_components: int,
    means: List[np.ndarray],
    cov_matrices: List[np.ndarray],
    weights: List[float],
    missingness_percentages: List[float],
    output_folder: str = "synthetic_gmm",
    random_state: int = 42
) -> None:
    """Generate GMM datasets with different missingness patterns on class variable."""
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Generate complete GMM data
    data, class_labels = generate_gmm_data(
        n_samples=n_samples,
        n_components=n_components,
        means=means,
        cov_matrices=cov_matrices,
        weights=weights,
        random_state=random_state
    )
    
    # Save complete data
    df_complete = data.copy()
    df_complete['class'] = class_labels
    complete_path = os.path.join(output_folder, "complete_data.csv")
    df_complete.to_csv(complete_path, index=False)
    print(f"Saved: {complete_path}")
    
    mechanisms = ["MCAR", "MAR", "MNAR", "LATENT"]
    
    for mechanism in mechanisms:
        if mechanism == "LATENT":
            # Only one latent version (no class variable at all)
            df_missing = inject_class_missingness(
                data=data,
                class_labels=class_labels,
                mechanism=mechanism,
                missingness_percentage=0,
                random_state=random_state
            )
            filename = f"{mechanism}_no_class.csv"
            filepath = os.path.join(output_folder, filename)
            df_missing.to_csv(filepath, index=False)
            print(f"Saved: {filepath}")
        else:
            for miss_pct in missingness_percentages:
                df_missing = inject_class_missingness(
                    data=data,
                    class_labels=class_labels,
                    mechanism=mechanism,
                    missingness_percentage=miss_pct,
                    random_state=random_state
                )
                filename = f"{mechanism}_missing_{int(miss_pct*100)}pct.csv"
                filepath = os.path.join(output_folder, filename)
                df_missing.to_csv(filepath, index=False)
                print(f"Saved: {filepath}")


if __name__ == "__main__":
    # 3-component GMM with 2D features
    n_components = 3
    means = [
        np.array([0, 0]),
        np.array([5, 5]),
        np.array([0, 5])
    ]
    cov_matrices = [
        np.array([[1.0, 0.3], [0.3, 1.0]]),
        np.array([[1.5, -0.5], [-0.5, 1.5]]),
        np.array([[1.0, 0.0], [0.0, 1.0]])
    ]
    weights = [0.3, 0.4, 0.3]
    
    generate_gmm_missing_files(
        n_samples=1000,
        n_components=n_components,
        means=means,
        cov_matrices=cov_matrices,
        weights=weights,
        missingness_percentages=[0.1, 0.2, 0.3, 0.5],
        output_folder="utils\\synthetic_GMM\\tests\\datasets_generated",
        random_state=42
    )