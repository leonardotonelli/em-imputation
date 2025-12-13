import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set style for academic plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'

def plot_error_heatmap(df, figsize=(12, 4)):
    """
    Create heatmap showing average error across all mechanisms.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Simulation results
    figsize : tuple
        Figure size
    """
    df_error = prepare_error_data(df)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    cmaps = ['RdYlGn_r', 'viridis', 'magma']
    
    for idx, mechanism in enumerate(['MCAR', 'MAR', 'MNAR']):
        data = df_error[df_error['mechanism'] == mechanism]
        pivot_data = data.pivot_table(
            values='error',
            index='method',
            columns='missingness_pct',
            aggfunc='mean'
        )
        
        # Reorder methods
        method_order = ['EM', 'RF', 'KNN', 'Mode']
        pivot_data = pivot_data.reindex(method_order)
        
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.3f',
            cmap=cmaps[idx],
            ax=axes[idx],
            cbar_kws={'label': 'Mean Error'},
            vmin=0,
            vmax=pivot_data.values.max()
        )
        
        axes[idx].set_title(mechanism, fontweight='bold', fontsize=12)
        axes[idx].set_xlabel('Missingness %', fontsize=10)
        
        if idx == 0:
            axes[idx].set_ylabel('Method', fontsize=10)
        else:
            axes[idx].set_ylabel('')
    
    plt.suptitle('Proportion Error Comparison Across Missingness Mechanisms', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def load_data(filepath='simulation_results_gmm.txt'):
    """Load and prepare simulation data."""
    df = pd.read_csv(filepath, sep='\t')
    return df

def prepare_error_data(df):
    """Prepare error data in long format for seaborn."""
    error_cols = {
        'pi_error': 'EM',
        'rf_imputation_prop_error': 'RF',
        'knn_imputation_prop_error': 'KNN',
        'mode_imputation_prop_error': 'Mode'
    }
    
    df_long = df.melt(
        id_vars=['mechanism', 'missingness_pct'],
        value_vars=list(error_cols.keys()),
        var_name='method',
        value_name='error'
    )
    
    df_long['method'] = df_long['method'].map(error_cols)
    df_long['missingness_pct'] = df_long['missingness_pct'] * 100
    
    return df_long

def apply_ordered_dodge(data, x_col, hue_col, dodge_width=0.8):
    """
    Apply dodge so that for each x value, the hue groups are ordered left→right.
    """
    data = data.copy()
    data[x_col + "_dodged"] = data[x_col]

    # For each x value, compute dodge offsets by sorted order of hue
    for x_val, group in data.groupby(x_col):
        methods = sorted(group[hue_col].unique())  # alphabetical = consistent order
        n = len(methods)
        offsets = {m: (i - (n - 1) / 2) * dodge_width for i, m in enumerate(methods)}

        idx = data[x_col] == x_val
        data.loc[idx, x_col + "_dodged"] += data.loc[idx, hue_col].map(offsets)

    return data, x_col + "_dodged"


def prepare_time_data(df):
    """Prepare time data in long format for seaborn."""
    time_cols = {
        'convergence_time': 'EM',
        'rf_imputation_time': 'RF',
        'knn_imputation_time': 'KNN',
        'mode_imputation_time': 'Mode'
    }
    
    df_long = df.melt(
        id_vars=['mechanism', 'missingness_pct'],
        value_vars=list(time_cols.keys()),
        var_name='method',
        value_name='time'
    )
    
    df_long['method'] = df_long['method'].map(time_cols)
    df_long['missingness_pct'] = df_long['missingness_pct'] * 100
    
    return df_long

def plot_error_comparison(df, mechanism='MCAR', figsize=(10, 6)):
    """
    Plot proportion error comparison with confidence intervals.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Simulation results
    mechanism : str
        Missingness mechanism ('MCAR', 'MAR', or 'MNAR')
    figsize : tuple
        Figure size
    """
    df_error = prepare_error_data(df)
    data = df_error[df_error['mechanism'] == mechanism]

    # Apply dodge
    data_dodged, x_dodged = apply_ordered_dodge(data, 'missingness_pct', 'method', dodge_width=0.3)

    fig, ax = plt.subplots(figsize=figsize)
    
    # Define color palette
    palette = {
        'EM': '#e74c3c',
        'RF': '#3498db',
        'KNN': '#2ecc71',
        'Mode': '#f39c12'
    }
    
    # Define line styles
    dashes = {
        'EM': '',
        'RF': '',
        'KNN': '',
        'Mode': ''
    }
    
    sns.lineplot(
        data=data_dodged,
        x=x_dodged,
        y='error',
        hue='method',
        style='method',
        markers=True,
        dashes=dashes,
        palette=palette,
        err_style='bars',
        errorbar=('se'),
        ax=ax,
        linewidth=2,
        markersize=8,
        err_kws={"capsize": 3}
    )

    ax.set_xlabel('Missingness Percentage (%)', fontsize=12)
    ax.set_ylabel('Proportion Error', fontsize=12)
    ax.set_title(f'Proportion Estimation Error by Missingness Pattern\nMechanism: {mechanism}',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Method', loc='best', frameon=True)

    plt.tight_layout()
    return fig

def plot_time_comparison(df, mechanism='MCAR', figsize=(10, 6)):
    """
    Plot computation time comparison with confidence intervals.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Simulation results
    mechanism : str
        Missingness mechanism ('MCAR', 'MAR', or 'MNAR')
    figsize : tuple
        Figure size
    """
    df_time = prepare_time_data(df)
    data = df_time[df_time['mechanism'] == mechanism]

    data_dodged, x_dodged = apply_ordered_dodge(data, 'missingness_pct', 'method', dodge_width=0.3)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define color palette
    palette = {
        'EM': '#e74c3c',
        'RF': '#3498db',
        'KNN': '#2ecc71',
        'Mode': '#f39c12'
    }
    
    # Define line styles
    dashes = {
        'EM': '',
        'RF': '',
        'KNN': '',
        'Mode': ''
    }
    
    sns.lineplot(
        data=data_dodged,
        x=x_dodged,
        y='time',
        hue='method',
        style='method',
        markers=True,
        dashes=dashes,
        palette=palette,
        err_style='bars',
        errorbar=('se'),
        ax=ax,
        linewidth=2,
        markersize=8,
        err_kws={"capsize": 3}
    )
    
    ax.set_xlabel('Missingness Percentage (%)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(f'Computation Time by Missingness Pattern\nMechanism: {mechanism}', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Method', loc='best', frameon=True)
    
    plt.tight_layout()
    return fig


def load_data(filepath='simulation_results_gmm.txt'):
    """Load and prepare simulation data."""
    df = pd.read_csv(filepath, sep='\t')
    return df

def prepare_error_data(df):
    """Prepare error data in long format for seaborn."""
    error_cols = {
        'pi_error': 'EM',
        'rf_imputation_prop_error': 'RF',
        'knn_imputation_prop_error': 'KNN',
        'mode_imputation_prop_error': 'Mode'
    }
    
    df_long = df.melt(
        id_vars=['mechanism', 'missingness_pct'],
        value_vars=list(error_cols.keys()),
        var_name='method',
        value_name='error'
    )
    
    df_long['method'] = df_long['method'].map(error_cols)
    df_long['missingness_pct'] = df_long['missingness_pct'] * 100
    
    return df_long

def apply_ordered_dodge(data, x_col, hue_col, dodge_width=0.8):
    """
    Apply dodge so that for each x value, the hue groups are ordered left→right.
    """
    data = data.copy()
    data[x_col + "_dodged"] = data[x_col]

    # For each x value, compute dodge offsets by sorted order of hue
    for x_val, group in data.groupby(x_col):
        methods = sorted(group[hue_col].unique())  # alphabetical = consistent order
        n = len(methods)
        offsets = {m: (i - (n - 1) / 2) * dodge_width for i, m in enumerate(methods)}

        idx = data[x_col] == x_val
        data.loc[idx, x_col + "_dodged"] += data.loc[idx, hue_col].map(offsets)

    return data, x_col + "_dodged"


def prepare_time_data(df):
    """Prepare time data in long format for seaborn."""
    time_cols = {
        'convergence_time': 'EM',
        'rf_imputation_time': 'RF',
        'knn_imputation_time': 'KNN',
        'mode_imputation_time': 'Mode'
    }
    
    df_long = df.melt(
        id_vars=['mechanism', 'missingness_pct'],
        value_vars=list(time_cols.keys()),
        var_name='method',
        value_name='time'
    )
    
    df_long['method'] = df_long['method'].map(time_cols)
    df_long['missingness_pct'] = df_long['missingness_pct'] * 100
    
    return df_long


def plot_sample_size_error(df, mechanism='MCAR', figsize=(10, 6)):
    """
    Plot proportion error vs sample size comparison.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Simulation results
    mechanism : str
        Missingness mechanism ('MCAR', 'MAR', or 'MNAR')
    figsize : tuple
        Figure size
    """
    # Prepare error data
    error_cols = {
        'pi_error': 'EM',
        'rf_imputation_prop_error': 'RF',
        'knn_imputation_prop_error': 'KNN',
        'mode_imputation_prop_error': 'Mode'
    }
    
    df_long = df.melt(
        id_vars=['mechanism', 'missingness_pct', 'n_samples'],
        value_vars=list(error_cols.keys()),
        var_name='method',
        value_name='error'
    )
    
    df_long['method'] = df_long['method'].map(error_cols)
    
    # Filter by mechanism
    data = df_long[df_long['mechanism'] == mechanism].copy()
    
    # Apply dodge
    data_dodged, x_dodged = apply_ordered_dodge(data, 'n_samples', 'method', dodge_width=15)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    palette = {
        'EM': '#e74c3c',
        'RF': '#3498db',
        'KNN': '#2ecc71',
        'Mode': '#f39c12'
    }
    
    dashes = {
        'EM': '',
        'RF': '',
        'KNN': '',
        'Mode': ''
    }
    
    sns.lineplot(
        data=data_dodged,
        x=x_dodged,
        y='error',
        hue='method',
        style='method',
        markers=True,
        dashes=dashes,
        palette=palette,
        err_style='bars',
        errorbar='se',
        ax=ax,
        linewidth=2,
        markersize=8,
        err_kws={"capsize": 3}
    )
    
    ax.set_xlabel('Sample Size', fontsize=12)
    ax.set_ylabel('Proportion Error', fontsize=12)
    ax.set_title(f'Proportion Error vs Sample Size\nMechanism: {mechanism}',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Method', loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_sample_size_time(df, mechanism='MCAR', figsize=(10, 6)):
    """
    Plot computation time vs sample size comparison.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Simulation results
    mechanism : str
        Missingness mechanism ('MCAR', 'MAR', or 'MNAR')
    figsize : tuple
        Figure size
    """
    # Prepare time data
    time_cols = {
        'convergence_time': 'EM',
        'rf_imputation_time': 'RF',
        'knn_imputation_time': 'KNN',
        'mode_imputation_time': 'Mode'
    }
    
    df_long = df.melt(
        id_vars=['mechanism', 'missingness_pct', 'n_samples'],
        value_vars=list(time_cols.keys()),
        var_name='method',
        value_name='time'
    )
    
    df_long['method'] = df_long['method'].map(time_cols)
    
    # Filter by mechanism
    data = df_long[df_long['mechanism'] == mechanism].copy()
    
    # Apply dodge
    data_dodged, x_dodged = apply_ordered_dodge(data, 'n_samples', 'method', dodge_width=15)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    palette = {
        'EM': '#e74c3c',
        'RF': '#3498db',
        'KNN': '#2ecc71',
        'Mode': '#f39c12'
    }
    
    dashes = {
        'EM': '',
        'RF': '',
        'KNN': '',
        'Mode': ''
    }
    
    sns.lineplot(
        data=data_dodged,
        x=x_dodged,
        y='time',
        hue='method',
        style='method',
        markers=True,
        dashes=dashes,
        palette=palette,
        err_style='bars',
        errorbar='se',
        ax=ax,
        linewidth=2,
        markersize=8,
        err_kws={"capsize": 3}
    )
    
    ax.set_xlabel('Sample Size', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(f'Computation Time vs Sample Size\nMechanism: {mechanism}',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Method', loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    # Use log scale for y-axis if range is large
    time_range = data['time'].max() / (data['time'].min() + 1e-10)
    if time_range > 100:
        ax.set_yscale('log')
    
    plt.tight_layout()
    return fig

def create_full_report_gmm(df, output_folder='tests'):
    """
    Generate a complete set of visualization reports for GMM missing data imputation analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the results with columns for mechanism, method, 
        missing_rate, proportion_error, computation_time, etc.
    output_folder : str, optional
        Base directory for saving visualizations (default: 'tests')
    
    Returns:
    --------
    None
        Saves all visualization files to the specified directory structure
    """
    import os
    
    # Create output folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    for mechanism in ['MCAR', 'MAR', 'MNAR']:
        os.makedirs(os.path.join(output_folder, mechanism), exist_ok=True)
    
    # Individual mechanism plots for proportion error
    for mechanism in ['MCAR', 'MAR', 'MNAR']:
        fig_error = plot_error_comparison(df, mechanism=mechanism)
        fig_error.savefig(f'{output_folder}\\{mechanism}\\error_comparison_{mechanism}_gmm.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig_time = plot_time_comparison(df, mechanism=mechanism)
        fig_time.savefig(f'{output_folder}\\{mechanism}\\time_comparison_{mechanism}_gmm.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig_sample_time = plot_sample_size_time(df, mechanism=mechanism)
        fig_sample_time.savefig(f'{output_folder}\\{mechanism}\\sample_time_comparison_{mechanism}_gmm.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig_sample_error = plot_sample_size_error(df, mechanism=mechanism)
        fig_sample_error.savefig(f'{output_folder}\\{mechanism}\\sample_error_comparison_{mechanism}_gmm.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Heatmap
    fig_heatmap = plot_error_heatmap(df)
    fig_heatmap.savefig(f'{output_folder}\\error_heatmap_all_gmm.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All visualizations generated successfully!")
    print(f"\nGenerated files in '{output_folder}':")
    print("- [MCAR/MAR/MNAR]/error_comparison_[mechanism]_gmm.png")
    print("- [MCAR/MAR/MNAR]/time_comparison_[mechanism]_gmm.png")
    print("- [MCAR/MAR/MNAR]/sample_time_comparison_[mechanism]_gmm.png")
    print("- [MCAR/MAR/MNAR]/sample_error_comparison_[mechanism]_gmm.png")
    print("- error_heatmap_all_gmm.png")

# Main execution
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("tests\\simulation_results_gmm.csv")
    
    create_full_report_gmm(df, output_folder='tests')


