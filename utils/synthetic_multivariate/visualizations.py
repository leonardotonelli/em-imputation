import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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
        data.loc[:,"missingness_pct"] = round(data.loc[:,"missingness_pct"], 2)
        # data.loc[:,"error"] = round(data.loc[:,"error"], 1)
        pivot_data = data.pivot_table(
            values='error',
            index='method',
            columns='missingness_pct',
            aggfunc='mean'
        )
        
        # Reorder methods
        method_order = ['EM', 'MICE', 'KNN', 'Median', 'Mean']
        pivot_data = pivot_data.reindex(method_order)
        
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.2f',
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
    
    plt.suptitle('Error Comparison Across Missingness Mechanisms', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def load_data(filepath='simulation_results.txt'):
    """Load and prepare simulation data."""
    df = pd.read_csv(filepath, sep='\t')
    return df

def prepare_error_data(df):
    """Prepare error data in long format for seaborn."""
    error_cols = {
        'mu_error': 'EM',
        'mice_imputation_error': 'MICE',
        'knn_imputation_error': 'KNN',
        'median_imputation_error': 'Median',
        'mean_imputation_error': 'Mean',
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
        'mice_imputation_time': 'MICE',
        'knn_imputation_time': 'KNN',
        'median_imputation_time': 'Median',
        'mean_imputation_time': 'Mean',
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
    df_error = prepare_error_data(df)
    data = df_error[df_error['mechanism'] == mechanism]

    # Apply dodge
    data_dodged, x_dodged = apply_ordered_dodge(data, 'missingness_pct', 'method', dodge_width=0.3)

    fig, ax = plt.subplots(figsize=figsize)

    
    # Define color palette
    palette = {
        'EM': '#e74c3c',
        'MICE': '#3498db',
        'KNN': '#2ecc71',
        'Median': '#f39c12',
        'Mean': '#95a5a6',
    }
    
    # Define line styles
    dashes = {
        'EM': '',
        'MICE': '',
        'KNN': '',
        'Median': '',
        'Mean': '',
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
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title(f'Estimation Error by Missingness Pattern\nMechanism: {mechanism}',
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
        'MICE': '#3498db',
        'KNN': '#2ecc71',
        'Median': '#f39c12',
        'Mean': '#95a5a6',
    }
    
    # Define line styles
    dashes = {
        'EM': '',
        'MICE': '',
        'KNN': '',
        'Median': '',
        'Mean': '',
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



def plot_sigma_error_comparison(df, mechanism='MCAR', figsize=(10, 6)):
    """
    Plot covariance (sigma) estimation error comparison with confidence intervals.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Simulation results
    mechanism : str
        Missingness mechanism ('MCAR', 'MAR', or 'MNAR')
    figsize : tuple
        Figure size
    """
    cov_cols = {
        'sigma_error': 'EM',
        'mice_imputation_cov_error': 'MICE',
        'knn_imputation_cov_error': 'KNN',
        'median_imputation_cov_error': 'Median',
        'mean_imputation_cov_error': 'Mean',
    }
    
    df_cov = df.melt(
        id_vars=['mechanism', 'missingness_pct'],
        value_vars=list(cov_cols.keys()),
        var_name='method',
        value_name='cov_error'
    )
    
    df_cov['method'] = df_cov['method'].map(cov_cols)
    df_cov['missingness_pct'] = df_cov['missingness_pct'] * 100
    
    data = df_cov[df_cov['mechanism'] == mechanism]

    data_dodged, x_dodged = apply_ordered_dodge(data, 'missingness_pct', 'method', dodge_width=0.3)

    
    fig, ax = plt.subplots(figsize=figsize)
    
    palette = {
        'EM': '#e74c3c',
        'MICE': '#3498db',
        'KNN': '#2ecc71',
        'Median': '#f39c12',
        'Mean': '#95a5a6',
    }
    
    dashes = {
        'EM': '',
        'MICE': '',
        'KNN': '',
        'Median': '',
        'Mean': '',
    }
    
    sns.lineplot(
        data=data_dodged,
        x=x_dodged,
        y='cov_error',
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
    ax.set_ylabel('Covariance Error', fontsize=12)
    ax.set_title(f'Covariance Estimation Error\nMechanism: {mechanism}', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Method', loc='best', frameon=True)
    
    plt.tight_layout()
    return fig

def plot_sample_size_cov_error(df, mechanism='MCAR', figsize=(10, 6)):
    """
    Plot mean error vs sample size comparison.
    
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
        'sigma_error': 'EM',
        'mice_imputation_cov_error': 'MICE',
        'knn_imputation_cov_error': 'KNN',
        'median_imputation_cov_error': 'Median',
        'mean_imputation_cov_error': 'Mean',
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
        'MICE': '#3498db',
        'KNN': '#2ecc71',
        'Median': '#f39c12',
        'Mean': '#95a5a6',
    }
    
    dashes = {
        'EM': '',
        'MICE': '',
        'KNN': '',
        'Median': '',
        'Mean': '',
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
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title(f'Mean Error vs Sample Size\nMechanism: {mechanism}',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Method', loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_sample_size_error(df, mechanism='MCAR', figsize=(10, 6)):
    """
    Plot mean error vs sample size comparison.
    
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
        'mu_error': 'EM',
        'mice_imputation_error': 'MICE',
        'knn_imputation_error': 'KNN',
        'median_imputation_error': 'Median',
        'mean_imputation_error': 'Mean',
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
        'MICE': '#3498db',
        'KNN': '#2ecc71',
        'Median': '#f39c12',
        'Mean': '#95a5a6',
    }
    
    dashes = {
        'EM': '',
        'MICE': '',
        'KNN': '',
        'Median': '',
        'Mean': '',
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
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title(f'Mean Error vs Sample Size\nMechanism: {mechanism}',
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
        'mice_imputation_time': 'MICE',
        'knn_imputation_time': 'KNN',
        'median_imputation_time': 'Median',
        'mean_imputation_time': 'Mean',
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
        'MICE': '#3498db',
        'KNN': '#2ecc71',
        'Median': '#f39c12',
        'Mean': '#95a5a6',
    }
    
    dashes = {
        'EM': '',
        'MICE': '',
        'KNN': '',
        'Median': '',
        'Mean': '',
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

def create_full_report(df, output_folder='tests'):
    """
    Generate a complete set of visualization reports for missing data imputation analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the results with columns for mechanism, method, 
        missing_rate, mean_error, computation_time, etc.
    
    Returns:
    --------
    None
        Saves all visualization files to the 'tests' directory
    """
    os.makedirs(output_folder, exist_ok=True)
    for mechanism in ['MCAR', 'MAR', 'MNAR']:
        os.makedirs(os.path.join(output_folder, mechanism), exist_ok=True)

    # Individual mechanism plots for mean error
    for mechanism in ['MCAR', 'MAR', 'MNAR']:
        fig_error = plot_error_comparison(df, mechanism=mechanism)
        fig_error.savefig(f'{output_folder}\\{mechanism}\\error_comparison_{mechanism}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig_time = plot_time_comparison(df, mechanism=mechanism)
        fig_time.savefig(f'{output_folder}\\{mechanism}\\time_comparison_{mechanism}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig_sigma = plot_sigma_error_comparison(df, mechanism=mechanism)
        fig_sigma.savefig(f'{output_folder}\\{mechanism}\\sigma_error_{mechanism}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Sample size plots
        fig_sample_error = plot_sample_size_error(df, mechanism=mechanism)
        fig_sample_error.savefig(f'{output_folder}\\{mechanism}\\sample_size_error_{mechanism}_gmm.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Sample size plots
        fig_sample_cov_error = plot_sample_size_cov_error(df, mechanism=mechanism)
        fig_sample_cov_error.savefig(f'{output_folder}\\{mechanism}\\sample_size_cov_error_{mechanism}_gmm.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig_sample_time = plot_sample_size_time(df, mechanism=mechanism)
        fig_sample_time.savefig(f'{output_folder}\\{mechanism}\\sample_size_time_{mechanism}_gmm.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Heatmap
    fig_heatmap = plot_error_heatmap(df)
    fig_heatmap.savefig(f'{output_folder}\\error_heatmap_all.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All visualizations generated successfully!")
    print("\nGenerated files:")
    print("- error_comparison_[MCAR/MAR/MNAR].png")
    print("- time_comparison_[MCAR/MAR/MNAR].png")
    print("- sigma_error_[MCAR/MAR/MNAR].png")
    print("- sample_size_error_[MCAR/MAR/MNAR]_gmm.png")
    print("- sample_size_cov_error_[MCAR/MAR/MNAR]_gmm.png")
    print("- sample_size_time_[MCAR/MAR/MNAR]_gmm.png")
    print("- error_heatmap_all.png")

def plot_sample_size_error_filtered(df, mechanism='MCAR', missingness_pct=0.3, figsize=(10, 6)):
    """
    Plot mean error vs sample size for a specific missingness percentage and mechanism.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Simulation results
    mechanism : str
        Missingness mechanism ('MCAR', 'MAR', or 'MNAR')
    missingness_pct : float
        Missingness percentage to filter for (e.g., 0.3 for 30%)
    figsize : tuple
        Figure size
    """
    # Prepare error data
    error_cols = {
        'mu_error': 'EM',
        'mice_imputation_error': 'MICE',
        'knn_imputation_error': 'KNN',
        'median_imputation_error': 'Median',
        'mean_imputation_error': 'Mean',
    }
    
    df_long = df.melt(
        id_vars=['mechanism', 'missingness_pct', 'n_samples'],
        value_vars=list(error_cols.keys()),
        var_name='method',
        value_name='error'
    )
    
    df_long['method'] = df_long['method'].map(error_cols)
    
    # Filter by mechanism and missingness percentage
    data = df_long[
        (df_long['mechanism'] == mechanism) & 
        (df_long['missingness_pct'] == missingness_pct)
    ].copy()
    
    # Apply dodge
    data_dodged, x_dodged = apply_ordered_dodge(data, 'n_samples', 'method', dodge_width=15)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    palette = {
        'EM': '#e74c3c',
        'MICE': '#3498db',
        'KNN': '#2ecc71',
        'Median': '#f39c12',
        'Mean': '#95a5a6',
    }
    
    dashes = {
        'EM': '',
        'MICE': '',
        'KNN': '',
        'Median': '',
        'Mean': '',
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
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title(f'Mean Error vs Sample Size\nMechanism: {mechanism}, Missingness: {missingness_pct*100:.0f}%',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Method', loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_time_per_iteration(df, 
                           mechanism=None, 
                           mean_idx=None, 
                           cov_idx=None,
                           n_samples=None,
                           missingness_pct=None,
                           x_axis='n_samples',  # or 'missingness_pct'
                           figsize=(10, 6)):
    """
    Plot time per iteration with flexible filtering options.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Simulation results
    mechanism : str or None
        Missingness mechanism ('MCAR', 'MAR', or 'MNAR'). If None, plots all mechanisms.
    mean_idx : int or None
        Mean configuration index to filter. If None, includes all.
    cov_idx : int or None
        Covariance configuration index to filter. If None, includes all.
    n_samples : int or None
        Sample size to filter. If None, includes all (used when x_axis='missingness_pct').
    missingness_pct : float or None
        Missingness percentage to filter (e.g., 0.3 for 30%). If None, includes all 
        (used when x_axis='n_samples').
    x_axis : str
        What to plot on x-axis: 'n_samples' or 'missingness_pct'
    figsize : tuple
        Figure size
    """
    # Start with full dataframe
    data = df.copy()
    
    # Apply filters
    filter_conditions = []
    filter_description = []
    
    if mechanism is not None:
        data = data[data['mechanism'] == mechanism]
        filter_description.append(f"Mechanism: {mechanism}")
    
    if mean_idx is not None:
        data = data[data['mean_idx'] == mean_idx]
        filter_description.append(f"Mean Config: {mean_idx}")
    
    if cov_idx is not None:
        data = data[data['cov_idx'] == cov_idx]
        filter_description.append(f"Cov Config: {cov_idx}")
    
    if n_samples is not None:
        data = data[data['n_samples'] == n_samples]
        filter_description.append(f"Sample Size: {n_samples}")
    
    if missingness_pct is not None:
        data = data[data['missingness_pct'] == missingness_pct]
        filter_description.append(f"Missingness: {missingness_pct*100:.0f}%")
    
    # Check if we have data
    if len(data) == 0:
        print("No data matching the specified filters!")
        return None
    
    # Prepare data based on x_axis choice
    fig, ax = plt.subplots(figsize=figsize)
    
    if x_axis == 'n_samples':
        # Plot time vs sample size
        if mechanism is None:
            # If mechanism not specified, use it as hue
            palette = {'MCAR': '#e74c3c', 'MAR': '#3498db', 'MNAR': '#2ecc71'}
            data_dodged, x_dodged = apply_ordered_dodge(data, 'n_samples', 'mechanism', dodge_width=15)
            
            sns.lineplot(
                data=data_dodged,
                x=x_dodged,
                y='time_per_iteration',
                hue='mechanism',
                style='mechanism',
                markers=True,
                palette=palette,
                err_style='bars',
                errorbar='se',
                ax=ax,
                linewidth=2,
                markersize=8,
                err_kws={"capsize": 3}
            )
        else:
            # Single line plot
            sns.lineplot(
                data=data,
                x='n_samples',
                y='time_per_iteration',
                marker='o',
                color='#e74c3c',
                err_style='bars',
                errorbar='se',
                ax=ax,
                linewidth=2,
                markersize=8,
                err_kws={"capsize": 3}
            )
        
        ax.set_xlabel('Sample Size', fontsize=12)
        
    elif x_axis == 'missingness_pct':
        # Plot time vs missingness percentage
        data['missingness_pct_plot'] = data['missingness_pct'] * 100
        
        if mechanism is None:
            # If mechanism not specified, use it as hue
            palette = {'MCAR': '#e74c3c', 'MAR': '#3498db', 'MNAR': '#2ecc71'}
            data_dodged, x_dodged = apply_ordered_dodge(
                data, 'missingness_pct_plot', 'mechanism', dodge_width=0.3
            )
            
            sns.lineplot(
                data=data_dodged,
                x=x_dodged,
                y='time_per_iteration',
                hue='mechanism',
                style='mechanism',
                markers=True,
                palette=palette,
                err_style='bars',
                errorbar='se',
                ax=ax,
                linewidth=2,
                markersize=8,
                err_kws={"capsize": 3}
            )
        else:
            # Single line plot
            sns.lineplot(
                data=data,
                x='missingness_pct_plot',
                y='time_per_iteration',
                marker='o',
                color='#e74c3c',
                err_style='bars',
                errorbar='se',
                ax=ax,
                linewidth=2,
                markersize=8,
                err_kws={"capsize": 3}
            )
        
        ax.set_xlabel('Missingness Percentage (%)', fontsize=12)
    
    ax.set_ylabel('Time per Iteration (seconds)', fontsize=12)
    
    # Build title
    title = 'EM Algorithm: Time per Iteration'
    if filter_description:
        title += '\n' + ', '.join(filter_description)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if mechanism is None:
        ax.legend(title='Mechanism', loc='best', frameon=True)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# Main execution
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("tests\\simulation_results.csv")    
    # Generate all visualizations
    
    create_full_report(df, output_folder='tests')