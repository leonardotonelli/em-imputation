import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SyntaxWarning)


# Set style for academic plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'

def plot_error_heatmap(df, figsize=(12, 4)):
    """
    Create heatmap showing average error across all mechanisms.
    """
    df_error = prepare_error_data(df)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    cmaps = ['RdYlGn_r', 'viridis_r', 'magma_r']
    
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
    ax.set_ylabel(r'Mean Estimation Error ($||\hat{\mu} - \mu||_2$)', fontsize=12)
    ax.set_title(f'MVN: Mean Vector Estimation Error vs Missingness Rate\nMechanism: {mechanism}',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Method', loc='best', frameon=True)

    plt.tight_layout()
    return fig

def plot_time_comparison(df, mechanism='MCAR', figsize=(10, 6)):
    """
    Plot computation time comparison with confidence intervals.
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
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_title(f'Computational Cost vs Missingness Rate\nMechanism: {mechanism}', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Method', loc='best', frameon=True)
    
    plt.tight_layout()
    return fig



def plot_sigma_error_comparison(df, mechanism='MCAR', figsize=(10, 6)):
    """
    Plot covariance (sigma) estimation error comparison with confidence intervals.
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
    ax.set_ylabel('Covariance Error (Frobenius Norm)', fontsize=12)
    ax.set_title(f'MVN: Covariance Matrix Estimation Error\nMechanism: {mechanism}', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Method', loc='best', frameon=True)
    
    plt.tight_layout()
    return fig

def plot_sample_size_cov_error(df, mechanism='MCAR', figsize=(10, 6)):
    """
    Plot mean error vs sample size comparison.
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
        # err_style='bars',
        errorbar=None,
        ax=ax,
        linewidth=2,
        markersize=8,
        # err_kws={"capsize": 3}
    )
    
    ax.set_xlabel('Sample Size', fontsize=12)
    ax.set_ylabel('Covariance Error (Frobenius Norm)', fontsize=12)
    ax.set_title(f'MVN: Covariance Matrix Estimation Error\nMechanism: {mechanism}',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Method', loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_sample_size_error(df, mechanism='MCAR', figsize=(10, 6)):
    """
    Plot mean error vs sample size comparison.
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
        y='error',        hue='method',
        style='method',
        markers=True,
        dashes=dashes,
        palette=palette,
        # err_style='bars',
        errorbar=None,
        ax=ax,
        linewidth=2,
        markersize=8,
        # err_kws={"capsize": 3}
    )
    
    ax.set_xlabel('Sample Size ($N$)', fontsize=12)
    ax.set_ylabel(r'Mean Estimation Error ($||\hat{\mu} - \mu||_2$)', fontsize=12)
    ax.set_title(f'MVN: Asymptotic Estimation Error Analysis\nMechanism: {mechanism}',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Method', loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_sample_size_time(df, mechanism='MCAR', figsize=(10, 6)):
    """
    Plot computation time vs sample size comparison.
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

def plot_method_comparison_flexible(df, 
                                   x_axis='n_samples',
                                   y_axis='time',
                                   mechanism=None,
                                   missingness_pct=None,
                                   n_samples=None,
                                   mean_idx=None,
                                   cov_idx=None,
                                   methods=None,
                                   log_scale=None,
                                   figsize=(10, 6)):
    """
    Flexible plotting function to compare different imputation methods.
    """
    # Define method column mappings based on y_axis
    if y_axis == 'time':
        method_cols = {
            'convergence_time': 'EM',
            'mice_imputation_time': 'MICE',
            'knn_imputation_time': 'KNN',
            'mode_imputation_time': 'Mode',
            'median_imputation_time': 'Median',
            'mean_imputation_time': 'Mean',
        }
    elif y_axis == 'error':
        method_cols = {
            'mu_error': 'EM',
            'mice_imputation_error': 'MICE',
            'knn_imputation_error': 'KNN',
            'mode_imputation_error': 'Mode',
            'median_imputation_error': 'Median',
            'mean_imputation_error': 'Mean',
        }
    elif y_axis == 'cov_error':
        method_cols = {
            'sigma_error': 'EM',
            'mice_imputation_cov_error': 'MICE',
            'knn_imputation_cov_error': 'KNN',
            'mode_imputation_cov_error': 'Mode',
            'median_imputation_cov_error': 'Median',
            'mean_imputation_cov_error': 'Mean',
        }
    else:
        print(f"y_axis='{y_axis}' not recognized. Use 'time', 'error', or 'cov_error'")
        return None
    
    # Filter to only existing columns
    method_cols = {k: v for k, v in method_cols.items() if k in df.columns}
    
    if not method_cols:
        print(f"No method columns found for y_axis='{y_axis}'")
        return None
    
    # Filter methods if specified
    if methods is not None:
        method_cols = {k: v for k, v in method_cols.items() if v in methods}
        if not method_cols:
            print(f"None of the specified methods {methods} found in data")
            return None
    
    # Melt data to long format
    id_vars = ['mechanism', 'missingness_pct', 'actual_missingness_pct', 'n_samples', 
               'mean_idx', 'cov_idx', 'simulation_id']
    id_vars = [col for col in id_vars if col in df.columns]
    
    df_long = df.melt(
        id_vars=id_vars,
        value_vars=list(method_cols.keys()),
        var_name='method',
        value_name='value'
    )
    
    df_long['method'] = df_long['method'].map(method_cols)
    
    # Apply filters
    data = df_long.copy()
    filter_description = []
    
    if mechanism is not None:
        if 'mechanism' in data.columns:
            data = data[data['mechanism'] == mechanism]
            filter_description.append(f"Mechanism: {mechanism}")
    
    if missingness_pct is not None:
        if 'missingness_pct' in data.columns:
            data = data[data['missingness_pct'] == missingness_pct]
            filter_description.append(f"Miss: {missingness_pct*100:.0f}%")
    
    if n_samples is not None:
        if 'n_samples' in data.columns:
            data = data[data['n_samples'] == n_samples]
            filter_description.append(f"n={n_samples}")
    
    if mean_idx is not None:
        if 'mean_idx' in data.columns:
            data = data[data['mean_idx'] == mean_idx]
            filter_description.append(f"Mean: {mean_idx}")
    
    if cov_idx is not None:
        if 'cov_idx' in data.columns:
            data = data[data['cov_idx'] == cov_idx]
            filter_description.append(f"Cov: {cov_idx}")
    
    # Check if we have data
    if len(data) == 0:
        print("No data matching the specified filters!")
        return None
    
    # Handle special transformations for x-axis
    if x_axis == 'missingness_pct':
        data['missingness_pct_plot'] = data['missingness_pct'] * 100
        x_col = 'missingness_pct_plot'
        x_label = 'Missingness Percentage (%)'
    elif x_axis == 'actual_missingness_pct':
        data['actual_missingness_pct_plot'] = data['actual_missingness_pct'] * 100
        x_col = 'actual_missingness_pct_plot'
        x_label = 'Actual Missingness Percentage (%)'
    else:
        x_col = x_axis
        x_label = x_axis.replace('_', ' ').title()
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define color palette
    palette = {
        'EM': '#e74c3c',
        'MICE': '#3498db',
        'KNN': '#2ecc71',
        'Mode': '#9b59b6',
        'Median': '#f39c12',
        'Mean': '#95a5a6',
    }
    
    # Filter palette to only included methods
    palette = {k: v for k, v in palette.items() if k in data['method'].unique()}
    
    # Apply dodge if using numeric x-axis
    if x_col in data.columns and data[x_col].dtype in ['int64', 'float64']:
        try:
            # Estimate dodge width based on x-axis range
            x_range = data[x_col].max() - data[x_col].min()
            x_unique = len(data[x_col].unique())
            dodge_width = x_range / (x_unique * 20) if x_unique > 0 else 15
            
            data_dodged, x_dodged = apply_ordered_dodge(data, x_col, 'method', dodge_width=dodge_width)
            
            sns.lineplot(
                data=data_dodged,
                x=x_dodged,
                y='value',
                hue='method',
                style='method',
                markers=True,
                palette=palette,
                err_style='bars',
                errorbar='se',
                ax=ax,
                linewidth=2,
                markersize=8,
                err_kws={"capsize": 3}
            )
        except NameError:
            # If apply_ordered_dodge not available, use standard seaborn
            sns.lineplot(
                data=data,
                x=x_col,
                y='value',
                hue='method',
                style='method',
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
        # For categorical x-axis
        sns.lineplot(
            data=data,
            x=x_col,
            y='value',
            hue='method',
            style='method',
            markers=True,
            palette=palette,
            err_style='bars',
            errorbar='se',
            ax=ax,
            linewidth=2,
            markersize=8,
            err_kws={"capsize": 3}
        )
    
    # Set labels
    ax.set_xlabel(x_label, fontsize=12)
    
    # Set appropriate y-axis label
    if y_axis == 'time':
        y_label = 'Time (seconds)'
    elif y_axis == 'error':
        y_label = 'Mean Parameter Error'
    elif y_axis == 'cov_error':
        y_label = 'Covariance Error'
    else:
        y_label = y_axis.replace('_', ' ').title()
    
    ax.set_ylabel(y_label, fontsize=12)
    
    # Build title
    title = f'{y_label} vs {x_label}'
    if filter_description:
        title += '\n' + ', '.join(filter_description)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title='Method', loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    # Handle log scale
    if log_scale is None:
        # Auto-decide based on range
        value_range = data['value'].max() / (data['value'].min() + 1e-10)
        if value_range > 100:
            ax.set_yscale('log')
    elif log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    return fig


def create_full_report(df, output_folder='tests'):
    """
    Generate a complete set of visualization reports for missing data imputation analysis.
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
        fig_sample_error.savefig(f'{output_folder}\\{mechanism}\\sample_size_error_{mechanism}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Sample size plots
        fig_sample_cov_error = plot_sample_size_cov_error(df, mechanism=mechanism)
        fig_sample_cov_error.savefig(f'{output_folder}\\{mechanism}\\sample_size_cov_error_{mechanism}.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig_sample_time = plot_sample_size_time(df, mechanism=mechanism)
        fig_sample_time.savefig(f'{output_folder}\\{mechanism}\\sample_size_time_{mechanism}.png', dpi=300, bbox_inches='tight')
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
    print("- sample_size_error_[MCAR/MAR/MNAR].png")
    print("- sample_size_cov_error_[MCAR/MAR/MNAR].png")
    print("- sample_size_time_[MCAR/MAR/MNAR].png")
    print("- error_heatmap_all.png")

def plot_sample_size_error_filtered(df, mechanism='MCAR', missingness_pct=0.3, figsize=(10, 6)):
    """
    Plot mean error vs sample size for a specific missingness percentage and mechanism.
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

def create_correlation_report(df, output_folder='plots'):
    """
    Generates specific comparison plots to analyze the impact of correlation 
    on the imputation error (Independence vs Strong Correlation).
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # --- Plot 1: Independence (Covariance Index 0) ---
    # In this scenario (Identity Matrix), EM should not perform significantly better than Mean
    fig_indep = plot_method_comparison_flexible(
        df,
        x_axis='missingness_pct',
        y_axis='error',
        mechanism='MAR',      # We compare under MAR
        cov_idx=0,            # Index 0 = Independence (from main.py config)
        methods=['EM', 'MICE', 'KNN', 'Mean'],
        figsize=(8, 6)
    )
    
    if fig_indep:
        # Overwrite title and labels for academic clarity
        ax = fig_indep.axes[0]
        
        # UPDATED TITLE: Specifies Independence and Sigma = I
        ax.set_title(r"Impact of Correlation: Independence ($\Sigma = I$)\nMechanism: MAR", 
                     fontweight='bold', fontsize=14)
        
        # UPDATED Y-LABEL: Specifies the error formula
        ax.set_ylabel(r'Mean Estimation Error ($||\hat{\mu} - \mu||_2$)', fontsize=12)
        
        # Save
        save_path = os.path.join(output_folder, 'error_independence_MAR.png')
        fig_indep.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig_indep)
        print(f"Saved: {save_path}")

    # --- Plot 2: Strong Correlation (Covariance Index 2) ---
    # In this scenario (rho=0.9), EM should significantly outperform Mean Imputation
    fig_strong = plot_method_comparison_flexible(
        df,
        x_axis='missingness_pct',
        y_axis='error',
        mechanism='MAR',
        cov_idx=2,            # Index 2 = Strong Correlation (from main.py config)
        methods=['EM', 'MICE', 'KNN', 'Mean'],
        figsize=(8, 6)
    )
    
    if fig_strong:
        # Overwrite title and labels for academic clarity
        ax = fig_strong.axes[0]
        
        # UPDATED TITLE: Specifies Rho = 0.9
        ax.set_title(r"Impact of Correlation: Strong Correlation ($\rho = 0.9$)" + "\nMechanism: MAR", 
                     fontweight='bold', fontsize=14)
        
        # UPDATED Y-LABEL
        ax.set_ylabel(r'Mean Estimation Error ($||\hat{\mu} - \mu||_2$)', fontsize=12)
        
        # Save
        save_path = os.path.join(output_folder, 'error_strong_correlation_MAR.png')
        fig_strong.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig_strong)
        print(f"Saved: {save_path}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    results_path = "utils\\synthetic_multivariate\\tests\\simulation_results\\simulation_results.csv"
    output_dir = "utils\\synthetic_multivariate\\tests\\plots"

    # 2. Generate Plots
    if os.path.exists(results_path):
        print(f"Loading data from: {results_path}")
        df = pd.read_csv(results_path)
        
        print("Generating Standard MVN Report...")
        create_full_report(df, output_folder=output_dir)
        
        print("Generating Correlation Analysis (Independence vs Strong)...")
        create_correlation_report(df, output_folder=output_dir)
        
        print(f"\nSuccess! All MVN plots are saved in: {output_dir}")
    else:
        print(f"ERROR: Could not find results file at: {results_path}")
        print("Please ensure you have run the MVN simulation in main.py first.")