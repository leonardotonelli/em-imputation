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
    """
    df_error = prepare_error_data(df)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    cmaps = ['RdYlGn_r', 'viridis_r', 'magma_r']
    # Define annotation keyword arguments to set a smaller font size
    annot_font_size = 8  # You can adjust this value
    annot_kws = {"fontsize": annot_font_size}
    
    for idx, mechanism in enumerate(['MCAR', 'MAR', 'MNAR']):
        data = df_error[df_error['mechanism'] == mechanism]
        
        # data.loc[:, 'missingness_pct'] = data.loc[:, 'missingness_pct'].astype(str)
        pivot_data = data.pivot_table(
            values='error',
            index='method',
            columns='missingness_pct',
            aggfunc='mean'
        )
        data.loc[:, 'missingness_pct'] = data.loc[:, 'missingness_pct'].astype(int)
        # Reorder methods
        method_order = ['EM', 'RF', 'KNN', 'Mode']
        pivot_data = pivot_data.reindex(method_order)
        
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.2f',
            cmap=cmaps[idx],
            ax=axes[idx],
            cbar_kws={'label': 'Mean Error'},
            vmin=0,
            vmax=pivot_data.values.max(),
            annot_kws=annot_kws
        )

        # --- NEW CODE BLOCK: Force x-axis labels to be integers ---
        # 1. Get the current labels (which are likely floats like '10.0', '20.0')
        x_labels = [label.get_text() for label in axes[idx].get_xticklabels()]
        
        # 2. Format them to remove the '.0' (e.g., '10.0' -> '10')
        integer_labels = [label.replace('.0', '') for label in x_labels]
        
        # 3. Set the new integer labels back to the axis
        axes[idx].set_xticklabels(integer_labels)
        # -----------------------------------------------------------

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

    if mechanism == 'MCAR':
        data = data[data["method"] != 'Mode']

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

    if mechanism == 'MCAR':
        data = data[data["method"] != 'Mode']
    
    # Apply dodge
    data_dodged, x_dodged = apply_ordered_dodge(data, 'n_samples', 'method', dodge_width=1)
    
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
    data_dodged, x_dodged = apply_ordered_dodge(data, 'n_samples', 'method', dodge_width=1)
    
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

def plot_sample_size_error_filtered_GMM(df, mechanism='MCAR', missingness_pct=0.3, figsize=(10, 6)):
    """ 
    Plot proportion error vs sample size for a specific missingness percentage and mechanism.
    """
    # Prepare error data
    error_cols = {
        'pi_error': 'EM',
        'rf_imputation_prop_error': 'RF',
        'knn_imputation_prop_error': 'KNN',
        # 'mode_imputation_prop_error': 'Mode'
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
    data_dodged, x_dodged = apply_ordered_dodge(data, 'n_samples', 'method', dodge_width=1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    palette = {
        'EM': '#e74c3c',
        'RF': '#3498db',
        'KNN': '#2ecc71',
        # 'Mode': '#f39c12'
    }
    
    dashes = {
        'EM': '',
        'RF': '',
        'KNN': '',
        # 'Mode': ''
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
    ax.set_title(f'Proportion Error vs Sample Size\nMechanism: {mechanism}, Missingness: {missingness_pct*100:.0f}%',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Method', loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_time_per_iteration_GMM(df, 
                                mechanism=None, 
                                config_idx=None,
                                mean_idx=None, 
                                cov_idx=None,
                                weight_idx=None,
                                n_components=None,
                                n_samples=None,
                                missingness_pct=None,
                                x_axis='n_samples',  # or 'missingness_pct'
                                figsize=(10, 6)):
    """
    Plot time per iteration for GMM with flexible filtering options.
    """
    # Start with full dataframe
    data = df.copy()
    
    # Apply filters
    filter_description = []
    
    if mechanism is not None:
        data = data[data['mechanism'] == mechanism]
        filter_description.append(f"Mechanism: {mechanism}")
    
    if config_idx is not None:
        data = data[data['config_idx'] == config_idx]
        filter_description.append(f"Config: {config_idx}")
    
    if mean_idx is not None:
        data = data[data['mean_idx'] == mean_idx]
        filter_description.append(f"Mean: {mean_idx}")
    
    if cov_idx is not None:
        data = data[data['cov_idx'] == cov_idx]
        filter_description.append(f"Cov: {cov_idx}")
    
    if weight_idx is not None:
        data = data[data['weight_idx'] == weight_idx]
        filter_description.append(f"Weight: {weight_idx}")
    
    if n_components is not None:
        data = data[data['n_components'] == n_components]
        filter_description.append(f"Components: {n_components}")
    
    if n_samples is not None:
        data = data[data['n_samples'] == n_samples]
        filter_description.append(f"n={n_samples}")
    
    if missingness_pct is not None:
        data = data[data['missingness_pct'] == missingness_pct]
        filter_description.append(f"Miss: {missingness_pct*100:.0f}%")
    
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
            legend_title = 'Mechanism'
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
            legend_title = None
        
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
            legend_title = 'Mechanism'
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
            legend_title = None
        
        ax.set_xlabel('Missingness Percentage (%)', fontsize=12)
    
    ax.set_ylabel('Time per Iteration (seconds)', fontsize=12)
    
    # Build title
    title = 'EM Algorithm: Time per Iteration'
    if filter_description:
        title += '\n' + ', '.join(filter_description)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if legend_title:
        ax.legend(title=legend_title, loc='best', frameon=True)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_GMM_flexible(df, 
                     x_axis='n_samples',
                     y_axis='time_per_iteration',
                     mechanism=None, 
                     config_idx=None,
                     mean_idx=None, 
                     cov_idx=None,
                     weight_idx=None,
                     n_components=None,
                     n_samples=None,
                     missingness_pct=None,
                     hue=None,
                     figsize=(10, 6)):
    """
    Flexible plotting function for GMM results with customizable x and y axes.
    """
    # Start with full dataframe
    data = df.copy()
    
    # Apply filters
    filter_description = []
    
    if mechanism is not None:
        data = data[data['mechanism'] == mechanism]
        filter_description.append(f"Mechanism: {mechanism}")
    
    if config_idx is not None:
        data = data[data['config_idx'] == config_idx]
        filter_description.append(f"Config: {config_idx}")
    
    if mean_idx is not None:
        data = data[data['mean_idx'] == mean_idx]
        filter_description.append(f"Mean: {mean_idx}")
    
    if cov_idx is not None:
        data = data[data['cov_idx'] == cov_idx]
        filter_description.append(f"Cov: {cov_idx}")
    
    if weight_idx is not None:
        data = data[data['weight_idx'] == weight_idx]
        filter_description.append(f"Weight: {weight_idx}")
    
    if n_components is not None:
        data = data[data['n_components'] == n_components]
        filter_description.append(f"Components: {n_components}")
    
    if n_samples is not None:
        data = data[data['n_samples'] == n_samples]
        filter_description.append(f"n={n_samples}")
    
    if missingness_pct is not None:
        data = data[data['missingness_pct'] == missingness_pct]
        filter_description.append(f"Miss: {missingness_pct*100:.0f}%")
    
    # Check if we have data
    if len(data) == 0:
        print("No data matching the specified filters!")
        return None
    
    # Create working copy for plotting
    plot_data = data.copy()
    
    # Handle special transformations for specific columns
    if x_axis == 'missingness_pct':
        plot_data['missingness_pct_plot'] = plot_data['missingness_pct'] * 100
        x_col = 'missingness_pct_plot'
        x_label = 'Missingness Percentage (%)'
    elif x_axis == 'actual_missingness_pct':
        plot_data['actual_missingness_pct_plot'] = plot_data['actual_missingness_pct'] * 100
        x_col = 'actual_missingness_pct_plot'
        x_label = 'Actual Missingness Percentage (%)'
    else:
        x_col = x_axis
        x_label = x_axis.replace('_', ' ').title()
    
    if y_axis == 'missingness_pct':
        plot_data['missingness_pct_plot_y'] = plot_data['missingness_pct'] * 100
        y_col = 'missingness_pct_plot_y'
        y_label = 'Missingness Percentage (%)'
    elif y_axis == 'actual_missingness_pct':
        plot_data['actual_missingness_pct_plot_y'] = plot_data['actual_missingness_pct'] * 100
        y_col = 'actual_missingness_pct_plot_y'
        y_label = 'Actual Missingness Percentage (%)'
    else:
        y_col = y_axis
        y_label = y_axis.replace('_', ' ').title()
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define color palette for common hue options
    palette = None
    if hue == 'mechanism':
        palette = {'MCAR': '#e74c3c', 'MAR': '#3498db', 'MNAR': '#2ecc71'}
    
    # Determine dodge width based on x_axis type
    if hue is not None and hue in plot_data.columns:
        # Estimate dodge width based on x-axis range
        x_range = plot_data[x_col].max() - plot_data[x_col].min()
        x_unique = len(plot_data[x_col].unique())
        dodge_width = x_range / (x_unique * 20) if x_unique > 0 else 15
        
        # Apply dodging if using apply_ordered_dodge function
        try:
            data_dodged, x_dodged = apply_ordered_dodge(plot_data, x_col, hue, dodge_width=dodge_width)
            sns.lineplot(
                data=data_dodged,
                x=x_dodged,
                y=y_col,
                hue=hue,
                style=hue,
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
                data=plot_data,
                x=x_col,
                y=y_col,
                hue=hue,
                style=hue,
                markers=True,
                palette=palette,
                err_style='bars',
                errorbar='se',
                ax=ax,
                linewidth=2,
                markersize=8,
                err_kws={"capsize": 3}
            )
        
        legend_title = hue.replace('_', ' ').title()
        ax.legend(title=legend_title, loc='best', frameon=True)
    else:
        # Single line plot
        sns.lineplot(
            data=plot_data,
            x=x_col,
            y=y_col,
            marker='o',
            color='#e74c3c',
            err_style='bars',
            errorbar='se',
            ax=ax,
            linewidth=2,
            markersize=8,
            err_kws={"capsize": 3}
        )
    
    # Set labels
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    
    # Build title
    title = f'{y_label} vs {x_label}'
    if filter_description:
        title += '\n' + ', '.join(filter_description)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_full_report_gmm(df, output_folder='tests'):
    """
    Generate a complete set of visualization reports for GMM missing data imputation analysis.
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
    df = pd.read_csv("utils\\synthetic_GMM\\tests\\simulation_results_gmm.csv")
    
    create_full_report_gmm(df, output_folder='utils\\synthetic_GMM\\tests')


