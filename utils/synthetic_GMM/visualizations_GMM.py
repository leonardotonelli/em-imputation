import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_convergence_comparison_gmm(results_df, save_path=None):
    """
    Plot convergence behavior across different mechanisms and missingness levels for GMM.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Filter out LATENT for some plots since it has 100% missingness
    results_no_latent = results_df[results_df['mechanism'] != 'LATENT']
    
    # 1. Iterations by mechanism and missingness
    ax1 = axes[0, 0]
    pivot_iter = results_no_latent.pivot_table(
        values='num_iterations',
        index='missingness_pct',
        columns='mechanism',
        aggfunc='mean'
    )
    pivot_iter.plot(kind='bar', ax=ax1, width=0.7)
    ax1.set_title('Average Iterations to Convergence', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Class Missingness Percentage', fontsize=12)
    ax1.set_ylabel('Number of Iterations', fontsize=12)
    ax1.legend(title='Mechanism', fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 2. Convergence time by mechanism (including LATENT)
    ax2 = axes[0, 1]
    mech_order = ['MCAR', 'MAR', 'MNAR', 'LATENT']
    mech_data = [results_df[results_df['mechanism'] == m]['convergence_time'].mean() 
                 for m in mech_order if m in results_df['mechanism'].values]
    mech_labels = [m for m in mech_order if m in results_df['mechanism'].values]
    
    ax2.bar(mech_labels, mech_data, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax2.set_title('Average Convergence Time by Mechanism', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Mechanism', fontsize=12)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.grid(alpha=0.3)
    
    # 3. Iterations vs sample size
    ax3 = axes[1, 0]
    for mechanism in results_df['mechanism'].unique():
        mech_data = results_df[results_df['mechanism'] == mechanism]
        grouped = mech_data.groupby('n_samples')['num_iterations'].mean()
        ax3.plot(grouped.index, grouped.values, marker='o', linewidth=2, 
                markersize=8, label=mechanism)
    ax3.set_title('Iterations vs Sample Size', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Sample Size', fontsize=12)
    ax3.set_ylabel('Number of Iterations', fontsize=12)
    ax3.legend(title='Mechanism', fontsize=10)
    ax3.grid(alpha=0.3)
    
    # 4. Box plot of iterations by mechanism
    ax4 = axes[1, 1]
    results_df.boxplot(column='num_iterations', by='mechanism', ax=ax4)
    ax4.set_title('Distribution of Iterations by Mechanism', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Mechanism', fontsize=12)
    ax4.set_ylabel('Number of Iterations', fontsize=12)
    plt.suptitle('')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_gmm_estimation_errors(results_df, save_path=None):
    """
    Visualize estimation errors for GMM parameters (pi, mu, Sigma).
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    results_no_latent = results_df[results_df['mechanism'] != 'LATENT']
    
    # 1. Mixing coefficient error by missingness
    ax1 = axes[0, 0]
    pivot_pi = results_no_latent.pivot_table(
        values='pi_error',
        index='missingness_pct',
        columns='mechanism',
        aggfunc='mean'
    )
    pivot_pi.plot(kind='bar', ax=ax1, width=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title('Mixing Coefficient Error', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Class Missingness Percentage', fontsize=12)
    ax1.set_ylabel('L1 Distance', fontsize=12)
    ax1.legend(title='Mechanism', fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 2. Mean error by missingness
    ax2 = axes[0, 1]
    pivot_mu = results_no_latent.pivot_table(
        values='mu_error',
        index='missingness_pct',
        columns='mechanism',
        aggfunc='mean'
    )
    pivot_mu.plot(kind='bar', ax=ax2, width=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_title('Component Mean Error', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Class Missingness Percentage', fontsize=12)
    ax2.set_ylabel('Average Euclidean Distance', fontsize=12)
    ax2.legend(title='Mechanism', fontsize=10)
    ax2.grid(alpha=0.3)
    
    # 3. Covariance error by missingness
    ax3 = axes[0, 2]
    pivot_sigma = results_no_latent.pivot_table(
        values='sigma_error',
        index='missingness_pct',
        columns='mechanism',
        aggfunc='mean'
    )
    pivot_sigma.plot(kind='bar', ax=ax3, width=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax3.set_title('Component Covariance Error', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Class Missingness Percentage', fontsize=12)
    ax3.set_ylabel('Average Frobenius Norm', fontsize=12)
    ax3.legend(title='Mechanism', fontsize=10)
    ax3.grid(alpha=0.3)
    
    # 4. Compare all errors including LATENT
    ax4 = axes[1, 0]
    mechanisms = results_df['mechanism'].unique()
    x = np.arange(len(mechanisms))
    width = 0.25
    
    pi_errors = [results_df[results_df['mechanism'] == m]['pi_error'].mean() for m in mechanisms]
    mu_errors = [results_df[results_df['mechanism'] == m]['mu_error'].mean() for m in mechanisms]
    sigma_errors = [results_df[results_df['mechanism'] == m]['sigma_error'].mean() for m in mechanisms]
    
    ax4.bar(x - width, pi_errors, width, label='Pi Error', alpha=0.8, color='#1f77b4')
    ax4.bar(x, mu_errors, width, label='Mu Error', alpha=0.8, color='#ff7f0e')
    ax4.bar(x + width, sigma_errors, width, label='Sigma Error', alpha=0.8, color='#2ca02c')
    ax4.set_title('Average Errors by Mechanism', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Mechanism', fontsize=12)
    ax4.set_ylabel('Error', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(mechanisms)
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)
    
    # 5. Error correlation scatter
    ax5 = axes[1, 1]
    for mechanism in mechanisms:
        mech_data = results_df[results_df['mechanism'] == mechanism]
        ax5.scatter(mech_data['mu_error'], mech_data['sigma_error'], 
                   alpha=0.6, s=100, label=mechanism)
    ax5.set_title('Mean Error vs Covariance Error', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Mean Error', fontsize=12)
    ax5.set_ylabel('Covariance Error', fontsize=12)
    ax5.legend(title='Mechanism', fontsize=10)
    ax5.grid(alpha=0.3)
    
    # 6. LATENT vs supervised comparison
    ax6 = axes[1, 2]
    latent_errors = results_df[results_df['mechanism'] == 'LATENT'][['pi_error', 'mu_error', 'sigma_error']].mean()
    supervised_errors = results_df[results_df['mechanism'] != 'LATENT'][['pi_error', 'mu_error', 'sigma_error']].mean()
    
    comparison = pd.DataFrame({
        'Fully Latent': latent_errors,
        'Semi-Supervised (avg)': supervised_errors
    })
    comparison.plot(kind='bar', ax=ax6, color=['#d62728', '#9467bd'], alpha=0.8)
    ax6.set_title('Fully Latent vs Semi-Supervised', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Error Type', fontsize=12)
    ax6.set_ylabel('Average Error', fontsize=12)
    ax6.set_xticklabels(['Pi', 'Mu', 'Sigma'], rotation=0)
    ax6.legend(fontsize=10)
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_mechanism_comparison_gmm(results_df, save_path=None):
    """
    Compare GMM algorithm performance across different class missingness mechanisms.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    mechanisms = results_df['mechanism'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. Heatmap: Mean error by mechanism and missingness
    ax1 = axes[0, 0]
    results_no_latent = results_df[results_df['mechanism'] != 'LATENT']
    heatmap_data = results_no_latent.pivot_table(
        values='mu_error',
        index='mechanism',
        columns='missingness_pct',
        aggfunc='mean'
    )
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax1, 
                cbar_kws={'label': 'Mean Error'})
    ax1.set_title('Mean Error Heatmap (Class Missingness)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Class Missingness Percentage', fontsize=12)
    ax1.set_ylabel('Mechanism', fontsize=12)
    
    # 2. Performance degradation with class missingness
    ax2 = axes[0, 1]
    for mechanism in results_no_latent['mechanism'].unique():
        mech_data = results_no_latent[results_no_latent['mechanism'] == mechanism]
        grouped = mech_data.groupby('missingness_pct')['mu_error'].mean()
        ax2.plot(grouped.index * 100, grouped.values, 
                marker='o', linewidth=2, markersize=8, label=mechanism)
    ax2.set_title('Mean Error vs Class Missingness', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Class Missingness (%)', fontsize=12)
    ax2.set_ylabel('Mean Error', fontsize=12)
    ax2.legend(title='Mechanism', fontsize=10)
    ax2.grid(alpha=0.3)
    
    # 3. Component-wise error comparison
    ax3 = axes[1, 0]
    summary_stats = results_df.groupby('mechanism').agg({
        'pi_error': 'mean',
        'mu_error': 'mean',
        'sigma_error': 'mean',
        'num_iterations': 'mean'
    })
    
    x = np.arange(len(mechanisms))
    width = 0.2
    
    ax3.bar(x - 1.5*width, summary_stats['pi_error'], width, 
           label='Pi Error', color='#1f77b4', alpha=0.8)
    ax3.bar(x - 0.5*width, summary_stats['mu_error'], width, 
           label='Mu Error', color='#ff7f0e', alpha=0.8)
    ax3.bar(x + 0.5*width, summary_stats['sigma_error'], width, 
           label='Sigma Error', color='#2ca02c', alpha=0.8)
    ax3.bar(x + 1.5*width, summary_stats['num_iterations'] / 100, width, 
           label='Iterations (÷100)', color='#d62728', alpha=0.8)
    
    ax3.set_title('Overall Performance by Mechanism', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Mechanism', fontsize=12)
    ax3.set_ylabel('Metric Value', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(mechanisms)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    
    # 4. Box plot comparison
    ax4 = axes[1, 1]
    data_to_plot = []
    positions = []
    labels = []
    for i, mech in enumerate(mechanisms):
        mech_data = results_df[results_df['mechanism'] == mech]['mu_error']
        data_to_plot.append(mech_data)
        positions.append(i)
        labels.append(mech)
    
    parts = ax4.violinplot(data_to_plot, positions=positions, showmeans=True, 
                           showmedians=True, widths=0.7)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.7)
    ax4.set_title('Mean Error Distribution by Mechanism', fontsize=14, fontweight='bold')
    ax4.set_xticks(positions)
    ax4.set_xticklabels(labels)
    ax4.set_ylabel('Mean Error', fontsize=12)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sample_size_effect_gmm(results_df, save_path=None):
    """
    Analyze how sample size affects GMM estimation accuracy.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sample_sizes = sorted(results_df['n_samples'].unique())
    
    # 1. Mean error vs sample size by mechanism
    ax1 = axes[0, 0]
    for mechanism in results_df['mechanism'].unique():
        mech_data = results_df[results_df['mechanism'] == mechanism]
        grouped = mech_data.groupby('n_samples')['mu_error'].mean()
        ax1.plot(grouped.index, grouped.values, marker='o', 
                linewidth=2, markersize=10, label=mechanism)
    ax1.set_title('Mean Error vs Sample Size', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sample Size', fontsize=12)
    ax1.set_ylabel('Mean Error', fontsize=12)
    ax1.legend(title='Mechanism', fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 2. Mixing coefficient error vs sample size
    ax2 = axes[0, 1]
    for mechanism in results_df['mechanism'].unique():
        mech_data = results_df[results_df['mechanism'] == mechanism]
        grouped = mech_data.groupby('n_samples')['pi_error'].mean()
        ax2.plot(grouped.index, grouped.values, marker='s', 
                linewidth=2, markersize=10, label=mechanism)
    ax2.set_title('Mixing Coefficient Error vs Sample Size', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sample Size', fontsize=12)
    ax2.set_ylabel('Pi Error', fontsize=12)
    ax2.legend(title='Mechanism', fontsize=10)
    ax2.grid(alpha=0.3)
    
    # 3. Convergence speed vs sample size
    ax3 = axes[1, 0]
    for n_samples in sample_sizes:
        subset = results_df[results_df['n_samples'] == n_samples]
        # Exclude LATENT for this plot
        subset = subset[subset['mechanism'] != 'LATENT']
        if len(subset) > 0:
            grouped = subset.groupby('missingness_pct')['num_iterations'].mean()
            ax3.plot(grouped.index * 100, grouped.values, marker='o', 
                    linewidth=2, markersize=8, label=f'N={n_samples}')
    ax3.set_title('Iterations vs Class Missingness (by Sample Size)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Class Missingness (%)', fontsize=12)
    ax3.set_ylabel('Number of Iterations', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    
    # 4. All parameters error vs sample size
    ax4 = axes[1, 1]
    for param in ['pi_error', 'mu_error', 'sigma_error']:
        errors = []
        for n in sample_sizes:
            error = results_df[results_df['n_samples'] == n][param].mean()
            errors.append(error)
        label_map = {'pi_error': 'Pi', 'mu_error': 'Mu', 'sigma_error': 'Sigma'}
        ax4.plot(sample_sizes, errors, marker='o', linewidth=2, 
                markersize=10, label=label_map[param])
    ax4.set_title('All Parameter Errors vs Sample Size', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Sample Size', fontsize=12)
    ax4.set_ylabel('Average Error', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_gmm_parameter_recovery(results_df, idx=0, save_path=None):
    """
    Visualize GMM parameter recovery for a specific simulation.
    Shows true vs estimated mixing coefficients, component means, and covariances.
    """
    row = results_df.iloc[idx]
    
    # Parse parameters
    true_pi = eval(row['true_pi'])
    est_pi = eval(row['estimated_pi'])
    true_means = eval(row['true_means'])
    est_means = eval(row['estimated_means'])
    
    n_components = len(true_pi)
    n_features = len(true_means[0])
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, n_components, hspace=0.4, wspace=0.3)
    
    # Title
    mechanism_str = row['mechanism']
    miss_str = 'Fully Latent' if mechanism_str == 'LATENT' else f"Miss={row['missingness_pct']*100:.0f}%"
    fig.suptitle(f"GMM Parameter Recovery - {mechanism_str} | "
                f"N={row['n_samples']} | {miss_str} | "
                f"K={n_components} | Iterations={row['num_iterations']}", 
                fontsize=16, fontweight='bold')
    
    # Row 1: Mixing coefficients
    ax_pi = fig.add_subplot(gs[0, :])
    x_pos = np.arange(n_components)
    width = 0.35
    ax_pi.bar(x_pos - width/2, true_pi, width, label='True', alpha=0.8, color='#1f77b4')
    ax_pi.bar(x_pos + width/2, est_pi, width, label='Estimated', alpha=0.8, color='#ff7f0e')
    ax_pi.set_title('Mixing Coefficients Comparison', fontsize=13, fontweight='bold')
    ax_pi.set_xlabel('Component', fontsize=11)
    ax_pi.set_ylabel('Weight', fontsize=11)
    ax_pi.set_xticks(x_pos)
    ax_pi.set_xticklabels([f'K={i+1}' for i in range(n_components)])
    ax_pi.set_ylim([0, 1])
    ax_pi.axhline(y=1/n_components, color='gray', linestyle='--', alpha=0.5, label='Uniform')
    ax_pi.legend()
    ax_pi.grid(alpha=0.3)
    
    # Row 2: Component means
    for k in range(n_components):
        ax = fig.add_subplot(gs[1, k])
        
        true_mu_k = np.array(true_means[k])
        est_mu_k = np.array(est_means[k])
        
        if n_features == 2:
            # 2D scatter plot
            ax.scatter(true_mu_k[0], true_mu_k[1], s=200, marker='*', 
                      color='#1f77b4', label='True', zorder=3, edgecolors='black', linewidths=2)
            ax.scatter(est_mu_k[0], est_mu_k[1], s=200, marker='o', 
                      color='#ff7f0e', label='Estimated', zorder=3, edgecolors='black', linewidths=2)
            ax.plot([true_mu_k[0], est_mu_k[0]], [true_mu_k[1], est_mu_k[1]], 
                   'r--', linewidth=2, alpha=0.5)
            ax.set_xlabel('Feature 1', fontsize=10)
            ax.set_ylabel('Feature 2', fontsize=10)
            ax.legend(fontsize=9)
        else:
            # Bar plot for higher dimensions
            x_pos = np.arange(n_features)
            width = 0.35
            ax.bar(x_pos - width/2, true_mu_k, width, label='True', alpha=0.8, color='#1f77b4')
            ax.bar(x_pos + width/2, est_mu_k, width, label='Estimated', alpha=0.8, color='#ff7f0e')
            ax.set_xlabel('Feature', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.set_xticks(x_pos)
            ax.legend(fontsize=9)
        
        ax.set_title(f'Component {k+1} Mean', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
    
    # Row 3: Component covariances (show as correlation if available)
    # Note: We don't have individual covariances in the results, so we'll show error info instead
    for k in range(n_components):
        ax = fig.add_subplot(gs[2, k])
        
        # Show summary statistics for this component
        info_text = f"Component {k+1} Info:\n"
        info_text += f"True π: {true_pi[k]:.3f}\n"
        info_text += f"Est π: {est_pi[k]:.3f}\n"
        info_text += f"True μ: {np.array(true_means[k])}\n"
        info_text += f"Est μ: {np.array(est_means[k])}\n"
        info_text += f"μ Error: {np.linalg.norm(np.array(true_means[k]) - np.array(est_means[k])):.4f}"
        
        ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='center', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')
        ax.set_title(f'Component {k+1} Summary', fontsize=11, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_class_missingness_pattern(data_path, filename, save_path=None):
    """
    Analyze and visualize the pattern of missing class labels in GMM data.
    """
    df = pd.read_csv(f"{data_path}/{filename}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    has_class = 'class' in df.columns
    
    if has_class:
        # 1. Class distribution (observed vs total)
        ax1 = axes[0, 0]
        class_col = df['class']
        observed_classes = class_col.dropna()
        
        if len(observed_classes) > 0:
            class_counts = observed_classes.value_counts().sort_index()
            class_counts.plot(kind='bar', ax=ax1, color='#2ca02c', alpha=0.7)
            ax1.set_title('Observed Class Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Class', fontsize=12)
            ax1.set_ylabel('Count', fontsize=12)
            ax1.grid(alpha=0.3)
            
            missing_count = class_col.isna().sum()
            total_count = len(class_col)
            ax1.text(0.95, 0.95, f'Missing: {missing_count}/{total_count} ({missing_count/total_count*100:.1f}%)',
                    transform=ax1.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
        else:
            ax1.text(0.5, 0.5, 'No observed class labels', transform=ax1.transAxes,
                    ha='center', va='center', fontsize=14)
            ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
        
        # 2. Feature space colored by observed classes
        ax2 = axes[0, 1]
        feature_cols = [col for col in df.columns if col != 'class']
        if len(feature_cols) >= 2:
            observed_mask = ~class_col.isna()
            missing_mask = class_col.isna()
            
            # Plot observations with known class
            if observed_mask.sum() > 0:
                for cls in observed_classes.unique():
                    cls_mask = (class_col == cls) & observed_mask
                    ax2.scatter(df.loc[cls_mask, feature_cols[0]], 
                              df.loc[cls_mask, feature_cols[1]],
                              label=f'Class {int(cls)}', alpha=0.6, s=50)
            
            # Plot observations with missing class
            if missing_mask.sum() > 0:
                ax2.scatter(df.loc[missing_mask, feature_cols[0]], 
                          df.loc[missing_mask, feature_cols[1]],
                          label='Missing Class', alpha=0.3, s=50, 
                          color='gray', marker='x')
            
            ax2.set_title('Feature Space (Class Labels)', fontsize=14, fontweight='bold')
            ax2.set_xlabel(feature_cols[0], fontsize=12)
            ax2.set_ylabel(feature_cols[1], fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(alpha=0.3)
    else:
        # Fully latent case
        ax1 = axes[0, 0]
        ax1.text(0.5, 0.5, 'Fully Latent\n(No class variable)', 
                transform=ax1.transAxes, ha='center', va='center', 
                fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        ax2 = axes[0, 1]
        feature_cols = [col for col in df.columns]
        if len(feature_cols) >= 2:
            ax2.scatter(df[feature_cols[0]], df[feature_cols[1]], 
                       alpha=0.5, s=50, color='gray')
            ax2.set_title('Feature Space (No Labels)', fontsize=14, fontweight='bold')
            ax2.set_xlabel(feature_cols[0], fontsize=12)
            ax2.set_ylabel(feature_cols[1], fontsize=12)
            ax2.grid(alpha=0.3)
    
    # 3. Feature distributions
    ax3 = axes[1, 0]
    feature_cols = [col for col in df.columns if col != 'class']
    for col in feature_cols[:min(4, len(feature_cols))]:
        ax3.hist(df[col].dropna(), alpha=0.5, bins=30, label=col)
    ax3.set_title('Feature Distributions', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Value', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    
# 4. Data summary
    ax4 = axes[1, 1]
    summary_text = f"Dataset Summary:\n\n"
    summary_text += f"Total samples: {len(df)}\n"
    summary_text += f"Features: {len(feature_cols)}\n\n"
    
    if has_class:
        missing_pct = class_col.isna().sum() / len(class_col) * 100
        summary_text += f"Class variable: Present\n"
        summary_text += f"Class missingness: {missing_pct:.1f}%\n"
        if len(observed_classes) > 0:
            summary_text += f"Unique classes: {len(observed_classes.unique())}\n"
            summary_text += f"Class balance:\n"
            for cls, count in class_counts.items():
                summary_text += f"  Class {int(cls)}: {count} ({count/len(observed_classes)*100:.1f}%)\n"
    else:
        summary_text += f"Class variable: Fully latent\n"
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax4.axis('off')
    ax4.set_title('Summary Statistics', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n=== GMM DATA SUMMARY ===")
    print(f"Total samples: {len(df)}")
    print(f"Features: {feature_cols}")
    if has_class:
        print(f"Class missingness: {class_col.isna().sum()}/{len(class_col)} ({class_col.isna().sum()/len(class_col)*100:.2f}%)")
    else:
        print("Class variable: Fully latent")


def create_full_report_gmm(results_df, output_folder='visualization_results_gmm'):
    """
    Generate a complete visual report of the GMM simulation study.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    print("Generating comprehensive GMM visualization report...")
    
    print("\n1. Convergence analysis...")
    plot_convergence_comparison_gmm(results_df, 
                                   save_path=f"{output_folder}/01_convergence_comparison.png")
    
    print("2. GMM estimation errors...")
    plot_gmm_estimation_errors(results_df, 
                              save_path=f"{output_folder}/02_gmm_estimation_errors.png")
    
    print("3. Mechanism comparison...")
    plot_mechanism_comparison_gmm(results_df, 
                                 save_path=f"{output_folder}/03_mechanism_comparison.png")
    
    print("4. Sample size effects...")
    plot_sample_size_effect_gmm(results_df, 
                               save_path=f"{output_folder}/04_sample_size_effect.png")
    
    print("5. Parameter recovery examples...")
    # Show best and worst cases, plus a LATENT case
    best_idx = results_df[results_df['mechanism'] != 'LATENT']['mu_error'].idxmin()
    worst_idx = results_df[results_df['mechanism'] != 'LATENT']['mu_error'].idxmax()
    latent_idx = results_df[results_df['mechanism'] == 'LATENT'].index[0] if 'LATENT' in results_df['mechanism'].values else None
    
    plot_gmm_parameter_recovery(results_df, idx=best_idx, 
                               save_path=f"{output_folder}/05_best_recovery.png")
    plot_gmm_parameter_recovery(results_df, idx=worst_idx, 
                               save_path=f"{output_folder}/06_worst_recovery.png")
    if latent_idx is not None:
        plot_gmm_parameter_recovery(results_df, idx=latent_idx, 
                                   save_path=f"{output_folder}/07_latent_recovery.png")
    
    print(f"\n✓ All visualizations saved to '{output_folder}/'")


if __name__ == "__main__":
    
    # Load simulation results
    results_df = pd.read_csv("tests/simulation_results_gmm.csv")
    
    print(f"Loaded {len(results_df)} simulation results")
    print(f"Mechanisms: {results_df['mechanism'].unique()}")
    print(f"Sample sizes: {sorted(results_df['n_samples'].unique())}")
    print(f"Missingness levels: {sorted(results_df['missingness_pct'].unique())}")
    print(f"Number of components: {sorted(results_df['n_components'].unique())}")
    
    # Generate all visualizations
    create_full_report_gmm(results_df, output_folder='tests')
    
    # Example: Analyze specific dataset missingness pattern
    print("\n6. Analyzing specific dataset patterns...")
    plot_class_missingness_pattern(
        data_path="tests",
        filename="missing_MAR_config0_n1000_miss30.csv",
        save_path="tests/08_missingness_pattern_example.png"
    )
    
    print("\n✓ Visualization script completed successfully!")