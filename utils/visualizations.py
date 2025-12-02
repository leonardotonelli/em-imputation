import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_convergence_comparison(results_df, save_path=None):
    """
    Plot convergence behavior across different mechanisms and missingness levels.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Iterations by mechanism and missingness
    ax1 = axes[0, 0]
    pivot_iter = results_df.pivot_table(
        values='num_iterations',
        index='missingness_pct',
        columns='mechanism',
        aggfunc='mean'
    )
    pivot_iter.plot(kind='bar', ax=ax1, width=0.7)
    ax1.set_title('Average Iterations to Convergence', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Missingness Percentage', fontsize=12)
    ax1.set_ylabel('Number of Iterations', fontsize=12)
    ax1.legend(title='Mechanism', fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 2. Convergence time by mechanism
    ax2 = axes[0, 1]
    pivot_time = results_df.pivot_table(
        values='convergence_time',
        index='missingness_pct',
        columns='mechanism',
        aggfunc='mean'
    )
    pivot_time.plot(kind='bar', ax=ax2, width=0.7)
    ax2.set_title('Average Convergence Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Missingness Percentage', fontsize=12)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.legend(title='Mechanism', fontsize=10)
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


def plot_estimation_errors(results_df, save_path=None):
    """
    Visualize estimation errors for mean and covariance parameters.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Mean error by missingness
    ax1 = axes[0, 0]
    pivot_mu = results_df.pivot_table(
        values='mu_error',
        index='missingness_pct',
        columns='mechanism',
        aggfunc='mean'
    )
    pivot_mu.plot(kind='bar', ax=ax1, width=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title('Mean Estimation Error', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Missingness Percentage', fontsize=12)
    ax1.set_ylabel('Euclidean Distance', fontsize=12)
    ax1.legend(title='Mechanism', fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 2. Covariance error by missingness
    ax2 = axes[0, 1]
    pivot_sigma = results_df.pivot_table(
        values='sigma_error',
        index='missingness_pct',
        columns='mechanism',
        aggfunc='mean'
    )
    pivot_sigma.plot(kind='bar', ax=ax2, width=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_title('Covariance Estimation Error', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Missingness Percentage', fontsize=12)
    ax2.set_ylabel('Normalized Frobenius Norm', fontsize=12)
    ax2.legend(title='Mechanism', fontsize=10)
    ax2.grid(alpha=0.3)
    
    # 3. Error comparison across sample sizes
    ax3 = axes[1, 0]
    for n_samples in sorted(results_df['n_samples'].unique()):
        subset = results_df[results_df['n_samples'] == n_samples]
        grouped = subset.groupby('missingness_pct')['mu_error'].mean()
        ax3.plot(grouped.index * 100, grouped.values, marker='o', 
                linewidth=2, markersize=8, label=f'N={n_samples}')
    ax3.set_title('Mean Error vs Missingness (by Sample Size)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Missingness (%)', fontsize=12)
    ax3.set_ylabel('Mean Error', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    
    # 4. Scatter: Mean error vs Sigma error
    ax4 = axes[1, 1]
    for mechanism in results_df['mechanism'].unique():
        mech_data = results_df[results_df['mechanism'] == mechanism]
        ax4.scatter(mech_data['mu_error'], mech_data['sigma_error'], 
                   alpha=0.6, s=100, label=mechanism)
    ax4.set_title('Mean Error vs Covariance Error', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Mean Error', fontsize=12)
    ax4.set_ylabel('Covariance Error', fontsize=12)
    ax4.legend(title='Mechanism', fontsize=10)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_mechanism_comparison(results_df, save_path=None):
    """
    Compare algorithm performance across different missingness mechanisms.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    mechanisms = results_df['mechanism'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 1. Violin plot: Error distribution by mechanism
    ax1 = axes[0, 0]
    positions = []
    data_to_plot = []
    labels = []
    for i, mech in enumerate(mechanisms):
        mech_data = results_df[results_df['mechanism'] == mech]['mu_error']
        data_to_plot.append(mech_data)
        positions.append(i)
        labels.append(mech)
    
    parts = ax1.violinplot(data_to_plot, positions=positions, showmeans=True, 
                           showmedians=True, widths=0.7)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    ax1.set_title('Mean Error Distribution by Mechanism', fontsize=14, fontweight='bold')
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Mean Error', fontsize=12)
    ax1.grid(alpha=0.3)
    
    # 2. Heatmap: Error by mechanism and missingness
    ax2 = axes[0, 1]
    heatmap_data = results_df.pivot_table(
        values='mu_error',
        index='mechanism',
        columns='missingness_pct',
        aggfunc='mean'
    )
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax2, 
                cbar_kws={'label': 'Mean Error'})
    ax2.set_title('Mean Error Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Missingness Percentage', fontsize=12)
    ax2.set_ylabel('Mechanism', fontsize=12)
    
    # 3. Line plot: Performance degradation with missingness
    ax3 = axes[1, 0]
    for mechanism in mechanisms:
        mech_data = results_df[results_df['mechanism'] == mechanism]
        grouped = mech_data.groupby('missingness_pct').agg({
            'mu_error': 'mean',
            'sigma_error': 'mean'
        })
        ax3.plot(grouped.index * 100, grouped['mu_error'], 
                marker='o', linewidth=2, markersize=8, label=mechanism)
    ax3.set_title('Performance Degradation with Missingness', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Missingness (%)', fontsize=12)
    ax3.set_ylabel('Mean Error', fontsize=12)
    ax3.legend(title='Mechanism', fontsize=10)
    ax3.grid(alpha=0.3)
    
    # 4. Bar plot: Relative performance
    ax4 = axes[1, 1]
    summary_stats = results_df.groupby('mechanism').agg({
        'mu_error': 'mean',
        'sigma_error': 'mean',
        'num_iterations': 'mean'
    })
    
    x = np.arange(len(mechanisms))
    width = 0.25
    
    ax4.bar(x - width, summary_stats['mu_error'], width, 
           label='Mean Error', color='#1f77b4', alpha=0.8)
    ax4.bar(x, summary_stats['sigma_error'] * 10, width, 
           label='Sigma Error (×10)', color='#ff7f0e', alpha=0.8)
    ax4.bar(x + width, summary_stats['num_iterations'] / 100, width, 
           label='Iterations (÷100)', color='#2ca02c', alpha=0.8)
    
    ax4.set_title('Overall Performance by Mechanism', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Mechanism', fontsize=12)
    ax4.set_ylabel('Normalized Metrics', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(mechanisms)
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sample_size_effect(results_df, save_path=None):
    """
    Analyze how sample size affects estimation accuracy.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sample_sizes = sorted(results_df['n_samples'].unique())
    
    # 1. Error vs sample size by mechanism
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
    ax1.set_xscale('log')
    
    # 2. Error reduction ratio
    ax2 = axes[0, 1]
    for mechanism in results_df['mechanism'].unique():
        mech_data = results_df[results_df['mechanism'] == mechanism]
        grouped = mech_data.groupby('n_samples')['sigma_error'].mean()
        ax2.plot(grouped.index, grouped.values, marker='s', 
                linewidth=2, markersize=10, label=mechanism)
    ax2.set_title('Covariance Error vs Sample Size', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sample Size', fontsize=12)
    ax2.set_ylabel('Covariance Error', fontsize=12)
    ax2.legend(title='Mechanism', fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xscale('log')
    
    # 3. Box plot by sample size
    ax3 = axes[1, 0]
    results_df.boxplot(column='mu_error', by='n_samples', ax=ax3)
    ax3.set_title('Error Distribution by Sample Size', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Sample Size', fontsize=12)
    ax3.set_ylabel('Mean Error', fontsize=12)
    plt.suptitle('')
    ax3.grid(alpha=0.3)
    
    # 4. Convergence efficiency
    ax4 = axes[1, 1]
    for n_samples in sample_sizes:
        subset = results_df[results_df['n_samples'] == n_samples]
        grouped = subset.groupby('missingness_pct')['num_iterations'].mean()
        ax4.plot(grouped.index * 100, grouped.values, marker='o', 
                linewidth=2, markersize=8, label=f'N={n_samples}')
    ax4.set_title('Iterations vs Missingness (by Sample Size)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Missingness (%)', fontsize=12)
    ax4.set_ylabel('Number of Iterations', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_parameter_recovery(results_df, idx=0, save_path=None):
    """
    Visualize how well the algorithm recovers true parameters.
    Shows true vs estimated means and covariances for a specific simulation.
    """
    row = results_df.iloc[idx]
    
    # Parse string representations back to arrays
    true_mu = np.array(eval(row['true_mu']))
    est_mu = np.array(eval(row['estimated_mu']))
    true_sigma = np.array(eval(row['true_sigma']))
    est_sigma = np.array(eval(row['estimated_sigma']))
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(f"Parameter Recovery - {row['mechanism']} | "
                f"N={row['n_samples']} | Miss={row['missingness_pct']*100:.0f}% | "
                f"Iterations={row['num_iterations']}", 
                fontsize=16, fontweight='bold')
    
    # 1. Mean comparison
    ax1 = fig.add_subplot(gs[0, 0])
    x_pos = np.arange(len(true_mu))
    width = 0.35
    ax1.bar(x_pos - width/2, true_mu, width, label='True', alpha=0.8, color='#1f77b4')
    ax1.bar(x_pos + width/2, est_mu, width, label='Estimated', alpha=0.8, color='#ff7f0e')
    ax1.set_title('Mean Vector Comparison', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Variable', fontsize=10)
    ax1.set_ylabel('Value', fontsize=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Var {i+1}' for i in range(len(true_mu))])
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Mean error per variable
    ax2 = fig.add_subplot(gs[0, 1])
    mean_errors = np.abs(true_mu - est_mu)
    ax2.bar(x_pos, mean_errors, color='#d62728', alpha=0.7)
    ax2.set_title('Absolute Error per Variable', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Variable', fontsize=10)
    ax2.set_ylabel('Absolute Error', fontsize=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Var {i+1}' for i in range(len(true_mu))])
    ax2.grid(alpha=0.3)
    
    # 3. True vs Estimated scatter
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(true_mu, est_mu, s=100, alpha=0.6, color='#9467bd')
    min_val = min(true_mu.min(), est_mu.min())
    max_val = max(true_mu.max(), est_mu.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Recovery')
    ax3.set_title('True vs Estimated Means', fontsize=12, fontweight='bold')
    ax3.set_xlabel('True Mean', fontsize=10)
    ax3.set_ylabel('Estimated Mean', fontsize=10)
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. True covariance heatmap
    ax4 = fig.add_subplot(gs[1, 0])
    sns.heatmap(true_sigma, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax4, cbar_kws={'label': 'Covariance'})
    ax4.set_title('True Covariance Matrix', fontsize=12, fontweight='bold')
    
    # 5. Estimated covariance heatmap
    ax5 = fig.add_subplot(gs[1, 1])
    sns.heatmap(est_sigma, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax5, cbar_kws={'label': 'Covariance'})
    ax5.set_title('Estimated Covariance Matrix', fontsize=12, fontweight='bold')
    
    # 6. Difference heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    diff_sigma = est_sigma - true_sigma
    sns.heatmap(diff_sigma, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, ax=ax6, cbar_kws={'label': 'Difference'})
    ax6.set_title('Covariance Error (Est - True)', fontsize=12, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_missingness_pattern_analysis(data_path, filename, save_path=None):
    """
    Analyze and visualize the pattern of missing data in a dataset.
    """
    df = pd.read_csv(f"{data_path}/{filename}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Missing data pattern
    ax1 = axes[0, 0]
    missing_matrix = df.isnull().astype(int)
    sns.heatmap(missing_matrix.T, cmap='RdYlGn_r', cbar_kws={'label': 'Missing'}, 
                ax=ax1, yticklabels=df.columns)
    ax1.set_title('Missing Data Pattern (rows × variables)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Variable', fontsize=12)
    
    # 2. Missing percentage per variable
    ax2 = axes[0, 1]
    missing_pct = df.isnull().sum() / len(df) * 100
    missing_pct.plot(kind='bar', ax=ax2, color='#d62728', alpha=0.7)
    ax2.set_title('Missingness per Variable', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Variable', fontsize=12)
    ax2.set_ylabel('Missing (%)', fontsize=12)
    ax2.axhline(y=missing_pct.mean(), color='blue', linestyle='--', 
                linewidth=2, label=f'Average: {missing_pct.mean():.1f}%')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Missing patterns combinations
    ax3 = axes[1, 0]
    pattern_counts = missing_matrix.groupby(list(missing_matrix.columns)).size().sort_values(ascending=False)
    top_patterns = pattern_counts.head(10)
    ax3.barh(range(len(top_patterns)), top_patterns.values, color='#2ca02c', alpha=0.7)
    ax3.set_yticks(range(len(top_patterns)))
    ax3.set_yticklabels([f'Pattern {i+1}' for i in range(len(top_patterns))])
    ax3.set_title('Top 10 Missing Patterns (frequency)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Count', fontsize=12)
    ax3.grid(alpha=0.3)
    
    # 4. Correlation of missingness
    ax4 = axes[1, 1]
    missing_corr = missing_matrix.corr()
    sns.heatmap(missing_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax4, cbar_kws={'label': 'Correlation'})
    ax4.set_title('Correlation of Missingness Between Variables', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n=== MISSING DATA SUMMARY ===")
    print(f"Total samples: {len(df)}")
    print(f"Total variables: {len(df.columns)}")
    print(f"Overall missingness: {df.isnull().sum().sum() / df.size * 100:.2f}%")
    print(f"\nMissingness per variable:")
    print(missing_pct)


def create_full_report(results_df, output_folder='visualization_results'):
    """
    Generate a complete visual report of the simulation study.
    """
    # Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    print("Generating comprehensive visualization report...")
    
    print("\n1. Convergence analysis...")
    plot_convergence_comparison(results_df, 
                               save_path=f"{output_folder}/01_convergence_comparison.png")
    
    print("2. Estimation errors...")
    plot_estimation_errors(results_df, 
                          save_path=f"{output_folder}/02_estimation_errors.png")
    
    print("3. Mechanism comparison...")
    plot_mechanism_comparison(results_df, 
                             save_path=f"{output_folder}/03_mechanism_comparison.png")
    
    print("4. Sample size effects...")
    plot_sample_size_effect(results_df, 
                           save_path=f"{output_folder}/04_sample_size_effect.png")
    
    print("5. Parameter recovery examples...")
    # Show best and worst cases
    best_idx = results_df['mu_error'].idxmin()
    worst_idx = results_df['mu_error'].idxmax()
    
    plot_parameter_recovery(results_df, idx=best_idx, 
                           save_path=f"{output_folder}/05_best_recovery.png")
    plot_parameter_recovery(results_df, idx=worst_idx, 
                           save_path=f"{output_folder}/06_worst_recovery.png")
    
    print(f"\n✓ All visualizations saved to '{output_folder}/'")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Load simulation results
    results_df = pd.read_csv("results/simulation_results.csv")
    
    print(f"Loaded {len(results_df)} simulation results")
    print(f"Mechanisms: {results_df['mechanism'].unique()}")
    print(f"Sample sizes: {sorted(results_df['n_samples'].unique())}")
    print(f"Missingness levels: {sorted(results_df['missingness_pct'].unique())}")
    
    # Generate all visualizations
    create_full_report(results_df, output_folder='plots\\synthetic_multivariate')
    
    # Or create individual plots
    # plot_convergence_comparison(results_df)
    # plot_estimation_errors(results_df)
    # plot_mechanism_comparison(results_df)
    # plot_sample_size_effect(results_df)
    # plot_parameter_recovery(results_df, idx=0)
    
    # Analyze specific dataset missingness pattern
    # plot_missingness_pattern_analysis(
    #     data_path="data/datasets",
    #     filename="missing_MAR_mean0_cov0_n500_miss10.csv"
    # )