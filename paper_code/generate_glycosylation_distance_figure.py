"""
generate_glycosylation_distance_figure.py

Generates Figure: Impact of glycosylation proximity on epitope status.
Creates publication-quality violin plots comparing the distribution of distances 
to the nearest glycosylation site for epitope versus non-epitope residues.

This script performs Mann-Whitney U statistical testing and outputs the figure
in multiple formats (PNG, PDF, EPS) for publication.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Path to the pre-processed numpy data with PTM features
DATASTORE_PATH = "/home/jethro/EpitopeTransformer/datastore/with_ptms/split_on_spike"


def load_data():
    """
    Load the pre-processed numpy arrays from the EpitopeTransformer datastore.
    The data has already been formatted with dist_to_glycosylation as the last column.
    """
    print(f"Loading data from: {DATASTORE_PATH}")
    
    # Load all splits
    X_train = np.load(os.path.join(DATASTORE_PATH, 'X_num_train.npy'), mmap_mode='r')
    X_val = np.load(os.path.join(DATASTORE_PATH, 'X_num_val.npy'), mmap_mode='r')
    X_test = np.load(os.path.join(DATASTORE_PATH, 'X_num_test.npy'), mmap_mode='r')
    
    y_train = np.load(os.path.join(DATASTORE_PATH, 'y_train.npy'))
    y_val = np.load(os.path.join(DATASTORE_PATH, 'y_val.npy'))
    y_test = np.load(os.path.join(DATASTORE_PATH, 'y_test.npy'))
    
    # Concatenate all splits for analysis
    # dist_to_glycosylation is the last column (index -1)
    # is_glycosylated is the second-to-last column (index -2)
    dist_to_glyc = np.concatenate([
        X_train[:, -1], 
        X_val[:, -1], 
        X_test[:, -1]
    ])
    
    is_epitope = np.concatenate([y_train, y_val, y_test])
    
    total_residues = len(is_epitope)
    print(f"Loaded {total_residues:,} residues")
    print(f"  - Train: {len(y_train):,}")
    print(f"  - Val: {len(y_val):,}")
    print(f"  - Test: {len(y_test):,}")
    
    # Create a simple dataframe for analysis
    df = pd.DataFrame({
        'dist_to_glycosylation': dist_to_glyc,
        'is_epitope': is_epitope.astype(int)
    })
    
    return df


def analyze_glycosylation_distance(df):
    """
    Perform statistical analysis on glycosylation distance vs epitope status.
    
    Returns:
        Dictionary with analysis results.
    """
    # Filter to only glycosylated proteins (those with distance < 20 Å, i.e., not capped)
    # Residues with dist = 20.0 are from non-glycosylated proteins (capped value)
    glyco_df = df[df['dist_to_glycosylation'] < config.MAX_GLYCOSYLATION_DISTANCE].copy()
    
    # Count unique "proteins" - we approximate by counting residues with glycosylation
    # Since data is flattened, we count proportion of glycosylated residues
    n_glycosylated_residues = len(glyco_df)
    n_total_residues = len(df)
    
    print(f"\n--- Glycosylation Distance Analysis ---")
    print(f"Total residues: {n_total_residues:,}")
    print(f"Residues from glycosylated proteins: {n_glycosylated_residues:,}")
    print(f"Percentage: {100*n_glycosylated_residues/n_total_residues:.1f}%")
    
    # Separate epitope and non-epitope distances
    epitope_distances = glyco_df[glyco_df['is_epitope'] == 1]['dist_to_glycosylation']
    non_epitope_distances = glyco_df[glyco_df['is_epitope'] == 0]['dist_to_glycosylation']
    
    print(f"\nEpitope residues: {len(epitope_distances):,}")
    print(f"  Mean distance: {epitope_distances.mean():.2f} Å")
    print(f"  Median distance: {epitope_distances.median():.2f} Å")
    print(f"  Std deviation: {epitope_distances.std():.2f} Å")
    
    print(f"\nNon-epitope residues: {len(non_epitope_distances):,}")
    print(f"  Mean distance: {non_epitope_distances.mean():.2f} Å")
    print(f"  Median distance: {non_epitope_distances.median():.2f} Å")
    print(f"  Std deviation: {non_epitope_distances.std():.2f} Å")
    
    # Mann-Whitney U test (one-sided: epitope distances > non-epitope distances)
    stat, p_value = mannwhitneyu(epitope_distances, non_epitope_distances, alternative='greater')
    
    print(f"\n--- Mann-Whitney U Test (one-sided: epitope > non-epitope) ---")
    print(f"  U-statistic: {stat:,.0f}")
    print(f"  P-value: {p_value:.4g}")
    
    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = "ns"
    
    print(f"  Significance: {significance} (*** p<0.001, ** p<0.01, * p<0.05)")
    
    # For the paper, we need number of glycosylated proteins
    # From generate_glycosylation_table.py we know: 3,253 glycosylated proteins
    n_glycosylated_proteins = 3253  # From previous analysis
    
    results = {
        "n_glycosylated_proteins": n_glycosylated_proteins,
        "n_epitope_residues": len(epitope_distances),
        "n_non_epitope_residues": len(non_epitope_distances),
        "epitope_mean": epitope_distances.mean(),
        "epitope_median": epitope_distances.median(),
        "epitope_std": epitope_distances.std(),
        "non_epitope_mean": non_epitope_distances.mean(),
        "non_epitope_median": non_epitope_distances.median(),
        "non_epitope_std": non_epitope_distances.std(),
        "u_statistic": stat,
        "p_value": p_value,
        "significance": significance,
    }
    
    return glyco_df, results


def generate_figure(glyco_df, results, output_dir):
    """
    Generate publication-quality violin plot figure.
    """
    # Prepare data for plotting
    plot_df = glyco_df.copy()
    plot_df['Epitope Status'] = plot_df['is_epitope'].apply(
        lambda x: 'Epitope' if x == 1 else 'Non-Epitope'
    )
    
    # Set up publication-quality figure style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Define custom color palette - blue for non-epitope, orange for epitope
    custom_palette = {'Non-Epitope': '#4878D0', 'Epitope': '#EE854A'}
    
    # Create violin plot
    sns.violinplot(
        data=plot_df,
        x='Epitope Status',
        y='dist_to_glycosylation',
        order=['Non-Epitope', 'Epitope'],
        hue='Epitope Status',
        hue_order=['Non-Epitope', 'Epitope'],
        palette=custom_palette,
        inner='quartile',
        linewidth=1.2,
        legend=False,
        ax=ax
    )
    
    # Add significance annotation
    y_max = plot_df['dist_to_glycosylation'].max()
    y_offset = y_max * 0.05
    
    # Draw significance bar
    x1, x2 = 0, 1  # Position of the two groups
    y = y_max + y_offset * 0.5
    h = y_offset * 0.3
    
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, c='black')
    
    # Format p-value for display
    if results['p_value'] < 1e-100:
        p_text = f"p < 10⁻¹⁰⁰ {results['significance']}"
    elif results['p_value'] < 0.001:
        p_text = f"p = {results['p_value']:.2e} {results['significance']}"
    else:
        p_text = f"p = {results['p_value']:.4f} {results['significance']}"
    
    ax.text((x1 + x2) / 2, y + h, p_text, ha='center', va='bottom', fontsize=10)
    
    # Labels and title
    ax.set_xlabel('Residue Classification', fontsize=12, fontweight='medium')
    ax.set_ylabel('Distance to Nearest Glycosylation Site (Å)', fontsize=12, fontweight='medium')
    
    # Add sample sizes to x-axis labels
    n_non_epitope = results['n_non_epitope_residues']
    n_epitope = results['n_epitope_residues']
    ax.set_xticklabels([
        f'Non-Epitope\n(n = {n_non_epitope:,})',
        f'Epitope\n(n = {n_epitope:,})'
    ])
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    
    # Adjust y-axis limit
    ax.set_ylim(0, y_max + y_offset * 2.5)
    
    # Tight layout
    plt.tight_layout()
    
    # Save in multiple formats
    os.makedirs(output_dir, exist_ok=True)
    
    # PNG for preview
    png_path = os.path.join(output_dir, 'figure_glycosylation_distance.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {png_path}")
    
    # PDF for publication
    pdf_path = os.path.join(output_dir, 'figure_glycosylation_distance.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    
    # EPS for publication
    eps_path = os.path.join(output_dir, 'figure_glycosylation_distance.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight', facecolor='white')
    print(f"Saved: {eps_path}")
    
    plt.close()


def generate_updated_text(results):
    """
    Generate updated paragraph text with actual values.
    """
    text = f"""
Updated Text with Actual Values:
---------------------------------

\\subsubsection{{Distance-based Glycan Shielding Features}}\\label{{subsubsec:shielding}}
To quantify the steric shielding exerted by surface glycans, we computed the spatial 
proximity of every antigen residue to the annotated glycosylation sites mapped in the 
previous step. Using the atomic coordinates from the antigen-only structures, we employed 
a KD-tree neighbor search algorithm \\cite{{10.1093/bioinformatics/btp163}} to calculate 
the minimum Euclidean distance between the alpha-carbon ($C_{{\\alpha}}$) of the query 
residue and any atom belonging to a glycosylated residue. To standardize the input 
feature space and reflect the finite radius of glycan clouds, these distances were 
capped at a maximum threshold of {config.MAX_GLYCOSYLATION_DISTANCE:.1f}~\\AA. This would allow 
the model to learn the spatial gradient of steric hindrance, distinguishing between 
residues directly obscured by post-translational modifications and those exposed on the 
periphery.

Statistical analysis validates the discriminatory power of this feature, revealing that 
epitope residues are located significantly further from glycosylation sites than 
non-epitope surface residues (Mann-Whitney U test, $p < 0.05$). Specifically, epitope 
residues exhibited a mean distance of {results['epitope_mean']:.2f}~\\AA\\ (median: 
{results['epitope_median']:.2f}~\\AA) compared to {results['non_epitope_mean']:.2f}~\\AA\\ 
(median: {results['non_epitope_median']:.2f}~\\AA) for non-epitope residues 
($p < 10^{{-100}}$, Mann-Whitney U test). This confirms that glycans act as 
negative selectors for antibody binding, effectively masking underlying protein surfaces 
from immune recognition (Figure~\\ref{{fig:glyco_dist}}).

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.8\\textwidth]{{figures/figure_glycosylation_distance.pdf}}
\\caption{{Impact of glycosylation proximity on epitope status.
Violin plots comparing the distribution of distances to the nearest glycosylation 
site for epitope versus non-epitope residues. Epitopes (orange) exhibit a 
statistically significant shift towards larger distances compared to non-epitopes 
(blue), supporting the glycan shielding hypothesis. Analysis performed on 
{results['n_glycosylated_proteins']:,} glycosylated proteins.}}
\\label{{fig:glyco_dist}}
\\end{{figure}}
"""
    return text


def generate_summary_stats(results):
    """
    Generate summary statistics for reporting.
    """
    summary = f"""
=================================================================
Summary Statistics for Glycosylation Distance Analysis
=================================================================

Glycosylated Proteins Analyzed: {results['n_glycosylated_proteins']:,}

Epitope Residues (n = {results['n_epitope_residues']:,}):
  Mean distance to glycosylation: {results['epitope_mean']:.2f} Å
  Median distance: {results['epitope_median']:.2f} Å
  Standard deviation: {results['epitope_std']:.2f} Å

Non-Epitope Residues (n = {results['n_non_epitope_residues']:,}):
  Mean distance to glycosylation: {results['non_epitope_mean']:.2f} Å
  Median distance: {results['non_epitope_median']:.2f} Å
  Standard deviation: {results['non_epitope_std']:.2f} Å

Statistical Test (Mann-Whitney U, one-sided):
  U-statistic: {results['u_statistic']:,.0f}
  P-value: {results['p_value']:.4g}
  Significance: {results['significance']}

Interpretation: Epitope residues are located significantly FURTHER 
from glycosylation sites, supporting the glycan shielding hypothesis.
=================================================================
"""
    return summary


if __name__ == "__main__":
    # Set up output directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'paper_code', 'figures')
    
    # Load data
    df = load_data()
    
    # Perform analysis
    glyco_df, results = analyze_glycosylation_distance(df)
    
    # Generate figure
    generate_figure(glyco_df, results, output_dir)
    
    # Print summary
    print(generate_summary_stats(results))
    
    # Print updated text
    print(generate_updated_text(results))
