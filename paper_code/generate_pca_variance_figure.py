"""
generate_pca_variance_figure.py

Generates Figure: Principal Component Analysis of protein language model embeddings.
Creates a two-panel figure showing cumulative variance explained for ESM-2 and ESM-1v,
suitable for publication in a scientific paper.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_pca_model(model_name, target_dim):
    """Load a PCA model from the cache directory."""
    model_path = os.path.join(config.PCA_MODEL_CACHE_DIR, f"{model_name}_pca_{target_dim}.pkl")
    if not os.path.exists(model_path):
        print(f"Warning: PCA model not found at {model_path}")
        return None
    
    with open(model_path, 'rb') as f:
        pca_model = pickle.load(f)
    
    return pca_model


def get_variance_data(pca_model):
    """Extract variance data from PCA model."""
    variance_ratio = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratio)
    return variance_ratio, cumulative_variance


def create_publication_figure(output_dir):
    """
    Create a publication-quality two-panel figure showing cumulative 
    variance explained for ESM-2 and ESM-1v embeddings.
    """
    # Load PCA models
    esm2_pca = load_pca_model('esm2', config.ESM2_DIM_TARGET)
    esm1v_pca = load_pca_model('esm1v', config.ESM1V_DIM_TARGET)
    
    if esm2_pca is None or esm1v_pca is None:
        print("Error: Could not load one or both PCA models.")
        print("Please ensure PCA models have been trained first.")
        return None
    
    # Get variance data
    esm2_var, esm2_cumvar = get_variance_data(esm2_pca)
    esm1v_var, esm1v_cumvar = get_variance_data(esm1v_pca)
    
    # Print statistics
    print("\n" + "="*60)
    print("PCA Variance Statistics")
    print("="*60)
    print(f"\nESM-2 ({config.ESM2_DIM_TARGET} components):")
    print(f"  Original dimensions: {esm2_pca.n_features_in_}")
    print(f"  Variance retained: {esm2_cumvar[-1]*100:.2f}%")
    print(f"  Top PC explains: {esm2_var[0]*100:.2f}%")
    
    print(f"\nESM-1v ({config.ESM1V_DIM_TARGET} components):")
    print(f"  Original dimensions: {esm1v_pca.n_features_in_}")
    print(f"  Variance retained: {esm1v_cumvar[-1]*100:.2f}%")
    print(f"  Top PC explains: {esm1v_var[0]*100:.2f}%")
    
    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Set style for publication
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.2
    
    # Color scheme
    line_color = '#2C3E50'
    fill_color = '#3498DB'
    threshold_90_color = '#E74C3C'
    threshold_95_color = '#F39C12'
    
    # Panel A: ESM-2
    ax1 = axes[0]
    n_components_esm2 = len(esm2_cumvar)
    x_esm2 = np.arange(1, n_components_esm2 + 1)
    
    ax1.plot(x_esm2, esm2_cumvar, linewidth=2, color=line_color, zorder=3)
    ax1.fill_between(x_esm2, esm2_cumvar, alpha=0.3, color=fill_color, zorder=2)
    ax1.axhline(y=0.90, color=threshold_90_color, linestyle='--', linewidth=1.5, 
                alpha=0.8, label='90% variance', zorder=1)
    ax1.axhline(y=0.95, color=threshold_95_color, linestyle='--', linewidth=1.5, 
                alpha=0.8, label='95% variance', zorder=1)
    
    # Mark the final variance retained
    ax1.scatter([n_components_esm2], [esm2_cumvar[-1]], color=line_color, s=50, zorder=4)
    ax1.annotate(f'{esm2_cumvar[-1]*100:.1f}%', 
                 xy=(n_components_esm2, esm2_cumvar[-1]),
                 xytext=(n_components_esm2 - 40, esm2_cumvar[-1] - 0.08),
                 fontsize=9, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
    
    ax1.set_xlabel('Number of Principal Components', fontsize=11)
    ax1.set_ylabel('Cumulative Variance Explained', fontsize=11)
    ax1.set_title('(A) ESM-2', fontsize=12, fontweight='bold', loc='left')
    ax1.set_xlim([0, n_components_esm2 + 10])
    ax1.set_ylim([0, 1.05])
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    
    # Panel B: ESM-1v
    ax2 = axes[1]
    n_components_esm1v = len(esm1v_cumvar)
    x_esm1v = np.arange(1, n_components_esm1v + 1)
    
    ax2.plot(x_esm1v, esm1v_cumvar, linewidth=2, color=line_color, zorder=3)
    ax2.fill_between(x_esm1v, esm1v_cumvar, alpha=0.3, color=fill_color, zorder=2)
    ax2.axhline(y=0.90, color=threshold_90_color, linestyle='--', linewidth=1.5, 
                alpha=0.8, label='90% variance', zorder=1)
    ax2.axhline(y=0.95, color=threshold_95_color, linestyle='--', linewidth=1.5, 
                alpha=0.8, label='95% variance', zorder=1)
    
    # Mark the final variance retained
    ax2.scatter([n_components_esm1v], [esm1v_cumvar[-1]], color=line_color, s=50, zorder=4)
    ax2.annotate(f'{esm1v_cumvar[-1]*100:.1f}%', 
                 xy=(n_components_esm1v, esm1v_cumvar[-1]),
                 xytext=(n_components_esm1v - 40, esm1v_cumvar[-1] - 0.08),
                 fontsize=9, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
    
    ax2.set_xlabel('Number of Principal Components', fontsize=11)
    ax2.set_ylabel('Cumulative Variance Explained', fontsize=11)
    ax2.set_title('(B) ESM-1v', fontsize=12, fontweight='bold', loc='left')
    ax2.set_xlim([0, n_components_esm1v + 10])
    ax2.set_ylim([0, 1.05])
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    
    plt.tight_layout()
    
    # Save figure in multiple formats
    os.makedirs(output_dir, exist_ok=True)
    
    # High-resolution PNG for general use
    png_path = os.path.join(output_dir, 'figure_pca_variance.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved PNG: {png_path}")
    
    # PDF for publication (vector graphics)
    pdf_path = os.path.join(output_dir, 'figure_pca_variance.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"✓ Saved PDF: {pdf_path}")
    
    # EPS for some journals
    eps_path = os.path.join(output_dir, 'figure_pca_variance.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight', facecolor='white')
    print(f"✓ Saved EPS: {eps_path}")
    
    plt.close()
    
    return {
        'esm2': {
            'n_components': n_components_esm2,
            'original_dim': esm2_pca.n_features_in_,
            'variance_retained': esm2_cumvar[-1]
        },
        'esm1v': {
            'n_components': n_components_esm1v,
            'original_dim': esm1v_pca.n_features_in_,
            'variance_retained': esm1v_cumvar[-1]
        }
    }


def generate_latex_figure_code():
    """Generate LaTeX code for including the figure in a paper."""
    latex = r"""
\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{figures/figure_pca_variance.pdf}
\caption{Principal Component Analysis of protein language model embeddings. 
The plots illustrate the cumulative explained variance ratio for (A) ESM-2 and 
(B) ESM-1v features, justifying the dimensionality reduction to 256 components 
to retain high informational content while reducing noise. Dashed lines indicate 
90\% and 95\% variance thresholds.}
\label{fig:pca_variance}
\end{figure}
"""
    return latex


if __name__ == "__main__":
    # Set output directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "paper_code", "figures")
    
    # Create the figure
    stats = create_publication_figure(output_dir)
    
    if stats:
        print("\n" + "="*60)
        print("LaTeX Code for Figure")
        print("="*60)
        print(generate_latex_figure_code())
