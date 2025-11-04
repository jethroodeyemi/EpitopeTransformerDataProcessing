#!/usr/bin/env python3
"""
PCA Analysis Script for ESM Embeddings

This script performs comprehensive analysis of PCA dimensionality reduction
applied to ESM embeddings, including:
- Variance explained analysis
- Cumulative variance plots
- Component correlation analysis
- Reconstruction error analysis
- Visualization of principal components
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_pca_model(model_name, target_dim):
    """Load a PCA model from the cache directory."""
    model_path = os.path.join(config.PCA_MODEL_CACHE_DIR, f"{model_name}_pca_{target_dim}.pkl")
    if not os.path.exists(model_path):
        print(f"❌ PCA model not found: {model_path}")
        return None
    
    with open(model_path, 'rb') as f:
        pca_model = pickle.load(f)
    
    print(f"✅ Loaded PCA model: {model_path}")
    return pca_model


def analyze_variance_explained(pca_model, model_name, target_dim):
    """Analyze and visualize variance explained by principal components."""
    print(f"\n{'='*80}")
    print(f"VARIANCE ANALYSIS: {model_name.upper()}")
    print(f"{'='*80}")
    
    n_components = pca_model.n_components_
    variance_ratio = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratio)
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"   Total components retained: {n_components}")
    print(f"   Original dimensionality: {pca_model.n_features_in_}")
    print(f"   Compression ratio: {n_components / pca_model.n_features_in_:.2%}")
    print(f"   Total variance explained: {cumulative_variance[-1]:.4f} ({cumulative_variance[-1]*100:.2f}%)")
    print(f"   Variance lost: {1 - cumulative_variance[-1]:.4f} ({(1-cumulative_variance[-1])*100:.2f}%)")
    
    # Top components analysis
    print(f"\n🔝 Top 10 Principal Components:")
    print(f"   {'Component':<12} {'Variance':<15} {'Cumulative':<15}")
    print(f"   {'-'*42}")
    for i in range(min(10, n_components)):
        print(f"   PC-{i+1:<10} {variance_ratio[i]:>8.4f} ({variance_ratio[i]*100:>5.2f}%)  "
              f"{cumulative_variance[i]:>8.4f} ({cumulative_variance[i]*100:>5.2f}%)")
    
    # Milestones
    print(f"\n🎯 Variance Milestones:")
    milestones = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    for threshold in milestones:
        n_comp_needed = np.argmax(cumulative_variance >= threshold) + 1
        if cumulative_variance[-1] >= threshold:
            print(f"   {threshold*100:>3.0f}% variance: {n_comp_needed} components")
        else:
            print(f"   {threshold*100:>3.0f}% variance: Not achievable (max: {cumulative_variance[-1]*100:.2f}%)")
    
    return variance_ratio, cumulative_variance


def plot_variance_explained(variance_ratio, cumulative_variance, model_name, output_dir):
    """Create variance explained plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'PCA Variance Analysis: {model_name.upper()}', fontsize=16, fontweight='bold')
    
    n_components = len(variance_ratio)
    
    # Plot 1: Individual variance explained
    ax1 = axes[0, 0]
    ax1.bar(range(1, n_components + 1), variance_ratio, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained Ratio')
    ax1.set_title('Individual Component Variance')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative variance explained
    ax2 = axes[0, 1]
    ax2.plot(range(1, n_components + 1), cumulative_variance, marker='o', 
             linewidth=2, markersize=4, color='darkgreen')
    ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90% threshold')
    ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
    ax2.fill_between(range(1, n_components + 1), cumulative_variance, alpha=0.3, color='green')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Explained')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # Plot 3: Log scale variance
    ax3 = axes[1, 0]
    ax3.semilogy(range(1, n_components + 1), variance_ratio, marker='s', 
                 linewidth=2, markersize=4, color='crimson')
    ax3.set_xlabel('Principal Component')
    ax3.set_ylabel('Variance Explained Ratio (log scale)')
    ax3.set_title('Component Variance (Log Scale)')
    ax3.grid(True, alpha=0.3, which='both')
    
    # Plot 4: First 20 components detailed view
    ax4 = axes[1, 1]
    n_show = min(20, n_components)
    colors = plt.cm.viridis(np.linspace(0, 1, n_show))
    bars = ax4.bar(range(1, n_show + 1), variance_ratio[:n_show], 
                   color=colors, edgecolor='black', alpha=0.8)
    ax4.set_xlabel('Principal Component')
    ax4.set_ylabel('Variance Explained Ratio')
    ax4.set_title(f'Top {n_show} Components (Detailed View)')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for i, (bar, val) in enumerate(zip(bars, variance_ratio[:n_show])):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{model_name}_variance_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved variance plot: {output_path}")
    plt.close()


def analyze_reconstruction_error(pca_model, model_name, embedding_dir, output_dir):
    """Analyze reconstruction error on actual embeddings."""
    print(f"\n{'='*80}")
    print(f"RECONSTRUCTION ERROR ANALYSIS: {model_name.upper()}")
    print(f"{'='*80}")
    
    # Find all embeddings for this model
    cache_pattern = f"*_{model_name}.npy"
    embedding_files = list(Path(embedding_dir).glob(cache_pattern))
    
    if not embedding_files:
        print(f"⚠️  No embedding files found matching pattern: {cache_pattern}")
        return None
    
    print(f"\n📁 Found {len(embedding_files)} embedding files")
    
    # Sample embeddings for analysis (to avoid memory issues)
    max_samples = min(50, len(embedding_files))
    sample_files = np.random.choice(embedding_files, max_samples, replace=False)
    
    reconstruction_errors = []
    relative_errors = []
    per_sample_errors = []
    
    print(f"📊 Computing reconstruction errors on {max_samples} samples...")
    
    for emb_file in tqdm(sample_files, desc="Processing"):
        try:
            original_emb = np.load(emb_file)
            
            # Transform and inverse transform
            transformed = pca_model.transform(original_emb)
            reconstructed = pca_model.inverse_transform(transformed)
            
            # Compute errors
            abs_error = np.abs(original_emb - reconstructed)
            mse = np.mean((original_emb - reconstructed) ** 2)
            mae = np.mean(abs_error)
            
            # Relative error (avoid division by zero)
            original_norm = np.linalg.norm(original_emb, axis=1, keepdims=True)
            original_norm = np.where(original_norm == 0, 1e-10, original_norm)
            rel_error = np.linalg.norm(original_emb - reconstructed, axis=1, keepdims=True) / original_norm
            
            reconstruction_errors.append(mse)
            relative_errors.extend(rel_error.flatten())
            per_sample_errors.append({
                'file': emb_file.name,
                'mse': mse,
                'mae': mae,
                'max_error': np.max(abs_error),
                'mean_rel_error': np.mean(rel_error)
            })
            
        except Exception as e:
            print(f"⚠️  Error processing {emb_file.name}: {e}")
            continue
    
    if not reconstruction_errors:
        print("❌ No successful reconstructions")
        return None
    
    # Print statistics
    reconstruction_errors = np.array(reconstruction_errors)
    relative_errors = np.array(relative_errors)
    
    print(f"\n📈 Reconstruction Error Statistics:")
    print(f"   Mean Squared Error (MSE):")
    print(f"      Mean:   {np.mean(reconstruction_errors):.6f}")
    print(f"      Median: {np.median(reconstruction_errors):.6f}")
    print(f"      Std:    {np.std(reconstruction_errors):.6f}")
    print(f"      Min:    {np.min(reconstruction_errors):.6f}")
    print(f"      Max:    {np.max(reconstruction_errors):.6f}")
    
    print(f"\n   Relative Error:")
    print(f"      Mean:   {np.mean(relative_errors):.4f} ({np.mean(relative_errors)*100:.2f}%)")
    print(f"      Median: {np.median(relative_errors):.4f} ({np.median(relative_errors)*100:.2f}%)")
    print(f"      95th percentile: {np.percentile(relative_errors, 95):.4f} ({np.percentile(relative_errors, 95)*100:.2f}%)")
    
    # Plot reconstruction errors
    plot_reconstruction_errors(reconstruction_errors, relative_errors, per_sample_errors, model_name, output_dir)
    
    return per_sample_errors


def plot_reconstruction_errors(mse_errors, rel_errors, per_sample_errors, model_name, output_dir):
    """Plot reconstruction error distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Reconstruction Error Analysis: {model_name.upper()}', fontsize=16, fontweight='bold')
    
    # Plot 1: MSE distribution
    ax1 = axes[0, 0]
    ax1.hist(mse_errors, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(np.mean(mse_errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mse_errors):.6f}')
    ax1.axvline(np.median(mse_errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(mse_errors):.6f}')
    ax1.set_xlabel('Mean Squared Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('MSE Distribution Across Samples')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Relative error distribution
    ax2 = axes[0, 1]
    ax2.hist(rel_errors, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax2.axvline(np.mean(rel_errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rel_errors):.4f}')
    ax2.axvline(np.median(rel_errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(rel_errors):.4f}')
    ax2.set_xlabel('Relative Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Relative Error Distribution (Per Residue)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plot of errors
    ax3 = axes[1, 0]
    error_data = [mse_errors, rel_errors]
    bp = ax3.boxplot(error_data, labels=['MSE', 'Relative Error'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['steelblue', 'coral']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_ylabel('Error Value')
    ax3.set_title('Error Distribution (Box Plot)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Top 20 samples by error
    ax4 = axes[1, 1]
    df_errors = pd.DataFrame(per_sample_errors)
    df_errors = df_errors.nlargest(20, 'mse')
    
    y_pos = np.arange(len(df_errors))
    ax4.barh(y_pos, df_errors['mse'].values, alpha=0.7, color='crimson', edgecolor='black')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in df_errors['file'].values], fontsize=8)
    ax4.set_xlabel('MSE')
    ax4.set_title('Top 20 Samples by Reconstruction Error')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.invert_yaxis()
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{model_name}_reconstruction_error.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved reconstruction error plot: {output_path}")
    plt.close()


def analyze_component_importance(pca_model, model_name, output_dir):
    """Analyze the importance and characteristics of principal components."""
    print(f"\n{'='*80}")
    print(f"COMPONENT IMPORTANCE ANALYSIS: {model_name.upper()}")
    print(f"{'='*80}")
    
    components = pca_model.components_
    n_components, n_features = components.shape
    
    print(f"\n📐 Component Matrix Shape: {n_components} components × {n_features} features")
    
    # Analyze component characteristics
    component_norms = np.linalg.norm(components, axis=1)
    component_sparsity = np.sum(np.abs(components) < 1e-3, axis=1) / n_features
    
    print(f"\n🔍 Component Characteristics:")
    print(f"   Component norms - Mean: {np.mean(component_norms):.4f}, Std: {np.std(component_norms):.4f}")
    print(f"   Sparsity (% near-zero) - Mean: {np.mean(component_sparsity)*100:.2f}%, Std: {np.std(component_sparsity)*100:.2f}%")
    
    # Find most important features for top components
    print(f"\n⭐ Top 5 Most Important Features per Component:")
    for i in range(min(5, n_components)):
        top_features = np.argsort(np.abs(components[i]))[-5:][::-1]
        print(f"\n   PC-{i+1} (Variance: {pca_model.explained_variance_ratio_[i]*100:.2f}%):")
        for rank, feat_idx in enumerate(top_features, 1):
            print(f"      {rank}. Feature {feat_idx}: {components[i, feat_idx]:+.4f}")
    
    # Plot component heatmap
    plot_component_heatmap(components, model_name, output_dir)
    
    return components


def plot_component_heatmap(components, model_name, output_dir):
    """Create a heatmap of principal components."""
    n_components, n_features = components.shape
    
    # Limit visualization to first 50 components and sample features
    n_comp_show = min(50, n_components)
    n_feat_show = min(100, n_features)
    
    # Sample features uniformly
    if n_features > n_feat_show:
        feat_indices = np.linspace(0, n_features-1, n_feat_show, dtype=int)
    else:
        feat_indices = np.arange(n_features)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle(f'Principal Component Heatmaps: {model_name.upper()}', fontsize=16, fontweight='bold')
    
    # Plot 1: Full heatmap (sampled)
    ax1 = axes[0]
    im1 = ax1.imshow(components[:n_comp_show, feat_indices], aspect='auto', cmap='RdBu_r', 
                     vmin=-np.percentile(np.abs(components), 95), 
                     vmax=np.percentile(np.abs(components), 95))
    ax1.set_xlabel('Feature Index (sampled)')
    ax1.set_ylabel('Principal Component')
    ax1.set_title(f'Component Weights (First {n_comp_show} components)')
    plt.colorbar(im1, ax=ax1, label='Weight')
    
    # Plot 2: Top 10 components in detail
    ax2 = axes[1]
    n_top = min(10, n_components)
    im2 = ax2.imshow(components[:n_top, feat_indices], aspect='auto', cmap='RdBu_r',
                     vmin=-np.percentile(np.abs(components[:n_top]), 95),
                     vmax=np.percentile(np.abs(components[:n_top]), 95))
    ax2.set_xlabel('Feature Index (sampled)')
    ax2.set_ylabel('Principal Component')
    ax2.set_title(f'Top {n_top} Components (Detailed)')
    ax2.set_yticks(range(n_top))
    ax2.set_yticklabels([f'PC-{i+1}' for i in range(n_top)])
    plt.colorbar(im2, ax=ax2, label='Weight')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{model_name}_component_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved component heatmap: {output_path}")
    plt.close()


def create_summary_report(all_analyses, output_dir):
    """Create a summary report comparing all models."""
    print(f"\n{'='*80}")
    print(f"SUMMARY REPORT")
    print(f"{'='*80}")
    
    summary_data = []
    for model_name, analysis in all_analyses.items():
        if analysis['pca_model'] is None:
            continue
        
        pca = analysis['pca_model']
        summary_data.append({
            'Model': model_name.upper(),
            'Original Dims': pca.n_features_in_,
            'Reduced Dims': pca.n_components_,
            'Compression': f"{pca.n_components_ / pca.n_features_in_:.2%}",
            'Variance Retained': f"{np.sum(pca.explained_variance_ratio_):.4f}",
            'Variance Lost': f"{1 - np.sum(pca.explained_variance_ratio_):.4f}",
            'Top Component': f"{pca.explained_variance_ratio_[0]:.4f}",
            'Components for 90%': np.argmax(analysis['cumulative_variance'] >= 0.9) + 1 
                                  if np.max(analysis['cumulative_variance']) >= 0.9 else 'N/A'
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    print("\n📋 Comparison Table:")
    print(df_summary.to_string(index=False))
    
    # Save summary to CSV
    csv_path = os.path.join(output_dir, 'pca_summary_comparison.csv')
    df_summary.to_csv(csv_path, index=False)
    print(f"\n💾 Saved summary table: {csv_path}")
    
    # Create comparison plot
    if len(summary_data) > 1:
        plot_model_comparison(df_summary, all_analyses, output_dir)


def plot_model_comparison(df_summary, all_analyses, output_dir):
    """Create comparison plots across models."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PCA Model Comparison', fontsize=16, fontweight='bold')
    
    models = list(all_analyses.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    # Plot 1: Variance explained comparison
    ax1 = axes[0, 0]
    for i, (model_name, analysis) in enumerate(all_analyses.items()):
        if analysis['pca_model'] is None:
            continue
        variance = analysis['variance_ratio']
        ax1.plot(range(1, len(variance)+1), variance, marker='o', label=model_name.upper(),
                color=colors[i], linewidth=2, markersize=3)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained')
    ax1.set_title('Individual Component Variance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative variance comparison
    ax2 = axes[0, 1]
    for i, (model_name, analysis) in enumerate(all_analyses.items()):
        if analysis['pca_model'] is None:
            continue
        cum_var = analysis['cumulative_variance']
        ax2.plot(range(1, len(cum_var)+1), cum_var, marker='s', label=model_name.upper(),
                color=colors[i], linewidth=2, markersize=3)
    ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% threshold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # Plot 3: Compression ratio
    ax3 = axes[1, 0]
    model_names = df_summary['Model'].values
    compression = [float(c.strip('%'))/100 for c in df_summary['Compression'].values]
    bars = ax3.bar(model_names, compression, color=colors[:len(model_names)], 
                   alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Compression Ratio')
    ax3.set_title('Dimensionality Compression')
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for bar, val in zip(bars, compression):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Variance retained
    ax4 = axes[1, 1]
    variance_retained = [float(v) for v in df_summary['Variance Retained'].values]
    variance_lost = [float(v) for v in df_summary['Variance Lost'].values]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, variance_retained, width, label='Retained',
                    color='green', alpha=0.7, edgecolor='black')
    bars2 = ax4.bar(x + width/2, variance_lost, width, label='Lost',
                    color='red', alpha=0.7, edgecolor='black')
    
    ax4.set_ylabel('Variance')
    ax4.set_title('Information Retention')
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 1.1])
    
    # Add percentage labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height*100:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved comparison plot: {output_path}")
    plt.close()


def main():
    """Main analysis function."""
    print("="*80)
    print("PCA DIMENSIONALITY REDUCTION ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing PCA models from: {config.PCA_MODEL_CACHE_DIR}")
    print(f"Using embeddings from: {config.EMBEDDING_CACHE_DIR}")
    
    # Create output directory for plots
    output_dir = os.path.join(config.OUTPUT_DIR, 'pca_analysis')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Models to analyze
    models_to_analyze = []
    
    if config.REDUCE_ESM2_DIM and 'esm2' in config.EMBEDDING_MODE:
        models_to_analyze.append(('esm2', config.ESM2_DIM_TARGET))
    
    if config.REDUCE_ESM1V_DIM and 'esm1v' in config.EMBEDDING_MODE:
        models_to_analyze.append(('esm1v', config.ESM1V_DIM_TARGET))
    
    if config.REDUCE_ESM_IF1_DIM and 'esm_if1' in config.EMBEDDING_MODE:
        models_to_analyze.append(('esm_if1', config.ESM_IF1_DIM_TARGET))
    
    if not models_to_analyze:
        print("\n⚠️  No PCA models configured for analysis.")
        print("Check your config.py settings for REDUCE_*_DIM flags.")
        return
    
    print(f"\n🔍 Models to analyze: {[m[0] for m in models_to_analyze]}")
    
    # Analyze each model
    all_analyses = {}
    
    for model_name, target_dim in models_to_analyze:
        print(f"\n{'#'*80}")
        print(f"# Analyzing {model_name.upper()} (Target dim: {target_dim})")
        print(f"{'#'*80}")
        
        # Load PCA model
        pca_model = load_pca_model(model_name, target_dim)
        
        if pca_model is None:
            all_analyses[model_name] = {'pca_model': None}
            continue
        
        # Variance analysis
        variance_ratio, cumulative_variance = analyze_variance_explained(
            pca_model, model_name, target_dim
        )
        
        # Plot variance
        plot_variance_explained(variance_ratio, cumulative_variance, model_name, output_dir)
        
        # Reconstruction error analysis
        reconstruction_errors = analyze_reconstruction_error(
            pca_model, model_name, config.EMBEDDING_CACHE_DIR, output_dir
        )
        
        # Component importance
        components = analyze_component_importance(pca_model, model_name, output_dir)
        
        # Store results
        all_analyses[model_name] = {
            'pca_model': pca_model,
            'variance_ratio': variance_ratio,
            'cumulative_variance': cumulative_variance,
            'reconstruction_errors': reconstruction_errors,
            'components': components
        }
    
    # Create summary comparison
    if len(all_analyses) > 0:
        create_summary_report(all_analyses, output_dir)
    
    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"All results saved to: {output_dir}")
    print("\n📊 Generated files:")
    for file in sorted(os.listdir(output_dir)):
        print(f"   - {file}")


if __name__ == "__main__":
    main()
