"""
generate_ablation_figure.py

Generates Figure: Impact of regularization on learning dynamics.
Plots training and validation loss curves for baseline and regularized models.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add parent directory to path to import config if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_results(path):
    """Load results from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Parse stringified lists if necessary
    for key in ['losses', 'val_losses', 'val_score', 'test_score']:
        if isinstance(data[key], str):
            data[key] = json.loads(data[key])
            
    return data

def generate_ablation_figure(baseline_path, regularized_path, output_dir):
    """Generate the ablation study figure."""
    
    # Load data
    baseline = load_results(baseline_path)
    regularized = load_results(regularized_path)
    
    # Extract losses
    b_train_loss = baseline['losses']
    b_val_loss = baseline['val_losses']
    r_train_loss = regularized['losses']
    r_val_loss = regularized['val_losses']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 11
    })
    
    # Plot A: Baseline
    epochs_b = range(1, len(b_train_loss) + 1)
    ax1.plot(epochs_b, b_train_loss, label='Training Loss', color='#2C3E50', linewidth=2)
    ax1.plot(epochs_b, b_val_loss, label='Validation Loss', color='#E74C3C', linewidth=2, linestyle='--')
    
    ax1.set_title('(A) Baseline (No Regularization)', fontsize=12, fontweight='bold', loc='left')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (Binary Cross-Entropy)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Highlight divergence point
    # Find min val loss index
    min_val_idx = np.argmin(b_val_loss)
    min_val_epoch = min_val_idx + 1
    
    ax1.axvline(x=min_val_epoch, color='gray', linestyle=':', alpha=0.8)
    ax1.text(min_val_epoch + 1, max(b_val_loss) * 0.9, f'Divergence\nEpoch {min_val_epoch}', 
             fontsize=9, color='#555')
    
    # Plot B: Regularized
    epochs_r = range(1, len(r_train_loss) + 1)
    ax2.plot(epochs_r, r_train_loss, label='Training Loss', color='#2C3E50', linewidth=2)
    ax2.plot(epochs_r, r_val_loss, label='Validation Loss', color='#27AE60', linewidth=2, linestyle='--')
    
    ax2.set_title('(B) With Latent Manifold Regularization', fontsize=12, fontweight='bold', loc='left')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss (Binary Cross-Entropy)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Ensure y-axis limits are comparable or reasonable
    y_max = max(max(b_train_loss), max(b_val_loss), max(r_train_loss), max(r_val_loss))
    y_min = min(min(b_train_loss), min(b_val_loss), min(r_train_loss), min(r_val_loss))
    
    # Add some padding
    y_range = y_max - y_min
    ax1.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    ax2.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_path = os.path.join(output_dir, 'figure_ablation_curves.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved PDF: {pdf_path}")
    
    png_path = os.path.join(output_dir, 'figure_ablation_curves.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved PNG: {png_path}")
    
    # Print metrics for text update
    print("\n--- Metrics for Text Update ---")
    
    # Generalization Gap
    b_gen_gap = baseline.get('gen_gap', 0)
    r_gen_gap = regularized.get('gen_gap', 0)
    print(f"Generalization Gap (Baseline): {b_gen_gap:.4f}")
    print(f"Generalization Gap (Regularized): {r_gen_gap:.4f}")
    
    # AUC-PR
    b_auc_pr = baseline.get('final_test_auc_pr', 0)
    r_auc_pr = regularized.get('final_test_auc_pr', 0)
    print(f"AUC-PR (Baseline): {b_auc_pr:.4f}")
    print(f"AUC-PR (Regularized): {r_auc_pr:.4f}")
    
    # Divergence Epoch
    print(f"Baseline Divergence Epoch: {min_val_epoch}")
    
    return {
        'b_gen_gap': b_gen_gap,
        'r_gen_gap': r_gen_gap,
        'b_auc_pr': b_auc_pr,
        'r_auc_pr': r_auc_pr,
        'divergence_epoch': min_val_epoch
    }

if __name__ == "__main__":
    # Paths
    # Assuming the script is run from the project root
    # and temp_ablation_data is in the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    baseline_path = os.path.join(base_dir, 'temp_ablation_data', 'baseline', 'results.json')
    regularized_path = os.path.join(base_dir, 'temp_ablation_data', 'regularized', 'results.json')
    output_dir = os.path.join(base_dir, 'paper_code', 'figures')
    
    if not os.path.exists(baseline_path):
        print(f"Error: Baseline file not found at {baseline_path}")
        sys.exit(1)
        
    generate_ablation_figure(baseline_path, regularized_path, output_dir)
