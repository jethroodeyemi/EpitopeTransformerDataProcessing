"""
generate_ablation_table.py

Generates Table: Ablation study results on the independent test set.
Reads the ablation results JSON files and outputs the LaTeX table.
"""

import os
import json
import sys

def load_results(path):
    """Load results from JSON file."""
    if not os.path.exists(path):
        return None
        
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Parse stringified lists if necessary
    for key in ['test_score']:
        if isinstance(data.get(key), str):
            data[key] = json.loads(data[key])
            
    return data

def get_metrics(data):
    """Extract metrics from data dictionary."""
    if data is None:
        return None
        
    # AUC-ROC (Final test score)
    auc_roc = data.get('final', 0)
    
    # AUC-PR
    auc_pr = data.get('final_test_auc_pr', 0)
    
    # Generalization Gap
    gen_gap = data.get('gen_gap', 0)
    
    return {
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr,
        'Gen. Gap': gen_gap
    }

def generate_latex_table(baseline_metrics, full_metrics):
    """Generate LaTeX table."""
    
    # Format values
    b_roc = f"{baseline_metrics['AUC-ROC']:.3f}"
    b_pr = f"{baseline_metrics['AUC-PR']:.3f}"
    b_gap = f"{baseline_metrics['Gen. Gap']:.2f}"
    
    f_roc = f"{full_metrics['AUC-ROC']:.3f}"
    f_pr = f"{full_metrics['AUC-PR']:.3f}"
    f_gap = f"{full_metrics['Gen. Gap']:.2f}"
    
    # Determine bolding (best values)
    # Full model should be better (higher ROC/PR, lower Gap)
    
    # ROC
    if full_metrics['AUC-ROC'] > baseline_metrics['AUC-ROC']:
        f_roc = f"\\textbf{{{f_roc}}}"
    else:
        b_roc = f"\\textbf{{{b_roc}}}"
        
    # PR
    if full_metrics['AUC-PR'] > baseline_metrics['AUC-PR']:
        f_pr = f"\\textbf{{{f_pr}}}"
    else:
        b_pr = f"\\textbf{{{b_pr}}}"
        
    # Gap (Lower is better)
    if full_metrics['Gen. Gap'] < baseline_metrics['Gen. Gap']:
        f_gap = f"\\textbf{{{f_gap}}}"
    else:
        b_gap = f"\\textbf{{{b_gap}}}"

    latex = r"""\begin{table}[h]
\caption{Ablation study results on the independent test set (split\_all).}\label{tab:ablation_metrics}%
\begin{tabular}{@{}lrrr@{}}
\toprule
Model Configuration & AUC-ROC & AUC-PR & Gen. Gap \\
\midrule
No Regularization (Baseline) & """ + b_roc + r""" & """ + b_pr + r""" & """ + b_gap + r""" \\
With Latent Manifold Regularization (Full) & """ + f_roc + r""" & """ + f_pr + r""" & """ + f_gap + r""" \\
\botrule
\end{tabular}
\end{table}
"""
    return latex

if __name__ == "__main__":
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    baseline_path = os.path.join(base_dir, 'temp_ablation_data', 'baseline', 'results.json')
    regularized_path = os.path.join(base_dir, 'temp_ablation_data', 'regularized', 'results.json')
    
    baseline_data = load_results(baseline_path)
    full_data = load_results(regularized_path)
    
    if baseline_data and full_data:
        b_metrics = get_metrics(baseline_data)
        f_metrics = get_metrics(full_data)
        
        print("--- Baseline Metrics ---")
        print(b_metrics)
        print("\n--- Full Model Metrics ---")
        print(f_metrics)
        
        print("\n--- LaTeX Table ---")
        print(generate_latex_table(b_metrics, f_metrics))
    else:
        print("Error: Could not load results files.")
