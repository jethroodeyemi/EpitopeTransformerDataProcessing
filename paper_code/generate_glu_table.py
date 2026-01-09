"""
generate_glu_table.py

Generates Table: Performance comparison of feed-forward architectures.
Reads results from ReLU, Sigmoid-GLU, and Tanh-GLU (Regularized) experiments.
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
            
    return data

def get_metrics(data):
    """Extract metrics from data dictionary."""
    if data is None:
        return {'AUC-ROC': 0, 'AUC-PR': 0}
        
    # AUC-ROC (Final test score - using 'final' key which is best val epoch)
    auc_roc = data.get('final', 0)
    
    # AUC-PR
    auc_pr = data.get('final_test_auc_pr', 0)
    
    return {
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr
    }

def calculate_improvement(baseline, current):
    if baseline == 0:
        return 0
    return (current - baseline) / baseline * 100

def generate_latex_table(relu_metrics, sigglu_metrics, tanhglu_metrics):
    """Generate LaTeX table."""
    
    # Format values
    r_roc = f"{relu_metrics['AUC-ROC']:.3f}"
    r_pr = f"{relu_metrics['AUC-PR']:.3f}"
    
    s_roc = f"{sigglu_metrics['AUC-ROC']:.3f}"
    s_pr = f"{sigglu_metrics['AUC-PR']:.3f}"
    
    t_roc = f"{tanhglu_metrics['AUC-ROC']:.3f}"
    t_pr = f"{tanhglu_metrics['AUC-PR']:.3f}"
    
    # Identify best
    best_roc_val = max(relu_metrics['AUC-ROC'], sigglu_metrics['AUC-ROC'], tanhglu_metrics['AUC-ROC'])
    best_pr_val = max(relu_metrics['AUC-PR'], sigglu_metrics['AUC-PR'], tanhglu_metrics['AUC-PR'])
    
    if relu_metrics['AUC-ROC'] == best_roc_val: r_roc = f"\\textbf{{{r_roc}}}"
    if sigglu_metrics['AUC-ROC'] == best_roc_val: s_roc = f"\\textbf{{{s_roc}}}"
    if tanhglu_metrics['AUC-ROC'] == best_roc_val: t_roc = f"\\textbf{{{t_roc}}}"
    
    if relu_metrics['AUC-PR'] == best_pr_val: r_pr = f"\\textbf{{{r_pr}}}"
    if sigglu_metrics['AUC-PR'] == best_pr_val: s_pr = f"\\textbf{{{s_pr}}}"
    if tanhglu_metrics['AUC-PR'] == best_pr_val: t_pr = f"\\textbf{{{t_pr}}}"

    # F1 score is removed as it's not available in the results
    latex = r"""\begin{table}[h]
\caption{Performance comparison of feed-forward architectures.}\label{tab:glu_comparison}%
\begin{tabular}{@{}lrr@{}}
\toprule
Architecture & AUC-ROC & AUC-PR \\
\midrule
Standard MLP (ReLU) & """ + r_roc + r""" & """ + r_pr + r""" \\
Gated Transformation (Sigmoid-GLU) & """ + s_roc + r""" & """ + s_pr + r""" \\
\textbf{Gated Transformation (Tanh-GLU)} & """ + t_roc + r""" & """ + t_pr + r""" \\
\botrule
\end{tabular}
\end{table}
"""
    return latex

def generate_update_stats(relu_metrics, tanhglu_metrics):
    """Generate text statistics."""
    imp_pr = calculate_improvement(relu_metrics['AUC-PR'], tanhglu_metrics['AUC-PR'])
    
    return {
        'relu_pr': relu_metrics['AUC-PR'],
        'tanhglu_pr': tanhglu_metrics['AUC-PR'],
        'improvement_pct': imp_pr
    }

if __name__ == "__main__":
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    relu_path = os.path.join(base_dir, 'temp_ablation_data', 'relu', 'results.json')
    sigglu_path = os.path.join(base_dir, 'temp_ablation_data', 'sigglu', 'results.json')
    tanhglu_path = os.path.join(base_dir, 'temp_ablation_data', 'regularized', 'results.json')
    
    relu_data = load_results(relu_path)
    sigglu_data = load_results(sigglu_path)
    tanhglu_data = load_results(tanhglu_path)
    
    if relu_data and sigglu_data and tanhglu_data:
        r_metrics = get_metrics(relu_data)
        s_metrics = get_metrics(sigglu_data)
        t_metrics = get_metrics(tanhglu_data)
        
        print("--- Metrics ---")
        print(f"ReLU: {r_metrics}")
        print(f"Sigmoid-GLU: {s_metrics}")
        print(f"Tanh-GLU: {t_metrics}")
        
        print("\n--- LaTeX Table ---")
        print(generate_latex_table(r_metrics, s_metrics, t_metrics))
        
        print("\n--- Text Stats ---")
        stats = generate_update_stats(r_metrics, t_metrics)
        print(f"Improvement in AUC-PR: {stats['improvement_pct']:.1f}%")
        print(f"Values: {stats['relu_pr']:.3f} vs {stats['tanhglu_pr']:.3f}")
        
    else:
        print("Error: Could not load results files.")
        if not relu_data: print(f"Missing: {relu_path}")
        if not sigglu_data: print(f"Missing: {sigglu_path}")
        if not tanhglu_data: print(f"Missing: {tanhglu_path}")
