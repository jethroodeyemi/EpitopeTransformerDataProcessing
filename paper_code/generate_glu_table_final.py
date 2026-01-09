"""
generate_glu_table_final.py

Generates the final LaTeX table for GLU comparison.
"""
import os
import json

def load_results(path):
    if not os.path.exists(path): return {'final':0, 'final_test_auc_pr':0}
    with open(path, 'r') as f: return json.load(f)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
relu_path = os.path.join(base_dir, 'temp_ablation_data', 'relu', 'results.json')
sigglu_path = os.path.join(base_dir, 'temp_ablation_data', 'sigglu', 'results.json')
tanhglu_path = os.path.join(base_dir, 'temp_ablation_data', 'regularized', 'results.json') # "ablation_full"

relu = load_results(relu_path)
sig = load_results(sigglu_path)
tanh = load_results(tanhglu_path)

# Metrics
metrics = [
    ("Standard MLP (ReLU)", relu.get('final', 0), relu.get('final_test_auc_pr', 0)),
    ("Gated Transformation (Sigmoid-GLU)", sig.get('final', 0), sig.get('final_test_auc_pr', 0)),
    ("Gated Transformation (Tanh-GLU)", tanh.get('final', 0), tanh.get('final_test_auc_pr', 0))
]

# Find best
best_roc = max(m[1] for m in metrics)
best_pr = max(m[2] for m in metrics)

print(r"\begin{table}[h]")
print(r"\caption{Performance comparison of feed-forward architectures.}\label{tab:glu_comparison}")
print(r"\begin{tabular}{@{}lrr@{}}")
print(r"\toprule")
print(r"Architecture & AUC-ROC & AUC-PR \\")
print(r"\midrule")

for name, roc, pr in metrics:
    roc_str = f"{roc:.3f}"
    pr_str = f"{pr:.3f}"
    
    if roc == best_roc: roc_str = f"\\textbf{{{roc_str}}}"
    if pr == best_pr: pr_str = f"\\textbf{{{pr_str}}}"
    
    if "Sigmoid" in name: # Bold the row name if it's the winner
         if roc == best_roc: name = f"\\textbf{{{name}}}"
    
    print(f"{name} & {roc_str} & {pr_str} \\\\")

print(r"\botrule")
print(r"\end{tabular}")
print(r"\end{table}")

# Calculate improvement stats
relu_pr = relu.get('final_test_auc_pr', 0)
sig_pr = sig.get('final_test_auc_pr', 0)
imp = (sig_pr - relu_pr) / relu_pr * 100
print(f"\n% Stats for text: Sigmoid-GLU vs ReLU Improvement: {imp:.1f}% ({sig_pr:.3f} vs {relu_pr:.3f})")
