"""
generate_ptm_table.py

Generates Table: Impact of PTM features on predictive performance.
Using actual HARDCODED results from the user's experiment to ensure accuracy.
"""

import pandas as pd

def generate_ptm_comparison_table():
    # User provided results
    # Global Test Set (AUC-ROC: 0.788 -> 0.810, AUC-PR: 0.214 -> 0.217)
    # Glycosylated Subset (AUC-ROC: 0.786 -> 0.808, AUC-PR: 0.210 -> 0.214)
    # Non-Glycosylated (AUC-ROC: 0.856 -> 0.870, AUC-PR: 0.388 -> 0.349)
    
    data = [
        ["Global Test Set", "AUC-ROC", 0.788, 0.810],
        ["Global Test Set", "AUC-PR", 0.214, 0.217],
        ["Glycosylated Subset", "AUC-ROC", 0.786, 0.808],
        ["Glycosylated Subset", "AUC-PR", 0.210, 0.214],
        ["Non-Glycosylated", "AUC-ROC", 0.856, 0.870],
        ["Non-Glycosylated", "AUC-PR", 0.388, 0.349],
    ]
    
    # Calculate improvement
    rows = []
    for row in data:
        subset, metric, base, aware = row
        imp = (aware - base) / base * 100
        imp_str = f"+{imp:.1f}\%" if imp > 0 else f"{imp:.1f}\%"
        
        # Determine bolding
        base_str = f"{base:.3f}"
        aware_str = f"{aware:.3f}"
        
        if aware > base:
            aware_str = f"\\textbf{{{aware_str}}}"
        elif base > aware:
            base_str = f"\\textbf{{{base_str}}}"
            
        rows.append([subset, metric, base_str, aware_str, imp_str])
        
    return rows

def generate_latex(rows):
    latex = r"""\begin{table}[h]
\caption{Impact of PTM features on predictive performance. Comparison of the baseline model versus the PTM-aware model across different antigen subsets. The 'Glycosylated' subset refers to proteins with $\ge 1$ annotated site.}\label{tab:ptm_comparison}%
\begin{tabular}{@{}llrrr@{}}
\toprule
Dataset Subset & Metric & Baseline (No PTMs) & Glyco-Aware (With PTMs) & Relative Improv. \\
\midrule
"""
    
    # Group by subset for merged cells
    current_subset = ""
    for idx, row in enumerate(rows):
        subset, metric, base, aware, imp = row
        
        if subset != current_subset:
            latex += f"\\multirow{{2}}{{*}}{{{subset}}} & {metric} & {base} & {aware} & {imp} \\\\\n"
            current_subset = subset
        else:
            latex += f" & {metric} & {base} & {aware} & {imp} \\\\\n"
            if idx < len(rows) - 1:
                latex += "\\midrule\n"
    
    latex += r"""\botrule
\end{tabular}
\end{table}
"""
    return latex

if __name__ == "__main__":
    rows = generate_ptm_comparison_table()
    print(generate_latex(rows))
