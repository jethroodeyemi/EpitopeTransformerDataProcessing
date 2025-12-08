"""
generate_curation_table.py

Generates Table 1: Dataset curation and filtering statistics.
This script replicates the filtering logic from data_preparation.py to 
calculate the exact counts at each stage of the data processing pipeline.
"""

import os
import pandas as pd

def generate_curation_statistics():
    """
    Calculates the number of complexes at each stage of the data curation pipeline.
    Returns a dictionary with processing stage names and their corresponding counts.
    """
    # File paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_tsv = os.path.join(base_dir, "dataset.tsv")
    outlier_file = os.path.join(base_dir, "utils", "outlier_pdb_ids_to_exclude.txt")
    
    # Read the initial dataset
    df = pd.read_csv(input_tsv, sep='\t')
    initial_count = len(df)
    
    # Ensure antigen_chain is string type
    df['antigen_chain'] = df['antigen_chain'].astype(str)
    
    # Stage 1: Single-chain Protein Filter
    # Filter out entries with multiple antigen chains, NA values, and non-protein antigens
    df_single_chain = df[
        (~df['antigen_chain'].str.contains(r'\|', na=False)) & 
        (df['antigen_chain'] != 'nan') &
        (df['antigen_type'] == 'protein')
    ]
    single_chain_count = len(df_single_chain)
    
    # Stage 2: PDB Deduplication
    # Remove duplicate PDB IDs, keeping the first instance
    df_deduped = df_single_chain.drop_duplicates(subset=['pdb'], keep='first').reset_index(drop=True)
    deduped_count = len(df_deduped)
    
    # Stage 3: Outlier Exclusion
    # Exclude PDBs where epitope accessibility is compromised by glycan shielding
    if os.path.exists(outlier_file):
        with open(outlier_file, 'r') as f:
            outlier_ids = {line.strip().lower() for line in f if line.strip()}
        
        df_final = df_deduped[~df_deduped['pdb'].str.lower().isin(outlier_ids)]
        final_count = len(df_final)
    else:
        print(f"Warning: Outlier file not found at {outlier_file}")
        final_count = deduped_count
    
    # Compile statistics
    stats = {
        "Initial Dataset": initial_count,
        "Single-chain Protein Filter": single_chain_count,
        "PDB Deduplication": deduped_count,
        "Outlier Exclusion": final_count,
        "Final Dataset": final_count
    }
    
    return stats


def print_table(stats):
    """
    Prints the curation statistics as a formatted table.
    """
    print("\n" + "="*60)
    print("Table 1: Dataset curation and filtering statistics.")
    print("="*60)
    print(f"{'Processing Stage':<35} | {'Number of Complexes':>20}")
    print("-"*60)
    for stage, count in stats.items():
        print(f"{stage:<35} | {count:>20,}")
    print("="*60)


def generate_latex_table(stats):
    """
    Generates LaTeX code for the table.
    """
    latex = r"""
\begin{table}[h]
\centering
\caption{Dataset curation and filtering statistics.}
\label{tab:curation_stats}
\begin{tabular}{l r}
\toprule
\textbf{Processing Stage} & \textbf{Number of Complexes} \\
\midrule
"""
    for stage, count in stats.items():
        latex += f"{stage} & {count:,} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_markdown_table(stats):
    """
    Generates Markdown code for the table.
    """
    md = "\n| Processing Stage | Number of Complexes |\n"
    md += "| :--- | ---: |\n"
    for stage, count in stats.items():
        md += f"| {stage} | {count:,} |\n"
    return md


if __name__ == "__main__":
    # Generate statistics
    stats = generate_curation_statistics()
    
    # Print formatted table
    print_table(stats)
    
    # Print Markdown version
    print("\n--- Markdown Format ---")
    print(generate_markdown_table(stats))
    
    # Print LaTeX version
    print("\n--- LaTeX Format ---")
    print(generate_latex_table(stats))
