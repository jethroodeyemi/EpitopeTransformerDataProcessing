"""
generate_split_statistics.py

Generates Table: Data partitioning statistics following cluster-based splitting.
Reads the pre-computed data splits and calculates the number of proteins 
and total residues for each split (train, val, test).
"""

import os
import json
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from tqdm import tqdm
import warnings

# Suppress PDB parser warnings
warnings.filterwarnings('ignore')


def count_residues_in_pdb(pdb_path):
    """
    Count the number of standard amino acid residues in a PDB file.
    
    Args:
        pdb_path: Path to the PDB file
        
    Returns:
        Number of residues in the structure
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
        residue_count = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Only count standard amino acids
                    if is_aa(residue, standard=True):
                        residue_count += 1
        return residue_count
    except Exception as e:
        print(f"Error parsing {pdb_path}: {e}")
        return 0


def generate_split_statistics():
    """
    Calculates split statistics from the pre-computed data splits JSON file.
    
    Returns:
        Dictionary with split names and their statistics (protein count, residue count)
    """
    # File paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    splits_file = os.path.join(base_dir, "output", "data_splits_on_spike_proteins_cluster.json")
    pdb_dir = os.path.join(base_dir, "cleaned_pdb_files")
    
    # Load the splits
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    stats = {}
    
    for split_name in ['train', 'val', 'test']:
        pdb_ids = splits.get(split_name, [])
        protein_count = len(pdb_ids)
        
        # Count total residues
        total_residues = 0
        missing_pdbs = []
        
        print(f"\nProcessing {split_name} split ({protein_count} proteins)...")
        
        for pdb_id in tqdm(pdb_ids, desc=f"Counting residues in {split_name}"):
            pdb_path = os.path.join(pdb_dir, f"{pdb_id}_cleaned.pdb")
            
            if os.path.exists(pdb_path):
                total_residues += count_residues_in_pdb(pdb_path)
            else:
                missing_pdbs.append(pdb_id)
        
        if missing_pdbs:
            print(f"  Warning: {len(missing_pdbs)} PDB files not found in {split_name} split")
        
        stats[split_name] = {
            'proteins': protein_count,
            'residues': total_residues
        }
    
    return stats


def print_table(stats):
    """
    Prints the split statistics as a formatted table.
    """
    print("\n" + "="*60)
    print("Table: Data partitioning statistics following cluster-based splitting.")
    print("="*60)
    print(f"{'Split':<15} | {'Number of Proteins':>20} | {'Total Residues':>15}")
    print("-"*60)
    
    # Map internal names to display names
    display_names = {'train': 'Training', 'val': 'Validation', 'test': 'Test'}
    
    for split_name in ['train', 'val', 'test']:
        data = stats[split_name]
        display_name = display_names[split_name]
        print(f"{display_name:<15} | {data['proteins']:>20,} | {data['residues']:>15,}")
    
    # Total row
    total_proteins = sum(s['proteins'] for s in stats.values())
    total_residues = sum(s['residues'] for s in stats.values())
    print("-"*60)
    print(f"{'Total':<15} | {total_proteins:>20,} | {total_residues:>15,}")
    print("="*60)


def generate_latex_table(stats):
    """
    Generates LaTeX code for the table in the required format.
    """
    display_names = {'train': 'Training', 'val': 'Validation', 'test': 'Test'}
    
    latex = r"""\begin{table}[h]
\caption{Data partitioning statistics following cluster-based splitting.}\label{tab:split_stats}%
\begin{tabular}{@{}lrr@{}}
\toprule
Split & Number of Proteins & Total Residues \\
\midrule
"""
    
    for split_name in ['train', 'val', 'test']:
        data = stats[split_name]
        display_name = display_names[split_name]
        latex += f"{display_name} & {data['proteins']:,} & {data['residues']:,} \\\\\n"
    
    latex += r"""\botrule
\end{tabular}
\end{table}
"""
    return latex


def generate_markdown_table(stats):
    """
    Generates Markdown code for the table.
    """
    display_names = {'train': 'Training', 'val': 'Validation', 'test': 'Test'}
    
    md = "\n| Split | Number of Proteins | Total Residues |\n"
    md += "| :--- | ---: | ---: |\n"
    
    for split_name in ['train', 'val', 'test']:
        data = stats[split_name]
        display_name = display_names[split_name]
        md += f"| {display_name} | {data['proteins']:,} | {data['residues']:,} |\n"
    
    return md


if __name__ == "__main__":
    # Generate statistics
    stats = generate_split_statistics()
    
    # Print formatted table
    print_table(stats)
    
    # Print Markdown version
    print("\n--- Markdown Format ---")
    print(generate_markdown_table(stats))
    
    # Print LaTeX version
    print("\n--- LaTeX Format ---")
    print(generate_latex_table(stats))
