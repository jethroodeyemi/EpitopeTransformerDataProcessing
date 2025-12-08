"""
generate_glycosylation_table.py

Generates Table: Statistics of retrieved glycosylation annotations.
Reads the glycosylation feature analysis CSV and calculates statistics
for the glycosylation annotation pipeline.
"""

import os
import sys
import pandas as pd

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def generate_glycosylation_statistics():
    """
    Calculates glycosylation statistics from the feature_rich_analysis.csv file.
    
    Returns:
        Dictionary with metric names and their corresponding values.
    """
    # File paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    glyco_csv = os.path.join(base_dir, config.GLYCOSYLATION_DATA_PATH)
    
    # Load the glycosylation analysis data
    df = pd.read_csv(glyco_csv)
    
    # Total proteins analyzed
    total_proteins = len(df)
    
    # Proteins with mapped UniProt IDs (those with non-empty glycosylation_info or is_glycosylated is valid)
    # If a protein has glycosylation info or is marked as glycosylated/not glycosylated, it has a UniProt mapping
    # Check for proteins where we successfully queried UniProt (even if no glycosylation was found)
    # The presence of 'Yes' or 'No' in is_glycosylated indicates successful UniProt mapping
    proteins_with_uniprot = df[df['is_glycosylated'].isin(['Yes', 'No'])].shape[0]
    
    # Glycosylated proteins identified
    glycosylated_proteins = df[df['is_glycosylated'] == 'Yes'].shape[0]
    
    # Total glycosylation sites
    # Count from glycosylation_info column - each site is separated by '; '
    total_sites = 0
    for idx, row in df.iterrows():
        if pd.notna(row['glycosylation_info']) and row['glycosylation_info'] != '':
            sites = str(row['glycosylation_info']).split('; ')
            # Filter out empty strings
            sites = [s for s in sites if s.strip()]
            total_sites += len(sites)
    
    # Count N-linked and O-linked separately
    n_linked_count = 0
    o_linked_count = 0
    unknown_type_count = 0
    
    for idx, row in df.iterrows():
        if pd.notna(row['glycan_classifications']) and row['glycan_classifications'] != '':
            classifications = str(row['glycan_classifications']).split('; ')
            for classification in classifications:
                if 'N-linked' in classification:
                    n_linked_count += 1
                elif 'O-linked' in classification:
                    o_linked_count += 1
                elif 'Unknown' in classification:
                    unknown_type_count += 1
    
    # Compile statistics
    stats = {
        "Total Proteins Analyzed": total_proteins,
        "Proteins with Mapped UniProt IDs": proteins_with_uniprot,
        "Glycosylated Proteins Identified": glycosylated_proteins,
        "Total Glycosylation Sites": total_sites,
    }
    
    # Additional detailed stats
    detailed_stats = {
        "N-linked Sites": n_linked_count,
        "O-linked Sites": o_linked_count,
        "Unknown Type Sites": unknown_type_count,
    }
    
    return stats, detailed_stats


def print_table(stats):
    """
    Prints the glycosylation statistics as a formatted table.
    """
    print("\n" + "="*60)
    print("Table: Statistics of retrieved glycosylation annotations.")
    print("="*60)
    print(f"{'Metric':<40} | {'Value':>15}")
    print("-"*60)
    for metric, value in stats.items():
        print(f"{metric:<40} | {value:>15,}")
    print("="*60)


def generate_latex_table(stats):
    """
    Generates LaTeX code for the table in the required format.
    """
    latex = r"""\begin{table}[h]
\caption{Statistics of retrieved glycosylation annotations.}\label{tab:glyco_stats}%
\begin{tabular}{@{}lr@{}}
\toprule
Metric & Value \\
\midrule
"""
    
    for metric, value in stats.items():
        latex += f"{metric} & {value:,} \\\\\n"
    
    latex += r"""\botrule
\end{tabular}
\end{table}
"""
    return latex


def generate_markdown_table(stats):
    """
    Generates Markdown code for the table.
    """
    md = "\n| Metric | Value |\n"
    md += "| :--- | ---: |\n"
    
    for metric, value in stats.items():
        md += f"| {metric} | {value:,} |\n"
    
    return md


def generate_updated_text(stats):
    """
    Generates the updated paragraph text with actual values.
    """
    text = f"""
Updated Paragraph with Actual Values:
--------------------------------------
To model the steric shielding effects of glycans, which are frequently disordered 
and absent from crystallographic coordinates, we established a dynamic mapping 
pipeline linking structural entries to sequence-based annotations. For each antigen 
chain, we queried the RCSB PDB GraphQL interface to resolve the specific polymer 
entity and retrieve its corresponding UniProtKB accession identifier. Using these 
identifiers, we extracted all annotated N-linked and O-linked glycosylation sites 
via the UniProt REST API. This pipeline successfully mapped {stats['Proteins with Mapped UniProt IDs']:,} 
of {stats['Total Proteins Analyzed']:,} proteins to UniProt identifiers, identifying 
{stats['Glycosylated Proteins Identified']:,} glycosylated proteins with a total of 
{stats['Total Glycosylation Sites']:,} annotated glycosylation sites (Table~\\ref{{tab:glyco_stats}}). 
This ensures that residues are characterized based on their biological potential for 
glycosylation rather than the presence of resolved heteroatoms in the PDB file, 
thereby accounting for flexible glycans lost during structural determination.
"""
    return text


if __name__ == "__main__":
    # Generate statistics
    stats, detailed_stats = generate_glycosylation_statistics()
    
    # Print formatted table
    print_table(stats)
    
    # Print detailed statistics
    print("\n--- Detailed Glycosylation Type Breakdown ---")
    for metric, value in detailed_stats.items():
        print(f"  {metric}: {value:,}")
    
    # Print Markdown version
    print("\n--- Markdown Format ---")
    print(generate_markdown_table(stats))
    
    # Print LaTeX version
    print("\n--- LaTeX Format ---")
    print(generate_latex_table(stats))
    
    # Print updated text
    print(generate_updated_text(stats))
