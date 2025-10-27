# visualize_glycosylation.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import random
import config # To get the path to your final data

def plot_glycosylation_for_one_chain(df: pd.DataFrame, pdb_id: str):
    """
    Generates a violin plot comparing the distribution of 'dist_to_glycosylation'
    for epitope vs. non-epitope residues for a single specified protein chain.

    Args:
        df (pd.DataFrame): The final dataframe containing all residue features.
        pdb_id (str): The PDB ID of the protein chain to plot (case-insensitive).
    """
    # --- 1. Data Preparation ---
    protein_df = df[df['pdb_id'].str.lower() == pdb_id.lower()].copy()

    if protein_df.empty:
        print(f"Error: PDB ID '{pdb_id}' not found in the DataFrame.")
        return

    if 'dist_to_glycosylation' not in protein_df.columns:
        print(f"Error: 'dist_to_glycosylation' column not found. Did you run feature engineering with glycosylation mode enabled?")
        return
        
    if protein_df['dist_to_glycosylation'].min() >= config.MAX_GLYCOSYLATION_DISTANCE:
        print(f"Warning: Protein '{pdb_id}' has no glycosylation sites within the specified distance. Plot will not be generated.")
        return
        
    protein_df['Epitope Status'] = protein_df['is_epitope'].apply(lambda x: 'Epitope' if x == 1 else 'Non-Epitope')

    # --- 2. Plotting ---
    plt.figure(figsize=(8, 7))
    sns.violinplot(data=protein_df, x='Epitope Status', y='dist_to_glycosylation', 
                   inner='quartile', palette='muted', hue='Epitope Status', legend=False)
    sns.stripplot(data=protein_df, x='Epitope Status', y='dist_to_glycosylation', color='black', alpha=0.5, jitter=0.2)
    
    # --- 3. Titles and Labels ---
    plt.title(f'Glycosylation Distance vs. Epitope Status for PDB: {pdb_id.upper()}', fontsize=16)
    plt.ylabel('Distance to Nearest Glycosylation Site (Å)', fontsize=12)
    plt.xlabel('Residue Type', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    print(f"Displaying plot for {pdb_id.upper()}. Close the plot window to continue.")
    plt.savefig(f'utils/glycosylation_distance_{pdb_id.lower()}.png', dpi=300)


def plot_glycosylation_for_multiple_chains(df: pd.DataFrame, num_chains: int = None, outlier_threshold: float = 5.0, output_file: str = "utils/outlier_pdb_ids_to_exclude.txt"):
    """
    Generates a violin plot and investigates outliers for multiple protein chains.

    Args:
        df (pd.DataFrame): The final dataframe containing all residue features.
        num_chains (int, optional): The number of glycosylated chains to randomly sample. If None, all are used.
        outlier_threshold (float, optional): The distance (in Å) below which an epitope is considered an outlier.
    """
    # --- 1. Data Preparation ---
    if 'dist_to_glycosylation' not in df.columns:
        print(f"Error: 'dist_to_glycosylation' column not found.")
        return
        
    glycosylated_pdb_ids = df[df['dist_to_glycosylation'] < config.MAX_GLYCOSYLATION_DISTANCE]['pdb_id'].unique()

    if len(glycosylated_pdb_ids) == 0:
        print("No glycosylated proteins found in the dataset to plot.")
        return

    plot_title = ""
    if num_chains is not None and num_chains < len(glycosylated_pdb_ids):
        selected_pdb_ids = random.sample(list(glycosylated_pdb_ids), num_chains)
        plot_title = f'Glycosylation Distance vs. Epitope Status (Random Sample of {num_chains} Chains)'
    else:
        selected_pdb_ids = glycosylated_pdb_ids
        plot_title = f'Glycosylation Distance vs. Epitope Status (All {len(glycosylated_pdb_ids)} Glycosylated Chains)'
    
    subset_df = df[df['pdb_id'].isin(selected_pdb_ids)].copy()
    subset_df['Epitope Status'] = subset_df['is_epitope'].apply(lambda x: 'Epitope' if x == 1 else 'Non-Epitope')

    # --- 2. NEW: Outlier Investigation ---
    print(f"\n--- Investigating Epitope Outliers (Distance < {outlier_threshold} Å) ---")
    
    # Define the columns to display, using their actual names from the DataFrame
    columns_to_inspect = [
        'pdb_id', 'antigen_chain', 'res_id', 'res_name', 
        'is_epitope', 'is_glycosylated', 'dist_to_glycosylation'
    ]
    
    # Check if all required columns exist before filtering
    if all(col in subset_df.columns for col in columns_to_inspect):
        outliers_df = df[
            (df['is_epitope'] == 1) & 
            (df['dist_to_glycosylation'] < outlier_threshold)
        ]
        total_epitopes = len(df[df['is_epitope'] == 1])

        if not outliers_df.empty:
            outlier_pdb_ids = outliers_df['pdb_id'].unique()
        
            print(f"Found {len(outliers_df)} of {total_epitopes} epitope residues closer than {outlier_threshold} Å to a glycan.")
            print(f"These outliers are found in {len(outlier_pdb_ids)} unique protein chains.")
            
            # --- NEW: Save the unique PDB IDs to a text file ---
            try:
                with open(output_file, 'w') as f:
                    for pdb_id in sorted(list(outlier_pdb_ids)):
                        f.write(f"{pdb_id}\n")
                print(f"Successfully saved the list of {len(outlier_pdb_ids)} outlier PDB IDs to '{output_file}'")
            except IOError as e:
                print(f"Error: Could not write to file '{output_file}'. Reason: {e}")
        else:
            print(f"No epitope residues found closer than {outlier_threshold} Å to a glycan.")
    else:
        print("Could not perform outlier analysis: one or more required columns are missing from the DataFrame.")
    print("--------------------------------------------------")

    # --- 3. Statistical Analysis ---
    epitope_distances = subset_df[subset_df['is_epitope'] == 1]['dist_to_glycosylation']
    non_epitope_distances = subset_df[subset_df['is_epitope'] == 0]['dist_to_glycosylation']

    if len(epitope_distances) > 0 and len(non_epitope_distances) > 0:
        stat, p_value = mannwhitneyu(epitope_distances, non_epitope_distances, alternative='two-sided')
        print(f"\n--- Mann-Whitney U Test Results ---")
        print(f"  P-value: {p_value:.4g}")
        if p_value < 0.05:
            print("  Result: The distributions are significantly different.")
        else:
            print("  Result: No significant difference found between the distributions.")
        print("-----------------------------------")
    else:
        p_value = float('nan')

    # --- 4. Plotting ---
    plt.figure(figsize=(10, 8))
    ax = sns.violinplot(data=subset_df, x='Epitope Status', y='dist_to_glycosylation', 
                        inner='quartile', palette='viridis', hue='Epitope Status', legend=False)
    
    # --- 5. Titles, Labels, and Annotations ---
    ax.set_title(plot_title, fontsize=16)
    ax.set_ylabel('Distance to Nearest Glycosylation Site (Å)', fontsize=12)
    ax.set_xlabel('Residue Type', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if not pd.isna(p_value):
        if p_value == 0.0:
            annotation_text = 'Mann-Whitney U p-value < 2.2e-308'
        else:
            annotation_text = f'Mann-Whitney U p-value = {p_value:.3g}'
        y_max = subset_df['dist_to_glycosylation'].max()
        ax.text(0.5, y_max * 0.85, annotation_text, 
                horizontalalignment='center', size='large', color='black', weight='semibold')
    
    print(f"\nDisplaying aggregate plot. Close the plot window to continue.")
    plt.savefig('utils/glycosylation_distance_multiple_chains.png', dpi=300)


if __name__ == '__main__':
    # --- Example Usage ---
    print("Loading final feature dataframe...")
    try:
        final_df = pd.read_pickle(config.FINAL_DATAFRAME_PATH)
        print("Data loaded successfully.")
        
        # Example 1: Plot for a single chain
        # pdb_to_inspect = '1a14' 
        # plot_glycosylation_for_one_chain(final_df, pdb_to_inspect)
        
        # Example 2: Plot for ALL chains and find outliers closer than 5 Å
        plot_glycosylation_for_multiple_chains(final_df, outlier_threshold=5.0)

        # Example 3: Plot for a sample of 20 chains and find outliers closer than 3 Å
        # plot_glycosylation_for_multiple_chains(final_df, num_chains=20, outlier_threshold=3.0)

    except FileNotFoundError:
        print(f"\nERROR: Could not find the data file at '{config.FINAL_DATAFRAME_PATH}'.")
        print("Please run the feature_engineering.py script first to generate this file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")