# data_preparation.py

import os
import pandas as pd
from Bio.PDB import PDBList
import requests
from tqdm import tqdm

def filter_and_deduplicate_tsv(input_path, output_path):
    """
    Reads the TSV, filters out entries with multiple antigen chains,
    and then removes rows with duplicate PDB IDs, keeping the first.
    """
    print("--- Step 1: Deduplicating and Filtering TSV ---")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input TSV file not found at: {input_path}")
        
    df = pd.read_csv(input_path, sep='\t')
    print(f"Original entries: {len(df)}")
    
    df['antigen_chain'] = df['antigen_chain'].astype(str)
    
    initial_count = len(df)
    # Filter out entries with multiple antigen chains and NA values
    df_single_chain = df[
        (~df['antigen_chain'].str.contains('\|', na=False)) & 
        (df['antigen_chain'] != 'nan') &
        (df['antigen_type'] == 'protein')
    ]
    
    removed_count = initial_count - len(df_single_chain)
    if removed_count > 0:
        print(f"Removed {removed_count} entries with multiple antigen chains.")
    print(f"Entries after filtering for single antigen chains: {len(df_single_chain)}")
    
    df_deduped = df_single_chain.drop_duplicates(subset=['pdb'], keep='first').reset_index(drop=True)
    print(f"Entries after PDB ID deduplication: {len(df_deduped)}")
    
    df_deduped.to_csv(output_path, sep='\t', index=False)
    print(f"Deduplicated and filtered TSV saved to: {output_path}")

     # --- EXCLUDE OUTLIER PDBs (These PDBs have epitopes on or too close to glycans) ---
    outlier_file = "utils/outlier_pdb_ids_to_exclude.txt"
    if os.path.exists(outlier_file):
        print(f"\n--- Excluding outlier PDBs listed in '{outlier_file}' ---")
        with open(outlier_file, 'r') as f:
            outlier_ids_to_exclude = {line.strip().lower() for line in f}
        
        initial_count = len(df_deduped)
        # Filter the DataFrame, making sure to compare lowercase IDs
        df_deduped = df_deduped[~df_deduped['pdb'].str.lower().isin(outlier_ids_to_exclude)]
        
        print(f"Removed {initial_count - len(df_deduped)} outlier chains from consideration.")
        print(f"Remaining chains for processing: {len(df_deduped)}")
    else:
        print(f"\nWarning: Outlier exclusion file '{outlier_file}' not found. Proceeding with all data.")

    return df_deduped

def download_pdbs(df, pdb_dir):
    """
    Downloads PDB files for each unique PDB ID in the dataframe.
    """
    print("\n--- Step 2: Downloading PDB files ---")
    pdb_list = PDBList()
    pdb_ids = df['pdb'].unique()
    
    for pdb_id in tqdm(pdb_ids, desc="Downloading PDBs"):
        output_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if not os.path.exists(output_path):
            try:
                # Retrieve and rename to a consistent format
                retrieved_file = pdb_list.retrieve_pdb_file(
                    pdb_id, pdir=pdb_dir, file_format='pdb', overwrite=False
                )
                if os.path.exists(retrieved_file):
                    os.rename(retrieved_file, output_path)
                    continue

                url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    with open(output_path, 'w') as f:
                        f.write(response.text)
                        
            except Exception as e:
                print(f"Could not download PDB {pdb_id}. Error: {e}")