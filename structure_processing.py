# structure_processing.py

import os
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Residue import Residue
from tqdm import tqdm

class ChainSelect(Select):
    """Helper class to select specific chains from a PDB structure."""
    def __init__(self, chains_to_keep):
        self.chains_to_keep = set(chains_to_keep)

    def accept_chain(self, chain):
        return chain.get_id() in self.chains_to_keep

    def accept_residue(self, residue: Residue) -> int:
        return 1 if residue.get_id()[0] == ' ' else 0

def clean_pdbs(df, pdb_dir, cleaned_pdb_dir, antigen_only_pdb_dir):
    """
    For each PDB, creates two cleaned versions:
    1. A file with antibody (H+L) and antigen chains for label generation (_cleaned.pdb).
    2. A file with ONLY the antigen chain for feature calculation to prevent data leakage (_antigen_only.pdb).
    """
    print("\n--- Step 3: Cleaning PDB files ---")
    parser = PDBParser()
    io = PDBIO()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning PDBs"):
        pdb_id = row['pdb']
        input_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        complex_output_path = os.path.join(cleaned_pdb_dir, f"{pdb_id}_cleaned.pdb")
        antigen_only_output_path = os.path.join(antigen_only_pdb_dir, f"{pdb_id}_antigen_only.pdb")

        if not os.path.exists(input_path):
            print(f"Warning: PDB file for {pdb_id} not found. Skipping cleaning.")
            continue
        
        chains_to_keep = [row['Hchain'], row['Lchain'], row['antigen_chain']]

        if os.path.exists(complex_output_path) and os.path.exists(antigen_only_output_path):
            continue
        
        try:
            structure = parser.get_structure(pdb_id, input_path)
            if not os.path.exists(complex_output_path):
                chains_for_complex = [row['Hchain'], row['Lchain'], row['antigen_chain']]
                io.set_structure(structure)
                io.save(complex_output_path, ChainSelect(chains_for_complex))

            if not os.path.exists(antigen_only_output_path):
                chains_for_antigen = [row['antigen_chain']]
                io.set_structure(structure)
                io.save(antigen_only_output_path, ChainSelect(chains_for_antigen))
                   
        except Exception as e:
            print(f"Could not process or clean PDB {pdb_id}. Error: {e}")