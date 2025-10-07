import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from tqdm import tqdm
import warnings
import sys

# --- Configuration ---
# Suppress PDBConstructionWarning, which is common with non-standard residues
warnings.filterwarnings("ignore", category=UserWarning)

PDB_DIR = Path("../cleaned_pdb_files")
REPORT_FILE = Path("./glycosylation_report.csv")
SABDAB_FILE = Path("../dataset.tsv") # The ground-truth file
OUTPUT_FILE = Path("./curated_complex_data.csv")

# --- Main Script ---

def main():
    """
    Main pipeline function to process PDBs, identify components via a direct
    one-to-one match with the SAbDab dataset, and map glycosylation data.
    """
    # --- Step 1: Load All Input Data ---
    if not PDB_DIR.is_dir():
        print(f"Error: PDB directory '{PDB_DIR}' not found.")
        sys.exit(1)
    if not REPORT_FILE.exists():
        print(f"Error: Glycosylation report '{REPORT_FILE}' not found.")
        sys.exit(1)
    if not SABDAB_FILE.exists():
        print(f"Error: SAbDab dataset '{SABDAB_FILE}' not found.")
        sys.exit(1)

    print(f"Loading glycosylation report from '{REPORT_FILE}'...")
    glycosylation_df = pd.read_csv(REPORT_FILE)
    
    print(f"Loading SAbDab antibody/antigen definitions from '{SABDAB_FILE}'...")
    sabdab_df = pd.read_csv(SABDAB_FILE, sep='\t', dtype={'antigen_chain': str})
    # Standardize PDB IDs to uppercase for consistent matching
    sabdab_df['pdb'] = sabdab_df['pdb'].str.upper()

    pdb_files = sorted(list(PDB_DIR.glob("*_cleaned.pdb")))
    if not pdb_files:
        print(f"Error: No '*_cleaned.pdb' files found in '{PDB_DIR}'.")
        sys.exit(1)
        
    print(f"Found {len(pdb_files)} PDB files to process.")

    # --- Step 2: Initialize PDB Parser and Results Storage ---
    parser = PDBParser(QUIET=True)
    processed_data = []

    # --- Step 3: Loop Through Each PDB File and Process ---
    for pdb_file in tqdm(pdb_files, desc="Processing PDB Complexes"):
        pdb_id = pdb_file.stem.replace("_cleaned", "").upper()

        report_entries = glycosylation_df[glycosylation_df['PDB_ID'] == pdb_id]
        sabdab_entries_for_pdb = sabdab_df[sabdab_df['pdb'] == pdb_id]

        if report_entries.empty or sabdab_entries_for_pdb.empty:
            continue # Skip PDBs that are not in both our reports

        try:
            structure = parser.get_structure(pdb_id, pdb_file)
            model = structure[0]
        except Exception as e:
            print(f"Warning: Could not parse {pdb_id}. Error: {e}")
            continue

        # --- Action 2.1.2: Isolate Components via Direct SAbDab Lookup ---
        target_antigen_chains = set(report_entries['Chain_ID'].astype(str))
        found_h_chains = set()
        found_l_chains = set()
        
        # For each antigen chain we care about in this PDB...
        for antigen_chain in target_antigen_chains:
            # Perform a direct, one-to-one string match in the SAbDab data for this PDB.
            match = sabdab_entries_for_pdb[sabdab_entries_for_pdb['antigen_chain'] == antigen_chain]
            
            if not match.empty:
                # Take the first match if there are duplicates
                sabdab_row = match.iloc[0]
                h_chain = sabdab_row['Hchain']
                l_chain = sabdab_row['Lchain']
                
                if pd.notna(h_chain) and h_chain != 'NA':
                    found_h_chains.add(str(h_chain))
                if pd.notna(l_chain) and l_chain != 'NA':
                    found_l_chains.add(str(l_chain))
        
        # --- Action 2.1.3: Map Glycosylation Data ---
        glycosylation_details = []
        glycosylated_antigens_in_complex = report_entries[report_entries['Is_Glycosylated'] == 'Yes']

        for _, row in glycosylated_antigens_in_complex.iterrows():
            chain_id = str(row['Chain_ID'])
            if chain_id not in [c.id for c in model]:
                continue
            
            res_nums_str = row.get('Glycosylation_Residue_Numbers')
            if pd.isna(res_nums_str) or not res_nums_str.strip():
                continue
            
            res_nums = [int(n.strip()) for n in res_nums_str.split(',')]
            antigen_chain_obj = model[chain_id]
            for res_num in res_nums:
                try:
                    residue = antigen_chain_obj[res_num]
                    glycosylation_details.append({
                        "chain_id": chain_id,
                        "residue_number": res_num,
                        "residue_name": residue.get_resname()
                    })
                except KeyError:
                    glycosylation_details.append({
                        "chain_id": chain_id,
                        "residue_number": res_num,
                        "residue_name": "NOT_IN_STRUCTURE"
                    })

        # --- Step 4: Consolidate and Store the Curation Results ---
        processed_data.append({
            "pdb_id": pdb_id,
            "antigen_chains": ",".join(sorted(list(target_antigen_chains))),
            "antibody_h_chains": ",".join(sorted(list(found_h_chains))),
            "antibody_l_chains": ",".join(sorted(list(found_l_chains))),
            "is_glycosylated": "Yes" if glycosylation_details else "No",
            "glycosylation_info": glycosylation_details
        })

    # --- Step 5: Convert to DataFrame and Save ---
    if not processed_data:
        print("No data was processed. Exiting.")
        return
        
    final_df = pd.DataFrame(processed_data)
    
    final_df['glycosylation_info_str'] = final_df['glycosylation_info'].apply(
        lambda details: "; ".join([f"{d['chain_id']}-{d['residue_name']}{d['residue_number']}" for d in details]) if details else ""
    )
    
    print(f"\nCuration complete. Saving structured data to '{OUTPUT_FILE}'...")
    
    final_df_to_save = final_df.drop(columns=['glycosylation_info'])
    final_df_to_save.to_csv(OUTPUT_FILE, index=False)
    
    pickle_output_file = OUTPUT_FILE.with_suffix('.pkl')
    final_df.to_pickle(pickle_output_file)
    
    print("\n--- Curation Summary ---")
    print(f"Total complexes processed: {len(final_df)}")
    found_ab_count = len(final_df[(final_df['antibody_h_chains'] != '') | (final_df['antibody_l_chains'] != '')])
    print(f"Complexes with antibody chains identified from SAbDab: {found_ab_count}")
    print(f"Complexes with mapped glycosylation sites: {len(final_df[final_df['is_glycosylated'] == 'Yes'])}")
    print("\n--- Data Preview ---")
    print(final_df_to_save.head())
    print(f"\nFull dataset saved to '{OUTPUT_FILE}'")
    print(f"Dataset with Python objects saved to '{pickle_output_file}' (recommended for next script).")

if __name__ == "__main__":
    main()