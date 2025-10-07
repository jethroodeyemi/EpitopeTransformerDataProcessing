import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser, NeighborSearch
import freesasa  # <-- IMPORT THE NEW LIBRARY
from tqdm import tqdm
import warnings
import sys

# --- Configuration ---
warnings.filterwarnings("ignore", category=UserWarning)

# --- INPUT FILES ---
CURATED_DATA_FILE = Path("./curated_complex_data.pkl")
PDB_DIR = Path("../cleaned_pdb_files")

# --- OUTPUT FILES ---
OUTPUT_FILE_CSV = Path("./interface_analysis_results.csv")
OUTPUT_FILE_PKL = Path("./interface_analysis_results.pkl")

# --- ANALYSIS PARAMETERS ---
DISTANCE_CUTOFFS = [4.0, 5.0, 6.0]

# --- Helper Functions ---

# !!! THIS FUNCTION IS REPLACED !!!
def calculate_bsa_with_freesasa_library(structure, antigen_chain_ids: list, antibody_chain_ids: list) -> float | None:
    """
    Calculates the Buried Surface Area (BSA) for a complex using the freesasa-python library.
    BSA = (SASA_antigen + SASA_antibody) - SASA_complex
    """
    if not antigen_chain_ids or not antibody_chain_ids:
        return None

    try:
        # 1. Calculate SASA for the full complex
        # The result object contains SASA values for each atom, residue, chain, etc.
        result_complex = freesasa.calc(structure)
        sasa_complex = result_complex.totalArea()

        # 2. Calculate SASA for the antigen chains only
        selection_antigen = "antigen, chain " + " or chain ".join(antigen_chain_ids)
        result_antigen = freesasa.calc(structure, freesasa.Classifier.select(selection_antigen))
        sasa_antigen = result_antigen.totalArea()

        # 3. Calculate SASA for the antibody chains only
        selection_antibody = "antibody, chain " + " or chain ".join(antibody_chain_ids)
        result_antibody = freesasa.calc(structure, freesasa.Classifier.select(selection_antibody))
        sasa_antibody = result_antibody.totalArea()

        # 4. Calculate BSA
        bsa = (sasa_antigen + sasa_antibody) - sasa_complex
        return bsa if bsa > 0 else 0.0

    except Exception as e:
        # This can happen if a structure is malformed in a way the library can't handle
        # print(f"Warning: FreeSASA library failed for structure. Error: {e}")
        return None

def get_interfacial_residues(antigen_chains: list, antibody_chains: list, cutoff: float) -> tuple[list, list]:
    """
    Identifies interfacial residues between antigen and antibody chains using NeighborSearch.
    """
    antigen_atoms = [atom for chain in antigen_chains for atom in chain.get_atoms()]
    antibody_atoms = [atom for chain in antibody_chains for atom in chain.get_atoms()]

    if not antigen_atoms or not antibody_atoms:
        return [], []

    neighbor_search = NeighborSearch(antibody_atoms)
    
    antigen_interface_residues = set()
    antibody_interface_residues_set = set()
    
    for atom in antigen_atoms:
        nearby_residues = neighbor_search.search(atom.get_coord(), cutoff, level='R')
        if nearby_residues:
            antigen_interface_residues.add(atom.get_parent())
            for res in nearby_residues:
                antibody_interface_residues_set.add(res)
    
    format_res = lambda r: f"{r.get_parent().id}-{r.get_resname()}{r.get_id()[1]}"
    
    antigen_interface_list = sorted(list(set(format_res(r) for r in antigen_interface_residues)))
    antibody_interface_list = sorted(list(set(format_res(r) for r in antibody_interface_residues_set)))
    
    return antigen_interface_list, antibody_interface_list

def analyze_single_complex(row: pd.Series, parser: PDBParser) -> dict:
    """
    Performs full interface analysis for a single complex (one row of the dataframe).
    """
    pdb_id = row['pdb_id'].lower()
    pdb_file = PDB_DIR / f"{pdb_id}_cleaned.pdb"
    analysis_results = {"pdb_id": pdb_id}
    if not pdb_file.exists():
        analysis_results["error"] = "PDB_FILE_NOT_FOUND"
        return analysis_results
        
    try:
        structure = parser.get_structure(pdb_id, pdb_file)
        model = structure[0]
    except Exception as e:
        analysis_results["error"] = f"PDB_PARSE_ERROR: {e}"
        return analysis_results

    antigen_chain_ids = row['antigen_chains'].split(',') if row['antigen_chains'] else []
    h_chain_ids = row['antibody_h_chains'].split(',') if row['antibody_h_chains'] else []
    l_chain_ids = row['antibody_l_chains'].split(',') if row['antibody_l_chains'] else []
    antibody_chain_ids = h_chain_ids + l_chain_ids

    antigen_chains = [model[cid] for cid in antigen_chain_ids if cid in model]
    antibody_chains = [model[cid] for cid in antibody_chain_ids if cid in model]

    if not antigen_chains or not antibody_chains:
        analysis_results["error"] = "CHAINS_NOT_FOUND_IN_STRUCTURE"
        return analysis_results

    # --- Identify Interfacial Residues at different cutoffs ---
    for cutoff in DISTANCE_CUTOFFS:
        ag_interface, ab_interface = get_interfacial_residues(antigen_chains, antibody_chains, cutoff)
        
        key_ag = f"antigen_interface_{cutoff}A"
        key_ab = f"antibody_interface_{cutoff}A"
        
        analysis_results[key_ag] = ag_interface
        analysis_results[f"{key_ag}_count"] = len(ag_interface)
        analysis_results[key_ab] = ab_interface
        analysis_results[f"{key_ab}_count"] = len(ab_interface)

    # --- Calculate Buried Surface Area (BSA) using the new library function ---
    bsa = calculate_bsa_with_freesasa_library(structure, antigen_chain_ids, antibody_chain_ids)
    analysis_results['buried_surface_area'] = bsa

    return analysis_results

# --- Main Pipeline ---

def main():
    if not CURATED_DATA_FILE.exists():
        print(f"Error: Curated data file not found at '{CURATED_DATA_FILE}'. Please run Step 2.1 first.")
        sys.exit(1)

    print(f"Loading curated complex data from '{CURATED_DATA_FILE}'...")
    curated_df = pd.read_pickle(CURATED_DATA_FILE)
    parser = PDBParser(QUIET=True)
    all_analysis_results = []

    for _, row in tqdm(curated_df.iterrows(), total=len(curated_df), desc="Analyzing Interfaces"):
        result = analyze_single_complex(row, parser)
        all_analysis_results.append(result)
    # print(all_analysis_results)
    analysis_df = pd.DataFrame(all_analysis_results)
    final_df = pd.merge(curated_df, analysis_df, on="pdb_id", how="left")

    print(f"\nAnalysis complete. Saving structured data...")
    
    df_for_csv = final_df.copy()
    for col in df_for_csv.columns:
        if df_for_csv[col].dropna().apply(type).eq(list).any():
             df_for_csv[col] = df_for_csv[col].apply(lambda x: "; ".join(map(str, x)) if isinstance(x, list) else x)

    df_for_csv.to_csv(OUTPUT_FILE_CSV, index=False)
    final_df.to_pickle(OUTPUT_FILE_PKL)
    
    print("\n--- Analysis Summary ---")
    print(f"Total complexes analyzed: {len(final_df)}")
    print(f"Average Buried Surface Area (BSA): {final_df['buried_surface_area'].mean():.2f} Å²")
    for cutoff in DISTANCE_CUTOFFS:
        avg_ag_res = final_df[f'antigen_interface_{cutoff}A_count'].mean()
        avg_ab_res = final_df[f'antibody_interface_{cutoff}A_count'].mean()
        print(f"Average interface residues at {cutoff}Å: Antigen={avg_ag_res:.1f}, Antibody={avg_ab_res:.1f}")

    print("\n--- Data Preview ---")
    preview_cols = ['pdb_id', 'antigen_chains', 'antibody_h_chains', 'buried_surface_area'] + [f'antigen_interface_{DISTANCE_CUTOFFS[0]}A_count']
    print(df_for_csv[preview_cols].head())
    
    print(f"\nFull dataset saved to '{OUTPUT_FILE_CSV}'")
    print(f"Dataset with Python objects saved to '{OUTPUT_FILE_PKL}' (recommended for the next script).")

if __name__ == "__main__":
    main()