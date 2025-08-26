# predict_standalone.py

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
import torch
import esm
import xgboost as xgb
import pickle
import requests
from Bio.PDB import PDBParser, PDBIO, Select, Polypeptide, SASA
from Bio.PDB.Residue import Residue
from Bio.SeqUtils import seq1
from tqdm import tqdm

# --- Configuration & Imports from other project files ---
# We bring in necessary components and configurations directly to make this script self-contained.
import config
import esm_embedding as esm_emb

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- HELPER FUNCTIONS FOR PACKAGED EXECUTABLE ---

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# --- HELPER FUNCTIONS (Adapted from other project files) ---

class ChainSelect(Select):
    """Helper class to select a specific chain from a PDB structure."""
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.get_id() == self.chain_id

    def accept_residue(self, residue: Residue) -> int:
        # Exclude HETATMs and other non-standard residues
        return 1 if residue.get_id()[0] == ' ' else 0

def download_and_clean_pdb(pdb_id, chain_id, pdb_dir, antigen_only_dir):
    """
    Downloads a PDB file, extracts and saves only the specified antigen chain.
    Returns the path to the cleaned, antigen-only PDB file.
    """
    pdb_id = pdb_id.lower()
    raw_pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    antigen_pdb_path = os.path.join(antigen_only_dir, f"{pdb_id}_{chain_id}_antigen_only.pdb")

    # Download if it doesn't exist
    if not os.path.exists(raw_pdb_path):
        print(f"Downloading PDB file for {pdb_id.upper()}...")
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Will raise an HTTPError for bad responses
            with open(raw_pdb_path, 'w') as f:
                f.write(response.text)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error: Could not download PDB {pdb_id.upper()}. Please check the ID and your internet connection.")
            print(f"Details: {e}")
            return None

    # Clean the PDB to keep only the specified antigen chain
    print(f"Extracting chain {chain_id} from {pdb_id.upper()}...")
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, raw_pdb_path)
        
        # Check if the chain exists in the structure
        chain_ids_in_pdb = [c.id for c in structure[0].get_chains()]
        if chain_id not in chain_ids_in_pdb:
            print(f"Error: Chain '{chain_id}' not found in PDB {pdb_id.upper()}.")
            print(f"Available chains are: {', '.join(chain_ids_in_pdb)}")
            return None

        io = PDBIO()
        io.set_structure(structure)
        io.save(antigen_pdb_path, ChainSelect(chain_id))
        print(f"Saved antigen-only structure to: {antigen_pdb_path}")
        return antigen_pdb_path
    except Exception as e:
        print(f"Error processing PDB file {raw_pdb_path}. Error: {e}")
        return None

# --- FEATURE GENERATION AND INFERENCE LOGIC (from inference.py) ---
# All functions from the original inference.py are kept here, mostly unchanged.

def load_models():
    """Loads the specified ESM models based on the config."""
    models = {}
    print("--- Loading Pre-trained Protein Language Models ---")
    print("(This may take a while on first run as models are downloaded...)")
    if config.EMBEDDING_MODE in ['esm2', 'both']:
        print(f"Loading ESM-2 model: {config.ESM2_MODEL_NAME}")
        model, alphabet = esm.pretrained.load_model_and_alphabet(config.ESM2_MODEL_NAME)
        models['esm2'] = (model.to(DEVICE).eval(), alphabet)
    if config.EMBEDDING_MODE in ['esm_if1', 'both']:
        print(f"Loading ESM-IF1 model: {config.ESM_IF1_MODEL_NAME}")
        model, alphabet = esm.pretrained.load_model_and_alphabet(config.ESM_IF1_MODEL_NAME)
        models['esm_if1'] = (model.eval(), alphabet)
    print(f"Models loaded and using device: {DEVICE}\n")
    return models

def get_amino_acid_one_hot(residue_name):
    """Generates a one-hot encoded vector for a given amino acid."""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_map = {aa: i for i, aa in enumerate(amino_acids)}
    try:
        one_letter = seq1(residue_name)
    except KeyError:
        return np.zeros(len(amino_acids), dtype=int)
    one_hot = np.zeros(len(amino_acids), dtype=int)
    if one_letter in aa_map:
        one_hot[aa_map[one_letter]] = 1
    return one_hot

def get_biophysical_features(structure, antigen_chain_id):
    """Calculates RSA and B-Factor for each residue in the antigen chain."""
    features = {}
    antigen_chain = structure[0][antigen_chain_id]
    sasa_calculator = SASA.ShrakeRupley()
    sasa_calculator.compute(structure, level="R")
    for res in antigen_chain.get_residues():
        if not Polypeptide.is_aa(res, standard=True):
            continue
        res_id_tuple = res.get_id()
        res_name = res.get_resname()
        res_id_str = f"{res_id_tuple[1]}{res_id_tuple[2]}".strip()
        try:
            sasa = res.sasa
            max_sasa = config.SASA_MAX_VALUES.get(seq1(res_name), 1.0)
            rsa = sasa / max_sasa if max_sasa > 0 else 0
            b_factor = np.mean([atom.get_bfactor() for atom in res.get_atoms()])
            features[res_id_str] = {"rsa": rsa, "b_factor": b_factor}
        except (AttributeError, KeyError):
            continue
    return features

def generate_features_for_single_protein(pdb_path, chain_id, esm_models):
    """
    Generates the complete feature matrix (X_arr) and metadata dataframe (df_stats)
    for a single protein chain from a PDB file.
    """
    print(f"--- Starting feature generation for {os.path.basename(pdb_path)} Chain {chain_id} ---")
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", pdb_path)
        antigen_chain = structure[0][chain_id]
    except Exception as e:
        print(f"Error parsing PDB file '{pdb_path}': {e}")
        return None, None
    residues = [res for res in antigen_chain if Polypeptide.is_aa(res, standard=True)]
    if not residues:
        print(f"Error: No standard amino acid residues found in chain '{chain_id}'.")
        return None, None
    seq = "".join([seq1(res.get_resname()) for res in residues])
    print(f"Found {len(residues)} residues in chain {chain_id}.")
    biophysical_feats = get_biophysical_features(structure, chain_id)
    print("Generating ESM embeddings...")
    esm2_embeddings, esm_if1_embeddings = None, None
    if config.EMBEDDING_MODE in ['esm2', 'both']:
        esm2_embeddings = esm_emb.get_esm2_embedding(esm_models['esm2'], seq)
    if config.EMBEDDING_MODE in ['esm_if1', 'both']:
        esm_if1_embeddings = esm_emb.get_esm_if1_embedding(esm_models['esm_if1'], pdb_path, chain_id)
    residue_data = []
    for i, res in enumerate(tqdm(residues, desc="Assembling features")):
        res_id_tuple = res.get_id()
        res_id_str = f"{res_id_tuple[1]}{res_id_tuple[2]}".strip()
        embedding = None
        if config.EMBEDDING_MODE == 'esm2':
            if esm2_embeddings is not None and i < len(esm2_embeddings):
                embedding = esm2_embeddings[i]
        elif config.EMBEDDING_MODE == 'esm_if1':
            if esm_if1_embeddings is not None and i < len(esm_if1_embeddings):
                embedding = esm_if1_embeddings[i]
        elif config.EMBEDDING_MODE == 'both':
            if esm2_embeddings is not None and esm_if1_embeddings is not None and i < len(esm2_embeddings) and i < len(esm_if1_embeddings):
                embedding = np.concatenate([esm2_embeddings[i], esm_if1_embeddings[i]])
        if embedding is None:
            print(f"Warning: Could not generate embedding for residue {res_id_str}. Skipping.")
            continue
        bio_feats = biophysical_feats.get(res_id_str, {"rsa": 0, "b_factor": 0})
        residue_data.append({
            "chain": chain_id, "res_id": res_id_str, "residue": res.get_resname(),
            "one_hot_amino_acid": get_amino_acid_one_hot(res.get_resname()),
            "rsa": bio_feats["rsa"], "b_factor": bio_feats["b_factor"], "embedding": embedding
        })
    if not residue_data:
        print("Error: Failed to assemble features for any residue.")
        return None, None
    df_stats = pd.DataFrame(residue_data)
    seq_length = len(df_stats)
    embeddings = np.vstack(df_stats['embedding'].values)
    seq_onehot = np.vstack(df_stats['one_hot_amino_acid'].values)
    b_factors = df_stats['b_factor'].values.reshape(-1, 1)
    seq_lengths = np.full((seq_length, 1), seq_length)
    rsas = df_stats['rsa'].values.reshape(-1, 1)
    X_arr = np.concatenate([embeddings, seq_onehot, b_factors, seq_lengths, rsas], axis=1)
    print("Feature generation complete.")
    return X_arr.astype(np.float32), df_stats.drop(columns=['embedding', 'one_hot_amino_acid'])

# --- MAIN PREDICTION PIPELINE ---

def run_prediction_pipeline(pdb_file, chain_id):
    """
    Main function to run the full inference pipeline on a given PDB file and chain.
    """
    # Use the helper function to find the model file
    MODEL_PATH = get_resource_path('models/final_model.json')

    # --- 1. Load Models (XGBoost and ESM) ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at '{MODEL_PATH}'.")
        print("This is a critical error. The model should have been packaged with the executable.")
        return
    print(f"--- Loading trained XGBoost model from '{MODEL_PATH}' ---")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    esm_models = load_models()
    
    # --- 2. Generate Features for the Input Protein ---
    X_protein, df_protein_stats = generate_features_for_single_protein(pdb_file, chain_id, esm_models)
    if X_protein is None or df_protein_stats is None:
        print("\n--- INFERENCE FAILED ---")
        return
        
    # --- 3. Run Inference ---
    print(f"\n--- Running prediction for {os.path.basename(pdb_file)} Chain {chain_id} ---")
    pred_probas = model.predict_proba(X_protein)[:, 1]
    
    # --- 4. Create and Display a Detailed Report ---
    df_protein_stats['prediction_score'] = pred_probas
    predicted_epitopes = df_protein_stats[df_protein_stats['prediction_score'] >= config.PREDICTION_THRESHOLD]
    print("\n" + "="*50)
    print("--- PREDICTION SUMMARY ---")
    print(f"Total residues analyzed: {len(df_protein_stats)}")
    print(f"Predicted epitope residues (score >= {config.PREDICTION_THRESHOLD}): {len(predicted_epitopes)}")
    print("="*50 + "\n")
    print(f"--- PREDICTED Epitope Residues (score >= {config.PREDICTION_THRESHOLD}) ---")
    if not predicted_epitopes.empty:
        print(predicted_epitopes[['chain', 'res_id', 'residue', 'prediction_score', 'rsa', 'b_factor']].to_string(index=False))
    else:
        print("None.")
    print("\n--- Top 10 Highest-Scoring Residues ---")
    print(df_protein_stats.sort_values(by='prediction_score', ascending=False).head(10)[['chain', 'res_id', 'residue', 'prediction_score', 'rsa', 'b_factor']].to_string(index=False))
    
    # Save results to a CSV file
    pdb_name = os.path.basename(pdb_file).split('.')[0]
    output_filename = f"{pdb_name}_{chain_id}_predictions.csv"
    df_protein_stats.sort_values(by='prediction_score', ascending=False).to_csv(output_filename, index=False)
    print(f"\nDetailed results saved to: {output_filename}")
    print("\n--- PIPELINE COMPLETE ---")

# --- USER INTERFACE AND MAIN EXECUTION BLOCK ---

def main():
    print("="*60)
    print("   XGBoost Conformational Epitope Prediction Tool")
    print("="*60)

    # Create necessary directories for caching and output
    for dir_path in [config.PDB_DIR, config.ANTIGEN_ONLY_PDB_DIR, config.EMBEDDING_CACHE_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    antigen_pdb_path = None
    chain_id = None

    while True:
        choice = input("Choose input method:\n1. Enter a PDB ID\n2. Provide a local PDB file path\nEnter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")

    if choice == '1':
        pdb_id = input("Enter the 4-character PDB ID (e.g., 4gxu): ").strip().upper()
        chain_id = input(f"Enter the antigen chain ID for {pdb_id}: ").strip().upper()
        antigen_pdb_path = download_and_clean_pdb(pdb_id, chain_id, config.PDB_DIR, config.ANTIGEN_ONLY_PDB_DIR)
    
    elif choice == '2':
        while True:
            file_path = input("Enter the full path to your antigen PDB file: ").strip()
            if os.path.exists(file_path):
                antigen_pdb_path = file_path
                chain_id = input(f"Enter the antigen chain ID within '{os.path.basename(file_path)}': ").strip().upper()
                break
            else:
                print("Error: File not found. Please check the path and try again.")
    
    if antigen_pdb_path and chain_id:
        run_prediction_pipeline(antigen_pdb_path, chain_id)
    else:
        print("\nCould not proceed with prediction due to previous errors.")

    print("\nPress Enter to exit.")
    input()


if __name__ == '__main__':
    main()