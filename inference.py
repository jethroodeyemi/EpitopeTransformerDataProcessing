# inference.py

import os
import argparse
import warnings
import pandas as pd
import numpy as np
import torch
import esm
import xgboost as xgb
import pickle
from Bio.PDB import PDBParser, Polypeptide, SASA
from Bio.SeqUtils import seq1
from tqdm import tqdm

# --- Configuration & Imports from other project files ---
# We bring in necessary components and configurations directly to make this script self-contained.
import config
import esm_embedding as esm_emb

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- HELPER FUNCTIONS (Adapted from feature_engineering.py) ---

def load_models():
    """Loads the specified ESM models based on the config."""
    models = {}
    print("--- Loading Pre-trained Protein Language Models ---")
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
             # Skip if sasa not calculated or residue name is non-standard
            continue
    return features

def generate_features_for_single_protein(pdb_path, chain_id, esm_models):
    """
    Generates the complete feature matrix (X_arr) and metadata dataframe (df_stats)
    for a single protein chain from a PDB file.
    This function mirrors the logic in feature_engineering.py.
    """
    print(f"--- Starting feature generation for {os.path.basename(pdb_path)} Chain {chain_id} ---")
    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure("protein", pdb_path)
        if chain_id not in [c.id for c in structure[0].get_chains()]:
            print(f"Error: Chain '{chain_id}' not found in '{pdb_path}'.")
            return None, None
        antigen_chain = structure[0][chain_id]
    except Exception as e:
        print(f"Error parsing PDB file '{pdb_path}': {e}")
        return None, None

    # --- Get Sequence and Residue Info ---
    residues = [res for res in antigen_chain if Polypeptide.is_aa(res, standard=True)]
    if not residues:
        print(f"Error: No standard amino acid residues found in chain '{chain_id}'.")
        return None, None
    seq = "".join([seq1(res.get_resname()) for res in residues])
    print(f"Found {len(residues)} residues in chain {chain_id}.")

    # --- Feature & Label Generation ---
    biophysical_feats = get_biophysical_features(structure, chain_id)

    # --- Generate Embeddings ---
    print("Generating ESM embeddings...")
    esm2_embeddings, esm_if1_embeddings = None, None
    if config.EMBEDDING_MODE in ['esm2', 'both']:
        esm2_embeddings = esm_emb.get_esm2_embedding(esm_models['esm2'], seq)
    if config.EMBEDDING_MODE in ['esm_if1', 'both']:
        esm_if1_embeddings = esm_emb.get_esm_if1_embedding(esm_models['esm_if1'], pdb_path, chain_id)

    # --- Assemble per-residue data ---
    residue_data = []
    for i, res in enumerate(tqdm(residues, desc="Assembling features")):
        res_id_tuple = res.get_id()
        res_id_str = f"{res_id_tuple[1]}{res_id_tuple[2]}".strip()

        # Combine embeddings based on config
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
            "chain": chain_id,
            "res_id": res_id_str,
            "residue": res.get_resname(),
            "one_hot_amino_acid": get_amino_acid_one_hot(res.get_resname()),
            "rsa": bio_feats["rsa"],
            "b_factor": bio_feats["b_factor"],
            "embedding": embedding
        })

    if not residue_data:
        print("Error: Failed to assemble features for any residue.")
        return None, None

    # --- Create final data structures (X_arr and df_stats) ---
    df_stats = pd.DataFrame(residue_data)
    seq_length = len(df_stats)
    
    # IMPORTANT: The order of concatenation must EXACTLY match the training script.
    # From feature_engineering.py `structure_data_to_dict`
    embeddings = np.vstack(df_stats['embedding'].values)
    seq_onehot = np.vstack(df_stats['one_hot_amino_acid'].values)
    b_factors = df_stats['b_factor'].values.reshape(-1, 1)
    seq_lengths = np.full((seq_length, 1), seq_length) # All rows have the same sequence length
    rsas = df_stats['rsa'].values.reshape(-1, 1)

    X_arr = np.concatenate([embeddings, seq_onehot, b_factors, seq_lengths, rsas], axis=1)

    print("Feature generation complete.")
    return X_arr.astype(np.float32), df_stats.drop(columns=['embedding', 'one_hot_amino_acid'])


# --- MAIN INFERENCE FUNCTION ---

def predict_epitopes(pdb_file, chain_id):
    """
    Main function to run the full inference pipeline on a given PDB file and chain.
    """
    MODEL_PATH = 'models/final_model.json'
    PREDICTION_THRESHOLD = 0.6 # Confidence score to be considered a predicted epitope

    # --- 1. Load Models (XGBoost and ESM) ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at '{MODEL_PATH}'.")
        print("Please run the training pipeline first to generate 'models/final_model.json'.")
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
    predicted_epitopes = df_protein_stats[df_protein_stats['prediction_score'] >= PREDICTION_THRESHOLD]

    print("\n--- Prediction Summary ---")
    print(f"Total residues analyzed: {len(df_protein_stats)}")
    print(f"Predicted epitope residues (score >= {PREDICTION_THRESHOLD}): {len(predicted_epitopes)}")

    print(f"\n--- PREDICTED Epitope Residues (score >= {PREDICTION_THRESHOLD}) ---")
    if not predicted_epitopes.empty:
        print(predicted_epitopes[['chain', 'res_id', 'residue', 'prediction_score', 'rsa', 'b_factor']].to_string(index=False))
    else:
        print("None.")

    print("\n--- Top 10 Highest-Scoring Residues ---")
    print(df_protein_stats.sort_values(by='prediction_score', ascending=False).head(10)[['chain', 'res_id', 'residue', 'prediction_score', 'rsa', 'b_factor']].to_string(index=False))
    
    # Save results to a CSV file
    output_filename = f"{os.path.basename(pdb_file).split('.')[0]}_{chain_id}_predictions.csv"
    df_protein_stats.sort_values(by='prediction_score', ascending=False).to_csv(output_filename, index=False)
    print(f"\nDetailed results saved to: {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict epitope residues on a single antigen protein chain.")
    parser.add_argument(
        "pdb_file",
        type=str,
        help="Path to the input PDB file containing the antigen structure."
    )
    parser.add_argument(
        "chain_id",
        type=str,
        help="The chain ID of the antigen within the PDB file to analyze."
    )
    args = parser.parse_args()

    # Create necessary cache directory if it doesn't exist
    os.makedirs(config.EMBEDDING_CACHE_DIR, exist_ok=True)
    
    predict_epitopes(args.pdb_file, args.chain_id.upper())