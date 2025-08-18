# inspect_protein.py

import json
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
import argparse

import config # To easily pass the PDB ID from the command line

def inspect_single_protein(pdb_id_to_inspect):
    """
    Loads the final model and the full dataset, runs prediction on a single
    specified protein from the test set, and prints a detailed report.
    """
    # --- 1. Load Model and Data ---
    if not os.path.exists(config.FINAL_MODEL_PATH) or not os.path.exists(config.STRUCTURED_DATA_PATH):
        print("Error: Ensure 'models/final_model.json' and 'output/structured_protein_data.pkl' exist.")
        print("Run the training and data structuring scripts first.")
        return

    print(f"--- Loading final model from '{config.FINAL_MODEL_PATH}' ---")
    model = xgb.XGBClassifier()
    model.load_model(config.FINAL_MODEL_PATH)

    print(f"--- Loading structured data from '{config.STRUCTURED_DATA_PATH}' ---")
    with open(config.STRUCTURED_DATA_PATH, 'rb') as f:
        protein_data_list = pickle.load(f)

    # --- 2. Identify the test set proteins ---
    with open(config.SPLITS_FILE_PATH, 'r') as f:
        splits = json.load(f)
    test_groups = splits['test']
    print("Test set PDBs loaded from splits file:", test_groups[:5], "...")

    # --- 3. Find and Prepare the Target Protein Data ---
    target_protein_data = None
    for protein_data in protein_data_list:
        if protein_data['pdb_id'] == pdb_id_to_inspect:
            target_protein_data = protein_data
            break

    if target_protein_data is None:
        print(f"Error: PDB ID '{pdb_id_to_inspect}' not found in the dataset.")
        return
        
    if pdb_id_to_inspect not in test_groups:
        print(f"Warning: PDB ID '{pdb_id_to_inspect}' was found, but it is in the TRAIN/VALIDATION set, not the TEST set.")
        print("The model has seen this protein during training. Predictions may be overly optimistic.")

    # Extract features (X_arr) and metadata (df_stats) for this protein
    X_protein = target_protein_data['X_arr']
    df_protein_stats = target_protein_data['df_stats'].copy()

    # --- 4. Run Inference ---
    print(f"\n--- Running prediction for PDB ID: {pdb_id_to_inspect} ---")
    pred_probas = model.predict_proba(X_protein)[:, 1]

    # --- 5. Create and Display a Detailed Report ---
    # Add the prediction scores to our metadata DataFrame
    df_protein_stats['prediction_score'] = pred_probas
    
    # Identify the true epitope residues
    true_epitopes = df_protein_stats[df_protein_stats['is_epitope'] == 1]
    
    # Identify residues predicted to be epitopes above a certain threshold
    predicted_epitopes = df_protein_stats[df_protein_stats['prediction_score'] >= config.PREDICTION_THRESHOLD]

    print("\n--- Summary ---")
    print(f"Total residues in protein: {len(df_protein_stats)}")
    print(f"Number of TRUE epitope residues: {len(true_epitopes)}")
    print(f"Number of PREDICTED epitope residues (score >= {config.PREDICTION_THRESHOLD}): {len(predicted_epitopes)}")

    print("\n--- TRUE Epitope Residues ---")
    if not true_epitopes.empty:
        # We select specific columns for a cleaner printout
        print(true_epitopes[['chain', 'res_id', 'residue', 'prediction_score', 'rsa', 'b_factor']].to_string(index=False))
    else:
        print("None.")

    print(f"\n--- PREDICTED Epitope Residues (score >= {config.PREDICTION_THRESHOLD}) ---")
    if not predicted_epitopes.empty:
        print(predicted_epitopes[['chain', 'res_id', 'residue', 'is_epitope', 'prediction_score', 'rsa', 'b_factor']].to_string(index=False))
    else:
        print("None.")
        
    print("\n--- Top 5 Highest-Scoring Residues ---")
    print(df_protein_stats.sort_values(by='prediction_score', ascending=False).head(5)[['chain', 'res_id', 'residue', 'is_epitope', 'prediction_score', 'rsa', 'b_factor']].to_string(index=False))


def get_random_test_protein_id():
    """Helper function to get a random PDB ID from the test set."""
    with open(config.SPLITS_FILE_PATH, 'r') as f:
        splits = json.load(f)
    test_groups = splits['test']
    return np.random.choice(test_groups)

if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Inspect epitope predictions for a single protein.")
    parser.add_argument(
        "pdb_id", 
        nargs='?', 
        default=None, 
        help="The PDB ID of the protein to inspect. If not provided, a random protein from the test set will be chosen."
    )
    args = parser.parse_args()
    
    if args.pdb_id:
        pdb_to_check = args.pdb_id
    else:
        print("No PDB ID provided. Choosing a random protein from the test set...")
        pdb_to_check = get_random_test_protein_id()

    # Run the inspection
    inspect_single_protein(pdb_to_check)