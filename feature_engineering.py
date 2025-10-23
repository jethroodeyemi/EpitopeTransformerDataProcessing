# feature_engineering.py

import os
import pandas as pd
import numpy as np
import torch
import esm
import warnings
from Bio.PDB import PDBParser, NeighborSearch, Polypeptide, SASA
from Bio.SeqUtils import seq1
from tqdm import tqdm
import esm_embedding as esm_emb
import pickle
import config # Import our configuration

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper Functions ---

def get_amino_acid_one_hot(residue_name):
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

def load_models():
    """Loads the specified ESM models based on the config."""
    models = {}
    if config.EMBEDDING_MODE in ['esm2', 'both']:
        print(f"Loading ESM-2 model: {config.ESM2_MODEL_NAME}")
        model, alphabet = esm.pretrained.load_model_and_alphabet(config.ESM2_MODEL_NAME)
        models['esm2'] = (model.to(DEVICE).eval(), alphabet)
    if config.EMBEDDING_MODE in ['esm_if1', 'both']:
        print(f"Loading ESM-IF1 model: {config.ESM_IF1_MODEL_NAME}")
        model, alphabet = esm.pretrained.load_model_and_alphabet(config.ESM_IF1_MODEL_NAME)
        models['esm_if1'] = (model.eval(), alphabet)
    print(f"Using device: {DEVICE}")
    return models

def get_biophysical_features(structure, antigen_chain_id):
    """Calculates RSA and B-Factor for each residue in the antigen chain."""
    features = {}
    antigen_chain = structure[0][antigen_chain_id]
    
    # Calculate SASA
    sasa_calculator = SASA.ShrakeRupley()
    sasa_calculator.compute(structure, level="R")

    for res in antigen_chain.get_residues():
        if not Polypeptide.is_aa(res, standard=True):
            continue

        res_id_tuple = res.get_id()
        res_name = res.get_resname()
        res_id_str = f"{res_id_tuple[1]}{res_id_tuple[2]}".strip()

        # RSA Calculation
        sasa = res.sasa if hasattr(res, 'sasa') else 0
        max_sasa = config.SASA_MAX_VALUES.get(seq1(res_name), 1.0)
        rsa = sasa / max_sasa if max_sasa > 0 else 0
        
        # B-Factor (average of all atoms in the residue)
        b_factor = np.mean([atom.get_bfactor() for atom in res.get_atoms()])

        features[res_id_str] = {"rsa": rsa, "b_factor": b_factor}
    return features

def identify_epitope_residues(structure, h_chain_id, l_chain_id, antigen_chain_id):
    """Identifies epitope residues using a distance threshold."""
    model = structure[0]
    antibody_atoms = []
    if h_chain_id in model:
        antibody_atoms.extend(list(model[h_chain_id].get_atoms()))
    if l_chain_id in model:
        antibody_atoms.extend(list(model[l_chain_id].get_atoms()))
    
    if not antibody_atoms:
        return set()

    antigen_residues = [res for res in model[antigen_chain_id] if Polypeptide.is_aa(res)]
    ns = NeighborSearch(antibody_atoms)
    epitope_residues = {
        res.get_id() for res in antigen_residues
        if any(ns.search(atom.get_coord(), config.DISTANCE_THRESHOLD, level='A') for atom in res)
    }
    return epitope_residues

def generate_features_and_labels(df, cleaned_pdb_dir, antigen_only_pdb_dir):
    """The main function to process structures and generate the final feature DataFrame."""
    print("\n--- Step 4: Generating Features and Labels ---")
    
    models = load_models()
    parser = PDBParser(QUIET=True)
    final_data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Structures"):
        pdb_id = row['pdb']
        antigen_chain_id = row['antigen_chain']
        h_chain_id, l_chain_id = row['Hchain'], row['Lchain']
        
        complex_path = os.path.join(cleaned_pdb_dir, f"{pdb_id}_cleaned.pdb")
        antigen_only_path = os.path.join(antigen_only_pdb_dir, f"{pdb_id}_antigen_only.pdb")

        if not os.path.exists(complex_path) or not os.path.exists(antigen_only_path):
            continue

        try:
            structure_for_labels = parser.get_structure(f"{pdb_id}_complex", complex_path)
            structure_for_features = parser.get_structure(f"{pdb_id}_antigen_only", antigen_only_path)
            antigen_chain = structure_for_features[0][antigen_chain_id]
            
            # --- Get Sequence and Residue Info ---
            residues = [res for res in antigen_chain if Polypeptide.is_aa(res, standard=True)]
            if not residues: continue
            
            seq = "".join([seq1(res.get_resname()) for res in residues])
            
            # --- Feature & Label Generation ---
            biophysical_feats = get_biophysical_features(structure_for_features, antigen_chain_id)
            epitope_ids = identify_epitope_residues(structure_for_labels, h_chain_id, l_chain_id, antigen_chain_id)
        
            esm2_embeddings = None
            esm_if1_embeddings = None
            embedding = None
            
            # ESM-2
            if config.EMBEDDING_MODE in ['esm2', 'both']:
                cache_path = os.path.join(config.EMBEDDING_CACHE_DIR, f"{pdb_id}_{antigen_chain_id}_esm2.npy")
                if os.path.exists(cache_path) and not config.FORCE_RECOMPUTE_EMBEDDINGS:
                    esm2_embeddings = np.load(cache_path)
                else:
                    esm2_embeddings = esm_emb.get_esm2_embedding(models['esm2'], seq)
                    if esm2_embeddings is not None:
                        np.save(cache_path, esm2_embeddings)
                        
                
            # ESM-IF1
            if config.EMBEDDING_MODE in ['esm_if1', 'both']:
                cache_path = os.path.join(config.EMBEDDING_CACHE_DIR, f"{pdb_id}_{antigen_chain_id}_esm_if1.npy")
                if os.path.exists(cache_path) and not config.FORCE_RECOMPUTE_EMBEDDINGS:
                    esm_if1_embeddings = np.load(cache_path)
                else:
                    esm_if1_embeddings = esm_emb.get_esm_if1_embedding(models['esm_if1'], antigen_only_path, antigen_chain_id)
                    if esm_if1_embeddings is not None:
                        np.save(cache_path, esm_if1_embeddings)
                        

            # --- Assemble per-residue data ---
            for i, res in enumerate(residues):
                res_id_tuple = res.get_id()
                res_id_str = f"{res_id_tuple[1]}{res_id_tuple[2]}".strip()
                
                # Combine embeddings based on config
                if config.EMBEDDING_MODE == 'esm2':
                    if esm2_embeddings is None or i >= len(esm2_embeddings): continue
                    embedding = esm2_embeddings[i]
                elif config.EMBEDDING_MODE == 'esm_if1':
                    if esm_if1_embeddings is None or i >= len(esm_if1_embeddings): continue
                    embedding = esm_if1_embeddings[i]
                elif config.EMBEDDING_MODE == 'both':
                    if esm2_embeddings is None or esm_if1_embeddings is None or i >= len(esm2_embeddings) or i >= len(esm_if1_embeddings): continue
                    embedding = np.concatenate([esm2_embeddings[i], esm_if1_embeddings[i]])
                
                bio_feats = biophysical_feats.get(res_id_str, {"rsa": 0, "b_factor": 0})

                final_data.append({
                    "pdb_id": pdb_id,
                    "antigen_chain": antigen_chain_id,
                    "res_id": res_id_str,
                    "res_name": res.get_resname(),
                    "is_epitope": 1 if res_id_tuple in epitope_ids else 0,
                    "one_hot_amino_acid": get_amino_acid_one_hot(res.get_resname()),
                    "rsa": bio_feats["rsa"],
                    "b_factor": bio_feats["b_factor"],
                    "seq_length": len(seq),
                    "embedding": embedding
                })

        except Exception as e:
            print(f"Failed to process {pdb_id}. Error: {e}")
            import traceback
            traceback.print_exc()

    final_df = pd.DataFrame(final_data)
    print(f"\nFinal dataframe created with {len(final_df)} residue entries.")
    print("DataFrame head:")
    print(final_df.head())

    final_df.to_pickle(config.FINAL_DATAFRAME_PATH)
    print(f"\nFinal DataFrame saved to: {config.FINAL_DATAFRAME_PATH}")

    # Convert to structured data format
    structure_data_to_dict(final_df)
    
    return final_df

def structure_data_to_dict(df):
    protein_data_list = []
    grouped = df.groupby('pdb_id')

    print("\nProcessing each protein into the desired dictionary structure...")
    for pdb_id, group in tqdm(grouped, desc="Processing Proteins"):
        L = len(group)
        embeddings = np.vstack(group['embedding'].values)
        seq_onehot = np.vstack(group['one_hot_amino_acid'].values)
        b_factors = group['b_factor'].values.reshape(-1, 1)
        seq_lengths = group['seq_length'].values.reshape(-1, 1)
        rsas = group['rsa'].values.reshape(-1, 1)
        X_arr = np.concatenate(
            [
                embeddings,             # (L, 1728)
                seq_onehot,             # (L, 20)
                b_factors,              # (L, 1)
                seq_lengths,            # (L, 1)
                rsas,                   # (L, 1)
            ],
            axis=1,
        )

        embed_dim = embeddings.shape[1]
        
        feature_idxs = {
            "embedding": range(0, embed_dim),
            "sequence_onehot": range(embed_dim, embed_dim + 20),
            "b_factor": range(embed_dim + 20, embed_dim + 21),
            "length": range(embed_dim + 21, embed_dim + 22),
            "rsa": range(embed_dim + 22, embed_dim + 23),
        }
        
        df_stats = pd.DataFrame({
            "pdb_id": pdb_id,
            "chain": group['antigen_chain'].values,
            "res_id": group['res_id'].values,
            "residue": group['res_name'].values,
            "is_epitope": group['is_epitope'].values,
            "rsa": group['rsa'].values,
            "b_factor": group['b_factor'].values, 
            "length": group['seq_length'].values,
        })
        
        output_dict = {
            "pdb_id": pdb_id,
            "X_arr": X_arr.astype(np.float32),
            "df_stats": df_stats,
            "length": L,
            "feature_idxs": feature_idxs,
        }
        
        protein_data_list.append(output_dict)

    with open(config.STRUCTURED_DATA_PATH, 'wb') as f:
        pickle.dump(protein_data_list, f)

    print(f"\nStructured protein data saved to: {config.STRUCTURED_DATA_PATH}")