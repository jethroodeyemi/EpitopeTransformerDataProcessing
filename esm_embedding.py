import esm.inverse_folding
import numpy as np
import torch
import os
import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_esm2_embedding(model_tuple, sequence: str) -> np.ndarray:
    """
    Generates ESM-2 embedding for a single sequence.

    Args:
        model_tuple: A tuple containing the loaded (model, alphabet).
        sequence: The amino acid sequence string.

    Returns:
        A numpy array of per-residue embeddings.
    """
    model, alphabet = model_tuple
    batch_converter = alphabet.get_batch_converter()
    
    # ESM-2 has a 1024 token limit; truncate if necessary.
    max_len = 1022 # 1024 minus start/end tokens
    if len(sequence) > max_len:
        print(f"Sequence truncated to {max_len} residues for ESM-2 embedding.")
        sequence = sequence[:max_len]

    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(DEVICE)

    # The layer number is specific to the model, e.g., 33 for esm2_t33_650M_UR50D
    try:
        layer = int(config.ESM2_MODEL_NAME.split('_')[1][1:])
    except (IndexError, ValueError):
        print(f"Could not parse layer from model name '{config.ESM2_MODEL_NAME}'. Defaulting to 33.")
        layer = 33
        
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[layer], return_contacts=False)
        
    embedding = results["representations"][layer].to("cpu").numpy()
    
    # Remove start/end tokens and batch dimension
    return torch.tensor(embedding[0, 1 : len(sequence) + 1]).float()
        

def get_esm_if1_embedding(model_tuple, pdb_path, chain_id) -> np.ndarray:
    model, alphabet = model_tuple

    try:
        structure = esm.inverse_folding.util.load_structure(pdb_path, chain_id)
        if not structure: return None
        coords, _ = esm.inverse_folding.util.extract_coords_from_structure(structure)
        with torch.no_grad():
            embedding = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)

        return torch.tensor(embedding).float()
    except Exception as e:
        print(f"Error getting ESM-IF1 embedding for {os.path.basename(pdb_path)} chain {chain_id}: {e}")
        return None


