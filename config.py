# config.py

# --- File and Directory Paths ---
INPUT_TSV = 'dataset.tsv'
OUTPUT_DIR = 'output'
FINAL_MODEL_PATH = 'models/final_model.json'
EVALUATION_OUTPUT_DIR = 'evaluation_results'
PDB_DIR = 'pdb_files'
CLEANED_PDB_DIR = 'cleaned_pdb_files'
ANTIGEN_ONLY_PDB_DIR = 'antigen_only_pdb_files'
EMBEDDING_CACHE_DIR = 'embedding_cache'
PCA_MODEL_CACHE_DIR = 'pca_models'
PCA_EMBEDDING_CACHE_DIR = 'pca_embedding_cache'
GLYCOSYLATION_DATA_PATH = 'glycosylation/feature_rich_analysis.csv'

DEDUPED_TSV = f'{OUTPUT_DIR}/epitopes_deduplicated.tsv'
FINAL_DATAFRAME_PATH = f'{OUTPUT_DIR}/antigen_residue_features.pkl'
STRUCTURED_DATA_PATH = f'{OUTPUT_DIR}/structured_protein_data.pkl'

FASTA_PATH = f'{OUTPUT_DIR}/all_antigen_sequences.fasta'
CLUSTER_FILE_PATH = f'{OUTPUT_DIR}/protein_clusters' # CD-HIT adds .clstr
SPLITS_FILE_PATH = f'{OUTPUT_DIR}/data_splits_on_spike_proteins_cluster.json'
CDHIT_THRESHOLD = 0.4
MAX_CLUSTER_SIZE = 50

PREDICTION_THRESHOLD = 0.6 # Confidence score to be considered a predicted epitope

# --- Analysis Parameters ---
DISTANCE_THRESHOLD = 6.0
SASA_MAX_VALUES = {
    "A": 106.0,
    "R": 248.0,
    "N": 157.0,
    "D": 163.0,
    "C": 135.0,
    "Q": 198.0,
    "E": 194.0,
    "G": 84.0,
    "H": 184.0,
    "I": 169.0,
    "L": 164.0,
    "K": 205.0,
    "M": 188.0,
    "F": 197.0,
    "P": 136.0,
    "S": 130.0,
    "T": 142.0,
    "W": 227.0,
    "Y": 222.0,
    "V": 142.0,
    "X": 169.55,
}

# --- Glycosylation Feature Engineering ---
# A list that can contain 'binary', 'distance', or both. Set to [] to disable.
GLYCOSYLATION_MODE = ['binary', 'distance']
MAX_GLYCOSYLATION_DISTANCE = 20.0 # Max distance for the distance feature (in Angstroms)

# --- Model Configuration ---
# Array of embedding models to use. Can include 'esm2', 'esm_if1', and/or 'esm1v'
EMBEDDING_MODE = ['esm2', 'esm_if1', 'esm1v']
FORCE_RECOMPUTE_EMBEDDINGS = False
ESM2_MODEL_NAME = "esm2_t33_650M_UR50D"
ESM_IF1_MODEL_NAME = "esm_if1_gvp4_t16_142M_UR50"
ESM1V_MODEL_NAME = "esm1v_t33_650M_UR90S_1"

# --- Dimensionality Reduction ---
REDUCE_ESM_IF1_DIM = False
ESM_IF1_DIM_TARGET = 64  # Target number of dimensions after PCA (e.g., 64, 128, 256)
REDUCE_ESM2_DIM = True
ESM2_DIM_TARGET = 64  # Target number of dimensions after PCA (e.g., 64, 128, 256)
REDUCE_ESM1V_DIM = True
ESM1V_DIM_TARGET = 64  # Target number of dimensions after PCA (e.g., 64, 128, 256)