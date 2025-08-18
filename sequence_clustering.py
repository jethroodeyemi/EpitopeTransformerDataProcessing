# sequence_clustering.py
import os
import subprocess
import json
import random
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio.PDB.Polypeptide import is_aa
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def extract_sequences_to_fasta(df, antigen_pdb_dir, fasta_path):
    """
    Parses antigen-only PDB files to extract sequences and saves them to a FASTA file.
    The FASTA header is formatted as 'pdb_id|chain_id' for easy parsing later.
    """
    print("\n--- Extracting sequences to FASTA for clustering ---")
    parser = PDBParser(QUIET=True)
    with open(fasta_path, 'w') as f_out:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting sequences"):
            pdb_id = row['pdb']
            chain_id = row['antigen_chain']
            pdb_path = os.path.join(antigen_pdb_dir, f"{pdb_id}_antigen_only.pdb")

            if not os.path.exists(pdb_path):
                continue

            try:
                structure = parser.get_structure(pdb_id, pdb_path)
                chain = structure[0][chain_id]
                residues = [res for res in chain if is_aa(res, standard=True)]
                sequence = "".join([seq1(res.get_resname()) for res in residues])
                
                if sequence:
                    # Use a unique header that we can parse later
                    header = f">{pdb_id}|{chain_id}\n"
                    f_out.write(header)
                    f_out.write(f"{sequence}\n")
            except Exception as e:
                print(f"Warning: Could not process {pdb_path} for sequence. Error: {e}")

def run_cd_hit(fasta_path, output_path, threshold):
    """
    Runs the CD-HIT command-line tool to cluster sequences.
    """
    print(f"\n--- Running CD-HIT with a {threshold*100:.0f}% identity threshold ---")
    # The -n 2 is for thresholds between 40-50%, a CD-HIT recommendation for speed/accuracy.
    word_size = 2 if 0.4 <= threshold <= 0.5 else 3
    
    cmd = [
        'cd-hit',
        '-i', fasta_path,
        '-o', output_path,
        '-c', str(threshold),
        '-n', str(word_size),
        '-d', '0' # Prevents long sequence names from being truncated
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"CD-HIT clustering complete. Output: {output_path}.clstr")
    except FileNotFoundError:
        print("\n*** ERROR: 'cd-hit' command not found. ***")
        print("Please install CD-HIT and ensure it is in your system's PATH.")
        raise
    except subprocess.CalledProcessError as e:
        print("\n*** ERROR: CD-HIT failed to run. ***")
        print("Command:", " ".join(e.cmd))
        print("Error message:\n", e.stderr)
        raise

def parse_clusters(cluster_file_path):
    """
    Parses the .clstr file from CD-HIT into a dictionary.
    Returns a dict mapping cluster_id -> [pdb_id, pdb_id, ...].
    """
    print(f"\n--- Parsing cluster file: {cluster_file_path} ---")
    clusters = {}
    current_cluster_id = None
    with open(cluster_file_path, 'r') as f:
        for line in f:
            if line.startswith('>Cluster'):
                current_cluster_id = int(line.strip().split()[-1])
                clusters[current_cluster_id] = []
            else:
                # Header format is >pdb_id|chain_id...
                pdb_id = line.split('>')[1].split('|')[0]
                if current_cluster_id is not None:
                    clusters[current_cluster_id].append(pdb_id)
    print(f"Found {len(clusters)} clusters.")
    return clusters

def subsample_clusters(clusters, max_size):
    """
    Subsamples large clusters to a maximum size to prevent training bias.
    """
    if max_size is None:
        return clusters
    
    print(f"\n--- Subsampling large clusters to a max size of {max_size} ---")
    subsampled_clusters = {}
    total_removed = 0
    for cid, members in clusters.items():
        if len(members) > max_size:
            total_removed += len(members) - max_size
            subsampled_clusters[cid] = random.sample(members, max_size)
        else:
            subsampled_clusters[cid] = members
    
    print(f"Subsampling complete. Removed {total_removed} proteins from oversized clusters.")
    return subsampled_clusters

def create_clustered_splits(clusters, test_size=0.2, val_size=0.1, random_state=42):
    """
    Creates train, validation, and test splits from the cluster dictionary.
    The split is performed on cluster IDs to ensure no homology leakage.
    """
    print("\n--- Creating train/validation/test splits based on clusters ---")
    cluster_ids = list(clusters.keys())
    
    # Split clusters into train_val and test
    train_val_cids, test_cids = train_test_split(
        cluster_ids, test_size=test_size, random_state=random_state
    )
    
    # Calculate the proportion of the original data to use for the validation set
    val_prop = val_size / (1 - test_size)
    
    # Split train_val clusters into train and val
    train_cids, val_cids = train_test_split(
        train_val_cids, test_size=val_prop, random_state=random_state
    )
    
    # Expand cluster IDs back to PDB IDs
    train_pdbs = [pdb for cid in train_cids for pdb in clusters[cid]]
    val_pdbs = [pdb for cid in val_cids for pdb in clusters[cid]]
    test_pdbs = [pdb for cid in test_cids for pdb in clusters[cid]]
    
    splits = {
        'train': train_pdbs,
        'val': val_pdbs,
        'test': test_pdbs
    }
    
    print(f"Split sizes (by PDB ID):")
    print(f"  Train: {len(train_pdbs)} proteins in {len(train_cids)} clusters")
    print(f"  Validation: {len(val_pdbs)} proteins in {len(val_cids)} clusters")
    print(f"  Test: {len(test_pdbs)} proteins in {len(test_cids)} clusters")
    
    return splits