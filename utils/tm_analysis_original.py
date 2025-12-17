#!/usr/bin/env python3
"""
TM-align Pairwise Structure Alignment Script

This script performs pairwise structural alignment between a reference structure
(6VXX - SARS-CoV-2 Spike protein) and all proteins in the data splits.

Outputs:
- RMSD: Root Mean Square Deviation (Å)
- TM-score: Template Modeling score (0-1, higher is more similar)
- Identity: Sequence identity percentage

Author: Generated for B-Cell Epitope Prediction Model validation
"""

import os
import sys
import subprocess
import json
import urllib.request
import tarfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

# Configuration
PROJECT_ROOT = Path(__file__).parent
ANTIGEN_PDB_DIR = PROJECT_ROOT / "antigen_only_pdb_files"
CLEANED_PDB_DIR = PROJECT_ROOT / "cleaned_pdb_files"
PDB_DIR = PROJECT_ROOT / "pdb_files"
SPLITS_FILE = PROJECT_ROOT / "output" / "data_splits_STRICT_no_viruses.json"
OUTPUT_DIR = PROJECT_ROOT / "tmalign_results"
TMALIGN_DIR = PROJECT_ROOT / "tmalign"
REFERENCE_PDB_ID = "6vxx"  # SARS-CoV-2 Spike protein

# TM-align download URL (source)
TMALIGN_URL = "https://zhanggroup.org/TM-align/TMalign.cpp"


def setup_tmalign() -> Path:
    """Download and compile TM-align if not already available."""
    
    tmalign_exe = TMALIGN_DIR / "TMalign"
    
    # Check if already exists
    if tmalign_exe.exists():
        print(f"✓ TM-align found at {tmalign_exe}")
        return tmalign_exe
    
    # Check if available in system PATH
    result = subprocess.run(["which", "TMalign"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ TM-align found in PATH: {result.stdout.strip()}")
        return Path(result.stdout.strip())
    
    result = subprocess.run(["which", "tmalign"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ TM-align found in PATH: {result.stdout.strip()}")
        return Path(result.stdout.strip())
    
    print("TM-align not found. Downloading and compiling...")
    
    # Create directory
    TMALIGN_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download TMalign.cpp
    cpp_file = TMALIGN_DIR / "TMalign.cpp"
    print(f"Downloading TM-align source from {TMALIGN_URL}...")
    
    try:
        urllib.request.urlretrieve(TMALIGN_URL, cpp_file)
    except Exception as e:
        print(f"Error downloading TM-align: {e}")
        print("\nPlease install TM-align manually:")
        print("  1. Download from https://zhanggroup.org/TM-align/")
        print("  2. Compile with: g++ -O3 -o TMalign TMalign.cpp")
        print("  3. Add to PATH or place in ./tmalign/")
        sys.exit(1)
    
    # Compile
    print("Compiling TM-align...")
    compile_result = subprocess.run(
        ["g++", "-O3", "-ffast-math", "-o", str(tmalign_exe), str(cpp_file)],
        capture_output=True,
        text=True
    )
    
    if compile_result.returncode != 0:
        print(f"Compilation error: {compile_result.stderr}")
        print("\nPlease install g++ and try again:")
        print("  sudo apt-get install g++")
        sys.exit(1)
    
    # Make executable
    os.chmod(tmalign_exe, 0o755)
    print(f"✓ TM-align compiled successfully at {tmalign_exe}")
    
    return tmalign_exe


def download_reference_pdb(pdb_id: str = REFERENCE_PDB_ID) -> Path:
    """Download the reference PDB structure if not available."""
    
    # Check various locations
    possible_paths = [
        ANTIGEN_PDB_DIR / f"{pdb_id}_antigen_only.pdb",
        CLEANED_PDB_DIR / f"{pdb_id}_cleaned.pdb",
        PDB_DIR / f"{pdb_id}.pdb",
        OUTPUT_DIR / f"{pdb_id}.pdb"
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✓ Reference PDB found at {path}")
            return path
    
    # Download from RCSB PDB
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{pdb_id}.pdb"
    
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"Downloading reference structure {pdb_id.upper()} from RCSB PDB...")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Downloaded {pdb_id.upper()}.pdb to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading {pdb_id}: {e}")
        sys.exit(1)


def parse_tmalign_output(output: str) -> Dict:
    """Parse TM-align output to extract metrics."""
    
    result = {
        "rmsd": None,
        "tm_score_chain1": None,  # Normalized by length of chain 1
        "tm_score_chain2": None,  # Normalized by length of chain 2
        "tm_score_avg": None,
        "seq_identity": None,
        "aligned_length": None,
        "length_chain1": None,
        "length_chain2": None
    }
    
    for line in output.split('\n'):
        # Parse aligned length and RMSD
        # Example: "Aligned length=  178, RMSD=   3.52, Seq_ID=n_identical/n_aligned=  0.096"
        if "Aligned length=" in line:
            match = re.search(r'Aligned length=\s*(\d+),\s*RMSD=\s*([\d.]+)', line)
            if match:
                result["aligned_length"] = int(match.group(1))
                result["rmsd"] = float(match.group(2))
            
            # Parse sequence identity
            id_match = re.search(r'Seq_ID=.*?=\s*([\d.]+)', line)
            if id_match:
                result["seq_identity"] = float(id_match.group(1)) * 100  # Convert to percentage
        
        # Parse TM-score
        # Example: "TM-score= 0.28583 (if normalized by length of Chain_1)"
        if "TM-score=" in line:
            tm_match = re.search(r'TM-score=\s*([\d.]+)', line)
            if tm_match:
                tm_score = float(tm_match.group(1))
                if "Chain_1" in line:
                    result["tm_score_chain1"] = tm_score
                elif "Chain_2" in line:
                    result["tm_score_chain2"] = tm_score
        
        # Parse chain lengths
        if "Length of Chain_1:" in line:
            match = re.search(r'Length of Chain_1:\s*(\d+)', line)
            if match:
                result["length_chain1"] = int(match.group(1))
        if "Length of Chain_2:" in line:
            match = re.search(r'Length of Chain_2:\s*(\d+)', line)
            if match:
                result["length_chain2"] = int(match.group(1))
    
    # Calculate average TM-score
    if result["tm_score_chain1"] is not None and result["tm_score_chain2"] is not None:
        result["tm_score_avg"] = (result["tm_score_chain1"] + result["tm_score_chain2"]) / 2
    
    return result


def run_tmalign(tmalign_exe: Path, pdb1: Path, pdb2: Path) -> Optional[Dict]:
    """Run TM-align on two PDB files."""
    
    try:
        result = subprocess.run(
            [str(tmalign_exe), str(pdb1), str(pdb2)],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout per alignment
        )
        
        if result.returncode != 0:
            return None
        
        return parse_tmalign_output(result.stdout)
        
    except subprocess.TimeoutExpired:
        print(f"Timeout for {pdb2.name}")
        return None
    except Exception as e:
        print(f"Error running TM-align on {pdb2.name}: {e}")
        return None


def find_pdb_file(pdb_id: str) -> Optional[Path]:
    """Find PDB file for a given PDB ID in various directories."""
    
    # Priority order: antigen_only > cleaned > original
    candidates = [
        ANTIGEN_PDB_DIR / f"{pdb_id}_antigen_only.pdb",
        CLEANED_PDB_DIR / f"{pdb_id}_cleaned.pdb", 
        PDB_DIR / f"{pdb_id}.pdb"
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    return None


def load_splits() -> Dict[str, List[str]]:
    """Load the data splits JSON file."""
    
    if not SPLITS_FILE.exists():
        print(f"Error: Splits file not found at {SPLITS_FILE}")
        sys.exit(1)
    
    with open(SPLITS_FILE, 'r') as f:
        splits = json.load(f)
    
    return splits


def main(test_limit: Optional[int] = None):
    """Main function to run TM-align analysis.
    
    Args:
        test_limit: If provided, only process this many proteins (for testing)
    """
    
    print("=" * 70)
    print("TM-align Pairwise Structure Alignment Analysis")
    print("=" * 70)
    print(f"\nReference structure: {REFERENCE_PDB_ID.upper()} (SARS-CoV-2 Spike)")
    print(f"Target: All proteins in {SPLITS_FILE.name}")
    if test_limit:
        print(f"*** TEST MODE: Processing only {test_limit} proteins ***")
    print()
    
    # Setup TM-align
    tmalign_exe = setup_tmalign()
    
    # Get reference PDB
    reference_pdb = download_reference_pdb()
    
    # Load splits
    splits = load_splits()
    
    # Get all unique PDB IDs from all splits
    all_pdb_ids = set()
    for split_name, pdb_list in splits.items():
        if isinstance(pdb_list, list):
            all_pdb_ids.update(pdb_list)
    
    print(f"\nTotal unique proteins in splits: {len(all_pdb_ids)}")
    
    # Find available PDB files
    available_pdbs = {}
    missing_pdbs = []
    
    for pdb_id in all_pdb_ids:
        pdb_path = find_pdb_file(pdb_id)
        if pdb_path:
            available_pdbs[pdb_id] = pdb_path
        else:
            missing_pdbs.append(pdb_id)
    
    print(f"Available PDB files: {len(available_pdbs)}")
    print(f"Missing PDB files: {len(missing_pdbs)}")
    
    if missing_pdbs and len(missing_pdbs) <= 10:
        print(f"Missing: {missing_pdbs}")
    
    # Apply test limit if specified
    if test_limit:
        pdb_items = list(available_pdbs.items())[:test_limit]
        available_pdbs = dict(pdb_items)
        print(f"\n*** Limited to {len(available_pdbs)} proteins for testing ***")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run TM-align for all proteins
    print("\n" + "=" * 70)
    print("Running TM-align alignments...")
    print("=" * 70)
    
    results = []
    
    for pdb_id, pdb_path in tqdm(available_pdbs.items(), desc="Aligning"):
        alignment = run_tmalign(tmalign_exe, reference_pdb, pdb_path)
        
        if alignment:
            results.append({
                "pdb_id": pdb_id,
                "pdb_path": str(pdb_path),
                "rmsd": alignment["rmsd"],
                "tm_score": alignment["tm_score_chain2"],  # Normalized by target length
                "tm_score_ref": alignment["tm_score_chain1"],  # Normalized by reference length
                "tm_score_avg": alignment["tm_score_avg"],
                "seq_identity": alignment["seq_identity"],
                "aligned_length": alignment["aligned_length"],
                "target_length": alignment["length_chain2"],
                "split": next((s for s, pdbs in splits.items() if pdb_id in pdbs), "unknown")
            })
        else:
            results.append({
                "pdb_id": pdb_id,
                "pdb_path": str(pdb_path),
                "rmsd": None,
                "tm_score": None,
                "tm_score_ref": None,
                "tm_score_avg": None,
                "seq_identity": None,
                "aligned_length": None,
                "target_length": None,
                "split": next((s for s, pdbs in splits.items() if pdb_id in pdbs), "unknown")
            })
    
    # Create DataFrame and save results
    df = pd.DataFrame(results)
    
    # Sort by TM-score (highest first)
    df_sorted = df.sort_values('tm_score', ascending=False, na_position='last')
    
    # Save to CSV
    output_csv = OUTPUT_DIR / f"tmalign_vs_{REFERENCE_PDB_ID}_results.csv"
    df_sorted.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to {output_csv}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    df_valid = df[df['tm_score'].notna()]
    
    print(f"\nTotal alignments attempted: {len(df)}")
    print(f"Successful alignments: {len(df_valid)}")
    print(f"Failed alignments: {len(df) - len(df_valid)}")
    
    if len(df_valid) > 0:
        print(f"\nRMSD (Å):")
        print(f"  Mean:   {df_valid['rmsd'].mean():.2f}")
        print(f"  Median: {df_valid['rmsd'].median():.2f}")
        print(f"  Min:    {df_valid['rmsd'].min():.2f}")
        print(f"  Max:    {df_valid['rmsd'].max():.2f}")
        
        print(f"\nTM-score (normalized by target length):")
        print(f"  Mean:   {df_valid['tm_score'].mean():.4f}")
        print(f"  Median: {df_valid['tm_score'].median():.4f}")
        print(f"  Min:    {df_valid['tm_score'].min():.4f}")
        print(f"  Max:    {df_valid['tm_score'].max():.4f}")
        
        print(f"\nSequence Identity (%):")
        print(f"  Mean:   {df_valid['seq_identity'].mean():.2f}%")
        print(f"  Median: {df_valid['seq_identity'].median():.2f}%")
        print(f"  Min:    {df_valid['seq_identity'].min():.2f}%")
        print(f"  Max:    {df_valid['seq_identity'].max():.2f}%")
        
        # TM-score interpretation
        print("\n" + "-" * 50)
        print("TM-score Interpretation:")
        print("-" * 50)
        print("  TM-score > 0.5  : Same fold (structurally similar)")
        print("  TM-score > 0.17 : Not random (some structural similarity)")
        print("  TM-score < 0.17 : Random structural similarity")
        
        high_similarity = df_valid[df_valid['tm_score'] > 0.5]
        medium_similarity = df_valid[(df_valid['tm_score'] > 0.17) & (df_valid['tm_score'] <= 0.5)]
        low_similarity = df_valid[df_valid['tm_score'] <= 0.17]
        
        print(f"\n  High similarity (TM > 0.5):   {len(high_similarity)} proteins ({100*len(high_similarity)/len(df_valid):.1f}%)")
        print(f"  Medium similarity (0.17-0.5): {len(medium_similarity)} proteins ({100*len(medium_similarity)/len(df_valid):.1f}%)")
        print(f"  Low similarity (TM < 0.17):   {len(low_similarity)} proteins ({100*len(low_similarity)/len(df_valid):.1f}%)")
        
        # Top 10 most similar proteins
        print("\n" + "-" * 50)
        print(f"Top 10 most structurally similar proteins to {REFERENCE_PDB_ID.upper()}:")
        print("-" * 50)
        top10 = df_sorted.head(10)
        print(top10[['pdb_id', 'tm_score', 'rmsd', 'seq_identity', 'split']].to_string(index=False))
        
        # Summary by split
        print("\n" + "-" * 50)
        print("Summary by data split:")
        print("-" * 50)
        for split_name in ['train', 'val', 'test']:
            split_data = df_valid[df_valid['split'] == split_name]
            if len(split_data) > 0:
                high_tm_split = split_data[split_data['tm_score'] > 0.5].sort_values('tm_score', ascending=False)
                print(f"\n{split_name.upper()} split:")
                print(f"  Count: {len(split_data)}")
                print(f"  Mean TM-score: {split_data['tm_score'].mean():.4f}")
                print(f"  Proteins with TM > 0.5: {len(high_tm_split)}")
                
                # Show all proteins with TM > 0.5 for this split
                if len(high_tm_split) > 0:
                    print(f"\n  All {len(high_tm_split)} proteins with TM > 0.5 in {split_name.upper()}:")
                    print("  " + "-" * 60)
                    # Format the output nicely
                    for idx, row in high_tm_split.iterrows():
                        print(f"    {row['pdb_id']:8s}  TM={row['tm_score']:.5f}  RMSD={row['rmsd']:.2f}Å  SeqID={row['seq_identity']:.1f}%")


def analyze_existing_results():
    """Analyze existing TM-align results without re-running."""
    
    output_csv = OUTPUT_DIR / f"tmalign_vs_{REFERENCE_PDB_ID}_results.csv"
    
    if not output_csv.exists():
        print(f"No existing results found at {output_csv}")
        print("Run the script without --analyze flag first.")
        return
    
    df = pd.read_csv(output_csv)
    print(f"Loaded {len(df)} results from {output_csv}")
    
    # Same summary statistics as in main()
    df_valid = df[df['tm_score'].notna()]
    
    if len(df_valid) > 0:
        print(f"\n{'='*70}")
        print("ANALYSIS OF EXISTING RESULTS")
        print(f"{'='*70}")
        
        print(f"\nTM-score distribution:")
        print(df_valid['tm_score'].describe())
        
        # Create histogram bins
        bins = [0, 0.17, 0.3, 0.5, 0.7, 1.0]
        labels = ['0-0.17', '0.17-0.3', '0.3-0.5', '0.5-0.7', '0.7-1.0']
        df_valid['tm_bin'] = pd.cut(df_valid['tm_score'], bins=bins, labels=labels)
        
        print("\nTM-score distribution by bins:")
        print(df_valid['tm_bin'].value_counts().sort_index())
        
        # Summary by split with detailed high TM proteins
        print("\n" + "-" * 50)
        print("Summary by data split:")
        print("-" * 50)
        for split_name in ['train']:
            split_data = df_valid[df_valid['split'] == split_name]
            if len(split_data) > 0:
                high_tm_split = split_data[split_data['tm_score'] > 0.5].sort_values('tm_score', ascending=False)
                print(f"\n{split_name.upper()} split:")
                print(f"  Count: {len(split_data)}")
                print(f"  Mean TM-score: {split_data['tm_score'].mean():.4f}")
                print(f"  Proteins with TM > 0.5: {len(high_tm_split)}")
                
                # Show all proteins with TM > 0.5 for this split
                if len(high_tm_split) > 0:
                    print(f"\n  All {len(high_tm_split)} proteins with TM > 0.5 in {split_name.upper()}:")
                    print("  " + "-" * 60)
                    for idx, row in high_tm_split.iterrows():
                        print(f"    {row['pdb_id']:8s}  TM={row['tm_score']:.5f}  RMSD={row['rmsd']:.2f}Å  SeqID={row['seq_identity']:.1f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TM-align pairwise structure alignment')
    parser.add_argument('--analyze', action='store_true', 
                        help='Analyze existing results without re-running alignments')
    parser.add_argument('--reference', type=str, default=REFERENCE_PDB_ID,
                        help=f'Reference PDB ID (default: {REFERENCE_PDB_ID})')
    parser.add_argument('--test', type=int, default=0,
                        help='Run quick test with N proteins only (e.g., --test 5)')
    
    args = parser.parse_args()
    
    if args.reference:
        REFERENCE_PDB_ID = args.reference.lower()
    
    if args.analyze:
        analyze_existing_results()
    else:
        main(test_limit=args.test if args.test > 0 else None)
