#!/usr/bin/env python3
"""
TM-align Pairwise Structure Alignment Script & Strict Split Generator

This script:
1. Performs pairwise structural alignment between a reference (SARS-CoV-2) and the dataset.
2. Generates a 'split_strict.json' by filtering out structural homologs/cousins.
   Criteria for Removal: SeqID > 25% AND TM-score > 0.5

Author: Updated for Zero-Shot Viral Prediction Validation
"""

import os
import sys
import subprocess
import json
import urllib.request
import re
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import pandas as pd

# Configuration
PROJECT_ROOT = Path(__file__).parent
ANTIGEN_PDB_DIR = PROJECT_ROOT / "antigen_only_pdb_files"
CLEANED_PDB_DIR = PROJECT_ROOT / "cleaned_pdb_files"
PDB_DIR = PROJECT_ROOT / "pdb_files"
INPUT_SPLITS_FILE = PROJECT_ROOT / "output" / "split_clean.json"
OUTPUT_STRICT_FILE = PROJECT_ROOT / "output" / "split_strict.json"
OUTPUT_DIR = PROJECT_ROOT / "tmalign_results"
TMALIGN_DIR = PROJECT_ROOT / "tmalign"
REFERENCE_PDB_ID = "6vxx"  # SARS-CoV-2 Spike protein

# TM-align download URL (source)
TMALIGN_URL = "https://zhanggroup.org/TM-align/TMalign.cpp"


def setup_tmalign() -> Path:
    """Download and compile TM-align if not already available."""
    tmalign_exe = TMALIGN_DIR / "TMalign"
    if tmalign_exe.exists(): return tmalign_exe
    
    # Check PATH
    for cmd in ["TMalign", "tmalign"]:
        result = subprocess.run(["which", cmd], capture_output=True, text=True)
        if result.returncode == 0: return Path(result.stdout.strip())
    
    print("TM-align not found. Downloading and compiling...")
    TMALIGN_DIR.mkdir(parents=True, exist_ok=True)
    cpp_file = TMALIGN_DIR / "TMalign.cpp"
    
    try:
        urllib.request.urlretrieve(TMALIGN_URL, cpp_file)
        subprocess.run(["g++", "-O3", "-ffast-math", "-o", str(tmalign_exe), str(cpp_file)], check=True)
        os.chmod(tmalign_exe, 0o755)
        return tmalign_exe
    except Exception as e:
        print(f"Error installing TM-align: {e}")
        sys.exit(1)


def download_reference_pdb(pdb_id: str = REFERENCE_PDB_ID) -> Path:
    """Download the reference PDB structure if not available."""
    # Check local paths first
    for path in [ANTIGEN_PDB_DIR, CLEANED_PDB_DIR, PDB_DIR, OUTPUT_DIR]:
        for f in path.glob(f"{pdb_id}*.pdb"):
            return f
            
    # Download
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{pdb_id}.pdb"
    print(f"Downloading reference {pdb_id}...")
    urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", output_path)
    return output_path


def parse_tmalign_output(output: str) -> Dict:
    """Parse TM-align output to extract metrics."""
    result = {
        "rmsd": None, "tm_score_chain2": None, "tm_score_avg": None, 
        "seq_identity": None, "aligned_length": None, "length_chain2": None
    }
    
    for line in output.split('\n'):
        if "Aligned length=" in line:
            m = re.search(r'Aligned length=\s*(\d+),\s*RMSD=\s*([\d.]+)', line)
            if m:
                result["aligned_length"] = int(m.group(1))
                result["rmsd"] = float(m.group(2))
            m_id = re.search(r'Seq_ID=.*?=\s*([\d.]+)', line)
            if m_id: result["seq_identity"] = float(m_id.group(1)) * 100
        
        if "TM-score=" in line:
            m_tm = re.search(r'TM-score=\s*([\d.]+)', line)
            if m_tm:
                val = float(m_tm.group(1))
                if "Chain_2" in line: result["tm_score_chain2"] = val
                # We use Chain 2 normalization (length of target) as standard
    
    return result


def run_tmalign(tmalign_exe: Path, pdb1: Path, pdb2: Path) -> Optional[Dict]:
    """Run TM-align on two PDB files."""
    try:
        # Run TMalign pdb1 pdb2 (Align pdb2 to pdb1)
        res = subprocess.run([str(tmalign_exe), str(pdb1), str(pdb2)], 
                           capture_output=True, text=True, timeout=120)
        if res.returncode == 0: return parse_tmalign_output(res.stdout)
    except: pass
    return None


def find_pdb_file(pdb_id: str) -> Optional[Path]:
    """Find PDB file for a given PDB ID."""
    for d in [ANTIGEN_PDB_DIR, CLEANED_PDB_DIR, PDB_DIR]:
        f = d / f"{pdb_id}_antigen_only.pdb"
        if f.exists(): return f
        f = d / f"{pdb_id}_cleaned.pdb"
        if f.exists(): return f
        f = d / f"{pdb_id}.pdb"
        if f.exists(): return f
    return None


def generate_strict_split_file(df: pd.DataFrame, original_splits: Dict):
    """
    Generate split_strict.json by removing homologs from Train/Val.
    Condition: SeqID > 25% AND TM > 0.5
    """
    print("\n" + "=" * 70)
    print("GENERATING STRICT SPLIT (Zero-Shot)")
    print("=" * 70)
    
    # Identify leakage candidates
    # Condition: High Sequence Identity (>25%) AND High Structural Similarity (>0.5)
    leakage_mask = (df['seq_identity'] > 25.0) & (df['tm_score'] > 0.5)
    leakage_pdbs = set(df[leakage_mask]['pdb_id'].tolist())
    
    print(f"Filtering Criteria: SeqID > 25% AND TM-score > 0.5")
    print(f"Total proteins flagged for removal: {len(leakage_pdbs)}")
    
    new_splits = {
        "train": [],
        "val": [],
        "test": original_splits.get("test", []) # Keep TEST exactly as is (SARS-CoV-2)
    }
    
    removed_counts = {"train": 0, "val": 0, "test": 0}
    
    # Process Train
    for pdb in original_splits.get("train", []):
        if pdb in leakage_pdbs:
            removed_counts["train"] += 1
            # Optional: Print what is being removed to verify
            row = df[df['pdb_id'] == pdb].iloc[0]
            print(f"  Removing from TRAIN: {pdb} (TM={row['tm_score']:.2f}, ID={row['seq_identity']:.1f}%)")
        else:
            new_splits["train"].append(pdb)
            
    # Process Val
    for pdb in original_splits.get("val", []):
        if pdb in leakage_pdbs:
            removed_counts["val"] += 1
        else:
            new_splits["val"].append(pdb)
            
    # Save
    with open(OUTPUT_STRICT_FILE, 'w') as f:
        json.dump(new_splits, f, indent=4)
        
    print(f"\nStrict Split Statistics:")
    print(f"  TRAIN: {len(new_splits['train'])} proteins (Removed {removed_counts['train']} homologs)")
    print(f"  VAL:   {len(new_splits['val'])} proteins (Removed {removed_counts['val']} homologs)")
    print(f"  TEST:  {len(new_splits['test'])} proteins (Unchanged - contains Target)")
    print(f"\n✓ Saved strict splits to: {OUTPUT_STRICT_FILE}")


def main(test_limit: Optional[int] = None, analyze_only: bool = False):
    
    # Load Splits
    if not INPUT_SPLITS_FILE.exists():
        print(f"Error: {INPUT_SPLITS_FILE} not found.")
        return
    with open(INPUT_SPLITS_FILE, 'r') as f:
        splits = json.load(f)
        
    output_csv = OUTPUT_DIR / f"tmalign_vs_{REFERENCE_PDB_ID}_results.csv"
    
    # 1. RUN OR LOAD ALIGNMENTS
    if analyze_only and output_csv.exists():
        print(f"Loading existing results from {output_csv}...")
        df = pd.read_csv(output_csv)
    else:
        # Setup and Run
        tmalign_exe = setup_tmalign()
        reference_pdb = download_reference_pdb()
        
        all_pdb_ids = set()
        for pdb_list in splits.values():
            all_pdb_ids.update(pdb_list)
            
        # Map PDBs to files
        available_pdbs = {}
        for pdb_id in all_pdb_ids:
            p = find_pdb_file(pdb_id)
            if p: available_pdbs[pdb_id] = p
            
        if test_limit:
            available_pdbs = dict(list(available_pdbs.items())[:test_limit])
            
        results = []
        print(f"Running TM-align on {len(available_pdbs)} proteins...")
        
        for pdb_id, pdb_path in tqdm(available_pdbs.items()):
            aln = run_tmalign(tmalign_exe, reference_pdb, pdb_path)
            
            # Determine split membership for this PDB
            current_split = "unknown"
            if pdb_id in splits.get("train", []): current_split = "train"
            elif pdb_id in splits.get("val", []): current_split = "val"
            elif pdb_id in splits.get("test", []): current_split = "test"
            
            if aln:
                results.append({
                    "pdb_id": pdb_id,
                    "tm_score": aln["tm_score_chain2"], # Target length normalized
                    "seq_identity": aln["seq_identity"],
                    "rmsd": aln["rmsd"],
                    "split": current_split
                })
            else:
                results.append({"pdb_id": pdb_id, "tm_score": 0, "seq_identity": 0, "split": current_split})
                
        df = pd.DataFrame(results)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.sort_values('tm_score', ascending=False).to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    # 2. GENERATE STRICT SPLIT
    if not df.empty:
        generate_strict_split_file(df, splits)
        
    # 3. PRINT SUMMARY
    print("\n" + "-"*50)
    print(f"Top Homologs (candidates for removal in Strict Split):")
    print("-"*50)
    # Filter for display: high TM + moderate SeqID
    candidates = df[(df['tm_score'] > 0.5) & (df['seq_identity'] > 25.0)].sort_values('tm_score', ascending=False)
    print(candidates[['pdb_id', 'tm_score', 'seq_identity', 'split']].head(15).to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze', action='store_true', help='Analyze existing CSV without running TM-align')
    parser.add_argument('--test', type=int, default=0, help='Test with N proteins')
    args = parser.parse_args()
    
    main(test_limit=args.test if args.test > 0 else None, analyze_only=args.analyze)