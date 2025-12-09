"""
Validation Script for Epitope Prediction Model Performance

This script provides two key validation approaches:
1. Calculate the random baseline to demonstrate model lift
2. Generate 3D visualization files for mapping predictions onto protein structures

Usage:
    python validate_model_performance.py

The script will:
- Calculate the random baseline AUC-PR based on class imbalance
- Generate a PyMOL script to visualize predictions on a SARS-CoV-2 structure (or any PDB)
- Provide detailed statistics comparing model performance to random chance
"""

import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

import config


def calculate_random_baseline_and_lift():
    """
    Calculate the random baseline AUC-PR for the test set and demonstrate model lift.
    
    For AUC-PR, the random baseline equals the fraction of positive samples.
    This shows that your model's 0.26 AUC-PR is a significant improvement over random.
    """
    print("=" * 70)
    print("  PART 1: CALCULATING RANDOM BASELINE AND MODEL LIFT")
    print("=" * 70)
    
    # --- 1. Load Data ---
    print("\n--- Loading structured data ---")
    with open(config.STRUCTURED_DATA_PATH, 'rb') as f:
        protein_data_list = pickle.load(f)

    print("--- Reconstructing flat data arrays ---")
    features, labels, groups = [], [], []
    for protein_data in tqdm(protein_data_list, desc="Reconstructing arrays"):
        features.append(protein_data['X_arr'])
        labels.append(protein_data['df_stats']['is_epitope'].values)
        groups.append(np.full(protein_data['length'], protein_data['pdb_id']))

    X = np.vstack(features)
    y = np.concatenate(labels)
    groups = np.concatenate(groups)

    # --- 2. Load the Pre-computed, Clustered Splits ---
    print(f"\n--- Loading clustered data splits from '{config.SPLITS_FILE_PATH}' ---")
    with open(config.SPLITS_FILE_PATH, 'r') as f:
        splits = json.load(f)
    
    test_groups = splits['test']
    test_mask = np.isin(groups, test_groups)

    y_test = y[test_mask]
    
    # --- 3. Calculate Statistics ---
    total_residues = len(y_test)
    positive_residues = np.sum(y_test == 1)
    negative_residues = np.sum(y_test == 0)
    positive_fraction = positive_residues / total_residues
    
    print("\n" + "=" * 70)
    print("  TEST SET CLASS DISTRIBUTION")
    print("=" * 70)
    print(f"  Total residues in test set:     {total_residues:,}")
    print(f"  Positive (epitope) residues:    {positive_residues:,}")
    print(f"  Negative (non-epitope) residues: {negative_residues:,}")
    print(f"  Positive fraction:               {positive_fraction:.4f} ({positive_fraction*100:.2f}%)")
    print(f"  Class imbalance ratio:           1:{negative_residues/positive_residues:.1f} (pos:neg)")
    
    # --- 4. Calculate Random Baseline ---
    # For AUC-PR (Average Precision), the random baseline = positive fraction
    random_baseline_aucpr = positive_fraction
    
    # Your model's performance
    model_aucpr = 0.2558  # From training output
    model_aucroc = 0.8268
    
    # Calculate lift
    lift = model_aucpr / random_baseline_aucpr
    improvement_absolute = model_aucpr - random_baseline_aucpr
    improvement_percent = (model_aucpr - random_baseline_aucpr) / random_baseline_aucpr * 100
    
    print("\n" + "=" * 70)
    print("  RANDOM BASELINE vs MODEL PERFORMANCE")
    print("=" * 70)
    print(f"\n  RANDOM BASELINE:")
    print(f"    AUC-PR (random classifier): {random_baseline_aucpr:.4f}")
    print(f"    AUC-ROC (random classifier): 0.5000")
    
    print(f"\n  YOUR MODEL:")
    print(f"    AUC-PR:  {model_aucpr:.4f}")
    print(f"    AUC-ROC: {model_aucroc:.4f}")
    
    print(f"\n  MODEL LIFT (how many times better than random):")
    print(f"    AUC-PR Lift:  {lift:.2f}x better than random")
    print(f"    AUC-ROC Lift: {model_aucroc/0.5:.2f}x better than random")
    
    print(f"\n  ABSOLUTE IMPROVEMENT:")
    print(f"    AUC-PR:  +{improvement_absolute:.4f} ({improvement_percent:.1f}% improvement)")
    print(f"    AUC-ROC: +{model_aucroc - 0.5:.4f} ({(model_aucroc - 0.5)/0.5*100:.1f}% improvement)")
    
    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)
    print(f"""
  With only {positive_fraction*100:.2f}% of residues being epitopes:
  
  ✓ A random classifier achieves AUC-PR of {random_baseline_aucpr:.4f}
  ✓ Your model achieves AUC-PR of {model_aucpr:.4f}
  ✓ This is a {lift:.1f}x improvement (lift) over random chance!
  
  This demonstrates that your model has learned meaningful patterns
  for distinguishing epitope residues from non-epitope residues.
  
  The {model_aucroc:.2f} AUC-ROC also shows excellent discrimination ability.
""")
    
    return {
        'total_residues': total_residues,
        'positive_residues': positive_residues,
        'positive_fraction': positive_fraction,
        'random_baseline_aucpr': random_baseline_aucpr,
        'model_aucpr': model_aucpr,
        'model_aucroc': model_aucroc,
        'lift': lift
    }


def visualize_predictions_on_structure(pdb_id, chain_id, output_dir='visualization_output'):
    """
    Generate predictions for a specific protein and create PyMOL visualization script.
    
    Args:
        pdb_id: PDB identifier (e.g., '6vxx', '7bnn')
        chain_id: Chain ID for the antigen
        output_dir: Directory to save output files
    """
    print("\n" + "=" * 70)
    print(f"  PART 2: GENERATING 3D VISUALIZATION FOR {pdb_id.upper()} Chain {chain_id}")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Load the trained model ---
    print(f"\n--- Loading trained model from '{config.FINAL_MODEL_PATH}' ---")
    model = xgb.XGBClassifier()
    model.load_model(config.FINAL_MODEL_PATH)
    
    # --- 2. Load structured data and find the protein ---
    print("--- Loading structured data ---")
    with open(config.STRUCTURED_DATA_PATH, 'rb') as f:
        protein_data_list = pickle.load(f)
    
    # Find the target protein
    target_protein = None
    for protein_data in protein_data_list:
        if protein_data['pdb_id'].lower() == pdb_id.lower():
            target_protein = protein_data
            break
    
    if target_protein is None:
        print(f"Error: Protein {pdb_id} not found in the dataset.")
        print("Available PDB IDs (first 20):", [p['pdb_id'] for p in protein_data_list[:20]])
        return None
    
    # --- 3. Run predictions ---
    print(f"--- Running predictions for {pdb_id} ---")
    X_protein = target_protein['X_arr']
    df_stats = target_protein['df_stats'].copy()
    
    pred_probas = model.predict_proba(X_protein)[:, 1]
    df_stats['prediction_score'] = pred_probas
    df_stats['pdb_id'] = pdb_id
    
    # --- 4. Print statistics ---
    print(f"\nProtein: {pdb_id}, Chain: {chain_id}")
    print(f"Total residues: {len(df_stats)}")
    print(f"Epitope residues (ground truth): {df_stats['is_epitope'].sum()}")
    print(f"Predicted epitopes (score >= 0.5): {(pred_probas >= 0.5).sum()}")
    print(f"Predicted epitopes (score >= 0.6): {(pred_probas >= 0.6).sum()}")
    print(f"Predicted epitopes (score >= 0.7): {(pred_probas >= 0.7).sum()}")
    
    print("\n--- Top 20 Highest-Scoring Residues ---")
    top_20 = df_stats.nlargest(20, 'prediction_score')[['res_id', 'residue', 'prediction_score', 'is_epitope', 'rsa']]
    print(top_20.to_string(index=False))
    
    # --- 5. Save predictions to CSV ---
    csv_path = os.path.join(output_dir, f'{pdb_id}_{chain_id}_predictions.csv')
    df_stats.to_csv(csv_path, index=False)
    print(f"\nPredictions saved to: {csv_path}")
    
    # --- 6. Generate PyMOL script ---
    pymol_script_path = os.path.join(output_dir, f'{pdb_id}_{chain_id}_visualization.pml')
    generate_pymol_script(df_stats, pdb_id, chain_id, pymol_script_path)
    
    # --- 7. Generate B-factor file for coloring ---
    bfactor_pdb_path = os.path.join(output_dir, f'{pdb_id}_{chain_id}_prediction_bfactor.pdb')
    generate_bfactor_pdb(df_stats, pdb_id, chain_id, bfactor_pdb_path)
    
    # --- 8. Create a prediction distribution plot ---
    plot_path = os.path.join(output_dir, f'{pdb_id}_{chain_id}_score_distribution.png')
    plot_prediction_distribution(df_stats, pdb_id, chain_id, plot_path)
    
    return df_stats


def generate_pymol_script(df_stats, pdb_id, chain_id, output_path):
    """
    Generate a PyMOL script to visualize predictions on the 3D structure.
    Colors residues from blue (low score) to red (high score).
    """
    script = f'''# PyMOL Visualization Script for Epitope Predictions
# PDB: {pdb_id}, Chain: {chain_id}
# Generated by validate_model_performance.py

# --- SETUP ---
# Fetch structure from PDB (or load local file)
fetch {pdb_id}, async=0

# Remove waters and other heteroatoms for clarity
remove solvent
remove hetatm

# Show cartoon representation
hide everything
show cartoon, chain {chain_id}
set cartoon_fancy_helices, 1
set cartoon_smooth_loops, 1

# Set background
bg_color white

# --- COLOR BY PREDICTION SCORE ---
# Color scheme: Blue (low) -> White (medium) -> Red (high)
# Using spectrum coloring based on prediction scores

# First, set all to gray
color gray, chain {chain_id}

# Color by prediction score (approximate using B-factor coloring)
# Load the B-factor modified PDB for accurate coloring
# Or use the selections below

# High confidence predictions (score >= 0.7) - RED
'''
    
    # Add high confidence residues
    high_conf = df_stats[df_stats['prediction_score'] >= 0.7]['res_id'].tolist()
    if high_conf:
        residue_list = '+'.join([str(r).replace(' ', '') for r in high_conf])
        script += f"select high_conf_epitopes, chain {chain_id} and resi {residue_list}\n"
        script += "color red, high_conf_epitopes\n"
        script += "show sticks, high_conf_epitopes\n\n"
    else:
        script += "# No high confidence predictions (>= 0.7)\n\n"
    
    # Add medium confidence residues
    script += "# Medium confidence predictions (0.5 <= score < 0.7) - ORANGE\n"
    med_conf = df_stats[(df_stats['prediction_score'] >= 0.5) & 
                        (df_stats['prediction_score'] < 0.7)]['res_id'].tolist()
    if med_conf:
        residue_list = '+'.join([str(r).replace(' ', '') for r in med_conf])
        script += f"select med_conf_epitopes, chain {chain_id} and resi {residue_list}\n"
        script += "color orange, med_conf_epitopes\n"
        script += "show sticks, med_conf_epitopes\n\n"
    else:
        script += "# No medium confidence predictions (0.5-0.7)\n\n"
    
    # Add low-medium confidence residues
    script += "# Low-medium confidence predictions (0.3 <= score < 0.5) - YELLOW\n"
    low_med_conf = df_stats[(df_stats['prediction_score'] >= 0.3) & 
                            (df_stats['prediction_score'] < 0.5)]['res_id'].tolist()
    if low_med_conf:
        residue_list = '+'.join([str(r).replace(' ', '') for r in low_med_conf])
        script += f"select low_med_epitopes, chain {chain_id} and resi {residue_list}\n"
        script += "color yellow, low_med_epitopes\n\n"
    else:
        script += "# No low-medium confidence predictions (0.3-0.5)\n\n"
    
    # Add ground truth epitopes for comparison
    script += "# --- GROUND TRUTH EPITOPES (for comparison) ---\n"
    gt_epitopes = df_stats[df_stats['is_epitope'] == 1]['res_id'].tolist()
    if gt_epitopes:
        residue_list = '+'.join([str(r).replace(' ', '') for r in gt_epitopes])
        script += f"select ground_truth_epitopes, chain {chain_id} and resi {residue_list}\n"
        script += "# Uncomment to show ground truth with different representation:\n"
        script += "# show spheres, ground_truth_epitopes\n"
        script += "# color green, ground_truth_epitopes\n\n"
    
    script += f'''
# --- FINALIZE VIEW ---
orient chain {chain_id}
zoom chain {chain_id}

# Add labels for highest scoring residues
'''
    
    # Add labels for top 5 predictions
    top_5 = df_stats.nlargest(5, 'prediction_score')
    for _, row in top_5.iterrows():
        script += f"label chain {chain_id} and resi {row['res_id']} and name CA, \"{row['res_id']}:{row['prediction_score']:.2f}\"\n"
    
    script += f'''
# --- SAVE IMAGES ---
set ray_opaque_background, 1
set antialias, 2
ray 1920, 1080
png {pdb_id}_{chain_id}_epitope_prediction.png, dpi=300

# --- LEGEND ---
# RED:    High confidence epitope (score >= 0.7)
# ORANGE: Medium confidence epitope (0.5 <= score < 0.7)
# YELLOW: Low-medium confidence (0.3 <= score < 0.5)
# GRAY:   Low confidence (score < 0.3)
# GREEN:  Ground truth epitopes (uncomment to show)

print "\\n=== Visualization Complete ==="
print "High confidence epitopes (red): {len(high_conf)}"
print "Medium confidence epitopes (orange): {len(med_conf)}"
print "Ground truth epitopes: {len(gt_epitopes)}"
'''
    
    with open(output_path, 'w') as f:
        f.write(script)
    
    print(f"PyMOL script saved to: {output_path}")
    print(f"  To use: pymol {output_path}")


def generate_bfactor_pdb(df_stats, pdb_id, chain_id, output_path):
    """
    Create a Python script to generate a modified PDB file with prediction scores as B-factors.
    This allows smooth spectrum coloring in PyMOL.
    """
    bfactor_script = f'''# B-factor Coloring Script
# Run this in PyMOL to color by prediction score using B-factors

# First, fetch the structure
fetch {pdb_id}, async=0

# Set all B-factors to 0
alter chain {chain_id}, b=0

# Set B-factors based on prediction scores
'''
    
    for _, row in df_stats.iterrows():
        res_id = str(row['res_id']).replace(' ', '')
        score = row['prediction_score'] * 100  # Scale to 0-100 for better visualization
        bfactor_script += f"alter chain {chain_id} and resi {res_id}, b={score:.1f}\n"
    
    bfactor_script += f'''
# Color by B-factor (spectrum)
spectrum b, blue_white_red, chain {chain_id}, minimum=0, maximum=100

# Show surface colored by prediction
show surface, chain {chain_id}
set transparency, 0.3

# Alternative: cartoon with putty (thickness = prediction score)
# hide surface
# show cartoon, chain {chain_id}
# cartoon putty, chain {chain_id}
# set cartoon_putty_scale_min, 0.5
# set cartoon_putty_scale_max, 2.0
# set cartoon_putty_radius, 0.2

orient chain {chain_id}
zoom chain {chain_id}

# Save the modified structure
save {pdb_id}_{chain_id}_with_bfactors.pdb, chain {chain_id}

print "B-factor coloring complete!"
print "Blue = low prediction score"
print "Red = high prediction score"
'''
    
    with open(output_path.replace('.pdb', '_bfactor_script.pml'), 'w') as f:
        f.write(bfactor_script)
    
    print(f"B-factor coloring script saved to: {output_path.replace('.pdb', '_bfactor_script.pml')}")


def plot_prediction_distribution(df_stats, pdb_id, chain_id, output_path):
    """
    Create a histogram showing the distribution of prediction scores.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Score distribution
    plt.subplot(1, 2, 1)
    plt.hist(df_stats['prediction_score'], bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0.5, color='r', linestyle='--', label='Threshold 0.5')
    plt.axvline(x=0.7, color='orange', linestyle='--', label='Threshold 0.7')
    plt.xlabel('Prediction Score')
    plt.ylabel('Number of Residues')
    plt.title(f'Distribution of Epitope Prediction Scores\n{pdb_id.upper()} Chain {chain_id}')
    plt.legend()
    
    # Plot 2: Score by epitope status
    plt.subplot(1, 2, 2)
    epitope_scores = df_stats[df_stats['is_epitope'] == 1]['prediction_score']
    non_epitope_scores = df_stats[df_stats['is_epitope'] == 0]['prediction_score']
    
    plt.hist(non_epitope_scores, bins=30, alpha=0.5, label=f'Non-Epitope (n={len(non_epitope_scores)})', color='blue')
    plt.hist(epitope_scores, bins=30, alpha=0.5, label=f'Epitope (n={len(epitope_scores)})', color='red')
    plt.xlabel('Prediction Score')
    plt.ylabel('Number of Residues')
    plt.title(f'Score Distribution by Ground Truth Label\n{pdb_id.upper()} Chain {chain_id}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Score distribution plot saved to: {output_path}")


def find_sars_cov2_structures():
    """
    Find SARS-CoV-2 related structures in the dataset for visual validation.
    """
    print("\n" + "=" * 70)
    print("  FINDING SARS-CoV-2 RELATED STRUCTURES FOR VISUALIZATION")
    print("=" * 70)
    
    # Common SARS-CoV-2 spike protein PDB IDs
    sars_cov2_spike_pdbs = [
        '6vxx',  # SARS-CoV-2 Spike glycoprotein
        '6vsb',  # Prefusion spike
        '6zge',  # Spike with ACE2
        '7bnn',  # Spike RBD
        '7c01',  # Spike with neutralizing antibody
        '6m0j',  # RBD-ACE2 complex
        '7kj2',  # Spike bound to antibody
        '6wpt',  # Spike protein closed state
    ]
    
    # Check which ones are in our dataset
    with open(config.STRUCTURED_DATA_PATH, 'rb') as f:
        protein_data_list = pickle.load(f)
    
    available_pdbs = {p['pdb_id'].lower() for p in protein_data_list}
    
    print("\nChecking for known SARS-CoV-2 structures in dataset:")
    found_structures = []
    for pdb in sars_cov2_spike_pdbs:
        if pdb.lower() in available_pdbs:
            print(f"  ✓ {pdb.upper()} - FOUND")
            found_structures.append(pdb)
        else:
            print(f"  ✗ {pdb.upper()} - not in dataset")
    
    # List some example structures from the test set
    print("\nSample structures from test set (first 10):")
    with open(config.SPLITS_FILE_PATH, 'r') as f:
        splits = json.load(f)
    
    for pdb_id in splits['test'][:10]:
        print(f"  - {pdb_id}")
    
    return found_structures


def main():
    """Main function to run all validation analyses."""
    print("\n" + "=" * 70)
    print("  EPITOPE PREDICTION MODEL VALIDATION SCRIPT")
    print("=" * 70)
    
    # Part 1: Calculate random baseline and lift
    stats = calculate_random_baseline_and_lift()
    
    # Part 2: Find available structures
    sars_structures = find_sars_cov2_structures()
    
    # Part 3: Generate visualization for available structure or first test structure
    # You can modify this to use any PDB ID you want to visualize
    
    # First, let's use a structure from the test set
    with open(config.SPLITS_FILE_PATH, 'r') as f:
        splits = json.load(f)
    
    # Try to find a suitable structure
    # Get the first few test structures for visualization
    test_structures = splits['test'][:5]
    
    print("\n" + "=" * 70)
    print("  GENERATING VISUALIZATIONS FOR TEST STRUCTURES")
    print("=" * 70)
    
    # Load structured data to get chain info
    with open(config.STRUCTURED_DATA_PATH, 'rb') as f:
        protein_data_list = pickle.load(f)
    
    pdb_to_chain = {}
    for protein_data in protein_data_list:
        pdb_id = protein_data['pdb_id'].lower()
        # Try to get chain from df_stats if available
        if 'chain' in protein_data['df_stats'].columns:
            chain = protein_data['df_stats']['chain'].iloc[0]
        else:
            chain = 'A'  # Default
        pdb_to_chain[pdb_id] = chain
    
    # Generate visualization for the first available test structure
    for pdb_id in test_structures[:1]:  # Just do one for now
        pdb_lower = pdb_id.lower()
        if pdb_lower in pdb_to_chain:
            chain_id = pdb_to_chain[pdb_lower]
            print(f"\nGenerating visualization for {pdb_id.upper()} Chain {chain_id}...")
            visualize_predictions_on_structure(pdb_id, chain_id)
        else:
            print(f"Skipping {pdb_id} - chain info not found")
    
    print("\n" + "=" * 70)
    print("  VALIDATION COMPLETE!")
    print("=" * 70)
    print("""
  Summary:
  --------
  1. Random Baseline Analysis: See above for detailed statistics
  2. Visualization Files: Check the 'visualization_output' folder for:
     - CSV file with per-residue predictions
     - PyMOL script for 3D visualization (.pml)
     - B-factor coloring script for smooth color gradients
     - Score distribution plots
  
  To visualize in PyMOL:
  ----------------------
  pymol visualization_output/<pdb_id>_<chain>_visualization.pml
  
  To use custom SARS-CoV-2 structure (e.g., 6VXX):
  ------------------------------------------------
  1. Download PDB file: wget https://files.rcsb.org/download/6VXX.pdb
  2. Run prediction: python predict_standalone.py 6VXX.pdb A
  3. Or modify this script to process your structure
""")


if __name__ == '__main__':
    main()
