# predict.py (with plot saving and per-protein metrics)

import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import config

# --- MODIFICATION 1: Update the data loading function to return groups ---
def load_data_and_create_test_set():
    """Loads pre-processed data, reconstructs arrays, and returns the test set."""
    print("--- Loading structured data and recreating test set ---")
    
    # This section assumes you have a single large pickle file with all data
    # If not, you might need to adjust this part.
    try:
        with open(config.STRUCTURED_DATA_PATH, 'rb') as f:
            protein_data_list = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Structured data file not found at '{config.STRUCTURED_DATA_PATH}'")
        print("Please ensure your pre-processing script creates this file.")
        exit()
        
    features, labels, groups = [], [], []
    for protein_data in tqdm(protein_data_list, desc="Reconstructing arrays"):
        features.append(protein_data['X_arr'])
        labels.append(protein_data['df_stats']['is_epitope'].values)
        groups.append(np.full(protein_data['length'], protein_data['pdb_id']))
        
    X = np.vstack(features)
    y = np.concatenate(labels)
    groups = np.concatenate(groups)

    with open(config.SPLITS_FILE_PATH, 'r') as f:
        splits = json.load(f)
        
    # We use the base PDB ID (without chain) for matching the JSON
    base_groups = np.array([g.split('_')[0] for g in groups])
    test_pdb_ids = set(splits['test'])
    print(f"Loaded {len(test_pdb_ids)} test PDB IDs from splits file: {list(test_pdb_ids)[:5]}...")

    test_mask = np.isin(base_groups, list(test_pdb_ids))
    X_test, y_test, test_groups_arr = X[test_mask], y[test_mask], groups[test_mask]
    
    print(f"Test set loaded: {len(X_test)} residues from {len(np.unique(test_groups_arr))} proteins.")
    return X_test, y_test, test_groups_arr

# --- NEW FUNCTION: To calculate per-protein metrics ---
def calculate_per_protein_metrics(df_results: pd.DataFrame):
    """Calculates key metrics for each protein individually."""
    print("\n--- Calculating Per-Protein Performance Metrics ---")
    per_protein_stats = []
    
    # Use the full PDB ID with chain for grouping
    for pdb_id in tqdm(df_results['pdb'].unique(), desc="Evaluating proteins"):
        df_pdb = df_results[df_results['pdb'] == pdb_id].copy()
        y_true = df_pdb['epitope'].values
        scores = df_pdb['score'].values
        
        # Skip proteins with no positive examples, as metrics are undefined
        if y_true.sum() == 0:
            continue
        
        # Calculate Epitope Rank Score
        epitope_scores = scores[y_true == 1]
        percentile_ranks = [np.mean(score >= scores) for score in epitope_scores]
        mean_epitope_rank_score = np.mean(percentile_ranks)

        per_protein_stats.append({
            'pdb': pdb_id,
            'auc_roc': roc_auc_score(y_true, scores),
            'auc_pr': average_precision_score(y_true, scores),
            'epitope_rank_score': mean_epitope_rank_score,
        })
    
    return pd.DataFrame(per_protein_stats)

# --- Plotting functions remain the same ---
def save_precision_recall_curve(y_true, y_pred_proba, output_path):
    """Plots and saves the Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where='post', label=f'AUC-PR = {avg_precision:.4f}')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve on Hold-Out Test Set')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Precision-Recall curve saved to: {output_path}")

def save_confusion_matrix(y_true, y_pred_class, output_path):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred_class)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Non-Epitope', 'Predicted Epitope'],
                yticklabels=['Actual Non-Epitope', 'Actual Epitope'])
    plt.title(f'Confusion Matrix on Hold-Out Test Set (Threshold={config.PREDICTION_THRESHOLD})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")

def main():
    os.makedirs(config.EVALUATION_OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(config.FINAL_MODEL_PATH):
        print(f"Error: Model file not found at '{config.FINAL_MODEL_PATH}'")
        return

    print(f"--- Loading final model from '{config.FINAL_MODEL_PATH}' ---")
    final_model = xgb.XGBClassifier()
    final_model.load_model(config.FINAL_MODEL_PATH)
    
    # --- MODIFICATION 2: Get test_groups back from the function ---
    X_test, y_test, test_groups = load_data_and_create_test_set()
    
    print("\n--- Evaluating model on the Hold-Out Test Set ---")
    y_test_pred_proba = final_model.predict_proba(X_test)[:, 1]

    # --- Global Metrics ---
    test_auc_pr = average_precision_score(y_test, y_test_pred_proba)
    test_auc_roc = roc_auc_score(y_test, y_test_pred_proba)

    print("\n--- Global Performance Metrics (on all test residues) ---")
    print("==============================================")
    print(f"  FINAL TEST SET AUC-PR: {test_auc_pr:.4f}")
    print(f"  FINAL TEST SET AUC-ROC: {test_auc_roc:.4f}")
    print("==============================================")
    
    # Save the visualizations
    save_precision_recall_curve(y_test, y_test_pred_proba, os.path.join(config.EVALUATION_OUTPUT_DIR, 'pr_curve.png'))
    y_pred_class = (y_test_pred_proba > config.PREDICTION_THRESHOLD).astype(int)
    save_confusion_matrix(y_test, y_pred_class, os.path.join(config.EVALUATION_OUTPUT_DIR, 'confusion_matrix.png'))

    # --- MODIFICATION 3: Calculate and display per-protein metrics ---
    # Create a DataFrame to hold all necessary data
    df_results = pd.DataFrame({
        'pdb': test_groups,
        'epitope': y_test,
        'score': y_test_pred_proba
    })

    # Calculate per-protein stats
    df_per_protein = calculate_per_protein_metrics(df_results)
    
    # Calculate and print the mean of the per-protein metrics
    mean_auc_roc = df_per_protein['auc_roc'].mean()
    mean_auc_pr = df_per_protein['auc_pr'].mean()
    mean_rank_score = df_per_protein['epitope_rank_score'].mean()

    print("\n--- Per-Protein Performance Metrics (Averaged) ---")
    print("==============================================")
    print(f"  Mean Per-Protein AUC-ROC: {mean_auc_roc:.4f}")
    print(f"  Mean Per-Protein AUC-PR:  {mean_auc_pr:.4f}")
    print(f"  Mean Epitope Rank Score:  {mean_rank_score:.4f}")
    print(f"  (True epitopes rank in the top {(1-mean_rank_score)*100:.1f}% on average)")
    print("==============================================")
    
    # Save the detailed per-protein results to a file for further inspection
    per_protein_path = os.path.join(config.EVALUATION_OUTPUT_DIR, 'per_protein_metrics.csv')
    df_per_protein.to_csv(per_protein_path, index=False)
    print(f"\nDetailed per-protein metrics saved to: {per_protein_path}")

if __name__ == '__main__':
    main()