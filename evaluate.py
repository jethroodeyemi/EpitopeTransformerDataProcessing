# predict.py (with plot saving)

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

# ... (load_data_and_create_test_set function remains the same) ...
def load_data_and_create_test_set():
    print("--- Loading structured data and recreating test set ---")
    with open(config.STRUCTURED_DATA_PATH, 'rb') as f:
        protein_data_list = pickle.load(f)
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
    test_groups = splits['test']
    print("Test set PDBs loaded from splits file:", test_groups[:5], "...")

    test_mask = np.isin(groups, test_groups)
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"Test set loaded: {len(X_test)} residues from {len(test_groups)} proteins")
    return X_test, y_test


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
    
    # *** THE CHANGE IS HERE ***
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close() # Close the figure to free up memory
    print(f"Precision-Recall curve saved to: {output_path}")

def save_confusion_matrix(y_true, y_pred_class, output_path):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred_class)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Non-Epitope', 'Predicted Epitope'],
                yticklabels=['Actual Non-Epitope', 'Actual Epitope'])
    plt.title('Confusion Matrix on Hold-Out Test Set (Threshold=0.5)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # *** THE CHANGE IS HERE ***
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close() # Close the figure to free up memory
    print(f"Confusion matrix saved to: {output_path}")


def main():
    os.makedirs(config.EVALUATION_OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(config.FINAL_MODEL_PATH):
        print(f"Error: Model file not found at '{config.FINAL_MODEL_PATH}'")
        return

    print(f"--- Loading final model from '{config.FINAL_MODEL_PATH}' ---")
    final_model = xgb.XGBClassifier()
    final_model.load_model(config.FINAL_MODEL_PATH)
    
    X_test, y_test = load_data_and_create_test_set()
    
    print("\n--- Evaluating model on the Hold-Out Test Set ---")
    y_test_pred_proba = final_model.predict_proba(X_test)[:, 1]

    test_auc_pr = average_precision_score(y_test, y_test_pred_proba)
    test_auc_roc = roc_auc_score(y_test, y_test_pred_proba)

    print("\n==============================================")
    print(f"  FINAL TEST SET AUC-PR: {test_auc_pr:.4f}")
    print(f"  FINAL TEST SET AUC-ROC: {test_auc_roc:.4f}")
    print("==============================================")
    
    # Save the visualizations instead of showing them
    save_precision_recall_curve(y_test, y_test_pred_proba, os.path.join(config.EVALUATION_OUTPUT_DIR, 'pr_curve.png'))

    y_pred_class = (y_test_pred_proba > 0.5).astype(int)
    save_confusion_matrix(y_test, y_pred_class, os.path.join(config.EVALUATION_OUTPUT_DIR, 'confusion_matrix.png'))

if __name__ == '__main__':
    main()