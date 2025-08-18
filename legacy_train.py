# train_epitope_predictor_final.py

import pandas as pd
import numpy as np
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def train_cv(X_train_val, y_train_val, groups_train_val):
    """
    Performs GroupKFold cross-validation on the training+validation set.
    Returns the list of trained models and the index of the best one.
    """
    print("--- Starting model training with GroupKFold cross-validation ---")
    
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'logloss', 'n_estimators': 1500,
        'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'gamma': 0.1, 'tree_method': 'hist',
        'random_state': 42
    }

    n_splits = 2
    gkf = GroupKFold(n_splits=n_splits)
    
    models, auc_pr_scores, auc_roc_scores = [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_val, y_train_val, groups=groups_train_val)):
        print(f"\n===== Fold {fold+1}/{n_splits} =====")
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        
        scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
        print(f"Scale Pos Weight for this fold: {scale_pos_weight:.2f}")
        
        model = xgb.XGBClassifier(**params, scale_pos_weight=scale_pos_weight, early_stopping_rounds=50)
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        auc_pr = average_precision_score(y_val, y_pred_proba)
        auc_roc = roc_auc_score(y_val, y_pred_proba)
        auc_pr_scores.append(auc_pr)
        auc_roc_scores.append(auc_roc)
        models.append(model)
        
        print(f"Fold {fold+1} AUC-PR: {auc_pr:.4f}, AUC-ROC: {auc_roc:.4f}")

    print("\n--- Cross-Validation Summary ---")
    print(f"Average AUC-PR: {np.mean(auc_pr_scores):.4f} ± {np.std(auc_pr_scores):.4f}")
    print(f"Average AUC-ROC: {np.mean(auc_roc_scores):.4f} ± {np.std(auc_roc_scores):.4f}")
    
    best_model_idx = np.argmax(auc_pr_scores)
    print(f"\nBest model found in Fold {best_model_idx + 1} with AUC-PR: {auc_pr_scores[best_model_idx]:.4f}")

    return models, best_model_idx

def train_final_model(X_train_val, y_train_val, best_params, best_iteration):
    """
    Trains one final model on the entire training+validation set.
    """
    print("\n--- Training one final model on all train+validation data ---")
    
    # Use the best number of trees found during the best CV fold
    final_params = best_params.copy()
    final_params.pop('scale_pos_weight', None)
    final_params['n_estimators'] = best_iteration
    # No early stopping needed as we already know the best number of estimators
    final_params.pop('early_stopping_rounds', None) 
    
    scale_pos_weight = np.sum(y_train_val == 0) / np.sum(y_train_val == 1)
    
    final_model = xgb.XGBClassifier(**final_params, scale_pos_weight=scale_pos_weight)
    final_model.fit(X_train_val, y_train_val, verbose=True)
    
    print("Final model training complete.")
    return final_model

def plot_feature_importance(model, feature_names, top_n=30):
    """Plots the feature importance from a trained XGBoost model."""
    print("\n--- Plotting Feature Importance from final model ---")
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.show()

def main():
    # --- 1. Load and Prepare Data ---
    print("--- Loading structured data ---")
    with open('output/structured_protein_data.pkl', 'rb') as f:
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

    feature_names = []
    feature_idxs = protein_data_list[0]['feature_idxs']
    sorted_features = sorted(feature_idxs.items(), key=lambda item: item[1].start)
    for name, idx_range in sorted_features:
        feature_names.extend([f"{name}_{i}" for i in range(len(idx_range))] if len(idx_range) > 1 else [name])

    # --- 2. Create the Hold-Out Test Set (Group-based Split) ---
    print("\n--- Creating Hold-Out Test Set (20% of proteins) ---")
    unique_groups = np.unique(groups)
    train_val_groups, test_groups = train_test_split(unique_groups, test_size=0.2, random_state=42)

    # Create boolean masks
    train_val_mask = np.isin(groups, train_val_groups)
    test_mask = np.isin(groups, test_groups)

    X_train_val, X_test = X[train_val_mask], X[test_mask]
    y_train_val, y_test = y[train_val_mask], y[test_mask]
    groups_train_val = groups[train_val_mask]

    print(f"Train/Val set size: {len(X_train_val)} residues from {len(train_val_groups)} proteins")
    print(f"Test set size: {len(X_test)} residues from {len(test_groups)} proteins")

    # --- 3. Cross-Validation on Train/Val set ---
    cv_models, best_model_idx = train_cv(X_train_val, y_train_val, groups_train_val)
    best_cv_model = cv_models[best_model_idx]
    
    # Save the best model from cross-validation
    os.makedirs('models', exist_ok=True)
    best_cv_model.save_model('models/best_cv_model.json')
    print("\nBest model from CV saved to 'models/best_cv_model.json'")

    # --- 4. Train Final Model on All Train/Val Data ---
    # Get parameters from the best CV model
    best_params = best_cv_model.get_params()
    # Get the optimal number of boosting rounds
    best_iteration = best_cv_model.best_iteration
    print(f"Optimal number of trees found: {best_iteration}")

    final_model = train_final_model(X_train_val, y_train_val, best_params, best_iteration)

    # Save the final model
    final_model.save_model('models/final_model.json')
    print("Final model trained on all data saved to 'models/final_model.json'")
    
    # --- 5. Final Evaluation on the Hold-Out Test Set ---
    print("\n--- Final Evaluation on the Hold-Out Test Set ---")
    y_test_pred_proba = final_model.predict_proba(X_test)[:, 1]

    test_auc_pr = average_precision_score(y_test, y_test_pred_proba)
    test_auc_roc = roc_auc_score(y_test, y_test_pred_proba)

    print("\n==============================================")
    print(f"  FINAL TEST SET AUC-PR: {test_auc_pr:.4f}")
    print(f"  FINAL TEST SET AUC-ROC: {test_auc_roc:.4f}")
    print("==============================================")
    
    # --- 6. Analyze Feature Importance from the Final Model ---
    plot_feature_importance(final_model, feature_names)

if __name__ == '__main__':
    main()