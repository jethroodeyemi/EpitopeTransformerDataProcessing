# run_pipeline.py

import json
import os
import traceback

# Import our custom modules
import config
import data_preparation as dp
import structure_processing as sp
import feature_engineering as fe
import sequence_clustering as sc

def main():
    """Main function to run the entire pipeline."""
    
    print("--- STARTING ANTIGEN-EPITOPE PROCESSING PIPELINE ---")
    
    # Create necessary directories
    for dir_path in [config.OUTPUT_DIR, config.PDB_DIR, config.CLEANED_PDB_DIR, config.ANTIGEN_ONLY_PDB_DIR, config.EMBEDDING_CACHE_DIR]:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory '{dir_path}' is ready.")

    try:
        # Step 1 & 2: Filter TSV and download PDBs
        deduped_df = dp.filter_and_deduplicate_tsv(config.INPUT_TSV, config.DEDUPED_TSV)
        #dp.download_pdbs(deduped_df, config.PDB_DIR)

        # Step 3: Clean PDB files
        #sp.clean_pdbs(deduped_df, config.PDB_DIR, config.CLEANED_PDB_DIR, config.ANTIGEN_ONLY_PDB_DIR)
        
        # Step 4: Generate features and labels
        fe.generate_features_and_labels(deduped_df, config.CLEANED_PDB_DIR, config.ANTIGEN_ONLY_PDB_DIR)

        # Step 5: Perform sequence clustering and create data splits ---
        # sc.extract_sequences_to_fasta(deduped_df, config.ANTIGEN_ONLY_PDB_DIR, config.FASTA_PATH)
        # sc.run_cd_hit(config.FASTA_PATH, config.CLUSTER_FILE_PATH, config.CDHIT_THRESHOLD)
        # clusters = sc.parse_clusters(f"{config.CLUSTER_FILE_PATH}.clstr")
        # subsampled_clusters = sc.subsample_clusters(clusters, config.MAX_CLUSTER_SIZE)
        # splits = sc.create_clustered_splits(subsampled_clusters)
        # with open(config.SPLITS_FILE_PATH, 'w') as f:
        #     json.dump(splits, f, indent=4)

        print("\n--- Pipeline finished successfully! ---")
        print(f"Final results are in: {config.FINAL_DATAFRAME_PATH}")

    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()