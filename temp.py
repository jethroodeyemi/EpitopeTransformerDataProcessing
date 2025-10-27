import os
import pickle
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import config
from utils.visualize_glycosylation import plot_glycosylation_for_one_chain, plot_glycosylation_for_multiple_chains

if __name__ == "__main__":
    with open(config.FINAL_DATAFRAME_PATH, 'rb') as f:
        final_df = pickle.load(f)
    # with pd.option_context('display.max_rows', None, 
    #                    'display.max_columns', None, 
    #                    'display.width', 1000,
    #                    'display.max_colwidth', None):
    #     print("--- Full DataFrame Preview ---")
    #     print(final_df[['res_id', 'res_name', 'is_epitope', 'is_glycosylated', 'dist_to_glycosylation']])
    plot_glycosylation_for_multiple_chains(final_df, num_chains=None, outlier_threshold=1)
    plot_glycosylation_for_one_chain(final_df, '1a14')
    print(final_df['embedding'].iloc[0].shape)



