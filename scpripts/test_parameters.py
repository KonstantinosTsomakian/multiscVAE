from integrate import *
from tqdm import tqdm
import torch 
import os

k_list = [1000, 2000, 3000, 4000, 5000]
work_with_pca_lsi_matrices_list = [True, False]
batch_size_list = [64, 128, 256, 1024]
epochs_list = [30, 50, 100, 200, 300]
learning_rate_list = [1e-4, 1e-5, 1e-6, 1e-7]
use_early_stopping = True
stopper_delta = 15
lattent_dim_list = [10,20,50,70,100]
embedding_df_to_viz = 'umap'


#Generate a list of a ll the possible combintaions
from itertools import product

k_list = [500, 1000, 1500, 2000]
work_with_pca_lsi_matrices_list = [True, False]
batch_size_list = [64, 128, 256, 1024]
epochs_list = [30, 50, 100, 200, 300]
learning_rate_list = [1e-4, 1e-5, 1e-6, 1e-7]

combinations = list(product(
    k_list,
    work_with_pca_lsi_matrices_list,
    batch_size_list,
    epochs_list,
    learning_rate_list,
    lattent_dim_list
))





# integrate_modalities(k = 2000, work_with_pca_lsi_matrices = True, batch_size = 64,
#                     epochs = 100, learning_rate = 1e-5, use_early_stopping = True,
#                     stopper_delta = 0.001, embedding_df_to_viz = 'umap', lattent_dim = 10)
print(f"Total combinations: {len(combinations)}")

for comb in tqdm(combinations):

    k, work_with_pca_lsi_matrices, batch_size, epochs, learning_rate, lattent_dim = comb
    if work_with_pca_lsi_matrices:
        k = 1000 #This is a summy value so the script do not run for all possble ks when work_with_pca_lsi_matrices is True
    path_to_save_fig = f'/home/ktsomakian/multimodal_genomic_data_integration/figures/k{k}_pcalsi{work_with_pca_lsi_matrices}_batch_size{batch_size}_epochs{epochs}_lr{learning_rate}/dimensionality_reduction.png'

    
    if not os.path.exists(path_to_save_fig):
        

        try:
            integrate_modalities(k = k, work_with_pca_lsi_matrices = work_with_pca_lsi_matrices, batch_size = batch_size,
                                epochs = epochs, learning_rate = learning_rate, use_early_stopping = use_early_stopping,
                                stopper_delta = stopper_delta, embedding_df_to_viz = embedding_df_to_viz, lattent_dim = lattent_dim)
            
        except Exception as e:
            print(f"Failed for {comb}")
            print(f"Error: {e}")
            continue
    else:
        print('Parameter set already tested!')