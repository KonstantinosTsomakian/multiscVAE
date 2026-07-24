import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import pandas as pd
import sys
sys.path.append("/home/ktsomakian/histone_modification_network/models")
import data_preprocessing_functions as prepro
import matplotlib.lines as mlines
import muon as mu
from early_stopage import EarlyStopping
from genomic_integration_vae import *
import subprocess
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###Parameters

# k = 2000
# work_with_pca_lsi_matrices = False
# batch_size = 512
# epochs = 30
# learning_rate = 1e-5
# use_early_stopping = True
# stopper_delta = 10
# input_length = 2000
# vae_encoder_parameter_dic = {'dense_block_1' : [input_length, 3],
#                     'transition_layer_1' : [input_length, input_length // 2],
#                     'dense_block_2' : [input_length // 2, 3],
#                     'transition_layer_2' : [input_length // 2, input_length // 4]}

# ####
# vae_decoder_parameter_dic = {'transition_layer_1' : [lattent_dim, input_length // 4],
#                     'dense_block_1' : [input_length // 4, 3],
#                     'transition_layer_2' : [input_length // 4, input_length // 2],
#                     'dense_block_2' : [input_length // 2, 3],
#                     'transition_layer_3' : [input_length // 2, input_length]}


# # embedding_pca, embedding_tsne, embedding_umap
# embedding_df_to_viz = 'umap'
# embedding_alg_name = 'UMAP'


def integrate_modalities(k, work_with_pca_lsi_matrices, batch_size, epochs, learning_rate, use_early_stopping, stopper_delta, embedding_df_to_viz, lattent_dim):
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_to_save_figs = f'/home/ktsomakian/multimodal_genomic_data_integration/figures/k{k}_pcalsi{work_with_pca_lsi_matrices}_batch_size{batch_size}_epochs{epochs}_lr{learning_rate}_ld{lattent_dim}/'
    subprocess.run(f'mkdir -p {path_to_save_figs}', text = True, shell = True)
    
    if work_with_pca_lsi_matrices:
        input_length = 49
    else:
        input_length = k
    
    vae_encoder_parameter_dic = {'dense_block_1' : [input_length, 3],
                        'transition_layer_1' : [input_length, input_length // 2],
                        'dense_block_2' : [input_length // 2, 3],
                        'transition_layer_2' : [input_length // 2, input_length // 4]}

    ####
    vae_decoder_parameter_dic = {'transition_layer_1' : [lattent_dim, input_length // 4],
                        'dense_block_1' : [input_length // 4, 3],
                        'transition_layer_2' : [input_length // 4, input_length // 2],
                        'dense_block_2' : [input_length // 2, 3],
                        'transition_layer_3' : [input_length // 2, input_length]}

    #Load the data
    mdata = mu.read("../data/pbmc10k/pbmc10k.h5mu")
    atac = mdata.mod['atac']
    rna = mdata.mod['rna']
    mdata


    # Filter the data
    #Because rna filtering produces a smaller set of variable genes i will subset the atac variable features
    # to a total of n features, where n is the total number of variable genes found in the rna data.
    rna_hvg = rna.var[rna.var['highly_variable'] == True].index
    atac_hvg = atac.var[atac.var['highly_variable'] == True].index[:len(rna_hvg)]
    print(f'\n--> In to total there are {len(rna_hvg)} features for the rna mdality and {len(atac_hvg)} features for the atac modality.')

    rna = rna[:, rna.var.index.isin(rna_hvg)].copy()
    atac = atac[:, atac.var.index.isin(atac_hvg)].copy()

    #Keep only the first k variable genes

    rna = rna[:, :k].copy()
    atac = atac[:, :k].copy()

    #Merge the modalities into one multimodal tensor
    atac_tensor = torch.Tensor(atac.X).unsqueeze(1)
    rna_tensor = torch.Tensor(rna.X).unsqueeze(1)

    multimodal_tensor = torch.cat([atac_tensor, rna_tensor], dim = 1)
    print(f'\n--> Input tensor is of shape {multimodal_tensor.shape}')



    #Define the data that the model will work with

    if work_with_pca_lsi_matrices:
        print('\nnWORKING WITH LSI MATRIX FOR ATAC DATA AND THE PCA MATRIX FOR THE RNA DATA\n')

        atac_matrix = atac.obsm['X_lsi']
        rna_matrix = rna.obsm['X_pca']

        #filter so both matrices have the same dimensions
        atac_components = atac_matrix.shape[1]
        rna_components = rna_matrix.shape[1]

        selected = atac_components if atac_components <= rna_components else rna_components

        atac_matrix = atac_matrix[:,:selected]
        rna_matrix = rna_matrix[:,:selected]

        atac_tensor = torch.Tensor(atac_matrix).unsqueeze(1)
        rna_tensor = torch.Tensor(rna_matrix).unsqueeze(1)

        multimodal_tensor = torch.cat([atac_tensor, rna_tensor], dim = 1)
        

    #Standardize the data
    print('\n--> Standardizing data!')
    standardized_data_tensor, computed_mean, compute_sd = prepro.standardize_multichannel_tensor(multimodal_tensor, dims = [0,2])



    # Split the data into training and test and convert to dataloaders
    print('\n--> Spplit data to train and test and convert to dataloaders!')
    from sklearn.model_selection import train_test_split

    indices = np.arange(multimodal_tensor.shape[0])

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    multimodal_tensor = standardized_data_tensor.to(device)
    train_tensor = multimodal_tensor[train_idx]
    test_tensor = multimodal_tensor[test_idx]

    train_tensor_list = [train_tensor[:,i,:] for i in range(train_tensor.shape[1])]
    test_tensor_list = [test_tensor[:,i,:] for i in range(test_tensor.shape[1])]

    train_dataset = torch.utils.data.TensorDataset(*train_tensor_list)
    test_dataset = torch.utils.data.TensorDataset(*test_tensor_list)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)


#     #Run the model



    early_stoper = EarlyStopping(patience=10, min_delta=stopper_delta,
                                restore_best_weights=True,accuracy=False)

    # Initialize model, optimizer

    print('\n--> Define the model.')
    model = VAE(
        latent_dim=lattent_dim,
        vae_encoder_parameter_dic = vae_encoder_parameter_dic,
        vae_decoder_parameter_dic = vae_decoder_parameter_dic,
        likelihood="gaussian",
        n_modalities=len(next(iter(train_loader)))
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    test_losses = []

    print('\n--> Train the model.')
    # Train the model
    for epoch in tqdm(range(1, epochs + 1)):
        epoch_loss, mean_test_loss, recons, jmu, jlogvar = train(epoch = epoch, model = model,
        train_loader=train_loader,test_loader=test_loader,
        optimizer = optimizer, device = device)
        train_losses.append(epoch_loss)
        test_losses.append(mean_test_loss)
        
        model_weights = model.state_dict()
        if use_early_stopping:
            early_stoper(mean_test_loss, model)
            if early_stoper.early_stop:
                model_weights = early_stoper.best_model_state
                break
        

    #Print models total prameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> Total trainable parameters: {total_params}")

    #Check if NaN values appear in hte results
    prepro.check_nan_weights(model)

    print('\n--> Ploting model performance!')
    fig, (ax1, ax2) = plt.subplots(figsize = [12,5], ncols=2)
    ax1.set_title('Train Losses over epochs', fontweight='bold')
    ax1.set_xlabel('Training Epochs', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.plot(train_losses, color = 'black')


    ax2.set_title('Test Losses over epochs', fontweight='bold')
    ax2.set_xlabel('Training Epochs', fontweight='bold')
    ax2.set_ylabel('Loss', fontweight='bold')
    ax2.plot(test_losses, color = 'black')
    plt.savefig(f'{path_to_save_figs}/model_performance', dpi=300, bbox_inches="tight")

    plt.show()

    print('\n--> Retrieve lattent representations of the data!')
    #Get all the embeddings of the data
    full_data = [multimodal_tensor[:,i,:] for i in range(multimodal_tensor.shape[1])]
    latent_embeddings = model(full_data)[1].cpu().detach()

    print('\n--> Perform dimensionality reduction on the lattent space!')
    #Perform dimensionality reduction on the embeddings and vizualize the results.
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap

    # Suppose latent_embeddings is your VAE output (batch_size x latent_dim)
    # and labels is the class labels (batch_size)


    # ---------- PCA ----------
    pca = PCA(n_components=min(lattent_dim - 1, 30))
    embedding_pca = pca.fit_transform(latent_embeddings)

    # ---------- t-SNE ----------
    tsne = TSNE(n_components=2, random_state=42)
    embedding_tsne = tsne.fit_transform(latent_embeddings)

    # ---------- UMAP ----------
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.01, n_components=min(lattent_dim - 1, 30))
    embedding_umap = reducer.fit_transform(latent_embeddings)

    # ---------- Visualization ----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PCA plot
    axes[0].scatter(embedding_pca[:,0], embedding_pca[:,1], s=2)
    axes[0].set_title("PCA", fontweight='bold')

    # t-SNE plot
    axes[1].scatter(embedding_tsne[:,0], embedding_tsne[:,1], s=2)
    axes[1].set_title("t-SNE", fontweight='bold')

    # UMAP plot
    axes[2].scatter(embedding_umap[:,0], embedding_umap[:,1], s=1)
    axes[2].set_title("UMAP", fontweight='bold')

    for ax in axes:
        ax.set_xlabel("Component 1", fontweight='bold')
        ax.set_ylabel("Component 2", fontweight='bold')

    plt.suptitle("Latent Space Visualizations", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{path_to_save_figs}/dimensionality_reduction', dpi=300, bbox_inches="tight")

    plt.show()


    #Retrieve the labels from the leiden clstering performed on the individual modalities
    atac_clusters = atac.obs['leiden'].values
    rna_clusters = rna.obs['leiden'].values

    # embedding_pca, embedding_tsne, embedding_umap
    if embedding_df_to_viz == 'tsne':
        embedding_df_to_viz = embedding_tsne

    elif embedding_df_to_viz == 'umap':
        embedding_df_to_viz = embedding_umap

    elif embedding_df_to_viz == 'pca':
        embedding_df_to_viz = embedding_pca
    embedding_alg_name = ''


    #Perform clustering on the joint representations
    import igraph as ig
    import leidenalg
    import networkx as nx
    from sklearn.neighbors import NearestNeighbors


    #Cluster latent representations
    # Step 1: Generate some synthetic data (e.g. embeddings or latent space)
    knn = NearestNeighbors(
    n_neighbors=30,
    metric="euclidean"
    )

    knn.fit(np.array(latent_embeddings))

    distances, indices = knn.kneighbors(np.array(latent_embeddings))

    # Create edges
    edges = []

    for i in range(indices.shape[0]):
        for j in indices[i]:
            if i != j:
                edges.append((i, j))

    # Create igraph graph
    G = ig.Graph(
        edges=edges,
        directed=False
    )

    # Leiden clustering
    partition = leidenalg.find_partition(
        G,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=0.5
    )

    # Cluster labels
    joint_labels = np.array(partition.membership)


    # Step 6: Visualize results
    plt.figure(figsize=(8,6))
    plt.scatter(embedding_df_to_viz[:,0], embedding_df_to_viz[:,1], c=joint_labels, cmap="tab20", s=10)
    plt.title("Louvain Clustering on kNN Graph")
    plt.savefig(f'{path_to_save_figs}/clustering', dpi=300, bbox_inches="tight")

    plt.show()


    #Plot the joint and the seperate modalities each time coloring the same cells that cluster together in the joint space
    import matplotlib.pyplot as plt
    print(f'\n--> Plotting {len(list(set(joint_labels)))} joint clusters agaiinst the joint, the RNA and the ATAC modalities')
    labels = sorted(set(joint_labels))
    n_labels = len(labels)

    fig, axes = plt.subplots(n_labels, 3, figsize=(9, 3 * n_labels))

    # Handle the case of a single label
    if n_labels == 1:
        axes = axes[None, :]

    for row, label_to_viz in enumerate(labels):

        colors = [
            "red" if i == label_to_viz else "black"
            for i in joint_labels
        ]

        # Joint UMAP
        axes[row, 0].scatter(
            embedding_df_to_viz[:, 0], embedding_df_to_viz[:, 1],
            c=colors, s=1
        )
        axes[row, 0].set_title(f"Joint {embedding_alg_name} (cluster {label_to_viz})", fontweight="bold")

        # RNA UMAP
        axes[row, 1].scatter(
            rna.obsm["X_umap"][:, 0], rna.obsm["X_umap"][:, 1],
            c=colors, s=1
        )
        axes[row, 1].set_title(f"RNA UMAP (cluster {label_to_viz})", fontweight="bold")

        # ATAC UMAP
        axes[row, 2].scatter(
            atac.obsm["X_umap"][:, 0], atac.obsm["X_umap"][:, 1],
            c=colors, s=1
        )
        axes[row, 2].set_title(f"ATAC UMAP (cluster {label_to_viz})", fontweight="bold")

        for ax in axes[row]:
            ax.set_xlabel("UMAP 1", fontweight="bold")
            ax.set_ylabel("UMAP 2", fontweight="bold")
            
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.tight_layout()
    plt.suptitle("JOINT ATAC+RNA cluster labels", fontsize=16, fontweight = 'bold', y = 1.01)
    plt.savefig(f'{path_to_save_figs}/joint_clusters', dpi=300, bbox_inches="tight")

    plt.show()


    ##################
    ##################
    print(f'\n--> Plotting {len(list(set(atac_clusters.codes)))} ATAC clusters agaiinst the joint, the RNA and the ATAC modalities')

    # Do the same for the cells in the clusters of the atac modality
    import matplotlib.pyplot as plt

    labels = sorted(set(atac_clusters.codes))
    n_labels = len(labels)

    fig, axes = plt.subplots(n_labels, 3, figsize=(9, 3 * n_labels))

    # Handle the case of a single label
    if n_labels == 1:
        axes = axes[None, :]

    for row, label_to_viz in enumerate(labels):

        colors = [
            "red" if i == label_to_viz else "black"
            for i in atac_clusters.codes
        ]

        # Joint UMAP
        axes[row, 0].scatter(
            embedding_df_to_viz[:, 0], embedding_df_to_viz[:, 1],
            c=colors, s=1
        )
        axes[row, 0].set_title(f"Joint {embedding_alg_name} (cluster {label_to_viz})", fontweight="bold")

        # RNA UMAP
        axes[row, 1].scatter(
            rna.obsm["X_umap"][:, 0], rna.obsm["X_umap"][:, 1],
            c=colors, s=1
        )
        axes[row, 1].set_title(f"RNA UMAP (cluster {label_to_viz})", fontweight="bold")

        # ATAC UMAP
        axes[row, 2].scatter(
            atac.obsm["X_umap"][:, 0], atac.obsm["X_umap"][:, 1],
            c=colors, s=1
        )
        axes[row, 2].set_title(f"ATAC UMAP (cluster {label_to_viz})", fontweight="bold")

        for ax in axes[row]:
            ax.set_xlabel("UMAP 1", fontweight="bold")
            ax.set_ylabel("UMAP 2", fontweight="bold")
            
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.tight_layout()
    plt.suptitle("ATAC modality cluster labels", fontsize=16, fontweight = 'bold', y = 1.02)
    plt.savefig(f'{path_to_save_figs}/atac_clusters', dpi=300, bbox_inches="tight")

    plt.show()

    #And for the rna
    print(f'\n--> Plotting {len(list(set(rna_clusters.codes)))} RNA clusters agaiinst the joint, the RNA and the ATAC modalities')

    import matplotlib.pyplot as plt

    labels = sorted(set(rna_clusters.codes))
    n_labels = len(labels)

    fig, axes = plt.subplots(n_labels, 3, figsize=(9, 3 * n_labels))

    # Handle the case of a single label
    if n_labels == 1:
        axes = axes[None, :]

    for row, label_to_viz in enumerate(labels):

        colors = [
            "red" if i == label_to_viz else "black"
            for i in rna_clusters.codes
        ]

        # Joint UMAP
        axes[row, 0].scatter(
            embedding_df_to_viz[:, 0], embedding_df_to_viz[:, 1],
            c=colors, s=1
        )
        axes[row, 0].set_title(f"Joint {embedding_alg_name} (cluster {label_to_viz})", fontweight="bold")

        # RNA UMAP
        axes[row, 1].scatter(
            rna.obsm["X_umap"][:, 0], rna.obsm["X_umap"][:, 1],
            c=colors, s=1
        )
        axes[row, 1].set_title(f"RNA UMAP (cluster {label_to_viz})", fontweight="bold")

        # ATAC UMAP
        axes[row, 2].scatter(
            atac.obsm["X_umap"][:, 0], atac.obsm["X_umap"][:, 1],
            c=colors, s=1
        )
        axes[row, 2].set_title(f"ATAC UMAP (cluster {label_to_viz})", fontweight="bold")

        for ax in axes[row]:
            ax.set_xlabel("UMAP 1", fontweight="bold")
            ax.set_ylabel("UMAP 2", fontweight="bold")
            
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.tight_layout()
    plt.suptitle("RNA modality cluster labels", fontsize=16, fontweight = 'bold', y = 1.02)
    plt.savefig(f'{path_to_save_figs}/rna_clusters', dpi=300, bbox_inches="tight")

    plt.show()




    def get_cluster_cell_sets(joint_cluster_num, rna_cluster_num, atac_cluster_num, mdata):
            
        #Build lists of doublets for all modalities
        #RNA
        rna_cells = mdata.obs.index[rna_clusters.codes == rna_cluster_num].to_list()

        rna_labels = ["RNA"] * len(rna_cells)

        rna_pairs = list(zip(rna_cells, rna_labels))

        #ATAC
        atac_cells = mdata.obs.index[atac_clusters.codes == atac_cluster_num].to_list()

        atac_labels = ["ATAC"] * len(atac_cells)

        atac_pairs = list(zip(atac_cells, atac_labels))

        #JOINT
        joint_cells = mdata.obs.index[joint_labels == joint_cluster_num].to_list()

        joint_pair_labels = ["JOINT"] * len(joint_cells)

        joint_pairs = list(zip(joint_cells, joint_pair_labels))
        return(rna_pairs, atac_pairs, joint_pairs)







        # print("RNA:", len(rna_cells))
        # print("ATAC:", len(atac_cells))
        # print("RNA ∩ ATAC:", len(rna_cells & atac_cells))

    print("\n--> Plotting maximum overlaps between clusters of the three modalities!")
    #Plot maximum overlap in clusters
    import math
    import matplotlib.pyplot as plt

    cluster_of_interest = 0
    mod_of_interest = 'joint'

    clusters = sorted(set(joint_labels))

    ncols = 4
    nrows = math.ceil(len(clusters) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = axes.ravel()

    for ax, cluster in zip(axes, clusters):

        result = compute_maximum_cluster_overlap(
            cluster_of_interest=cluster,
            mod_of_interest=mod_of_interest,
            rna_clusters=rna_clusters,
            atac_clusters=atac_clusters,
            joint_labels=joint_labels
        )

        joint_cluster_num, rna_cluster_num, atac_cluster_num = result[0][0]


    # Hide unused subplots
    for ax in axes[len(clusters):]:
        ax.axis("off")

    plt.suptitle("Cluster overlaps", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f'{path_to_save_figs}/max_overlap_venn', dpi=300, bbox_inches="tight")

    plt.show()
    print('\nEND OF ANALYSIS\n')