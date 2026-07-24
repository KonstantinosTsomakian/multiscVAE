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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Build a class that can be used to build residual dense linear blocks of different size
class dense_block(nn.Module):
	def __init__(self, input_dim, blocksize = 1):
		super(dense_block, self).__init__()
		self.input_dim = input_dim
		self.blocksize = blocksize


		self.layers = nn.ModuleDict({})
		for i in range(self.blocksize):
			self.layers[f'Linnear{i+1}'] = nn.Linear(in_features = input_dim, out_features=input_dim)
		
		self.bns = nn.ModuleDict({})

		for i in range(self.blocksize):
			self.layers[f'Linnear{i+1}'] = nn.Linear(input_dim, input_dim)
			self.bns[f'BN{i+1}'] = nn.BatchNorm1d(input_dim)



	def forward(self, x):
		initial = x	
		for i in range(self.blocksize):				
			x = F.relu(self.bns[f'BN{i+1}'](self.layers[f'Linnear{i+1}'](x)))

		return(x + initial)


# Define the encoder that takes the input and reutns the mean and the variance (log variance) of the latent dimension.
class Encoder(nn.Module):
    def __init__(self, latent_dim, vae_encoder_parameter_dic):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.vae_encoder_parameter_dic = vae_encoder_parameter_dic

		#Define dense blocks
        self.encoder_layers = nn.ModuleDict({})
        for i, (layer_name, parameters) in enumerate(self.vae_encoder_parameter_dic.items()):
            # print(parameters)
            if 'dense' in layer_name:
                dimension, size = parameters
                final_output_dim = dimension
                self.encoder_layers[f'dense_{i}'] = dense_block(input_dim=dimension, blocksize=size)
            elif 'transition' in layer_name:
                input_dim, output_dim = parameters
                self.encoder_layers[f'transition_{i}'] = nn.Linear(input_dim, output_dim)
                final_output_dim = output_dim
                
        # print(self.encoder_layers)
        self.encoder_mu = nn.Linear(final_output_dim, latent_dim) 
        # Log variance of latent space  
        # I use the log variance and not the variance for numerical stability
        self.encoder_logvar = nn.Linear(final_output_dim, latent_dim)

    def encode(self, x):
        for layer in self.encoder_layers.values():
            x = F.relu(layer(x))
 
        return self.encoder_mu(x), self.encoder_logvar(x)      
    def forward(self, x):
        mu, logvar = self.encode(x)  
        return(mu, logvar)
    


#Define the decoder. This is a network that takes as input the a vector of the size of the 
# latent dim and upscales to a vector of size of the inpt dim that represents the mean vector of hte output space. 
# I assume that the decoder(lekelyood follows a gaussian distribution)
class Decoder(nn.Module):
    def __init__(self, vae_decoder_parameter_dic):

        super(Decoder, self).__init__()
        
        self.vae_decoder_parameter_dic = vae_decoder_parameter_dic
        
		# Decoder layers
        self.decoder_layers = nn.ModuleDict({})
        for i, (layer_name, parameters) in enumerate(self.vae_decoder_parameter_dic.items()):
            if 'dense' in layer_name:
                dimension, size = parameters
                final_output_dim = dimension
                self.decoder_layers[f'dense_{i}'] = dense_block(input_dim=dimension, blocksize=size)
            elif 'transition' in layer_name:
                input_dim, output_dim = parameters
                self.decoder_layers[f'transition_{i}'] = nn.Linear(input_dim, output_dim)
                final_output_dim = output_dim
        # map to likelihood parameters
        # I assume that the likelihood (decoder) follows anormal or a bernoulli distribution
        # The decoder outputs the parameters for the chosen distribution
        # For gaussian the decoder outputs the mean vector or/and the log variance
        # For bernoulli it ouputs th e probbability of being 1
       

    def decode(self, z):
        # This function takes the latent representation of the input and gets it through the decoder to output the mean vector of the p(x|z) distribution.
        # I can consider the mean vector of that distribution as the most probable reconstruction.
        for layer in self.decoder_layers.values():
            z = F.relu(layer(z))
		
        return(z)
    
    def forward(self, z):
        likelyhood = self.decode(z)
        return(likelyhood)
        

class VAE(nn.Module):
    def __init__(
        self, latent_dim, vae_encoder_parameter_dic, vae_decoder_parameter_dic, n_modalities = 1,
        likelihood="gaussian" #The distribution of the likelihood(decoder)
    ):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim # dimension in latent space
        self.likelihood = likelihood
        self.n_modalities = n_modalities
        self.vae_encoder_parameter_dic = vae_encoder_parameter_dic
        self.vae_decoder_parameter_dic = vae_decoder_parameter_dic

		# Define a separate encoder and a corresponding decoder for each modality.
        self.encoders_dic = nn.ModuleDict({f'encoder{i}': Encoder(latent_dim = self.latent_dim, vae_encoder_parameter_dic = self.vae_encoder_parameter_dic) for i in range(self.n_modalities)}).to(device)
        self.decoders_dic = nn.ModuleDict({f'decoder{i}': Decoder(vae_decoder_parameter_dic = self.vae_decoder_parameter_dic) for i in range(self.n_modalities)}).to(device)
              
    # Given the parameters of a Gausiian distribution sample a datapoint from th latent space
    def sample(self, mu, logvar):
        """Sample from Gaussian posterior of z with reparametrization"""
        # At this step I mimic sampling from q(z|x)
        # Because in the reparametrization trick i need the standard deviation and the encoder outputs logvar
        # To get the variance i should convert to exp(logvar) # exp(logvar) = var
		# from var to get the standard deviation in need to get the square root
		# So finally i have sd = sqrt(var) = sqrt(exp(logvar)) <=> sd = exp(logvar/2) 
        std = torch.exp(0.5 * logvar) 
        eps = torch.randn_like(std) #This samples from a N(0,I) with the same shape as std
        return mu + eps * std #This is a random sample from the latnt distribution.
    
    
	#Each one of the encoders produces a mean and a variance. I need a way to combine all of them 
	# into a unifiead distribution with a specific single mean and a single variance.
    # Instead of averaging the gaussians I can multiply them.
    # A good point about the power of experts is that it can handle missing modalities.
    # The inout of the function is a set of gaussians ans the result is a single gaussian.
    
    def poe_diagonal_gaussians(self, mus, logvars):
        # Compute precisions: λ = 1 / σ^2 = exp(-logvar)
        precisions = [torch.exp(-lv) for lv in logvars]
        
        # Sum of precisions
        lambda_sum = torch.stack(precisions, dim=0).sum(dim=0)

        # Precision-weighted sum of means
        eta_sum = torch.stack([p * m for p, m in zip(precisions, mus)], dim=0).sum(dim=0)

        # Final parameters
        mu_poe = eta_sum / lambda_sum
        logvar_poe = -torch.log(lambda_sum)

        return mu_poe, logvar_poe

    def forward(self, x):
        mus, logvars = zip(*[self.encoders_dic[f'encoder{i}'](input) for i,input in enumerate(x)])

        jmu, jlogvar = self.poe_diagonal_gaussians(mus=mus, logvars=logvars)
        z = self.sample(jmu, jlogvar) # sample with reparametrization
 
        recons = [self.decoders_dic[f'decoder{i}'](z) for i in range(self.n_modalities)]
        # print(recons[0].shape)
        return(recons, jmu, jlogvar)


def loss_function(recons, x, jmu, jlogvar, beta=1.0):

    recon_losses = [F.mse_loss(recon, input, reduction="sum") for recon, input in zip(recons, x) ] #The reduction="sum" argument tells torch.nn.functional.mse_loss how to aggregate the individual squared errors.


    KLD = -0.5 * torch.sum(1 + jlogvar - jmu.pow(2) - jlogvar.exp())	

    return sum(recon_losses) + beta * KLD



### Assemble the train function that will be used to train the model
def train(epoch, model, train_loader, test_loader, optimizer, device, test_grads = False):
    """Trains the VAE for one epoch."""
    model.train()
    train_loss = 0
    likelihood = model.likelihood

    for batch_idx, data in enumerate(train_loader):
        
        optimizer.zero_grad()
        # For the train batch
        recons, jmu, jlogvar = model(data)
        loss = loss_function(recons = recons, x = data, jmu = jmu, jlogvar=jlogvar, beta=0.5)
        train_loss += loss.item()
        

        
		#Backpropagate
        loss.backward()        
        optimizer.step()

        if test_grads:
            # run forward + loss + backward first
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"{name} has no grad!")
                else:
                    print(f"{name} grad mean: {param.grad.abs().mean().item():.6f}")
    total_batches = len(train_loader)
    mean_loss = train_loss / total_batches

    
	# For the test set
    test_recons, test_jmu, test_jlogvar = model(next(iter(test_loader))) #I use next(iter()) because there is only one batch
    test_loss = loss_function(recons = test_recons, x = next(iter(test_loader)), jmu = test_jmu, jlogvar= test_jlogvar)
    test_loss = test_loss.item()

    return(mean_loss, test_loss, recons, jmu, jlogvar)
    

# For the cluster of interest of hte modality of interest find the cluster of the distinct modalities that have the maximum cell overlap
def compute_maximum_cluster_overlap(cluster_of_interest, mod_of_interest, rna_clusters, atac_clusters, joint_labels):
        from collections import Counter
        order_dic = {'joint' : 0,
                'rna' : 1,
                'atac' : 2} #This is the order that the cluster labels will be joined into triplets.
        
        pair_list = list(zip(joint_labels, rna_clusters.codes, atac_clusters.codes)) #Merge the labels of the cells into triplets

        #Each triplet represents a cell that is once clustered based on its joint embedding, once on its rna embedding and once on its atac embedding.
        triplet_counts = dict(Counter(pair_list)) # For each triplet compute the number of times that appears in the list
        list_of_triplets = [] #Initialize an empty list that will later on store the triplets of the desired cluster
        
        for triplet in triplet_counts.keys():
            
            if triplet[order_dic[mod_of_interest]] == cluster_of_interest:
                list_of_triplets.append([triplet, triplet_counts[triplet]]) #Keep only the triplets of the desired cluster

        max_overlap = max(list_of_triplets, key = lambda x: x[1]) #Get the clustering combination that has the most cells 
        total_cells_on_joint = list(joint_labels).count(max_overlap[0][0]) #For the triplet keep find the number of cells clustered at each modality
        total_cells_on_rna = list(rna_clusters.codes).count(max_overlap[0][1])
        total_cells_on_atac = list(atac_clusters.codes).count(max_overlap[0][2])
        return(max_overlap,{'total_cells_on_joint' : total_cells_on_joint,
            'total_cells_on_rna' : total_cells_on_rna,
            'total_cells_on_atac' : total_cells_on_atac,
            'max_overlap' : max_overlap[-1]})

# Build a list of doublets where for each doublet the first entry is the cell barcode and the second enty is the label
def get_cluster_cell_sets(joint_cluster_num, rna_cluster_num, atac_cluster_num):
    ### NOTE ###
    # mdata, rna_clusters, atac_clusters, joint labels should be defined in the main script
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


#Plot the overlaps in the form of venn diagrams
def plot_cluster_overlap(joint_cluster_num, rna_cluster_num, atac_cluster_num, ax, title):
    
    #Build lists of doublets for all modalities
    joint_pairs, atac_pairs, rna_pairs = get_cluster_cell_sets(joint_cluster_num, rna_cluster_num, atac_cluster_num)
    


    #Merge to one list of doublets where each doublet contains the cell barcode at position 0 and the modality label at position 1
    #MERGE
    merged_pairs = joint_pairs + atac_pairs + rna_pairs


    #PLOT
    from collections import defaultdict

    modality_cells = defaultdict(set) #Initialize a dictionary sets

    for cell, modality in merged_pairs:
        modality_cells[modality].add(cell) #Build a dictionary where the keys are the modalities and the values are lists of cells


    # Recompute the lists of cells for each modality.
    rna_cells = modality_cells["RNA"]
    atac_cells = modality_cells["ATAC"]
    joint_cells = modality_cells["JOINT"]


    print(rna_cells, atac_cells, joint_cells)
    from matplotlib_venn import venn3

    #Plot the overlaps
    venn3(
        [set(joint_cells), set(rna_cells), set(atac_cells)],
        set_labels=("Joint", "RNA", "ATAC"),
        ax=ax,
    )
    ax.set_title(title, fontweight="bold")
