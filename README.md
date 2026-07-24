### **<ins> Variational Inerence, VAE and Multimodal single cell data integration**

Sequencing technologies have shed light into biological sytems like any other technology has before. Current technologies allow profiling of cells transcriptome, epigenome, proteome, chromatin occupancy, regulatory elements and others. At the edge of the field is the profiling of single cells in a multilevel perspective. For example there are technologies that capture both the chromatin accessibility and the transcriptome of a single cell. Although these technologies provide the potential for an even better undestanding of the subject biological system there is also the need for appropriate computational tools that are able to pull out this potential  the technologies provide.

Although not complete this project represents an efort to integrate data from multiple modalities into a single latent representation that captures the information of all different modalities all at once, thus allowing a better more complete representation of cells.
For this purpose a **<ins> custom deep generative model** is built using python's pytorch library. The model architecture comprises a **<ins>Variational Auto Encoder(VAE)** with residual connections and a modular user defined architecture.

### **<ins> Variational Inerence and VAE**

Variational Inference is a probabilistic approach of approximating distributions that cannot be computed.
First we assume that the observed data $x$ are somehow related to a latent variable $z$. So we define a joint distribution of p(x,z) which essentially means that some samples of $x$ co exist with some samples of $z$

$$p(x,z) = p(x|z)p(z)$$ 
where, 
$p(z)$ is the prior distribution of the latent varible $z$. This is a user defined/assumed distribution.
$p(x|z)$ is the distribution of $x$ given a sampled $z$. Essentially this is a mathematical formula that given the latent representation of a sample is a able to generate the sample. In VAE this model is a neural network that takes as input the parameters of a distibution and roduces a sample.

*<ins>The problem:*
From the Bayes theorem if we want to compute $p(x|z)$ we need to solve the following:

$$p(z|x) = \frac{p(x|z)p(z)}{p(x)}$$

Because I do not know/observe $z$ in order ot compuute $p(x)$ I need to marginalize the joint distribution

$$
p(x)=\int p(x,z)\,dz
$$
or using the law of total probability
$$
p(x)=\int p(x|z)p(z)\,dz
$$

The problem here is that we have to marginalize over all possible $z$. Most of the times $z$ is multidimensional so integrating over all possible dimensions can't be done due to the huge number of combinations. This do not means that the integral can not be solved. Theoretically it can be solved. However it is too expensive computationally which makes either not practical or even impossible.

The solution is that instead of computing the posterior $p(z|x)$ i can aproximate by intoducing a new distribution $q(z|x)$. In VAE $(z|x)$ is enother neural network the encoder. The encoder takes as inputs the observed data and produces the parameters of $q(z|x)$.

Overall there is an Encoder and a decoder. The encoder produces a distribution and the decoder produces samples by sampling from $q(z|x)$ that the encoder produced. 

To find the optimal parameters of $q(z|x)$ we define the loss function of the full VAE as the:

$$
\mathrm{ELBO}
=
\mathbb{E}_{q_\phi(z|x)}
\left[
\log p_\theta(x|z)
\right]
-
D_{\mathrm{KL}}
\left(
q_\phi(z|x)\,\|\,p(z)
\right)
$$
which essentially is the reconstruction error plus the KL divergence of the $q(z|x)$ from the prior $p(z)$.
    
### **<ins> Multimodal single cell data integration**

Combining everything together this project considers cell modalities as different data distributions. Building on the rational that these modalities might arise from different unobserved cell states (different cell types, different expressoin profiles, different developmenta stages) it aims to represent this states as a latent distribution given the the observed set of different modalities $q(hidden\ cell\ state| observed\ modalities)$. For this purpose a VAE model was implemented in such a way were the user is able to :
* Use different training parameters.
* Easily alter the models architecture depending on the needs of the data.
* Input several different modalities.
