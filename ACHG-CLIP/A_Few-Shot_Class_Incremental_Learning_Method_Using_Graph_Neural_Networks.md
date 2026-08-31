## A Few-Shot Class Incremental Learning Method Using Graph Neural Networks

Yuqian Ma, Youfa Liu , and Bo Du , Senior Member, IEEE [URL 🔗](https://orcid.org/0000-0002-3540-5775)

Abstract—Few-shot class incremental learning (FSCIL) aims to continuously learn new classes from limited training samples while retaining previously acquired knowledge. Existing approaches are not fully capable of balancing stability and plasticity in dynamic scenarios. To overcome this limitation, we introduce a novel FSCIL framework that leverages graph neural networks (GNNs) to model interdependencies between different categories and enhance cross-modal alignment. Our framework incorporates three key components: 1) a Graph Isomorphism Network (GIN) to propagate contextual relationships among prompts; 2) a Hamiltonian Graph Network with Energy Conservation (HGN-EC) to stabilize training dynamics via energy conservation constraints; and 3) an Adversarially Constrained Graph Autoencoder (ACGA) to enforce latent space consistency. By integrating these components with a parameter-efficient CLIP backbone, our method dynamically adapts graph structures to model semantic correlations between textual and visual modalities. Additionally, contrastive learning with energy-based regularization is employed to mitigate catastrophic forgetting and improve generalization. Comprehensive experiments on benchmark datasets validate the framework’s incremental accuracy and stability compared to state-of-the-art baselines. This work advances FSCIL by unifying graph-based relational reasoning with physics-inspired optimization, offering a scalable and interpretable framework. Code is available at: https://github.com/aries-yqian/ACHG-CLIP

Index Terms—Few-shot class incremental learning, CLIP, graph neural networks, Hamiltonian graph network, graph autoencoder.

W ITH the wide application of machine learning tech- nology in various fields, the ability of models to continuously learn new information in dynamic environments has become a key focus of research [1], [2]. Few-Shot Class Incremental Learning (FSCIL), as an important research direc- tion in this field, endeavors to enable models to learn new categories with limited training samples and avoid forgetting [URL 🔗](#page-0)

Received 25 July 2025; revised 27 November 2025; accepted 11 January 2026. Date of publication 28 January 2026; date of current version 3 February 2026. This work was supported in part by the National Key Research and Development Program of China under Grant 2023YFC2705702, in part by the National Natural Science Foundation of China under Grant 62576257 and Grant 62225113, in part by the National Key Research and Development Pro- gram of China under Grant 2024YFF1207300, in part by Yunnan Provincial Key Science and Technology Project under Grant 202502AS080002, and in part by the Wuhan University (WHU)-Kingsoft Joint Laboratory. The associate editor coordinating the review of this article and approving it for publication was Prof. Guosen Xie. (Corresponding authors: Youfa Liu; Bo Du.)

The authors are with the School of Computer Science, National Engi- neering Research Center for Multimedia Software, Institute of Artificial Intelligence, Hubei Key Laboratory of Multimedia and Network Commu- nication Engineering, Wuhan University, Wuhan 430072, China (e-mail: yqian402@whu.edu.cn; liuyfa1991@whu.edu.cn; dubo@whu.edu.cn).

Digital Object Identifier 10.1109/TIP.2026.3657170

1941-0042 © 2026 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and

similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.

See https://www.ieee.org/publications/rights/index.html for more information.

## I. INTRODUCTION

previously learned categories, that is, to solve the stability- plasticity dilemma [3], [4], [5], [6]. In practical application scenarios, in the field of autonomous driving, new road sce- narios, traffic signs, or vehicle types require the model to learn quickly [7], [8], [9]. However, it is difficult to obtain a large amount of labeled data. In the area of financial fraud detection, new fraud patterns emerge continuously, and labeling these [URL 🔗](#page-0)

new cases is costly and time-consuming

[[10].](#page-0)

At the same

time, old data needs to be retained to identify common fraud behaviors. In these situations, FSCIL technology can effectively improve the system performance.

From the perspective of the research field, FSCIL belongs to the category of continual learning. It requires models to con- tinuously update knowledge while maintaining memory of his- torical knowledge during the process of continuously receiving new data [11]. In the research of FSCIL, how to achieve effi- cient class incremental learning under the condition of scarce samples is the core issue that researchers focus on. Prototype learning [12], [13] and attention mechanisms [14] are explored to address FSCIL challenges. Prototype learning represents classes with prototypes from few sample feature vectors for classification [15], but precise prototype definition is hard in complex FSCIL data, making similar class discrimination dif- ficult [16], [17]. Attention mechanisms focus on relevant input data, like highlighting key image regions in classification, yet may not fully solve catastrophic forgetting as new data can dis- rupt the attention distribution and cause forgetting of old class [URL 🔗](#page-0)

features

[18]. Additionally, Graph Neural Networks (GNNs), [URL 🔗](#page-0)

as a powerful tool, can effectively handle data with complex relationships and have achieved remarkable results in multiple fields in recent years [19], [20], [21], [22], [23], [24], [25]. Introducing GNNs into FSCIL provides a new perspective and method for solving the problems of class relationship modeling and knowledge transfer in few-shot learning, and has become a highly promising research direction [26]. [URL 🔗](#page-0)

Regarding the application of GNNs in FSCIL, although there have been some attempts, it is still in its infancy. Some studies [27], [28] use GNNs to model the relational structure of data. However, these methods often do not fully consider the data scarcity and the dynamic characteristics of class increment in few-shot learning. For example, some GNN- based methods cannot effectively distinguish between similar categories when dealing with new categories, resulting in a decrease in classification accuracy [29]. During the knowledge transfer process, due to the failure to rationally utilize graph structure information, the learning efficiency of the model for new categories is low. [URL 🔗](#page-0)


Based on the above research status, this study aims to build an innovative few-shot class incremental learning structure based on graph neural networks to overcome the shortcom- ings of existing methods. Specifically, we are committed to designing a model that can effectively utilize graph structure information, enhance the model’s learning ability for new categories in the case of few-shot samples, and improve the memory retention ability for previously learned categories, thereby enhancing the overall performance of the model in FSCIL tasks. The realization of this goal is of great sig- nificance for promoting the application of machine learning in practical scenarios and can provide effective solutions for model training and optimization in resource-constrained environments.

Specifically, to effectively address the above issues and achieve more efficient learning, this study proposes a mul- timodal framework named Adversarially Constrained Hamil- tonian Graph-CLIP (ACHG-CLIP). The framework is mainly based on two key innovations. It updates the graph embeddings of learnable prompts [30], [31] to help the model learn features more effectively. The introduction of learnable prompts plays a certain role in FSCIL. It can not only enhance the model’s ability to capture the features of new categories in the case of few-shot samples, alleviating the learning difficulties caused by insufficient samples, but also retain the knowledge of old cat- egories during the learning of new categories, reducing catas- trophic forgetting, and thus improving the overall performance of the model at different learning stages. Specifically, the first innovation of the ACHG-CLIP framework is the adoption of the Adversarially Constrained Graph Autoencoder (ACGA). In the ACGA module, the Graph Isomorphism Network (GIN) layer [32] is integrated into the adversarial training paradigm. The graph autoencoder is responsible for reconstructing node features, and the discriminator is used to enhance the con- sistency of the latent space. During this process, the graph embeddings of learnable prompts are updated, enabling the model to better capture and learn data features, effectively avoiding overfitting on few-shot data and better preserving the topological relationships between old and new categories. The second innovation is the introduction of the Hamiltonian Graph Network with Energy Conservation (HGN-EC). Inspired by Hamiltonian dynamics [33], HGN-EC models the feature evo- lution through symplectic dynamics. During the model learn- ing process, by maintaining “energy” conservation, it ensures a smooth transition in the incremental learning stage. During this process, the graph embeddings of learnable prompts are also updated to further optimize the model’s feature-learning abil- ity, thereby reducing the occurrence of catastrophic forgetting. In terms of specific implementation, this study is based on the CLIP (Contrastive Language-Image Pre-Training) model [34]. It freezes the backbone network of CLIP and only trains a small number of GIN-based modules. This approach enables the ACHG-CLIP framework to fully exert its advantages in few-shot class incremental learning tasks, efficiently improving the overall performance of the model. The contributions of this paper can be summarized as follows: [URL 🔗](#page-0)

- By combining adversarial regularization and Hamiltonian dynamics and applying them to the FSCIL framework

based on the Graph Isomorphism Network (GIN), a new

- framework is provided for few-shot class incremental learning. This integration helps the model better balance stability and plasticity in complex few-shot learning envi- ronments.

- Through experiments on benchmark datasets such as CIFAR-100, miniImageNet, and CUB200, the effective- ness of our method is verified. Compared with existing methods, the performance has been improved.

The subsequent content of this paper is organized as fol- lows. In section II, we will review in detail the research progress in few-shot learning, incremental learning, graph neural networks, and parameter-efficient learning related to this study. In section III, we will precisely elaborate on the problem definition and related requirements of few-shot class incremental learning. In section IV, we will deeply introduce the specific architecture, module design, and training process of the ACHG-CLIP framework. In section V, we will show the experimental results on multiple benchmark datasets, conduct comparative analysis with existing methods, and perform sensitivity and ablation studies on the model. Finally, in section VI, we will summarize the research results, discuss the limitations of the research, and look ahead to future research directions. [URL 🔗](#page-0)

## II. RELATED WORKS

## A. Few-Shot Class Incremental Learning

In Few-Shot Class Incremental Learning (FSCIL), existing research has primarily encountered challenges in balancing catastrophic forgetting and overfitting when handling sparse samples and new classes [35], [36]. Although some researchers have attempted to address this issue through meta-learning approaches, such as in pedestrian attribute recognition, these methods still exhibit limitations in open-ended FSCIL scenar- ios [37]. While other studies have proposed few-shot detection models, yet these approaches are unable to incrementally learn new targets in open environments [38]. [URL 🔗](#page-0)

Parallelly, in the domain of FSCIL with Graph Neural Net- works (GNNs), the existing methods face multiple limitations as well. As elaborated in [39], many approaches grapple with the dual challenges of data scarcity and the dynamic nature of classes in FSCIL scenarios. For instance, when dealing with datasets where new classes arrive incrementally with only a few samples, these methods often fail to adequately represent the new class information. In terms of leveraging graph structure information, traditional GNN-based FSCIL methods typically use static graph construction techniques. They do not adapt well to the changing relationships between classes over time [40]. As a result, when faced with classes that are visually or semantically similar, the model’s performance degrades sig- nificantly. Also, capturing the evolving relationships between old and new classes, which is crucial for continuous learning [41], remains a weak point in these methods. [URL 🔗](#page-0)

To address the above issues, our study introduces ACHG- CLIP, a novel framework integrating adversarial regularization and Hamiltonian dynamics. By updating learnable prompts via graph embeddings, it captures new class features from few samples, reducing overfitting and catastrophic forgetting.

This method is applicable in dynamic scenarios. Our work

Authorized licensed use limited to: Indian Institute of Information technology Sricity. Downloaded on June 23,2026 at 07:06:10 UTC from IEEE Xplore. Restrictions apply.


contributes to FSCIL by offering a more robust solution for real-world applications with data scarcity and class dynamism.

## B. Hamiltonian Graph Network

Existing Hamiltonian-based graph networks, including Hamiltonian Graph Neural Networks (HGNNs) [42] and Symplectic Graph Neural Networks (SGNNs) [43], have demonstrated promising capabilities in modeling dynamic systems and capturing intricate relational patterns within graph-structured data. However, their application to Few-Shot Continual Incremental Learning (FSCIL) encounters signifi- cant challenges stemming from the non-stationary nature of data distributions and the evolving dependencies between fea- tures in incremental learning scenarios. Traditional HGNNs, which leverage message-passing mechanisms inspired by Hamiltonian mechanics, struggle to adapt to the dynamic shifts inherent in FSCIL, where data distributions evolve over time and new classes are introduced with extremely limited labeled training samples [44]. Similarly, while SGNNs pre- serve symplectic structures to enhance stability, they often lack the necessary flexibility to accommodate the complex feature evolutions and non-stationary distributions that characterize FSCIL tasks. To address these limitations, our work draws inspiration from the HamiltoniAN Graph diffusion (HANG) model [33] to propose the “Hamiltonian Graph Network with Energy Conservation (HGN-EC)”, an architecture specifically tailored for FSCIL scenarios. HGN-EC incorporates evolu- tion mechanisms based on symplectic dynamics derived from Hamiltonian mechanics, ensuring smooth and stable transitions between successive incremental learning phases. By explicitly enforcing energy conservation across learning sessions, HGN- EC effectively mitigates the issue of catastrophic forgetting, enabling the model to adapt to newly introduced classes while retaining proficiency in previously learned concepts, thus establishing itself as a robust solution for dynamic learning environments. [URL 🔗](#page-0)

## C. Regularized Adversarial Training

Existing regularized adversarial training approaches within the Few-Shot Class-Incremental Learning (FSCIL) paradigm suffer from notable limitations that hinder their practical effi- cacy. For example, the regularization strategy proposed in [45] struggles to strike a balanced equilibrium between mitigating overfitting risks and enabling sufficient adaptation to newly introduced classes. In practical scenarios, such as biological classification tasks involving novel species datasets, this imbal- ance leads to excessive constraints on the model’s adaptive capacity, impairing its ability to capture distinctive traits of emerging categories [46]. Similarly, the method proposed in [47], imposes overly rigid restrictions on model flexibility during training with datasets featuring emerging art styles, resulting in suboptimal performance when processing diverse and novel artistic data—this rigidity prevents the model from effectively encoding the subtle, nuanced characteristics that define these evolving styles. In contrast, drawing inspiration from the insights in [48], our ACHG-CLIP framework incor- porates the Adversarially Constrained Graph Autoencoder (ACGA) to innovatively integrate regularization mechanisms [URL 🔗](#page-0)

Authorized licensed use limited to: Indian Institute of Information technology Sricity. Downloaded on June 23,2026 at 07:06:10 UTC from IEEE Xplore. Restrictions apply.

with adversarial training paradigms, thereby enhancing the model’s capability to capture intricate data features while maintaining a robust balance between stability and adaptability in incremental learning scenarios.

## III. PRELIMINARIES

## A. FSCIL

Few-Shot Class Incremental Learning (FSCIL) presents a challenging scenario where a model must sequentially learn new classes from limited training samples while maintaining performance on previously learned classes. This problem is formalized as follows: the model is first trained on an initial dataset D0 = {(xi, yi)}N0 i=1, where N0 is large and yi ∈ C0 (base classes). This foundational training establishes the model’s ability to recognize the initial set of classes. Subsequently, at each incremental step t ≥ 1, the model receives a small dataset, the model receives a small dataset Dt = {(xi, yi)}Nt i=1, where Nt  N0 (e.g., 5 samples per class) and new classes Ct. The new classes Ct are disjoint from all previously learned classes Cτ for τ < t, ensuring that the model must learn entirely new categories without revisiting old data. Additionally, the model operates under constraints: it must infer class labels without knowing the session of origin (class-incremental setting) and cannot store or revisit previous data D0, . . . , Dt−1, enforcing strict memory efficiency. This formulation highlights the need for models that balance stability (retaining knowledge from previous tasks) and plasticity (adapting to new classes with limited samples), making FSCIL a critical area of research for real-world applications such as autonomous systems, medical diagnosis, and personalized learning.

## B. GNNs

Graph Neural Networks (GNNs), as important tools for processing graph-structured data, their core lies in accu- rately capturing complex graph structures and the associative relationships between nodes by learning the embeddings of nodes, edges, and graphs [49]. The following is a rigorous problem formulation for graph neural networks: Given a graph G = {V, E, X}, where V represents the set of nodes, E is the set of edges, and the node-feature matrix X ∈ Rn×d (n is the number of nodes, and d is the feature dimension). The goal is to construct a function fθ that maps nodes or graphs to an embedding space Z ∈ Rn×k (k is the embedding dimension), and in this process, the topological structure and semantic information of the graph need to be completely preserved. In the computational mechanism of graph neural networks, the adjacency matrix of edges A ∈ {0, 1}n×n plays a crucial role. Usually, it needs to be mapped to the embedding space to capture the connection relationships between nodes. Take Graph Convolutional Networks (GCN) as an example [50]. It uses the normalized adjacency matrix A˜ = D− 1 2 AD− 1 2 for mapping, where D is the degree matrix. This mapping method can effectively maintain the integrity of the graph’s topological structure when aggregating neighbor-node information. [URL 🔗](#page-0)

Advanced GNN models such as Graph Attention Networks [51] (GAT) dynamically adjust the adjacency matrix by learn- ing attention weights, thus more flexibly capturing the complex dependency relationships between nodes. The dynamically [URL 🔗](#page-0)


*Fig. 1. Framework of ACHG-CLIP. On the left, text and vision encoders process textual and image inputs separately. The middle section uses GIN to handle graph-structured learnable prompts for both text and vision. On the right, encoder-decoder structure of ACGA generates optimized representations. Finally, feature compression, Hamiltonian module, and restoration module produce updated learnable prompts, which are fed back into the text and vision encoders for subsequent learning. The overall process emphasizes the fusion and updating of multi-modal information.*

adjusted adjacency matrix is denoted as A ∈ Rn×n, and the matrix element A i j reflects the attention weight between node i and node j.

Graph Isomorphism Network (GIN) is a unique graph neural network architecture with several key advantages [52]. GIN updates node features by aggregating features from neighboring nodes, enabling each node to effectively integrate information from its neighbors and enhancing the model’s expressive power for graph-structured data [32]. GIN intro- [URL 🔗](#page-0)

duces a learnable parameter



to control the fusion ratio of a

node’s own features and its neighbors’ features, which allows the model to automatically adjust the weights of these features for optimal representation [53]. GIN is robust to structural changes in graphs, making it effective in handling noise and uncertainties in graph data [54]. Its design enables it to capture complex patterns and dependencies in graph-structured data, leading to strong generalization performance across various graph-related tasks. [URL 🔗](#page-0)

## IV. METHOD

Our proposed framework, as shown in Fig. 1, integrates Graph Neural Networks (GNNs) with a CLIP backbone to tackle Few-Shot Class Incremental Learning (FSCIL). By constructing a dynamic graph where learnable prompts act as nodes, the framework models semantic correlations between textual and visual modalities. The adjacency matrix is built using cosine similarity with a sparsity-inducing threshold. Three key components enhance the model: a Graph Iso- morphism Network (GIN) propagates information across the graph, enriching prompt representations through neighbor aggregation; an Adversarially Constrained Graph Autoencoder (ACGA) ensures latent space consistency through recon- struction; and a Hamiltonian Graph Network with Energy Conservation (HGN-EC) stabilizes training by maintaining consistent energy levels. Together, these components enable the model to update prompt embeddings while preserving semantic relationships, ensuring stability and effective gener- alization to new classes with limited samples. [URL 🔗](#page-0)

## A. CLIP Encoder

Contrastive Language-Image Pre-Training (CLIP) jointly trains text and vision encoders to build a cross-modal semantic

alignment embedding space. Its core architecture includes symmetric text and vision encoders, both based on improved Transformer structures, mathematically formulated as follows.

1) Text Encoder: It takes tokenized sentences tokens = [t1, t2, . . . , tn] as input and generates semantic embeddings through three stages:

- 1) Embedding Representation: Discrete tokens are first mapped to continuous vectors

X = ET(tokens) ∈ Rn×d

where ET is word embedding matrix and n denotes the sequence length of tokenized text inputs in the CLIP text encoder. Positional encoding is then added

X ← X+ PT(n),

PT ∈ Rn×d

In the formula, PT refers to the positional encoding matrix.

- 2) Feature Interaction: Context modeling is achieved through L layers of Transformer blocks, with each layer defined as

The multi-head attention mechanism MHA(Q, K, V) = Concat(head1, . . . , headh)WOutput, with a single head cal- culated as

where W is learnable weight matrix, dk denotes the dimension of the query/key vectors in the multi-head attention mechanism of the CLIP text encoder, used to scale the dot-product attention scores for stable training.

- 3) Semantic Projection: The global average vector of the last layer’s output is taken, and a linear transformation is applied to obtain the normalized text embedding

WT · AvgPool(X(L))2

where de represents the embedding dimension of the final text and visual features.


2) Vision Encoder: It takes image patch sequences patches = [p1, p2, . . . , pm] as input. Its processing workflow is symmetric to the text encoder but has two key differences:

- 1) Structured Input: Image patches are linearly projected and concatenated with a [CLS] token

where EV ∈ Rd×|p| is the vision embedding matrix, m refers to the number of image patches in the CLIP vision encoder, and PV ∈ R(m+1)×d is the positional encoding.

- 2) Feature Extraction: After L layers of identical Trans- former blocks, the feature corresponding to the [CLS] token is extracted

- 3) Learnable Prompts: To enhance the model’s expressive- ness and adaptability, we introduces learnable prompts that are processed and updated in each Transformer layer. The prompt generation process is as follows:

- 1) Text Prompt Generation: Text prompts g are generated through a learnable parameter matrix G ∈ RL×M×d, Prompts are inserted into the text embeddings at each layer where learnable prompts, and L is the number of layers, d is the embedding dimension. M is the number of

- 2) Vision Prompt Generation: Vision prompts gV are gener- ated through a learnable parameter matrix GV ∈ RL×M×d and inserted into the vision embeddings at each layer

Prompts directly replace the input of each layer.

4) Cross-Modal Alignment: Achieved through a contrastive loss. Given N image-text pairs {(h(i) V , h(i) T )}N i=1 in a batch, the similarity matrix Sim ∈ RN×N is defined as

where τ is a learnable temperature coefficient. The classifica- tion loss function is cross-entropy CE(S , y) [55]measures the difference between the predicted similarity matrix S and the true labels y [URL 🔗](#page-0)

log

PN

where

yi

is the index of the correct text corresponding to the

i-th image.

## B. GIN

Textual and visual learnable prompts are processed through the Graph Isomorphism Network (GIN) module to construct node feature matrices X ∈ RN×D and adjacency matrices A ∈ RN×N, respectively, which are passed into the following ARGA module.

GIN is a neural network architecture designed for graph- structured data, updating node features by aggregating features

from neighboring nodes. The mathematical formulation for a GIN layer is expressed as

@(1 + (k))h(k−1) v + u∈N(v) h(k−1) u

A

Here, h(k) v denotes the feature of node v at the k-th layer, N(v) is the set of neighboring nodes for v, (k) is a learnable coefficient enabling GIN to distinguish non-isomorphic graphs by flexibly controlling the fusion ratio of a node’s own features and its neighbors’ features, and MLP(k) represents a multi- layer perceptron that not only introduces non-linearity to the aggregated features, but also supports adaptive relational modeling via shared parameter design, ensuring both strong expressiveness and parameter efficiency, which is critical for scenarios like FSCIL with limited samples.

1) Adjacency Matrix Construction: The adjacency matrix, fundamental for representing node connections in graph data, is constructed through several steps. Initially, a cosine simi- larity matrix is computed using

This matrix is then binarized via thresholding

To ensure symmetry, the adjacency matrix is symmetrized using

followed by normalization

where D is the node degree matrix. Optionally, an attention mechanism

can be incorporated to enhance the model’s expressiveness, resulting in a final adjacency matrix of

Specifically, in each incremental stage of FSCIL, features of new nodes are added to the graph, while features of old nodes are kept unchanged to retain knowledge from previous tasks.

2) GIN Layer Feature Update:

In the GIN layer, node

features are updated through the aggregation of neighboring features

u∈N(v) X

combined with the learnable coefficient  to form

and subsequently transformed by the multi-layer perceptron

This process equips GIN with the capability to effectively capture complex patterns and dependencies in graph-structured data, leading to strong performance across various graph- related tasks.


## C. ACGA

Adversarially Constrained Graph Autoencode (ACGA), takes as input the sample features and associated topological structure output by the previous module, specifically, a node feature matrix X ∈ RN×D, and an adjacency matrix A ∈ {0, 1}N×N that describes the associative relationships among samples.

On this basis, as a graph embedding framework that integrates graph autoencoders with adversarial training mech- anisms, ACGA aims to effectively address the problem of latent representation degradation in traditional graph embed- ding methods by jointly optimizing topological reconstruction and latent distribution alignment. The core architecture of this framework consists of three components: graph convolutional encoder, structural reconstruction decoder, and adversarial regularization module.

1) Graph Convolutional Encoder: The encoder employs an improved Graph Isomorphism Network (GIN), which fuses node features and topological information through a multi- layer message-passing mechanism. The node representation at layer l is updated according to the following formula

where the specific formula for GINLayer is shown in Eq.(13), (l) ∈ R is a learnable coefficient, and MLP is composed of linear layers, batch normalization, and an activation function (GELU). Finally, the encoded output is Z = GIN(X, A) ∈ RN×K, where N denotes the number of nodes in the graph and K represents the dimension of the latent space, controlling the complexity and capacity of the learned representations. N and K are independent, with no functional relationship between them. [URL 🔗](#page-0)

2) Structural Reconstruction Decoder: The decoder recon- structs the adjacency matrix through the inner product of latent representations and models the probability of edge existence

1

using the Sigmoid function σ(x) =

Aˆ

The reconstruction loss takes the form of negative log- likelihood

X  Ai j log ˆAi j + (1 − Ai j) log(1 −

where E− represents the set of negatively sampled edges.

3) Adversarial Regularization Module: A discriminator D : RK → [0, 1] is introduced to force the latent representation Z to match the prior distribution pz = N(0, I). The discriminator is composed of two fully-connected layers, and its specific form is

The adversarial loss takes the form of Wasserstein distance to enhance the stability of the training process

whereq(Z|X, A) is the empirical distribution of the encoder output.

1+e−x

= σ

ZZT

D. HGN-EC

HGN-EC receives the node feature matrix X ∈ RN×D and the adjacency matrix A ∈ RN×N from the ARGA module, where N is the number of nodes and D is the feature dimension.

HGN-EC is a graph neural network architecture based on Hamiltonian mechanics, designed to model the dynamic evolu- tion of graph data through the energy-conservation properties of physical systems. Beyond stabilizing training dynamics via smooth incremental transitions, the energy conservation constraint in HGN-EC aligns with the core demands of FSCIL by preserving the “energy state” of learned features, analogous to maintaining the integrity of physical systems’ energy, thereby preventing the degradation of base classes feature representations that often causes catastrophic forgetting. This energy preservation ensures that the semantic and topological relationships of base classes, encoded in their initial energy states, remain consistent as new classes are learned, while the symplectic dynamics-inspired feature evolution enables the model to adapt to novel classes without disrupting existing knowledge, ultimately fostering better generalization across incremental sessions by balancing the stability of prior knowl- edge and the plasticity of new learning.

1) Initial State Formation: HGN-EC aggregates neighbor information using the adjacency matrix adj and concatenates the node features X with the aggregated neighbor features to form the initial state

2) Feature Compression: HGN-EC applies a linear layer to compress the state, mapping the high-dimensional state vector to a lower-dimensional space

where Wcompress is the weight matrix and bcompress is the bias vector.

3) System Initialization: To initialize the generalized coor- dinates q and generalized momenta p of the system, HGN-EC assigns the compressed feature vector compressed to both q and p. Here, q and p represent the generalized coordinates and momenta of the system, respectively, which are core variables in Hamiltonian mechanics used to describe the system’s state.

4) Hamiltonian Energy Function: The Hamiltonian energy function H is modeled by a multi-layer perceptron (MLP) that takes q and p as inputs

where H net is an MLP consisting of GIN layers and activa- tion functions.

5) State Updates: The Hamiltonian equations define the rules for state updates

These equations compute gradients via automatic differentia- tion and are solved numerically using the Symplectic Euler method


*TABLE I*

*EVALUATION ON CIFAR100. ∆PD REPRESENTS THE DIFFERENCE BETWEEN THE CLASSIFICATION ACCURACY OF THE MODEL ON THE BASE SESSION*

*AND THE LAST INCREMENTAL SESSION. THE SMALLER THE ∆PD VALUE, THE LOWER THE MODEL’S*

*CATASTROPHIC FORGETTING OF OLD CLASSES*

where dt is the time step size.

6) State Restoration: The updated state is then restored to the original dimensionality via another linear layer

where Wrestore and brestore are the weight matrix and bias vector, respectively.

7) Energy Conservation: To ensure energy conservation in the system, HGN-EC introduces an energy-conservation loss

where Hinitial is the energy of the system in its initial state, MSE stands for Mean Squared Error and Hfinal is the energy of the system after dynamic changes.

8) Final Output: q final serves as the updated learnable prompt, which is passed into the vision and text encoders for subsequent learning tasks. This design leverages the Hamil- tonian energy function and state-update equations, combined with the Symplectic Euler method for solving ODEs, to ensure energy conservation in the system, thereby enhancing the model’s stability and performance.

## E. Loss Function

The total loss function is formulated as a weighted sum of the aforementioned partial losses. The mathematical expres- sion is given by

where

- contributions of the partial losses λ1, λ2, and λ3 are hyperparameters that balance the

- LCE • Lrecon represents the classification loss

- denotes the reconstruction loss

- Ladv is the adversarial loss for discriminator

- Lenergy stands for the energy loss

This formulation enables our model to effectively integrate classification tasks with graph structure regularization, thereby enhancing both the generalization capability and robustness of the model.

## V. EXPERIMENTS

We evaluated our model ACHG-CLIP on three datasets: CIFAR-100 [56], miniImageNet [57], and CUB-200-2011 [58], and compared it with other FSCIL methods (as shown in table I-III). [URL 🔗](#page-0)

## A. Evaluation Datasets

In our experiments, we utilized the following three datasets, each configured for class-incremental learning with specific few-shot settings.

- CIFAR100: This dataset consists of 60,000 32×32 RGB images across 100 classes. We designated 60 classes as base classes and the remaining 40 as incremental classes. These incremental classes are divided into 8 sessions, each introducing 5 new classes with 5 samples per class, employing a 5-way 5-shot class-incremental learning setup.

- miniImageNet: With 60,000 84 × 84 RGB images over 100 classes, we similarly assigned 60 classes as base classes and the remaining 40 as incremental. These are also split into 8 sessions, each with 5 new classes and 5 samples per class, following the same 5-way 5-shot class- incremental learning approach.

- CUB-200-2011: This dataset comprises a total of 11,788 RGB images with a resolution of 224 covering 200 distinct bird species. Here, 100 classes serve as the base, and the other 100 are used for incremen- tal learning. They are divided into 10 sessions, each introducing 10 new classes with 5 samples each, thus implementing a 10-way 5-shot class-incremental learning scenario. × 224 pixels,

## B. Implementation Details

In our experiments, we implemented a class-incremental learning model using the PyTorch framework. We constructed a 4-layer GIN network structure, with the number of neurons in the hidden layer being 16. The total loss function is a combination of the classification, ACGA loss including adversarial and reconstruction losses, and HGN-EC energy conservation loss. The coefficients assigned to the ACGA loss


*TABLE II*

*EVALUATION ON MINIIMAGENET. ∆PD REPRESENTS THE DIFFERENCE BETWEEN THE CLASSIFICATION ACCURACY OF THE MODEL ON THE BASE*

*SESSION AND THE LAST INCREMENTAL SESSION. THE SMALLER THE ∆PD VALUE, THE LOWER THE MODEL’S*

*CATASTROPHIC FORGETTING OF OLD CLASSES*

*TABLE III*

*EVALUATION ON CUB200. ∆PD REPRESENTS THE DIFFERENCE BETWEEN THE CLASSIFICATION ACCURACY OF THE MODEL ON THE BASE SESSION AND THE LAST INCREMENTAL SESSION. THE SMALLER THE ∆PD VALUE, THE LOWER THE MODEL’S CATASTROPHIC FORGETTING OF OLD CLASSES*

and the HGN-EC loss within the total loss function were set to

0.04. For optimization, we employed the Lion optimizer

[[59]](#page-0)

with a learning rate of 0.000325 and a weight decay of 1e-3, combined with the CosineAnnealingWarmupRestarts sched-

uler

[[60]](#page-0)

for dynamic learning rate adjustment. To stabilize

the training process, we adopted gradient accumulation over 3 steps and gradient clipping with a maximum norm of 4.0. For the training of base classes, we set the batch size to 4 and the number of training epochs to 3. In the incremental learning phase, the batch size and the number of training epochs were set to 4 and 5, respectively.

## C. Comparisons With State-of-the-Arts

In our experiments, the proposed model outperformed the

existing SOTA methods

[65], [66], [67], [68], [69], [70], [71] [URL 🔗](#page-0)

In CIFAR100, our model achieved an average accuracy of 82.30% and a ∆PD value of 9.72%. In miniImageNet, it delivered an average accuracy of 85.05% and a ∆PD value of 8.42%. In the challenging CUB200 dataset, the model excelled with an average accuracy of 69.54% and a ∆PD value of 17.67%, marking increases of 0.36% and reductions of 0.05% respectively. Detailed results are in Tables I, II, and III. [URL 🔗](#page-0)

These results show that our approach effectively retains the memory of old classes while quickly adapting to new ones, demonstrating superior performance and broad applicability in class-incremental learning tasks.

[11], [30], [40], [61], [62], [63], [64], [URL 🔗](#page-0)

on multiple datasets.

## TABLE IV

*COMPARISON WITH OTHER METHODS ON*

*CUB200*

*DATASET. HERE,*

*TRAIN. TIME REFERS TO THE COMPUTATIONAL COST OF THE ENTIRE EXPERIMENT, MEASURED IN MINUTES. PARAMS DENOTES THE*

*NUMBER OF PARAMETERS IN THE MODEL.*

*Abase*

*DENOTES THE*

*ACCURACY IN THE BASE SESSION PHASE,*

*Alast*

*REPRE-*

*SENTS THE ACCURACY IN THE LAST INCREMENTAL*

*SES-*

*SION. MEAN*

*IS*

*THE AVERAGE ACCURACY ACROSS*

*THE BASE SESSION AND ALL INCREMENTAL*

*SESSIONS, AND*

*∆Alast*

*INDICATES THE*

*DIF-*

*FERENCE IN THE LAST INCREMENTAL SESSION ACCURACY BETWEEN EACH METHOD AND THE METHOD AT*

*THE TOP OF THE TABLE*

In addition, we compare the computational overhead and parameters of our proposed framework and other methods. Under the same experimental conditions, we conducted the following experiments on CUB200 dataset using one single GPU (NVIDIA GeForce RTX 2080Ti), as shown in Table IV. [URL 🔗](#page-0)

## D. Sensitivity Study

We conduct sensitivity studies to assess the model’s response to input parameter changes, aiming to identify the most influential parameters for optimizing the model structure,


*Fig. 2. Sensitivity Analysis of “selection of GNNs”, “number of learnable prompts” and “optimizer”. Line charts are used to demonstrate the influence of these three choices on model performance.*

improving prediction accuracy, and pinpointing key factors in system performance.

1) Selection of Graph Neural Networks: We compared different graph neural networks, including GCN [72], Graph SAGE [73], GAT [74], and GIN [52]. As illustrated in Fig. 2(a), although all GNN variants exhibit similar forget- ting trends across incremental sessions (where each session incrementally introduces new classes), GIN exhibited the most robust performance, consistently maintaining relatively high accuracy. This superiority stems from GIN’s ability to effec- tively capture fine-grained structural and semantic information in graph-structured data through its isomorphism-based node embedding mechanism, which preserves the essential topolog- ical and feature similarities critical for stable representation learning in dynamic incremental scenarios, making it well- suited for our task and thus selected as the final graph neural network. Notably, its benefit lies not in altering the forgetting pattern, but in mitigating its impact through stronger representational capacity, which directly aligns with our goal of enhancing model performance in FSCIL. [URL 🔗](#page-0)

For GAT, its notably low performance can be attributed to its attention-based mechanism. In the dynamic setting of few-shot class-incremental learning, where data distributions shift as new classes with limited samples are added, GAT’s node-level attention struggles to stably identify and prioritize critical features. The fluctuating data patterns disrupt the optimal adjustment of attention weights, leading to suboptimal feature aggregation and propagation. Consequently, GAT fails to robustly adapt to the evolving feature dependencies across sessions, resulting in a continuous decline in accuracy and underperforming compared to other models like GIN.

Regarding GCN and GraphSAGE, their performance lags behind GIN. GCN relies on fixed-form spectral graph con- volutions, which restrict its capacity to flexibly capture diverse local structural variations emerging in incremental ses- sions. GraphSAGE, while adopting a sampling-based inductive strategy, has inherent limitations in comprehensively aggre- gating node features—its neighbor-sampling and aggregation schemes cannot fully preserve the fine-grained semantic cor- relations that GIN’s isomorphism-driven approach effectively encodes. These structural and methodological constraints cause GCN and GraphSAGE to lose more critical information during feature propagation across dynamic sessions, resulting in lower accuracy compared to GIN.

*Fig. 3. Heatmap of the impact of different adjacency matrix threshold values on model performance.*

2) Adjacency Matrix Threshold: We investigated the impact of different threshold settings for the adjacency matrix, includ- ing fixed thresholds in the set of {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}. The results (Fig. 3) indicated that a fixed threshold of 0.8 achieved the optimal balance between graph connectivity and noise reduction. When the threshold is less than 0.8, the graph structure might be sparse, leading to insufficient propagation of inter-class information and reduced performance, While the threshold is greater than 0.8, the graph structure becomes overly dense, introducing redundant con- nections and noise, which degrades the model’s generalization ability. Therefore, 0.8 was selected for our model. [URL 🔗](#page-0)

3) Loss Function Coefficients: We conducted ablation stud- ies on the coefficients of the combined loss of reconstruction loss, adversarial loss, and energy conservation loss. After evaluating coefficients in the set of {0.01, 0.02, 0.04, 0.08}, we found that a coefficient of 0.04 provided the best trade- off between stability and performance improvement, and was therefore chosen for our final model (as shown in Fig. 4). [URL 🔗](#page-0)

4) Learnable Prompts: Learnable prompts dynamically adapt feature representations to bridge the domain gap between base and incremental sessions. We tested prompt quantities in {1, 2, 3, 4} while keeping other parameters consistent. Fig. 2(b) shows the model achieves optimal comprehensive perfor- mance with 1 learnable prompt, yielding the highest accuracy. Increasing prompt numbers leads to performance decline. This is due to redundant feature dimensions from excessive prompts (causing overfitting on incremental data) and amplified stability-plasticity dilemma in continual learning (impairing [URL 🔗](#page-0)


## TABLE V

*ABLATION STUDY. HGN-GC AND ACGA ARE TWO COMPONENTS UNDER ABLATION. THE TABLE REPORTS THE CLASSIFICATION ACCURACY ACROSS 11 SESSIONS (BASE SESSION AND 10 INCREMENTAL SESSIONS). ∆PD REPRESENTS THE DIFFERENCE BETWEEN THE CLASSIFICATION ACCURACY OF THE MODEL ON THE BASE SESSION AND THE LAST INCREMENTAL SESSION. THE SMALLER THE ∆PD VALUE, THE LOWER THE*

*MODEL’S CATASTROPHIC FORGETTING OF OLD CLASSES*

*Fig. 4. Curve graph of sensitivity study of loss function coefficients, where the blue and orange curves represent Last Session Accuracy and Avg Accuracy respectively, varying with different values.*

old knowledge retention), confirming 1 learnable prompt as the optimal balance between retention and adaptation.

5) Optimizer: We further evaluated four representative optimizers (Lion [59], SGD with momentum [75], Adam [76], Adan [77]) under identical settings to identify the best strategy. As shown in Fig. 2(c), the Lion optimizer outperforms others with 69.54% mean accuracy and 17.67% ∆PD, demonstrating superiority in incremental learning. Com- pared to SGD (66.55% mean, 18.04% ∆PD), Lion improves average accuracy and reduces catastrophic forgetting via its sign momentum update, enhancing gradient consistency and noise robustness. Against Adam (57.76% mean, 29.76% ∆PD), Lion achieves higher accuracy by eliminating Adam’s second- moment estimation. It also maintains a slight edge over Adan (63.67% mean, 26.23% ∆PD) via better convergence-speed and parameter-stability trade-off. Lion’s unique sign-function update (converting continuous gradients to discrete direction indicators) strengthens consistent feature learning and sim- plifies tuning, aligning with incremental learning’s need for efficient updates and stable retention, thus it is selected as the default optimizer. [URL 🔗](#page-0)

## E. Visualization

To validate the proposed method’s effectiveness in learning discriminative and stable features for SCIL, we use t-SNE [78] (as shown in Fig. 5) to project high-dimensional features into 2D space. t-SNE minimizes KL divergence between similarity distributions, enabling clear visualization of cluster structures that linear methods often miss. In the plot, samples from different tasks use distinct shapes, and classes use unique colors to reflect feature separation and stability across tasks. [URL 🔗](#page-0)

*Fig. 5. t-SNE Visualization. It projects high-dimensional features into a 2D space, where different colors distinguish old classes and novel classes. It can be observed that both old and novel classes form compact, well-separated clusters, indicating the method’s capability to learn discriminative features for FSCIL while mitigating catastrophic forgetting.*

Our method’s old-class features remain compact and cohe- sive after incremental learning, showing effective mitigation of catastrophic forgetting. New-class features (learned with 5-shot samples) form well-separated clusters, distinguishable from old and other new classes, proving efficient discrimina- tive feature extraction from limited samples. Features exhibit higher intra-class compactness and inter-class separability in t-SNE space, that is, intra-class samples aggregate tightly, while inter-class samples stay widely spaced. This aligns with quantitative metrics, providing visual evidence that our method balances old-class knowledge retention and new-class information acquisition, addressing FSCIL’s core challenges.

## F. Ablation Study

In the ablation study (TABLE V), we systematically evalu- ated the contributions of various components of our model to validate their effectiveness on CUB200 dataset. [URL 🔗](#page-0)

We explored four configurations to incorporate learnable prompts into the graph embedding process: (1) without HGN- EC and ACGA, (2) with HGN-EC alone, (3) with ACGA alone, and (4) the complete model with both HGN-EC and ACGA.

The complete model consistently outperformed the other configurations, and this outcome is derived from the inher- ent limitations of single-component usage and the powerful


synergy when components are combined. When only a single module, be it HGN-EC or ACGA, is employed, performance suffers. For HGN-EC alone, while it excels at capturing dynamic category relationships and can mitigate catastrophic forgetting and overfitting by maintaining robust representations of both old and new categories, it lacks the mechanisms to effectively enhance cross-modal alignment. In real-world scenarios, data often comes from multiple modalities, and without proper alignment, the model struggles to fully uti- lize diverse information, leading to suboptimal generalization, especially when dealing with few samples. On the other hand, ACGA, when used in isolation, can enhance cross-modal alignment through adversarial regularization and Hamiltonian dynamics, which aids in improving generalization from few samples. However, it fails to establish and preserve the kind of stable, dynamic category relationships that HGN-EC spe- cializes in. Without these relationships, the model has difficulty distinguishing between different categories over time, and old knowledge can be easily overwritten as new categories are introduced, resulting in degraded performance.

In contrast, when HGN-EC and ACGA work in tandem, they create a comprehensive solution. HGN-EC lays the groundwork by capturing those crucial dynamic category rela- tionships and safeguarding against catastrophic forgetting and overfitting. Simultaneously, ACGA builds on this foundation to boost cross-modal alignment, further refining the model’s ability to generalize from limited samples. Their combined efforts leverage the strengths of graph-based learning (via HGN-EC) and adversarial strategies (via ACGA), addressing the key challenges of class-incremental learning. Thus, the ablation study not only underscores the importance of each component in our model but also validates the effectiveness of our design choices, demonstrating that the synergy between distinct architectural elements is pivotal for achieving superior performance in complex learning tasks.

## VI. CONCLUSION

This paper presents a novel Few-Shot Class-Incremental Learning (FSCIL) framework tailored to balance stability and plasticity in dynamic learning scenarios, where models must incrementally learn new classes from scarce labeled data while preserving existing knowledge. The framework innovatively combines Graph Neural Networks (GNNs) with physically inspired optimization techniques, leveraging GNNs’ structural learning strengths to model complex relational patterns and physical system principles to enhance learning robustness. It integrates three key graph components: Graph Isomorphism Networks (GIN) for capturing fine-grained feature similarities across instances, Hamiltonian Graph Networks with Graph Convolution (HGN-EC) to embed dynamic semantic rela- tionship evolution via Hamiltonian mechanics analogies, and Adversarially-Constrained Graph Autoencoders (ACGA) to refine cross-modal semantic associations through adversarial training, bridging inter-modal gaps. To address catastrophic forgetting, the framework employs energy-conservation con- straints, which preserve prior knowledge-related energy states akin to physical energy conservation, and contrastive learning, which strengthens old-new class separability by maximizing

intra-class similarity and minimizing inter-class similarity. Empirical results show the proposed approach outperforms baseline models on standard benchmark datasets.

## LIMITATIONS AND FUTURE WORK

While the current dataset coverage provides a solid basis for initial validation, future research will prioritize expanding benchmark diversity to enhance the framework’s generaliz- ability across varied data distributions. This includes targeted validation in specialized domains such as medical imaging, where data exhibits unique pathological feature patterns, and remote-sensing data, characterized by complex spatial hetero- geneity. To strengthen empirical rigor, systematic evaluations across broader datasets will be conducted, complemented by extending the framework to more complex vision tasks including object detection and semantic segmentation—tasks that demand structured output prediction and richer contextual understanding. These efforts aim to not only demonstrate the framework’s versatility in real-world applications but also deepen theoretical foundations by addressing task-specific challenges. Ultimately, such work seeks to advance practical, efficient solutions for few-shot class-incremental learning, fostering the development of more scalable and generalizable learning systems.

## ACKNOWLEDGMENT

The numerical calculations in this article have been done on the supercomputing systems in the Supercomputing Center, Wuhan University.

## REFERENCES

- [1] X. Shu, J. Tang, G.-J. Qi, W. Liu, and J. Yang, “Hierarchical long short-term concurrent memory for human interaction recognition,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 43, no. 3, pp. 1110–1118, Mar. 2021. [URL 🔗](#page-0)

- [2] R. Yan, L. Xie, X. Shu, L. Zhang, and J. Tang, “Progressive instance- aware feature learning for compositional action recognition,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 8, pp. 10317–10330, Aug. 2023. [URL 🔗](#page-0)

- [3] K. Hu, Y. Wang, Y. Zhang, and X. Gao, “Progressive learning strategy for few-shot class-incremental learning,” IEEE Trans. Cybern., vol. 55, no. 3, pp. 1210–1223, Mar. 2025. [URL 🔗](#page-0)

- [4] L. Zhao et al., “Few-shot class-incremental learning via class-aware bilateral distillation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2023, pp. 11838–11847. [URL 🔗](#page-0)

- [5] J. Zhang, L. Liu, O. Silv´en, M. Pietik¨ainen, and D. Hu, “Few-shot class- incremental learning for classification and object detection: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 4, pp. 2924–2945, Apr. 2025. [URL 🔗](#page-0)

- [6] H. Qu, R. Yan, X. Shu, H. Gao, P. Huang, and G. Xie, “MVP-shot: Multi-velocity progressive-alignment framework for few-shot action recognition,” IEEE Trans. Multimedia, vol. 27, pp. 6593–6605, 2025. [URL 🔗](#page-0)

- [7] Y. Tan and X. Xiang, “Cross-domain few-shot incremental learning for point-cloud recognition,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), Jan. 2024, pp. 2296–2305. [URL 🔗](#page-0)

- [8] A. K. Tiwari and G. K. Sharma, “FS-3DSSN: An efficient few-shot learning for single-stage 3D object detection on point clouds,” Vis. Comput., vol. 40, no. 11, pp. 8125–8139, Jan. 2024, doi: 10.1007/ s00371-023-03228-8. [URL 🔗](#page-0)

- [9] H. Yi, “Few-shot class-incremental learning with class centers and contrastive learning for incremental vehicle recognition,” in Proc. Int. Joint Conf. Neural Netw. (IJCNN), Jun. 2024, pp. 1–8. [URL 🔗](#page-0)

- [10] A. Jayanthkumar et al., “Fraud detection in financial transactions using machine learning—A comparative study,” in Proc. 3rd, Ed., IEEE Delhi Sect. Flagship Conf. (DELCON), Nov. 2024, pp. 1–4. [URL 🔗](#page-0)


- [11] C. Zhang, N. Song, G. Lin, Y. Zheng, P. Pan, and Y. Xu, “Few- shot incremental learning with continually evolved classifiers,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2021, pp. 12450–12459. [URL 🔗](#page-0)

- [12] X. Qiu, H. Zhu, X. Fang, J. Liang, B. Chen, and H. Lin, “Advancing few-shot class-incremental learning with virtual prototype guidance prompting,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), Apr. 2025, pp. 1–5. [URL 🔗](#page-0)

- [13] Y. Zhao, L. Zhao, D. Ding, D. Hu, G. Kuang, and L. Liu, “Few-shot class-incremental SAR target recognition via cosine prototype learning,” IEEE Trans. Geosci. Remote Sens., vol. 61, 2023. [URL 🔗](#page-0)

- [14] C. Liu et al., “Few-shot class incremental learning with attention-aware self-adaptive prompt,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 1–18. [URL 🔗](#page-0)

- [15] Y. Wei, J. Ye, Z. Huang, J. Zhang, and H. Shan, “Online prototype learning for online continual learning,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct. 2023, pp. 18718–18728. [URL 🔗](#page-0)

- [16] X. Xu, S. Niu, Z. Wang, W. Guo, L. Jing, and H. Yang, “Multi-feature space similarity supplement for few-shot class incremental learning,” Knowl.-Based Syst., vol. 265, Apr. 2023, Art. no. 110394, doi: 10.1016/ j.knosys.2023.110394. [URL 🔗](#page-0)

- [17] W. Zhang and X. Gu, “Few shot class incremental learning via efficient prototype replay and calibration,” Entropy, vol. 25, no. 5, p. 776, May 2023. [URL 🔗](#page-0)

- [18] M.-H. Guo et al., “Attention mechanisms in computer vision: A survey,” Comput. Vis. Media, vol. 8, no. 3, pp. 331–368, 2022. [URL 🔗](#page-0)

- [19] L. Waikhom and R. Patgiri, “Recurrent convolution based graph neural network for node classification in graph structure data,” in Proc. 12th Int. Conf. Cloud Comput., Data Sci. Eng. (Confluence), Jan. 2022, pp. 201–206. [URL 🔗](#page-0)

- [20] S. Muppidi, A. Angadi, and S. K. Gorripati, “Semi-supervised label propagation community detection on graphs with graph neural network,” in Proc. 1st Int. Conf. Artif. Intell. Trends Pattern Recognit. (ICAITPR), Mar. 2022, pp. 1–6. [URL 🔗](#page-0)

- [21] J. Shi, S. Chaudhari, and J. M. F. Moura, “Graph convolutional neural networks in the companion model,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), Apr. 2024, pp. 7045–7049. [URL 🔗](#page-0)

- [22] C. Wang, Y. Qiu, D. Gao, and S. Scherer, “Lifelong graph learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Aug. 2022, pp. 13719–13728. [URL 🔗](#page-0)

- [23] X. Shu, L. Zhang, G.-J. Qi, W. Liu, and J. Tang, “Spatiotemporal co-attention recurrent neural networks for human-skeleton motion prediction,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 6, pp. 3300–3315, Jun. 2022. [URL 🔗](#page-0)

- [24] J. Tang, X. Shu, R. Yan, and L. Zhang, “Coherence constrained graph LSTM for group activity recognition,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 2, pp. 636–647, Feb. 2022. [URL 🔗](#page-0)

- [25] R. Yan, L. Xie, J. Tang, X. Shu, and Q. Tian, “HiGCIN: Hierarchical graph-based cross inference network for group activity recognition,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 6, pp. 6955–6968, Jun. 2023. [URL 🔗](#page-0)

- [26] H. Chen, L. Li, F. Lyu, F. Hu, Z. Xia, and F. Xu, “Two-level graph net- work for few-shot class-incremental learning,” 2023, arXiv:2303.13862. [URL 🔗](#page-0)

- [27] Z. Gu, C. Xu, and Z. Cui, “Grassmann graph embedding for few- shot class incremental learning,” in Proc. 6th Chin. Conf. Pattern Recognit. Comput. Vis., Xiamen, China. Berlin, Germany: Springer, 2023, pp. 179–191, doi: 10.1007/978-981-99-8543-2 15. [URL 🔗](#page-0)

- [28] F. Hu, J. Zhang, F. Lyu, L. Li, and F. Xu, “Constructing sample-to-class graph for few-shot class-incremental learning,” 2023, arXiv:2310.20268. [URL 🔗](http://dx.doi.org/10.1007/978-981-99-8543-2%5F15)

- [29] T. Yu, S. He, Y.-Z. Song, and T. Xiang, “Hybrid graph neural networks for few-shot learning,” in Proc. AAAI Conf. Artif. Intell., 2022, vol. 36, no. 3, pp. 3179–3187. [URL 🔗](#page-0)

- [30] M. D’Alessandro, A. Alonso, E. Calabr´es, and M. Galar, “Multimodal parameter-efficient few-shot class incremental learning,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. Workshops (ICCVW), Oct. 2023, pp. 3385–3395. [URL 🔗](#page-0)

- [31] P. Li, X. Shu, C.-M. Feng, Y. Feng, W. Zuo, and J. Tang, “Surgical video workflow analysis via visual-language learning,” Npj Health Syst., vol. 2, no. 1, p. 5, Jan. 2025. [URL 🔗](#page-0)

- [32] C. Xue, Y. Li, F. Zargari, and Y. Li, “Graph isomorphism network: A learning-based workflow for converter inverse design problem,” in Proc. IEEE Energy Convers. Congr. Expo. (ECCE), Oct. 2023, pp. 6589–6591. [URL 🔗](#page-0)

- [33] K. Zhao, Q. Kang, Y. Song, R. She, S. Wang, and W. P. Tay, “Adversarial robustness in graph neural networks: A Hamiltonian approach,” in Proc. Adv. Neural Inf. Process. Syst., 2023, pp. 3338–3361. [URL 🔗](#page-0)

- [34] A. Radford et al., “Learning transferable visual models from natural language supervision,” in Proc. Int. Conf. Mach. Learn., vol. 139, 2021, pp. 8748–8763. [URL 🔗](#page-0)

- [35] Y. Cui, Z. Yu, W. Peng, Q. Tian, and L. Liu, “Rethinking few-shot class- incremental learning with open-set hypothesis in hyperbolic geometry,” IEEE Trans. Multimedia, vol. 26, pp. 5897–5910, 2024. [URL 🔗](#page-0)

- [36] J. Kim, Y. Ku, D. Han, and S. Baek, “Beyond synthetic replays: Turning diffusion features into few-shot class-incremental learning knowledge,” 2025, arXiv:2503.23402. [URL 🔗](#page-0)

- [37] L. Xiang, X. Jin, G. Ding, J. Han, and L. Li, “Incremental few-shot learning for pedestrian attribute recognition,” in Proc. 28th Int. Joint Conf. Artif. Intell., Aug. 2019, pp. 3912–3918. [URL 🔗](#page-0)

- [38] G. Han and S.-N. Lim, “Few-shot object detection with foundation models,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2024, pp. 28608–28618. [URL 🔗](#page-0)

- [39] Y. Li, P. Moghadam, C. Peng, N. Ye, and P. Koniusz, “Inductive graph few-shot class incremental learning,” in Proc. 18th ACM Int. Conf. Web Search Data Mining. New York, NY, USA: Association for Computing Machinery, Mar. 2025, pp. 466–474, doi: 10.1145/3701551.3703578. [URL 🔗](#page-0)

- [40] X. Li et al., “Mamba-FSCIL: Dynamic adaptation with selective state space model for few-shot class-incremental learning,” 2024, arXiv:2407.06136. [URL 🔗](#page-0)

- [41] X. Li, J. Wu, Y. Yu, L. Nie, and M. Zhang, “Continuous knowledge- preserving decomposition with adaptive layer selection for few-shot class-incremental learning,” 2025, arXiv:2501.05017. [URL 🔗](#page-0)

- [42] W. Wang, “QGHNN: A quantum graph Hamiltonian neural network,” 2025, arXiv:2501.07986. [URL 🔗](#page-0)

- [43] A. J. Varghese, Z. Zhang, and G. E. Karniadakis, “SympGNNs: Symplectic graph neural networks for identifying high-dimensional Hamiltonian systems and node classification,” Neural Netw., vol. 187, Jul. 2025, Art. no. 107397. [URL 🔗](#page-0)

- [44] Q. Kang, K. Zhao, Y. Song, S. Wang, R. She, and W. P. Tay, “Node embedding from Hamiltonian information propagation in graph neural networks,” 2023, arXiv:2303.01030. [URL 🔗](#page-0)

- [45] Q.-F. Wang et al., “Covariance-based space regularization for few-shot class incremental learning,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), Feb. 2025, pp. 9566–9576. [URL 🔗](#page-0)

- [46] X.-Y. Liu, S. Wang, H. Zhang, H. Zhang, Z.-Y. Yang, and Y. Liang, “Novel regularization method for biomarker selection and cancer classification,” IEEE/ACM Trans. Comput. Biol. Bioinf., vol. 17, no. 4, pp. 1329–1340, Jul. 2020. [URL 🔗](#page-0)

- [47] B. Kim, Y. Ko, and J. Seo, “Novel regularization method for the class imbalance problem,” Expert Syst. Appl., vol. 188, Feb. 2022, Art. no. 115974. [URL 🔗](#page-0)

- [48] S. Pan, R. Hu, G. Long, J. Jiang, L. Yao, and C. Zhang, “Adversarially regularized graph autoencoder for graph embedding,” 2018, arXiv:1802.04407. [URL 🔗](#page-0)

- [49] J. Zhou et al., “Graph neural networks: A review of methods and applications,” AI open, vol. 1, pp. 57–81, Aug. 2020. [URL 🔗](#page-0)

- [50] T. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in Proc. Int. Conf. Learn. Represent., 2016. [Online]. Available: https://openreview.net/forum?id=SJU4ayYgl [URL 🔗](#page-0)

- [51] P. Veliˇckovi´c, G. Cucurull, A. Casanova, A. Romero, P. Li´o, and Y. Bengio, “Graph attention networks,” in Proc. 6th Int. Conf. Learn. Represent., Vancouver, BC, Canada, May 2018. [Online]. Available: https://openreview.net/forum?id=rJXMpikCZ [URL 🔗](#page-0)

- [52] K. Xu, W. Hu, J. Leskovec, and S. Jegelka, “How powerful are graph neural networks?,” in Proc. 7th Int. Conf. Learn. Represent., New Orleans, LA, USA, 2018. [Online]. Available: https://openreview.net/ forum?id=ryGs6iA5Km [URL 🔗](#page-0)

- [53] Y. Peng, Y. Lin, X.-Y. Jing, H. Zhang, Y. Huang, and G. S. Luo, “Enhanced graph isomorphism network for molecular ADMET prop- erties prediction,” IEEE Access, vol. 8, pp. 168344–168360, 2020. [URL 🔗](#page-0)

- [54] C. Wu, Y. Lou, J. Li, L. Wang, S. Xie, and G. Chen, “A multitask network robustness analysis system based on the graph isomorphism network,” IEEE Trans. Cybern., vol. 54, no. 11, pp. 6630–6642, Nov. 2024. [URL 🔗](#page-0)

- [55] A. Mao, M. Mohri, and Y. Zhong, “Cross-entropy loss functions: Theoretical analysis and applications,” in Proc. 40th Int. Conf. Mach. Learn., 2023, pp. 23803–23828. [URL 🔗](#page-0)

- [56] A. Krizhevsky and G. Hinton, “Learning multiple layers of features from tiny images,” Univ. Toronto, Toronto, ON, Canada, Tech. Rep. UTML TR 2009-007, 2009. [URL 🔗](#page-0)

- [57] O. Russakovsky et al., “ImageNet large scale visual recognition challenge,” Int. J. Comput. Vis., vol. 115, no. 3, pp. 211–252, Dec. 2015. [URL 🔗](#page-0)

- [58] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie, “The Caltech-UCSD Birds-200-2011 Dataset,” California Inst. Technol., Pasadena, CA, USA, Tech. Rep. CNS-TR-2011-001, 2011. [URL 🔗](#page-0)


- [59] X. Chen et al., “Symbolic discovery of optimization algorithms,” in Proc. 37th Int. Conf. Neural Inf. Process. Syst., 2023, vol. 36, pp. 49205–49233. [URL 🔗](#page-0)

- [60] I. Loshchilov and F. Hutter, “SGDR: Stochastic gradient descent with warm restarts,” in Proc. 5th Int. Conf. Learn. Represent., Toulon, France, Apr. 2016. [Online]. Available: https://openreview.net/ forum?id=Skq89Scxx [URL 🔗](#page-0)

- [61] Z. Chi, L. Gu, H. Liu, Y. Wang, Y. Yu, and J. Tang, “MetaFSCIL: A meta-learning approach for few-shot class incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2022, pp. 14146–14155. [URL 🔗](#page-0)

- [62] M. Hersche, G. Karunaratne, G. Cherubini, L. Benini, A. Sebastian, and A. Rahimi, “Constrained few-shot class-incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2022, pp. 9047–9057. [URL 🔗](#page-0)

- [63] D.-W. Zhou, H.-J. Ye, L. Ma, D. Xie, S. Pu, and D.-C. Zhan, “Few- shot class-incremental learning by sampling multi-phase tasks,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 11, pp. 12816–12831, Nov. 2023. [URL 🔗](#page-0)

- [64] D.-W. Zhou, F.-Y. Wang, H.-J. Ye, L. Ma, S. Pu, and D.-C. Zhan, “Forward compatible few-shot class-incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2022, pp. 9036–9046. [URL 🔗](#page-0)

- [65] Q. Wang, D.-W. Zhou, Y. Zhang, D. Zhan, and H.-J. Ye, “Few-shot class- incremental learning via training-free prototype calibration,” in Proc. 37th Int. Conf. Neural Inf. Process. Syst. Red Hook, NY, USA: Curran Associates Inc., 2023, pp. 15060–15076. [URL 🔗](#page-0)

- [66] J. Li, S. Dong, Y. Gong, Y. He, and X. Wei, “Analogical learning-based few-shot class-incremental learning,” IEEE Trans. Circuits Syst. Video Technol., vol. 34, no. 7, pp. 5493–5504, Jul. 2024. [URL 🔗](#page-0)

- [67] S. Roy, C. Park, A. Fahrezi, and A. Etemad, “A bag of tricks for few- shot class-incremental learning,” 2024, arXiv:2403.14392. [URL 🔗](#page-0)

- [68] Y. Zou, S. Zhang, H. Zhou, Y. Li, and R. Li, “Compositional few-shot class-incremental learning,” in Proc. 41st Int. Conf. Mach. Learn., 2024, pp. 62964–62977. [URL 🔗](#page-0)

- [69] N. Ahmed, A. Kukleva, and B. Schiele, “OrCo: Towards better gener- alization via orthogonality and contrast for few-shot class-incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2024, pp. 28762–28771. [URL 🔗](#page-0)

- [70] K. Zhu, Y. Cao, W. Zhai, J. Cheng, and Z.-J. Zha, “Self-promoted prototype refinement for few-shot class-incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2021, pp. 6797–6806. [URL 🔗](#page-0)

- [71] W. Wang et al., “Gradient guided multiscale feature collabora- tion networks for few-shot class-incremental remote sensing scene classification,” IEEE Trans. Geosci. Remote Sens., vol. 62, 2024. [URL 🔗](#page-0)

- [72] J. Bruna, W. Zaremba, A. Szlam, and Y. LeCun, “Spectral networks and locally connected networks on graphs,” in Proc. 2nd Int. Conf. Learn. Represent., Banff, AB, Canada, Apr. 2013. [URL 🔗](#page-0)

- [73] W. L. Hamilton, R. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” in Proc. 31st Int. Conf. Neural Inf. Process. Syst., vol. 30. Red Hook, NY, USA: Curran Associates Inc., 2017, pp. 1024–1034. [URL 🔗](#page-0)

- [74] P. Veliˇckovi´c, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio, “Graph attention networks,” 2017, arXiv:1710.10903. [URL 🔗](#page-0)

- [75] Y. Liu, Y. Gao, and W. Yin, “An improved analysis of stochastic gradient descent with momentum,” in Proc. Adv. Neural Inf. Process. Syst., 2020, pp. 18261–18271. [URL 🔗](#page-0)

- [76] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” 2014, arXiv:1412.6980. [URL 🔗](#page-0)

- [77] X. Xie, P. Zhou, H. Li, Z. Lin, and S. Yan, “Adan: Adaptive Nesterov momentum algorithm for faster optimizing deep models,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 12, pp. 9508–9520, Dec. 2024. [URL 🔗](#page-0)

- [78] L. van der Maaten and G. E. Hinton, “Visualizing data using t-SNE,” J. Mach. Learn. Res., vol. 9, no. 86, pp. 2579–2605, Nov. 2008. [URL 🔗](#page-0)

Yuqian Ma received the B.A. degree from Wuhan University, Wuhan, China, in 2022, and the M.S. degree from Nanyang Technological University, Sin- gapore, in 2023. She is currently pursuing the Ph.D. degree with the School of Computer Science, Wuhan University. Her research interests include few-shot class-incremental learning and multimodal learning.

Youfa Liu received the M.S. degree in mathematics from Chinese Academy of Sciences, Beijing, China, in 2017, and the Ph.D. degree in computer science from Wuhan University, Wuhan, China, in 2020. He has published some papers in top journals and con- ferences, such as IJCV, IEEE TRANSACTIONS ON

IMAGE PROCESSING, IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS,

ACM MM, and ECCV. His research interests include machine learning and computer vision.

Bo Du (Senior Member, IEEE) received the Ph.D. degree in photogrammetry and remote sensing from the State Key Laboratory of Information Engineer- ing in Surveying, Mapping and Remote Sensing, Wuhan University, Wuhan, China, in 2010. He is currently a Professor with the School of Com- puter Science, Wuhan University. He has more than 60 research articles published in IEEE TRANSAC-

TIONS ON GEOSCIENCE AND REMOTE SENSING, IEEE TRANSACTIONS ON IMAGE PROCESSING, IEEE JOURNAL OF SELECTED TOPICS IN APPLIED

EARTH OBSERVATIONS AND REMOTE SENSING, and IEEE GEOSCIENCE

AND REMOTE SENSING LETTERS. 13 of them are ESI hot papers or highly cited papers. His research interests include pattern recognition, hyperspectral image processing, and signal processing. He was a Senior PC Member of the International Joint Conference on Artificial Intelligence and the Association for the Advancement of Artificial Intelligence. He was a recipient of the Best Reviewer Awards from IEEE GRSS for his service to IEEE JOURNAL

OF SELECTED TOPICS IN APPLIED EARTH OBSERVATIONS AND REMOTE

SENSING in 2011 and the ACM Rising Star Award for his academic progress in 2015, the International Joint Conferences on Artificial Intelligence (IJCAI) Distinguished Paper Prize, the IEEE Data Fusion Contest Champion, and the IEEE Workshop on Hyperspectral Image and Signal Processing Best Paper Award in 2018. He was the Session Chair of the 2016 International Geoscience and Remote Sensing Symposium (IGARSS) and the 4th IEEE GRSS Workshop on Hyperspectral Image and Signal Processing: Evolution in Remote Sensing (WHISPERS). He was a reviewer for 20 science citation index (SCI) magazines, including IEEE TRANSACTIONS ON GEOSCIENCE

AND REMOTE SENSING, IEEE TRANSACTIONS ON IMAGE PROCESSING, IEEE JOURNAL OF SELECTED TOPICS IN APPLIED EARTH OBSERVATIONS AND REMOTE SENSING, and IEEE GEOSCIENCE AND REMOTE SENSING LETTERS.
