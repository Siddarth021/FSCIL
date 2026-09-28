# Paper Specification: ACHG-CLIP

**Source paper:** Yuqian Ma, Youfa Liu, Bo Du, "A Few-Shot Class Incremental Learning Method
Using Graph Neural Networks," *IEEE Transactions on Image Processing*, vol. 35, 2026, pp. 1337–1349.
DOI: 10.1109/TIP.2026.3657170.

**Proposed method name:** Adversarially Constrained Hamiltonian Graph-CLIP (ACHG-CLIP).

This document is a faithful extraction of the specification stated in the paper. It does not
add, correct, or infer any detail beyond what the paper states or what is directly legible in
Figure 1. Every unclear or underspecified item referenced here is cross-linked to
`ambiguity_log.md` rather than resolved.

---

## 1. FSCIL Problem Definition (Section III.A)

- The model is first trained on an initial (base) dataset `D0 = {(xi, yi)}_{i=1..N0}`, where `N0`
  is large and `yi ∈ C0` (base classes).
- At each incremental step `t ≥ 1`, the model receives a small dataset `Dt = {(xi, yi)}_{i=1..Nt}`,
  where `Nt << N0` (e.g., 5 samples per class) and new classes `Ct`.
- `Ct` is disjoint from all previously seen classes `Cτ` for `τ < t` — the model must learn
  entirely new categories without revisiting old data.
- Constraints:
  - The model must infer class labels **without knowing the session of origin** (class-incremental
    setting, not task-incremental).
  - The model **cannot store or revisit** previous data `D0, …, D_{t-1}` (strict memory efficiency).
- This formalizes the **stability–plasticity dilemma**: stability = retaining knowledge of
  previous tasks; plasticity = adapting to new classes from few samples.

## 2. Stability–Plasticity Problem

- Named explicitly in the Introduction as the central FSCIL challenge (citing [3]–[6]).
- Prior approaches surveyed and critiqued by the paper:
  - **Prototype learning** ([12],[13],[15]) — hard to define precise prototypes in complex FSCIL
    data, making discrimination between similar classes difficult ([16],[17]).
  - **Attention mechanisms** ([14],[18]) — focus on relevant regions but do not fully prevent
    catastrophic forgetting, since new data can disrupt attention distributions.
  - **GNN-based FSCIL methods** ([27],[28],[39],[40]) — do not fully account for data scarcity and
    dynamic class increments; cannot always distinguish similar new categories ([29]); static graph
    construction fails to adapt to changing inter-class relationships over time ([40]).

## 3. ACHG-CLIP Purpose

ACHG-CLIP is proposed to balance stability and plasticity in FSCIL by:
- Modeling inter-category relationships and cross-modal alignment via GNNs.
- Updating **learnable prompt** embeddings through graph structure, rather than updating the CLIP
  backbone itself.
- Combining three components: GIN (relational propagation), HGN-EC (Hamiltonian, energy-conserving
  dynamics for stable transitions across sessions), and ACGA (adversarially regularized graph
  autoencoder for latent consistency and cross-modal alignment).
- Freezing the CLIP backbone and training only a small number of GIN-based modules
  (parameter-efficient tuning).

## 4. Overall Architecture (Section IV, Fig. 1)

Per Figure 1, the pipeline is:

1. **CLIP backbone** (frozen): separate Text Encoder and Vision Encoder produce embeddings for
   class label text (e.g., "Bird", "Flower", "Car", …) and input images.
2. **Learnable Prompts**: a shared block holds text prompts `g` and vision prompts `g̃'` (and their
   updated counterparts), inserted into both encoders at every Transformer layer (Eqs. 9–10). Two
   `MLP` blocks sit between the Learnable Prompts block and the GIN modules — one for text prompts,
   one for vision prompts (see Fig. 1; exact MLP mapping is not otherwise specified in text — see
   `ambiguity_log.md`).
3. **GIN module** (Section IV.B): consumes text prompts and vision prompts separately, each
   producing a node feature matrix `X` and adjacency matrix `A`, and constructs two graphs,
   `G1` (text) and `G2` (vision).
4. **ACGA module** (Section IV.C): takes `(X, A)` for each modality's graph and:
   - Encodes with a GIN-based graph convolutional encoder → latent `Z` (labeled `z_fake` in Fig. 1).
   - Decodes via inner-product decoder → reconstructed adjacency `Â` (graphs `G'1`, `G'2` in Fig. 1)
     with reconstruction loss `L_recon`.
   - Regularizes the latent space adversarially against a prior `z_real ~ N(0, I)` via a
     discriminator, producing `L_adv`.
5. **HGN-EC module** (Section IV.D): takes node features `X` and adjacency `A` (paper text names
   the source as "the ARGA module" here — see `ambiguity_log.md` for the ACGA/ARGA naming
   inconsistency) and:
   - Aggregates neighbor information and forms an initial state.
   - Compresses the state (Feature Compress) and initializes generalized coordinates `q` and
     momenta `p`.
   - Runs a Hamiltonian energy function `H` (an MLP built from GIN layers — shown in Fig. 1 as a
     "Hamiltonian GIN Layer") and updates `(q, p)` via the Symplectic Euler method.
   - Restores dimensionality (Feature Restore) to produce the updated prompt `q_final`.
   - Enforces an energy-conservation loss `L_energy` between `H_initial` and `H_final`.
6. The output `q_final` (updated learnable prompt) is fed back ("Update", top and bottom arrows in
   Fig. 1) into the text and vision encoders for subsequent learning.
7. Cross-modal alignment between the resulting text and vision embeddings is trained with a
   contrastive classification loss `L_CE` (Eqs. 11–12).

## 5. CLIP Backbone (Section IV.A)

- Contrastive Language–Image Pre-Training model (Radford et al. [34]).
- Jointly trained text and vision encoders, both based on "improved Transformer structures."
- The CLIP backbone is **frozen**; only GIN-based modules (GIN, ACGA, HGN-EC) and the prompt/MLP
  parameters are trained.

## 6. Text Encoder (Section IV.A.1, Eqs. 1–6)

1. **Embedding Representation:** tokens → continuous vectors via word-embedding matrix `E_T`
   (Eq. 1), then positional encoding `P_T(n)` is added (Eq. 2).
2. **Feature Interaction:** `L` Transformer blocks, each with multi-head attention (Eq. 3, Eq. 5)
   followed by a feed-forward network with residual/LayerNorm (Eq. 4).
3. **Semantic Projection:** global average pool of the final layer output, linearly projected and
   L2-normalized to give `h*_T` (Eq. 6).

## 7. Vision Encoder (Section IV.A.2, Eqs. 7–8)

- Symmetric to the text encoder, with two differences:
  1. **Structured Input:** patch embeddings concatenated with a `[CLS]` token and positional
     encoding `P_V(m+1)` (Eq. 7).
  2. **Feature Extraction:** after `L` Transformer layers, the `[CLS]` token's final-layer feature
     is projected and normalized to give `h*_V` (Eq. 8).

## 8. Learnable Text Prompts (Section IV.A.3.1, Eq. 9)

- Generated from a learnable parameter matrix `G ∈ R^{L×M×d}` (`L` = number of layers, `M` = number
  of learnable prompts, `d` = embedding dimension).
- Inserted into text embeddings at each layer: `X = [X_CLS, g^(l), X_tokens]` (Eq. 9).

## 9. Learnable Vision Prompts (Section IV.A.3.2, Eq. 10)

- Generated from a learnable parameter matrix `GV ∈ R^{L×M×d}`.
- Inserted into vision embeddings at each layer: `X = [X_CLS, gV^(l), X_patches]` (Eq. 10).
- Paper states vision prompts "directly replace the input of each layer" — see
  `ambiguity_log.md` regarding whether this differs from the text-prompt insertion mechanism.

## 10. Graph Construction (Section IV.B.1, Eqs. 14–18)

1. Cosine similarity matrix: `sim_matrix_{i,j} = cos(x_i, x_j)` (Eq. 14).
2. Binarization by threshold: `A_{i,j} = 1` if `sim_matrix_{i,j} > adj_threshold`, else `0` (Eq. 15).
3. Symmetrization: `Z = (A + A^T) / 2` (Eq. 16).
4. Normalization: `Ã = D^{-1/2} · Z · D^{-1/2}`, where `D` is the degree matrix (Eq. 17).
5. Optional attention step: `attention_{i,j} = softmax(sim_matrix_{i,j})`, then
   `Ã = Ã · attention` (Eq. 18). The paper marks this step as "optional" without stating whether it
   is used in the final reported model — see `ambiguity_log.md`.
6. In each FSCIL incremental stage: new-node features are appended to the graph; old-node features
   are kept unchanged, to retain prior-task knowledge.

## 11. Cosine Similarity

- Defined in Eq. 14, used for both adjacency-matrix construction (Section IV.B.1) and cross-modal
  similarity in the CLIP contrastive loss (Eq. 11: `Sim_{ij} = τ · cos(h_V^{(i)}, h_T^{(j)})`).

## 12. Adjacency Threshold

- Symbol: `adj_threshold`, used in Eq. 15.
- Value used in the final model: **0.8** (selected via sensitivity study over
  `{0.1, …, 0.9}`, Section V.D.2, Fig. 3).

## 13. GIN (Graph Isomorphism Network) (Section IV.B, Eqs. 13, 19–21)

- Layer update: `h_v^(k) = MLP^(k)( (1 + ε^(k)) · h_v^(k-1) + Σ_{u∈N(v)} h_u^(k-1) )` (Eq. 13),
  equivalently decomposed in the paper into:
  - Aggregation: `agg_v = Σ_{u∈N(v)} h_u^(k-1)` (Eq. 19)
  - Combination: `combined_v = (1 + ε^(k)) h_v^(k-1) + agg_v` (Eq. 20)
  - Transformation: `h_v^(k) = MLP^(k)(combined_v)` (Eq. 21)
- `ε^(k)` is a learnable scalar per layer controlling self-vs-neighbor feature fusion.
- Used to process both text and vision learnable prompts into node feature matrices `X ∈ R^{N×D}`
  and adjacency matrices `A ∈ R^{N×N}`, which feed into the ACGA module (paper text says "ARGA" at
  this specific point — see naming ambiguity below).
- Known architecture hyperparameter (Section V.B): **4 layers**, hidden dimension **16**.

## 14. ACGA (Adversarially Constrained Graph Autoencoder) (Section IV.C)

Three sub-components:

1. **Graph Convolutional Encoder** (Eq. 22): an "improved GIN" — `Z^(l+1) = GINLayer(Z^(l), A)`,
   using the GIN layer formula of Eq. 13, with `ε^(l) ∈ R` learnable and MLP composed of linear
   layers, batch normalization, and GELU activation. Final output: `Z = GIN(X, A) ∈ R^{N×K}`, where
   `N` = number of graph nodes, `K` = latent dimension ("N and K are independent, with no
   functional relationship between them" — explicit paper statement).
2. **Structural Reconstruction Decoder** (Eq. 23): inner-product decoder,
   `Â = σ(Z Z^T)`, with `σ` the sigmoid function.
3. **Adversarial Regularization Module** (Eqs. 25–26): discriminator
   `D(z) = Sigmoid(W2 · GELU(W1 z + b1) + b2)` composed of two fully-connected layers, trained to
   match the encoder's latent distribution `q(Z|X,A)` to a prior `p_z = N(0, I)` using a
   Wasserstein-distance-style adversarial loss.

## 15. Graph Encoder

- See item 14.1 above (Graph Convolutional Encoder). Also referred to generically as "Encoder" in
  Fig. 1, producing `z_fake`.

## 16. Reconstruction Decoder

- See item 14.2 above (Structural Reconstruction Decoder). Reconstructs the **adjacency matrix**
  (not node features) via `Â = σ(ZZ^T)` (Eq. 23).

## 17. Adversarial Discriminator

- See item 14.3 above. Two fully-connected layers with GELU activation and sigmoid output
  (Eq. 25).

## 18. Reconstruction Loss (Eq. 24)

```
L_recon = - Σ_{(i,j)∈E∪E-} [ A_ij·log(Â_ij) + (1 - A_ij)·log(1 - Â_ij) ]
```
- `E` = positively-sampled (real) edges; `E-` = negatively-sampled edges.
- The negative-sampling ratio/strategy for `E-` is not specified — see `ambiguity_log.md`.

## 19. Adversarial Loss (Eq. 26)

```
L_adv = E_{z~p_z}[D(z)] - E_{z~q(Z|X,A)}[D(z)]
```
- Framed by the paper as a "Wasserstein distance" form for training stability. Standard WGAN
  training details (critic weight clipping / gradient penalty, number of discriminator steps per
  generator step) are not stated — see `ambiguity_log.md`.

## 20. HGN-EC (Hamiltonian Graph Network with Energy Conservation) (Section IV.D)

- Receives node feature matrix `X ∈ R^{N×D}` and adjacency matrix `A ∈ R^{N×N}` — paper text says
  from "the ARGA module" (naming inconsistency, see `ambiguity_log.md`).
- Purpose: model dynamic evolution of graph data via Hamiltonian-mechanics-inspired,
  energy-conserving state updates, to stabilize training and prevent degradation of base-class
  feature representations (catastrophic forgetting) as new classes are learned.
- Steps (see items 21–26 below for the equations): Initial State Formation → Feature Compression →
  System Initialization (q, p) → Hamiltonian Energy Function → State Updates (Symplectic Euler) →
  State Restoration → Energy Conservation loss → Final Output (`q_final`).

## 21. Feature Compression (Section IV.D.2, Eq. 28)

1. **Initial state formation** (Eq. 27):
   `aggregated = A · X`, `state = [X, aggregated]` (concatenation of node features with
   neighbor-aggregated features).
2. **Compression** (Eq. 28): `compressed = W_compress · state + b_compress`, a single linear layer
   mapping the (higher-dimensional, concatenated) state vector to a lower-dimensional space. Exact
   output dimensionality is not numerically specified — see `ambiguity_log.md`.

## 22. q and p States (Section IV.D.3)

- `q` (generalized coordinates) and `p` (generalized momenta) are the core state variables of the
  Hamiltonian system.
- **Both `q` and `p` are initialized to the same value**: the compressed feature vector
  `compressed` is assigned to both `q` and `p` ("HGN-EC assigns the compressed feature vector
  compressed to both q and p" — explicit paper statement).

## 23. Hamiltonian Network (Section IV.D.4, Eq. 29)

```
H = H_net(cat(q, p))
```
- `H_net` is an MLP "consisting of GIN layers and activation functions." Figure 1 depicts this as a
  "Hamiltonian GIN Layer." Exact number of GIN layers / hidden width inside `H_net` is not
  separately specified from the main GIN's "4 layers, hidden dim 16" — see `ambiguity_log.md`.

## 24. Hamiltonian Equations and Symplectic Euler Update (Section IV.D.5, Eqs. 30–31)

- Hamilton's equations (computed via automatic differentiation):
```
q̇ = ∂H/∂p,   ṗ = -∂H/∂q
```
- Solved numerically with the Symplectic Euler method:
```
p_new = p + dt · ṗ
q_new = q + dt · q̇
```
- `dt` (the time-step size) is referenced symbolically but **no numeric value is given anywhere in
  the paper** — see `ambiguity_log.md`.

## 25. State Restoration (Section IV.D.6, Eq. 32)

```
q_final = W_restore · q_new + b_restore
```
- A single linear layer restoring `q_new` to the original (pre-compression) dimensionality.
- `q_final` is the module's final output, serving as the updated learnable prompt fed back into
  the text and vision encoders (Section IV.D.8).

## 26. Energy Conservation (Section IV.D.7, Eq. 33)

```
L_energy = MSE(H_initial, H_final) = (1/n) Σ_{i=1}^{n} (H_initial,i - H_final,i)^2
```
- `H_initial` = system energy at the initial state; `H_final` = system energy after dynamic
  (Hamiltonian) changes. `n` is the number of elements being averaged over (not otherwise defined
  numerically — presumably batch/node count, not stated explicitly).

## 27. Total Loss (Section IV.E, Eq. 34)

```
L_total = L_CE + λ1·L_recon + λ2·L_adv + λ3·L_energy
```
- `λ1, λ2, λ3` are hyperparameters balancing the partial losses.
- `L_CE`: CLIP contrastive classification loss (Eq. 12).
- `L_recon`: ACGA reconstruction loss (Eq. 24).
- `L_adv`: ACGA adversarial loss (Eq. 26).
- `L_energy`: HGN-EC energy-conservation loss (Eq. 33).
- Reported coefficient value for **both** the ACGA loss terms and the HGN-EC energy loss term is
  **0.04** (Section V.B): "The coefficients assigned to the ACGA loss and the HGN-EC loss within
  the total loss function were set to 0.04." It is not explicitly stated whether this means
  `λ1 = λ2 = λ3 = 0.04`, or whether `λ1`/`λ2` (both "ACGA loss") share one value distinct from `λ3`
  — see `ambiguity_log.md`.

## 28. FSCIL Training Procedure

- **Base session:** trained on `D0` (base classes) with a full-size labeled dataset.
- **Incremental sessions:** for `t = 1 … T`, trained on `Dt` (5-shot per new class), while
  retaining performance on all previously seen classes, without replaying old data (Section III.A).
- Per Section IV.B.1: at each incremental stage, new node features are added to the graph while old
  node features are kept unchanged.
- Optimization uses Lion optimizer with `CosineAnnealingWarmupRestarts` scheduler, gradient
  accumulation, and gradient clipping (see reproduction_protocol.md and known hyperparameters
  below).

## 29. Base Session (Section V.B)

- Batch size: **4**
- Training epochs: **3**

## 30. Incremental Sessions (Section V.B)

- Batch size: **4**
- Training epochs: **5**

## 31. Datasets (Section V.A)

| Dataset | Images | Total classes | Base classes | Incremental classes | Sessions | Classes/session | Shot | Image size |
|---|---|---|---|---|---|---|---|---|
| CIFAR-100 | 60,000 | 100 | 60 | 40 | 8 | 5 | 5 | 32×32 |
| miniImageNet | 60,000 | 100 | 60 | 40 | 8 | 5 | 5 | 84×84 |
| CUB-200-2011 | 11,788 | 200 | 100 | 100 | 10 | 10 | 5 | 224×224 |

## 32. Metrics

- **Per-session classification accuracy (%)** across all sessions (base session `0` through the
  final incremental session).
- **Mean accuracy** — average accuracy across the base session and all incremental sessions.
- **ΔPD** ("performance drop") — difference between the base-session accuracy and the last
  incremental-session accuracy: smaller ΔPD indicates less catastrophic forgetting. (Note: the
  paper's Table IV separately defines `ΔA_last` as a *comparative* metric between methods, distinct
  from the `ΔPD` intra-method metric of Tables I–III — see `reproduction_protocol.md`.)
- Table IV additionally reports **Train. Time** (minutes, "computational cost of the entire
  experiment") and **Params** (number of model parameters), measured "under the same experimental
  conditions… using one single GPU (NVIDIA GeForce RTX 2080Ti)."

## 33. Sensitivity Experiments (Section V.D)

1. **Selection of Graph Neural Network** (Fig. 2(a)): compares GCN, GraphSAGE, GAT, GIN as the
   choice of GNN. GIN selected as most robust.
2. **Adjacency Matrix Threshold** (Fig. 3): sweeps `{0.1, …, 0.9}`; **0.8** selected as optimal.
3. **Loss Function Coefficients** (Fig. 4): sweeps `{0.01, 0.02, 0.04, 0.08}` for the combined
   ACGA/HGN-EC loss coefficient(s); **0.04** selected.
4. **Number of Learnable Prompts** (Fig. 2(b)): sweeps `{1, 2, 3, 4}`; **1** selected as optimal
   (more prompts caused overfitting on incremental data and worsened the stability–plasticity
   trade-off).
5. **Optimizer** (Fig. 2(c)): compares Lion, SGD (with momentum), Adam, Adan; **Lion** selected
   (69.54% mean accuracy, 17.67% ΔPD on CUB200, versus 66.55%/18.04% for SGD, 57.76%/29.76% for
   Adam, 63.67%/26.23% for Adan).

## 34. Ablation Experiments (Section V.F, Table V)

- Conducted on **CUB200**, evaluated across 11 sessions (base + 10 incremental).
- Four configurations of learnable-prompt graph embedding:
  1. Without HGN-EC and without ACGA (both blank in Table V row 1)
  2. With ACGA alone
  3. With HGN-EC alone (paper text calls the column "HGN-GC" in the table header — naming
     inconsistency, see `ambiguity_log.md`)
  4. With both HGN-EC and ACGA (full model)
- Full model achieves the best Mean (69.54%) and lowest ΔPD (17.67%) among the four configurations.

## 35. Reported Results (Tables I–IV)

### Table I — CIFAR100 (Ours row)
Per-session accuracy (%) for sessions 0–8: 88.02, 85.89, 84.69, 81.81, 81.23, 81.02, 80.34, 79.38,
78.30. **Mean = 82.30%, ΔPD = 9.72%.**

### Table II — miniImageNet (Ours row)
Per-session accuracy (%) for sessions 0–8: 90.28, 88.77, 86.13, 85.05, 85.32, 83.25, 82.50, 82.33,
81.86. **Mean = 85.05%, ΔPD = 8.42%.**

### Table III — CUB200 (Ours row)
Per-session accuracy (%) for sessions 0–10: 80.80, 76.58, 75.13, 70.14, 70.03, 68.96, 66.80, 66.22,
64.46, 62.70, 63.13. **Mean = 69.54%, ΔPD = 17.67%.**

### Table IV — CUB200 comparison (compute/params)
| Method | Train. Time (min) | Params | A_base | A_last | Mean | ΔA_last |
|---|---|---|---|---|---|---|
| CEC [11] | 57 | 12.3M | 75.85 | 52.28 | 61.33 | — |
| LIMIT [63] | 61 | 12.3M | 75.89 | 57.41 | 65.48 | +5.13 |
| CPE-CLIP [30] | 26 | 400K | 80.56 | 62.84 | 69.18 | +10.56 |
| Ours | 30 | 800K | 80.80 | 63.13 | 69.54 | +10.85 |

### Table V — Ablation on CUB200 (Ours full-model row)
Per-session accuracy (%) for sessions 0–10: 80.80, 76.58, 75.13, 70.14, 70.03, 68.96, 66.80, 66.22,
64.46, 62.70, 63.13. **Mean = 69.54%, ΔPD = 17.67%.** (Identical to Table III's "Ours" row.)

## 36. All Explicitly Stated Hyperparameters

| Hyperparameter | Value | Source |
|---|---|---|
| GIN layers | 4 | Section V.B |
| GIN hidden dimension | 16 | Section V.B |
| Adjacency threshold | 0.8 | Section V.D.2 |
| Number of learnable prompts (M) | 1 | Section V.D.4 |
| Optimizer | Lion | Section V.B |
| Learning rate | 0.000325 | Section V.B |
| Weight decay | 1e-3 | Section V.B |
| LR scheduler | CosineAnnealingWarmupRestarts | Section V.B |
| Gradient accumulation | 3 steps | Section V.B |
| Gradient clipping (max norm) | 4.0 | Section V.B |
| Base session batch size | 4 | Section V.B |
| Base session epochs | 3 | Section V.B |
| Incremental session batch size | 4 | Section V.B |
| Incremental session epochs | 5 | Section V.B |
| Loss coefficient (ACGA + HGN-EC terms) | 0.04 | Section V.B |
| GPU (Table IV experiments only) | NVIDIA GeForce RTX 2080Ti | Section V.C |

No other numeric hyperparameters (e.g., `d`, `d_e`, `N`, `K`, `dt`, MLP widths inside `H_net`,
discriminator hidden width, negative-sampling ratio for `L_recon`, optimizer betas, scheduler
warmup length/restart period) are explicitly stated in the paper text. These are logged in
`ambiguity_log.md` and must not be guessed.
