# Equation → Code Mapping

This table maps every equation stated in the paper (Section IV) to its meaning, tensor I/O, and a
proposed implementation location. Dimensions are filled in **only** where the paper explicitly
states them; all other cells say "Not specified in paper" and are cross-referenced to
`ambiguity_log.md`. No dimension is invented.

Notation used below: `d` = CLIP token embedding dim, `d_e` = final projected embedding dim,
`n` = text sequence length, `m` = number of vision patches, `L` = number of Transformer/prompt
layers, `M` = number of learnable prompts (paper value: 1), `N` = number of graph nodes,
`D` = node feature dimension, `K` = ACGA latent dimension.

| Eq. | Meaning | Input tensors | Output tensors | Dimensions (if stated) | Proposed module/function | Implementation notes |
|---|---|---|---|---|---|---|
| (1) | Token → embedding lookup | `tokens` (int ids) | `X` | `E_T`: not specified; `X ∈ R^{n×d}` | `models/clip/text_encoder.py :: TextEmbedding.forward` | `n` = tokenized sequence length; `d` = CLIP embedding dim. CLIP backbone is frozen — `E_T` is a pretrained, not newly-trained, parameter. |
| (2) | Add positional encoding | `X ∈ R^{n×d}`, `P_T ∈ R^{n×d}` | `X` (in-place add) | `P_T ∈ R^{n×d}` (stated) | `models/clip/text_encoder.py :: TextEmbedding.forward` | Standard learned/sinusoidal positional encoding; paper does not say which — see ambiguity log. |
| (3) | Transformer block: attention + residual + LayerNorm | `X`, `X_query/key/value` | `X'` | Not specified beyond `d` | `models/clip/transformer_block.py :: TransformerBlock.forward` | Standard pre/post-LN Transformer sublayer; "improved Transformer structures" per Section IV.A is not otherwise defined — see ambiguity log. |
| (4) | Transformer block: FFN + residual + LayerNorm | `X'` | `X''` | Not specified | `models/clip/transformer_block.py :: TransformerBlock.forward` | FFN hidden dim not stated. |
| (5) | Multi-head attention, single head | `X`, `W_Query/Key/Value^i` | `head_i` | `d_k` = per-head query/key dim (symbolic only, no value given) | `models/clip/attention.py :: multi_head_attention` | Number of heads `h` not stated. |
| (6) | Text semantic projection | `X^(L)`, `W_T ∈ R^{d_e×d}` | `h*_T ∈ R^{d_e}` (per sample) | `W_T ∈ R^{d_e×d}` (stated); `d_e` value not given | `models/clip/text_encoder.py :: TextEncoder.forward` | `AvgPool` over the sequence dimension of the last layer, then linear projection + L2 normalize. |
| (7) | Vision structured input (patch embed + CLS + pos enc) | `patches`, `E_V ∈ R^{d×|p|}`, `P_V ∈ R^{(m+1)×d}` | `X ∈ R^{(m+1)×d}` | `E_V ∈ R^{d×|p|}`, `P_V ∈ R^{(m+1)×d}` (stated) | `models/clip/vision_encoder.py :: VisionEmbedding.forward` | `|p|` = flattened patch pixel dim; patch size not stated. |
| (8) | Vision feature extraction (CLS token projection) | `X^(L)_[CLS]`, `W_V ∈ R^{d_e×d}` | `h*_V ∈ R^{d_e}` | `W_V ∈ R^{d_e×d}` (stated) | `models/clip/vision_encoder.py :: VisionEncoder.forward` | Symmetric to Eq. 6 but on the CLS token, not average-pooled. |
| (9) | Text prompt insertion at layer `l` | `X_[CLS]`, `g^(l)`, `X_tokens`, `G ∈ R^{L×M×d}` | `X` (concatenated) | `G ∈ R^{L×M×d}` (stated) | `models/prompts/text_prompt.py :: TextPromptInjector.forward` | Concatenation-style insertion per paper wording ("Prompts are inserted into the text embeddings at each layer"). |
| (10) | Vision prompt insertion at layer `l` | `X_[CLS]`, `gV^(l)`, `X_patches`, `GV ∈ R^{L×M×d}` | `X` | `GV ∈ R^{L×M×d}` (stated) | `models/prompts/vision_prompt.py :: VisionPromptInjector.forward` | Paper states vision prompts "directly replace the input of each layer" — differs in wording from text-prompt insertion; exact mechanism ambiguous, see ambiguity log. |
| (11) | Cross-modal similarity matrix | `h_V^(i)`, `h_T^(j)` for `N` pairs, `τ` (learnable) | `Sim ∈ R^{N×N}` | `N` = batch size (contextual) | `losses/contrastive_loss.py :: clip_similarity_matrix` | `τ` is a learnable temperature scalar, standard CLIP-style. |
| (12) | CLIP contrastive classification loss | `Sim ∈ R^{N×N}`, `y` (labels/indices) | `L_CE` (scalar) | — | `losses/contrastive_loss.py :: clip_contrastive_loss` | Standard cross-entropy over rows of `Sim`; paper's formula sums per-image log-softmax over the correct text index. |
| (13) | GIN layer update (general form) | `h_v^(k-1)`, neighbor features `{h_u^(k-1)}`, `ε^(k)` | `h_v^(k)` | Hidden dim = 16, 4 layers (stated in Section V.B) | `models/gnn/gin_layer.py :: GINLayer.forward` | Equivalent operationally to Eqs. 19–21; implement once and reuse. |
| (14) | Cosine similarity for adjacency construction | `x_i, x_j` (node/prompt features) | `sim_matrix ∈ R^{N×N}` | — | `models/graph/adjacency.py :: cosine_similarity_matrix` | Standard cosine similarity. |
| (15) | Threshold binarization | `sim_matrix`, `adj_threshold` (=0.8) | `A ∈ {0,1}^{N×N}` | `adj_threshold = 0.8` (stated) | `models/graph/adjacency.py :: threshold_binarize` | Strict `>` per paper ("if sim_matrix > adj_threshold"). |
| (16) | Symmetrization | `A ∈ {0,1}^{N×N}` | `Z ∈ R^{N×N}` | — | `models/graph/adjacency.py :: symmetrize` | `Z = (A + A^T)/2`. |
| (17) | Degree normalization | `Z`, degree matrix `D` | `Ã ∈ R^{N×N}` | — | `models/graph/adjacency.py :: normalize_adjacency` | `Ã = D^{-1/2} Z D^{-1/2}`; degree computed from `Z`. |
| (18) | Optional attention re-weighting of adjacency | `Ã`, `attention = softmax(sim_matrix)` | `Ã` (updated) | — | `models/graph/adjacency.py :: apply_attention_reweight` | Paper marks this "optional" — whether it is used in the final reported model is not stated; implement as a togglable config flag, default **off**, per ambiguity log entry (do not assume it is used). |
| (19) | Neighbor aggregation | `{h_u^(k-1)} for u∈N(v)` | `agg_v` | — | `models/gnn/gin_layer.py :: GINLayer._aggregate` | Sum aggregation, matches GIN definition. |
| (20) | Combine self + aggregated features | `h_v^(k-1)`, `agg_v`, `ε^(k)` | `combined_v` | — | `models/gnn/gin_layer.py :: GINLayer._combine` | `(1+ε^(k))·h_v^(k-1) + agg_v`. |
| (21) | MLP transform | `combined_v` | `h_v^(k)` | Hidden dim 16 (Section V.B, for the "main" 4-layer GIN) | `models/gnn/gin_layer.py :: GINLayer._transform` | MLP architecture (number of internal linear layers, activation) not specified beyond hidden dim 16 — see ambiguity log. |
| (22) | ACGA graph-conv encoder layer | `Z^(l) ∈ R^{N×·}`, `A` | `Z^(l+1)` | Final `Z ∈ R^{N×K}` (K = latent dim, unspecified value) | `models/acga/encoder.py :: ACGAEncoder.forward` | Reuses GIN layer formula (Eq. 13) with MLP = Linear + BatchNorm + GELU, per paper text. |
| (23) | Inner-product adjacency decoder | `Z ∈ R^{N×K}` | `Â ∈ R^{N×N}` | — | `models/acga/decoder.py :: InnerProductDecoder.forward` | `Â = σ(ZZ^T)`. |
| (24) | ACGA reconstruction loss | `A` (real edges `E`), `Â`, negative edges `E-` | `L_recon` (scalar) | — | `losses/acga_losses.py :: reconstruction_loss` | Negative-sampling strategy/ratio for `E-` not specified — see ambiguity log. |
| (25) | Discriminator | `z ∈ R^{K}` | `D(z) ∈ [0,1]` | Two FC layers (stated); widths not specified | `models/acga/discriminator.py :: Discriminator.forward` | `Sigmoid(W2·GELU(W1 z + b1) + b2)`. |
| (26) | Adversarial (Wasserstein-style) loss | `z ~ p_z = N(0,I)`, `z ~ q(Z\|X,A)` (encoder output) | `L_adv` (scalar) | — | `losses/acga_losses.py :: adversarial_loss` | Paper calls this a Wasserstein-distance form but does not specify weight clipping / gradient penalty / critic update ratio — see ambiguity log. |
| (27) | HGN-EC initial state formation | `A ∈ R^{N×N}`, `X ∈ R^{N×D}` | `state = [X, aggregated]` | — | `models/hgn_ec/state_init.py :: build_initial_state` | `aggregated = A·X`; `state` is the concatenation `[X, aggregated]` along the feature axis. |
| (28) | Feature compression | `state`, `W_compress`, `b_compress` | `compressed` | Output dim not specified | `models/hgn_ec/compress.py :: FeatureCompressor.forward` | Single linear layer; target compressed dimensionality is not given numerically — see ambiguity log. |
| — | q/p initialization | `compressed` | `q`, `p` (both = `compressed`) | Same shape as `compressed` | `models/hgn_ec/state_init.py :: init_q_p` | Explicit paper statement: both `q` and `p` initialized to the same `compressed` vector. |
| (29) | Hamiltonian energy function | `q`, `p` | `H` (scalar/scalar-per-node) | `H_net` = MLP of GIN layers + activations; exact layer count/width not specified beyond the main "4 layers, hidden 16" GIN config, which may or may not be reused here | `models/hgn_ec/hamiltonian.py :: HamiltonianNet.forward` | `H = H_net(cat(q,p))`. Fig. 1 depicts this as a single "Hamiltonian GIN Layer" block, which may indicate fewer than 4 layers are used inside `H_net` — see ambiguity log. |
| (30) | Hamilton's equations (via autodiff) | `H(q,p)` | `q̇ = ∂H/∂p`, `ṗ = -∂H/∂q` | — | `models/hgn_ec/hamiltonian.py :: hamiltonian_gradients` | Computed via `torch.autograd.grad` on `H` w.r.t. `q` and `p`. |
| (31) | Symplectic Euler update | `q, p, q̇, ṗ, dt` | `q_new, p_new` | `dt` value **not specified anywhere in paper** | `models/hgn_ec/integrator.py :: symplectic_euler_step` | `p_new = p + dt·ṗ`; `q_new = q + dt·q̇`. Number of integration steps (single step vs. multiple) also not specified — see ambiguity log. |
| (32) | State restoration | `q_new`, `W_restore`, `b_restore` | `q_final` | Restored to "original dimensionality" (i.e., matches `state`'s or the original prompt's dim — not numerically specified) | `models/hgn_ec/restore.py :: FeatureRestorer.forward` | Single linear layer, inverse in spirit to Eq. 28's compression. |
| (33) | Energy conservation loss | `H_initial` (energy at initial `q,p`), `H_final` (energy after update) | `L_energy` (scalar) | `n` = number of averaged elements (not defined numerically) | `losses/hgn_ec_losses.py :: energy_conservation_loss` | MSE between initial and final Hamiltonian energy values. |
| (34) | Total loss | `L_CE, L_recon, L_adv, L_energy, λ1, λ2, λ3` | `L_total` (scalar) | `λ1=λ2=λ3=0.04` per Section V.B, but see ambiguity log re: whether λ1/λ2 vs λ3 truly share the same value | `losses/total_loss.py :: total_loss` | Weighted sum: `L_CE + λ1·L_recon + λ2·L_adv + λ3·L_energy`. |

## Notes on equations appearing twice

- Eq. (13) and Eqs. (19)–(21) describe the *same* GIN node-update rule — once as a single compact
  formula (13), once decomposed into aggregate/combine/transform steps (19–21). They should be
  implemented as a single `GINLayer` module reused everywhere a GIN layer is required (main GIN
  block, Eq. 22's ACGA encoder, and the "Hamiltonian GIN Layer" inside Eq. 29's `H_net`), consistent
  with rule #8 ("do not silently correct inconsistencies") — the paper's re-statement is treated as
  emphasis, not two different modules.
