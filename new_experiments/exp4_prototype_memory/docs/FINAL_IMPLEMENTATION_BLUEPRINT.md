# FINAL IMPLEMENTATION BLUEPRINT — ACHG-CLIP

This document is the authoritative software-architecture handoff. It builds on
`docs/FINAL_RESEARCH_DECISIONS.md` (frozen scientific specification) and does not reopen any
resolved scientific decision except the five blockers explicitly targeted below. A separate coding
agent should be able to implement the project from this document alone, without making further
scientific judgment calls.

No code is written in this document.

---

## PART 0 — TARGETED BLOCKER-RESOLUTION PASS

Scope: only the five TRUE BLOCKERS from `FINAL_RESEARCH_DECISIONS.md` Part E. All other frozen
decisions are unchanged. CPE-CLIP was re-consulted directly (arXiv:2303.04751 / ICCVW 2023 text,
via web search of the paper's own equations/prose, not merely its abstract) since it is the most
relevant reference ACHG-CLIP cites for prompt mechanics.

### Blocker 1 — Graph node definition / N

PAPER EVIDENCE: §IV.B: "Textual and visual learnable prompts are processed through the GIN module
to construct node feature matrices X∈R^{N×D}... respectively." §V.D.4: M=1 optimal. No sentence
anywhere equates N to M, L, or L×M.
REFERENCE EVIDENCE: CPE-CLIP has no graph/node structure at all — it is pure prompt-tuning with no
GNN component. It cannot resolve this by structural analogy. ARGA (Pan et al. 2018) and the
Hamiltonian references likewise define generic graph-node formalisms without giving ACHG-CLIP's
specific N. No usable reference evidence exists for this blocker.
FINAL DECISION: **IMPLEMENTATION-CHOICE.** Adopt `N = L` (one graph node per Transformer/prompt
layer, per modality), i.e., `X_text ∈ R^{L×D}`, `X_vision ∈ R^{L×D}`. Rationale: with `M=1`,
`G ∈ R^{L×M×d} = R^{L×1×d}` naturally decomposes along its `L` axis into `L` per-layer prompt
vectors — the only decomposition of the existing prompt tensor that yields a non-degenerate,
multi-node graph without inventing new state. The alternative (`N=M=1`) is provably degenerate
(single-node graph, no edges, defeats the entire GIN/adjacency machinery) and is therefore rejected
as inconsistent with the paper's own emphasis on graph structure. This is a substantive reproduction
risk, isolated behind `configs/model/graph.yaml: num_nodes_mode: per_layer`.
EVIDENCE TYPE: IMPLEMENTATION-CHOICE (rejection of the degenerate alternative is JUSTIFIED-INFERENCE;
the specific choice of N=L beyond "non-degenerate" is not otherwise evidenced).
CONFIDENCE: Low.

### Blocker 2 — Vision prompt insertion (concatenate vs. replace)

PAPER EVIDENCE: Eq.9 (text) and Eq.10 (vision) both use identical concatenation notation
`[X_CLS, prompt, X_seq]`. Immediately after Eq.10, prose states vision prompts "directly replace the
input of each layer."
REFERENCE EVIDENCE (new, from direct CPE-CLIP text): CPE-CLIP explicitly assigns **replacement to
the language/text branch** and **accumulation (i.e., concatenation building up across layers) to the
vision branch** — the exact opposite modality assignment from ACHG-CLIP's prose. CPE-CLIP's own
words: "an accumulation method, as an alternative to the replacement method used in the language
branch" (vision = accumulate, language = replace). If ACHG-CLIP's "replace" sentence were a direct,
correctly-labeled borrowing from CPE-CLIP, it should have been attached to the *text* prompts, not
vision. This is concrete, citable evidence that ACHG-CLIP's modality-role assignment for "replace" is
inconsistent with its own most relevant reference, strengthening (but not proving) the reading that
the sentence is a mislabeling/editing error rather than an intentional divergent design choice.
FINAL DECISION: **IMPLEMENTATION-CHOICE.** Implement both text and vision prompt insertion as
concatenation (matching Eq.9 and Eq.10's literal, identical notation), which is also the reading
consistent with CPE-CLIP's actual vision-branch design (accumulation ≈ concatenation-based growth).
Flag the "replace" sentence in code comments as an apparent, evidence-supported but unconfirmed
paper inconsistency; do not implement a literal token-replacement path for vision prompts.
EVIDENCE TYPE: JUSTIFIED-INFERENCE, informed by REFERENCE-FACT (CPE-CLIP's actual role assignment).
CONFIDENCE: Medium (higher than the previous pass, due to the new specific CPE-CLIP mismatch).

### Blocker 3 — Figure-1 MLP blocks

PAPER EVIDENCE: Fig. 1 shows two unlabeled-in-prose MLP boxes between "Learnable Prompts" and the
GIN modules (one per modality per prior figure reading). No sentence in §IV mentions them.
REFERENCE EVIDENCE (new): CPE-CLIP bridges its language G-Prompt and vision prompts with an explicit
**learnable linear projection** `f^PROJ: R^{d_NLP} → R^{d_CV}` ("we bridge the gap between language
and vision prompts by explicitly expressing the latter as a function of the former"). This confirms
the *general pattern* of a learned projection sitting between a prompt representation and a
downstream consumer is well-precedented in the closest cited reference — but CPE-CLIP's version is
(a) a single **linear** layer, not a multi-layer MLP, and (b) bridges *text→vision* prompts directly,
not *prompt-space→graph-node-space* within a single modality. ACHG-CLIP's own diagram explicitly
labels these blocks "MLP" (implying more than one layer / nonlinearity), and shows **two** of them
(one per modality) rather than one cross-modal bridge — a different shape of solution than CPE-CLIP's.
FINAL DECISION: **IMPLEMENTATION-CHOICE.** Interpret the two MLP blocks as per-modality projections
from prompt space to the GIN's expected node-feature space: `MLP_text: R^{d} → R^{D}`,
`MLP_vision: R^{d} → R^{D}`, applied to each layer's prompt vector before graph construction (i.e.,
immediately upstream of Eq.13's `X`). This is the most direct interpretation of their position in
Fig. 1 (between "Learnable Prompts" and "GIN") and is a *different* function from CPE-CLIP's
cross-modal bridge, which is explicitly noted as a difference, not a confirmation. Two internal
linear layers with GELU (mirroring the MLP composition explicitly given for ACGA's encoder, Eq.22, as
the paper's only fully-specified MLP recipe) are used as the concrete default.
EVIDENCE TYPE: IMPLEMENTATION-CHOICE (CPE-CLIP evidence establishes plausibility of *a* prompt-space
bridging projection existing in this family of models, not this specific one).
CONFIDENCE: Low.

### Blocker 4 — ACGA → HGN-EC data connection

PAPER EVIDENCE: §IV.D states HGN-EC receives `X∈R^{N×D}` and `A∈R^{N×N}` "from the ARGA module" —
identical symbols/shapes to GIN's own output (§IV.B), not ACGA's defined outputs `Z∈R^{N×K}`,
`Â∈R^{N×N}`. Fig. 1's caption describes a serial GIN→ACGA→(compress/Hamiltonian/restore) flow.
REFERENCE EVIDENCE: Not applicable — this is an ACHG-CLIP-internal wiring question about its own
novel ACGA/HGN-EC modules; no cited reference (CPE-CLIP, ARGA, HNN) defines this connection, since
none of them combine a graph autoencoder with a Hamiltonian module in series. No reference resolves
this.
FINAL DECISION: **IMPLEMENTATION-CHOICE.** Adopt the reading that is literally consistent with
§IV.D's stated inputs without requiring a second unexplained typo: ACGA computes `Z` and `Â` **only**
for its own loss terms (`L_recon`, `L_adv`); the node features `X` and adjacency `A` that GIN produced
pass through to HGN-EC **unchanged** (ACGA acts as a parallel auxiliary/regularization head off the
same `(X,A)`, not as a transformation stage that HGN-EC consumes downstream of). This preserves
Fig. 1's left-to-right visual layout (GIN feeds both ACGA and HGN-EC) while respecting §IV.D's literal
`(X,A)` input claim.
EVIDENCE TYPE: IMPLEMENTATION-CHOICE.
CONFIDENCE: Low.

### Blocker 5 — Per-modality vs. joint ACGA/HGN-EC processing

PAPER EVIDENCE: §IV.B uses "respectively" (dual per-modality GIN processing, confirmed). §IV.C
(ACGA) and §IV.D (HGN-EC) never repeat equivalent per-modality language, using singular `X, A, Z, H`
throughout. Fig. 1 shows one ACGA-Module box and one HGN-EC-Module box.
REFERENCE EVIDENCE: Not applicable — CPE-CLIP has no ACGA/HGN-EC analogue; the surgical
visual-language reference (already logged in `research_control_log.md` §15) supports dual
text/vision graphs in general but not specifically whether a *shared downstream module* is applied
twice or graphs are fused.
FINAL DECISION: **IMPLEMENTATION-CHOICE.** Adopt option A (fully independent per-modality
processing) with **shared weights**: one set of ACGA parameters and one set of HGN-EC parameters,
each invoked twice per training step (once on the text graph, once on the vision graph), producing
two independent `L_recon`/`L_adv`/`L_energy` contributions that are summed before weighting by
`λ1,λ2,λ3` in Eq.34. Rejected alternatives: fusing the two graphs before ACGA/HGN-EC would require an
unstated fusion operator (concatenation? cross-attention? no such operation appears in §IV.C–D), so
options B/C are not adopted absent evidence. Singular Fig.1 boxes are read as "one weight-shared
module, diagrammed once" rather than as evidence of joint processing.
EVIDENCE TYPE: IMPLEMENTATION-CHOICE.
CONFIDENCE: Low-Medium.

### Summary of Part 0 outcomes

None of the five blockers is resolved to PAPER-FACT or REFERENCE-FACT confidence. All five are
now converted into **explicit, individually-justified IMPLEMENTATION-CHOICEs** so that the blueprint
below can specify concrete tensor shapes and data flow. Blockers 2 and 3 benefited from new,
specific CPE-CLIP textual evidence (raising blocker 2 to Medium confidence); blockers 1, 4, and 5
have no usable reference evidence and remain Low confidence, IMPLEMENTATION-CHOICE only. **These
choices are reproduction risks and must be revisited if the authors' code or errata become
available.** They are recorded once here as the canonical decisions the rest of this blueprint
depends on; they are not re-justified in every section below.

---

## PART 1 — COMPLETE PROJECT STRUCTURE

```text
ACHG-CLIP/
├── configs/
│   ├── model/
│   │   ├── clip_backbone.yaml       # IMPLEMENTATION-CHOICE: CLIP variant, d, d_e, d_k
│   │   ├── prompts.yaml             # PAPER-FACT: M=1, L; IMPLEMENTATION-CHOICE: insertion=concat (Blocker 2)
│   │   ├── graph.yaml               # PAPER-FACT: threshold=0.8, cosine sim; IMPLEMENTATION-CHOICE: N=L (Blocker 1)
│   │   ├── gin.yaml                 # PAPER-FACT: layers=4, hidden=16
│   │   ├── mlp_bridge.yaml          # IMPLEMENTATION-CHOICE: Fig.1 MLP blocks (Blocker 3)
│   │   ├── acga.yaml                # PAPER-FACT: encoder/decoder form; IMPLEMENTATION-CHOICE: K, discriminator width
│   │   └── hgn_ec.yaml              # PAPER-FACT: q/p/H/Symplectic Euler form; IMPLEMENTATION-CHOICE: dt, steps, compress dim
│   ├── optim/
│   │   └── optim.yaml               # PAPER-FACT: Lion, lr=0.000325, wd=1e-3, grad_accum=3, clip=4.0; IMPLEMENTATION-CHOICE: scheduler warmup/restart
│   ├── loss/
│   │   └── loss.yaml                # PAPER-FACT: λ1=λ2=λ3=0.04 (flagged assumption, A10)
│   ├── data/
│   │   ├── cifar100.yaml            # PAPER-FACT: 60/40, 8 sessions, 5-way-5-shot
│   │   ├── mini_imagenet.yaml       # PAPER-FACT: 60/40, 8 sessions, 5-way-5-shot
│   │   └── cub200.yaml              # PAPER-FACT: 100/100, 10 sessions, 10-way-5-shot
│   └── targets/
│       └── reported_results.yaml    # read-only reference values; never asserted as our output
├── data/
│   ├── datasets/                    # dataset classes (CIFAR100FSCIL, MiniImageNetFSCIL, CUB200FSCIL)
│   ├── splits/                      # IMPLEMENTATION-CHOICE: standard published FSCIL class-split files
│   └── loaders.py                   # session-aware DataLoader construction (base vs incremental)
├── models/
│   ├── clip/
│   │   ├── text_encoder.py          # Eqs. 1-6
│   │   ├── vision_encoder.py        # Eqs. 7-8
│   │   ├── transformer_block.py     # Eqs. 3-5
│   │   └── attention.py             # Eq. 5
│   ├── prompts/
│   │   ├── text_prompt.py           # Eq. 9 (concatenation, per Blocker 2 decision)
│   │   ├── vision_prompt.py         # Eq. 10 (concatenation, per Blocker 2 decision)
│   │   └── mlp_bridge.py            # Fig.1 MLP blocks (per Blocker 3 decision)
│   ├── graph/
│   │   ├── node_builder.py          # constructs N=L nodes per modality (per Blocker 1 decision)
│   │   └── adjacency.py             # Eqs. 14-18
│   ├── gnn/
│   │   └── gin_layer.py             # Eqs. 13, 19-21 — single reusable GINLayer
│   ├── acga/
│   │   ├── encoder.py               # Eq. 22
│   │   ├── decoder.py               # Eq. 23
│   │   └── discriminator.py         # Eq. 25
│   ├── hgn_ec/
│   │   ├── state_init.py            # Eq. 27, q/p init
│   │   ├── compress.py              # Eq. 28
│   │   ├── hamiltonian.py           # Eq. 29 (H_net, reuses GINLayer)
│   │   ├── integrator.py            # Eqs. 30-31 (Symplectic Euler)
│   │   └── restore.py               # Eq. 32
│   └── achg_clip.py                 # top-level module wiring everything per Part 0 decisions
├── losses/
│   ├── contrastive_loss.py          # Eqs. 11-12
│   ├── acga_losses.py               # Eqs. 24, 26
│   ├── hgn_ec_losses.py             # Eq. 33
│   └── total_loss.py                # Eq. 34
├── training/
│   ├── base_session.py              # base-session training loop
│   ├── incremental_session.py       # incremental-session training loop
│   ├── freeze.py                    # CLIP-freezing / trainable-parameter bookkeeping
│   └── scheduler.py                 # CosineAnnealingWarmupRestarts wrapper
├── evaluation/
│   ├── metrics.py                   # A_base, A_last, Mean, ΔPD, ΔA_last (kept as two distinct functions, A11)
│   └── evaluator.py                 # per-session cumulative evaluation
├── utils/
│   ├── seed.py                      # reproduction-safety: seeding
│   ├── logging.py                   # structured per-epoch/per-session logging
│   ├── config_tracking.py           # PAPER-FACT vs IMPLEMENTATION-CHOICE provenance tagging
│   └── checkpoint.py                # checkpoint save/load + metadata
├── tests/
│   ├── test_tensor_shapes.py
│   ├── test_graph_construction.py
│   ├── test_gin.py
│   ├── test_acga.py
│   ├── test_hgn_ec.py
│   ├── test_prompt_insertion.py
│   ├── test_losses.py
│   ├── test_incremental_sessions.py
│   └── test_checkpoint.py
├── scripts/
│   ├── run_base_session.py
│   ├── run_incremental_session.py
│   ├── run_full_experiment.py       # base + all incremental sessions for one dataset
│   ├── smoke_test.py                # Part 10 synthetic smoke test
│   └── evaluate.py
├── checkpoints/                     # gitignored; populated at runtime
├── results/                         # gitignored; per-run logs/metrics/configs snapshot
└── docs/                            # this file + all prior research docs (unchanged)
```

---

## PART 2 — MODULE MAPPING

| Paper component | File | Class/function | Inputs | Outputs | Shapes (per Part 0/config) |
|---|---|---|---|---|---|
| CLIP text embedding+PE | `models/clip/text_encoder.py` | `TextEmbedding.forward` | token ids | `X` | `(B, n, d)` |
| CLIP text Transformer | `models/clip/transformer_block.py` | `TransformerBlock.forward` | `X` | `X''` | `(B, n, d)` |
| CLIP text projection | `models/clip/text_encoder.py` | `TextEncoder.forward` | `X^(L)` | `h*_T` | `(B, d_e)` |
| CLIP vision embed+PE | `models/clip/vision_encoder.py` | `VisionEmbedding.forward` | patches | `X` | `(B, m+1, d)` |
| CLIP vision projection | `models/clip/vision_encoder.py` | `VisionEncoder.forward` | `X^(L)_[CLS]` | `h*_V` | `(B, d_e)` |
| Text prompt insertion | `models/prompts/text_prompt.py` | `TextPromptInjector.forward` | `X, g^(l)` | `X` (concat) | `(B, n+M, d)` per layer |
| Vision prompt insertion | `models/prompts/vision_prompt.py` | `VisionPromptInjector.forward` | `X, gV^(l)` | `X` (concat, per Blocker 2) | `(B, m+1+M, d)` per layer |
| Fig.1 MLP bridge (text) | `models/prompts/mlp_bridge.py` | `PromptToNodeMLP.forward` (text) | `g^(l) ∈ R^d` | node feature `∈ R^D` | `(L, D)` per Blocker 3/1 |
| Fig.1 MLP bridge (vision) | `models/prompts/mlp_bridge.py` | `PromptToNodeMLP.forward` (vision) | `gV^(l) ∈ R^d` | node feature `∈ R^D` | `(L, D)` per Blocker 3/1 |
| Node/graph construction | `models/graph/node_builder.py`, `adjacency.py` | `build_nodes`, `cosine_similarity_matrix`, `threshold_binarize`, `symmetrize`, `normalize_adjacency` | MLP-bridged node features | `X∈R^{N×D}`, `A∈R^{N×N}` | `N=L` per modality (Blocker 1) |
| GIN layer (shared) | `models/gnn/gin_layer.py` | `GINLayer.forward` | `X, A, ε` | `X'` | `(N, D_hidden=16)`, 4 layers |
| ACGA encoder | `models/acga/encoder.py` | `ACGAEncoder.forward` | `X, A` | `Z∈R^{N×K}` | `K` = config (Issue 12) |
| ACGA decoder | `models/acga/decoder.py` | `InnerProductDecoder.forward` | `Z` | `Â∈R^{N×N}` | — |
| ACGA discriminator | `models/acga/discriminator.py` | `Discriminator.forward` | `z∈R^K` | `D(z)∈[0,1]` | 2 FC + GELU + sigmoid |
| HGN-EC state init | `models/hgn_ec/state_init.py` | `build_initial_state`, `init_q_p` | `X, A` (per Blocker 4: GIN's X,A, not ACGA's Z) | `state=[X,agg]`, `q=p=compressed` | `state ∈ R^{N×2D}` |
| HGN-EC compress | `models/hgn_ec/compress.py` | `FeatureCompressor.forward` | `state` | `compressed` | `R^{N×D_c}`, `D_c`=config |
| Hamiltonian net | `models/hgn_ec/hamiltonian.py` | `HamiltonianNet.forward`, `hamiltonian_gradients` | `q,p` | `H`, `q̇,ṗ` | `H` shape per config default (scalar-per-graph, Issue 19) |
| Symplectic Euler | `models/hgn_ec/integrator.py` | `symplectic_euler_step` | `q,p,q̇,ṗ,dt` | `q_new,p_new` | same shape as `q,p` |
| State restoration | `models/hgn_ec/restore.py` | `FeatureRestorer.forward` | `q_new` | `q_final` | restored to `state` dim (Issue 25 default reading) |
| Top-level wiring | `models/achg_clip.py` | `ACHGCLIP.forward` | image, text | logits, all loss components | see Part 4 |
| CE/contrastive loss | `losses/contrastive_loss.py` | `clip_similarity_matrix`, `clip_contrastive_loss` | `h*_V, h*_T` | `L_CE` | scalar |
| Reconstruction loss | `losses/acga_losses.py` | `reconstruction_loss` | `A, Â, E-` | `L_recon` | scalar |
| Adversarial loss | `losses/acga_losses.py` | `adversarial_loss` | `D(z_real), D(z_fake)` | `L_adv` | scalar |
| Energy loss | `losses/hgn_ec_losses.py` | `energy_conservation_loss` | `H_initial, H_final` | `L_energy` | scalar |
| Total loss | `losses/total_loss.py` | `total_loss` | all 4 losses, λ1,λ2,λ3 | `L_total` | scalar |
| FSCIL sessions | `training/base_session.py`, `incremental_session.py` | `run_base_session`, `run_incremental_session` | dataset, model, config | trained model, logs | — |
| Evaluation | `evaluation/evaluator.py` | `evaluate_session` | model, cumulative test set | per-session accuracy | — |

---

## PART 3 — TENSOR CONTRACT

| Name | Shape | Dtype | Device | Producer | Consumer |
|---|---|---|---|---|---|
| `tokens` | `(B, n)` | `int64` | GPU | tokenizer | `TextEmbedding` |
| `X_text_seq` | `(B, n, d)` | `float32` | GPU | `TextEmbedding` | `TransformerBlock` |
| `patches` | `(B, m, patch_dim)` | `float32` | GPU | patch extractor | `VisionEmbedding` |
| `X_vision_seq` | `(B, m+1, d)` | `float32` | GPU | `VisionEmbedding` | `TransformerBlock` |
| `G` (text prompt param) | `(L, M=1, d)` | `float32`, learnable | GPU | model init | `TextPromptInjector`, `PromptToNodeMLP` |
| `GV` (vision prompt param) | `(L, M=1, d)` | `float32`, learnable | GPU | model init | `VisionPromptInjector`, `PromptToNodeMLP` |
| `h*_T` | `(B, d_e)` | `float32` | GPU | `TextEncoder` | `clip_similarity_matrix` |
| `h*_V` | `(B, d_e)` | `float32` | GPU | `VisionEncoder` | `clip_similarity_matrix` |
| `X_node_text` | `(N=L, D)` | `float32` | GPU | `PromptToNodeMLP` (text) | `adjacency.py`, `GINLayer` |
| `X_node_vision` | `(N=L, D)` | `float32` | GPU | `PromptToNodeMLP` (vision) | `adjacency.py`, `GINLayer` |
| `A_text`, `A_vision` | `(L, L)` | `float32`, `{0,1}` pre-norm | GPU | `adjacency.py` | `GINLayer`, HGN-EC state init |
| `Z_text`, `Z_vision` | `(L, K)` | `float32` | GPU | `ACGAEncoder` | `InnerProductDecoder`, `Discriminator` |
| `Â_text`, `Â_vision` | `(L, L)` | `float32`, `[0,1]` | GPU | `InnerProductDecoder` | `reconstruction_loss` |
| `state_text`, `state_vision` | `(L, 2D)` | `float32` | GPU | `build_initial_state` (consumes GIN's `X,A` per Blocker 4) | `FeatureCompressor` |
| `compressed_*` | `(L, D_c)` | `float32` | GPU | `FeatureCompressor` | `init_q_p` |
| `q_*`, `p_*` | `(L, D_c)` | `float32` | GPU | `init_q_p` | `HamiltonianNet`, `symplectic_euler_step` |
| `H_initial`, `H_final` | config-dependent (default scalar per graph, i.e. `(1,)`, per Issue 19 default) | `float32` | GPU | `HamiltonianNet` | `energy_conservation_loss` |
| `q_final_*` | `(L, 2D)` (restored to `state` dim, per Issue 25 default reading) | `float32` | GPU | `FeatureRestorer` | reshaped into `G`/`GV` update (see Part 4 note) |
| `L_CE, L_recon, L_adv, L_energy, L_total` | `()` scalar | `float32` | GPU | respective loss fns | `total_loss`, optimizer |

**No undocumented reshaping rule:** every reshape between `(L, 2D)` (post-restoration) and
`(L, M=1, d)` (prompt-tensor form needed for feedback, per Blocker 3's node=layer choice) must go
through an explicitly named, tested function (`models/prompts/mlp_bridge.py`'s inverse path or a
dedicated `restore_to_prompt_shape` function) — this exact reshape is itself part of Blocker 3/26's
IMPLEMENTATION-CHOICE and must be unit-tested (`test_prompt_insertion.py`) for round-trip shape
consistency, not assumed to "just work."

---

## PART 4 — DATA FLOW

```text
image, text_tokens
  → CLIP frozen embed + PE                         (Eqs. 1-2, 7)
  → per-layer: [prompt_insert (Blocker 2: concat) → TransformerBlock]   (Eqs. 3-5, 9-10)  ×L
  → final-layer projection → h*_T, h*_V             (Eqs. 6, 8)
  → contrastive similarity + L_CE                   (Eqs. 11-12)

  [in parallel, once per modality — text and vision graphs built independently, per Blocker 5]
  G (or GV) prompt tensor (L, 1, d)
  → PromptToNodeMLP (Blocker 3)                      → X_node (L, D)
  → cosine similarity + threshold + symmetrize + normalize (Eqs. 14-18) → A (L, L)
  → GIN ×4 layers, hidden=16                         (Eqs. 13, 19-21)  → X_gin (L, D)
     ├──→ ACGA encoder → Z (L,K) → decoder → Â; discriminator → L_recon, L_adv   (Eqs. 22-26)
     │      [Blocker 4: Z/Â used ONLY for L_recon/L_adv, do NOT overwrite X_gin/A]
     └──→ HGN-EC: state_init(X_gin, A) → compress → q=p=compressed             (Eqs. 27-28, §IV.D.3)
             → H_net(q,p) = H_initial                                          (Eq. 29)
             → Hamilton gradients (autograd) → Symplectic Euler step(s) (dt: config)  (Eqs. 30-31)
             → H_net(q_new,p_new) = H_final → L_energy                          (Eq. 33)
             → restore(q_new) = q_final                                        (Eq. 32)
  → reshape q_final (L, 2D) → prompt-tensor form (L, 1, d) → update G / GV      ("Update" arrows, Fig.1)
  → L_total = L_CE + λ1·L_recon + λ2·L_adv + λ3·L_energy                       (Eq. 34)
  → backward, optimizer step (Lion), gradient clip, gradient accumulation
```

Note: this sequence is derived from Part 0's frozen blocker decisions, not assumed independently.
If any Part-0 decision changes (e.g., authors' code surfaces later), this section must be revised
before recoding.

---

## PART 5 — TRAINING FLOW (pseudocode)

### Base session
```text
load frozen CLIP backbone (config-selected variant)
initialize G, GV (L, 1, d), all GIN/ACGA/HGN-EC/MLP-bridge weights
freeze: all CLIP backbone parameters
trainable: G, GV, GIN weights (shared across ACGA-encoder/H_net reuse per Part 0), MLP bridges,
           ACGA decoder/discriminator, HGN-EC compress/restore/H_net-specific weights
for epoch in 1..3 (base_epochs, PAPER-FACT):
    for batch in D0 (batch_size=4, PAPER-FACT):
        forward pass (Part 4)
        compute L_total
        backward; accumulate gradients over 3 steps (PAPER-FACT); clip grad norm to 4.0 (PAPER-FACT)
        optimizer.step() every 3 accumulated batches
        scheduler.step()
save checkpoint: "session_00_base.ckpt" with full config snapshot + provenance tags
evaluate on D0's test set → A_0 (session-0 accuracy)
```

### Incremental session t (1..T)
```text
load checkpoint from session t-1
add new-class node features to the graph (append to X_node); keep old node features frozen
  (PAPER-FACT, §IV.B.1) — implemented as a boolean mask / requires_grad=False on old node rows
for epoch in 1..5 (incremental_epochs, PAPER-FACT):
    for batch in Dt (5-shot per new class, batch_size=4, PAPER-FACT):
        forward pass (Part 4) — restricted to available data per FSCIL constraints (§III.A):
          no replay of D0..D_{t-1} raw data (PAPER-FACT)
        compute L_total; backward; accumulate/clip/step as in base session
save checkpoint: "session_{t:02d}.ckpt"
evaluate on cumulative test set (all classes seen through session t) → A_t
```

### Evaluation
```text
for each session s in 0..T:
    load "session_{s:02d}.ckpt" (or continue in-memory if run end-to-end)
    run forward pass in eval mode (no dropout/no grad) over the cumulative test set for classes 0..s
    compute per-session accuracy A_s
after all sessions:
    Mean = average(A_0 ... A_T)               (PAPER-FACT definition)
    ΔPD = A_0 - A_T                            (PAPER-FACT definition; distinct function from ΔA_last, A11)
    write results/{dataset}/metrics.json (never overwrite configs/targets/reported_results.yaml)
```

### Checkpoint loading
```text
checkpoint contains: model state_dict, optimizer state_dict, scheduler state_dict,
                      current session index, graph node registry (which node rows are "old"/frozen),
                      full resolved config (with PAPER-FACT/IMPLEMENTATION-CHOICE tags), RNG state
load_checkpoint(path) → restores all of the above; raises if config hash mismatches expected
  architecture (prevents silently loading into a differently-shaped model)
```

### Model freezing/unfreezing
```text
freeze_clip_backbone(model)      # sets requires_grad=False on all CLIP weights, once, at init
freeze_old_graph_nodes(model, session_idx)  # called at the start of each incremental session
  before training: marks node rows belonging to sessions < session_idx as requires_grad=False
```

---

## PART 6 — CONFIGURATION

Every hyperparameter lives in `configs/`. Each YAML key is tagged in comments (and mirrored in
`utils/config_tracking.py`'s schema) as one of: `paper_fact`, `implementation_choice`,
`justified_inference`. Example (`configs/model/graph.yaml`):

```yaml
adjacency_threshold: 0.8          # paper_fact — Section V.D.2 / Fig. 3
cosine_similarity: true           # paper_fact — Eq. 14
optional_attention_reweight: false # paper_fact-default — Eq. 18 marked "optional", usage unconfirmed (A7)
num_nodes_mode: per_layer         # implementation_choice — Blocker 1, N = L
attention_reweight_enabled: false  # duplicate flag name avoided; single source of truth above
```

Nothing scientifically important is hard-coded in Python; all seven config groups in Part 1's
`configs/` tree are loaded once at startup and the *resolved* config (with defaults applied) is
snapshotted into every checkpoint and every `results/` run directory, per Part 8.

`configs/targets/reported_results.yaml` is read-only reference data (paper_spec.md §35's transcribed
numbers) and is never written to by training/evaluation code, consistent with
`reproduction_protocol.md` §15.4.

---

## PART 7 — TEST PLAN

| Test | File | Verifies | Expected shape/result |
|---|---|---|---|
| Tensor shapes end-to-end | `test_tensor_shapes.py` | full forward pass produces every tensor in Part 3's contract at the documented shape | no shape assertion failures for a synthetic batch |
| Graph construction | `test_graph_construction.py` | cosine similarity + threshold + symmetrize + normalize pipeline | `A` symmetric, `Ã` row-sums consistent with `D^{-1/2} Z D^{-1/2}` identity on a hand-built 4-node example |
| Cosine similarity | (same file) | Eq. 14 correctness | matches `torch.nn.functional.cosine_similarity` on known vectors |
| Thresholding | (same file) | Eq. 15 strict `>` | boundary value exactly at threshold → 0, not 1 |
| GIN | `test_gin.py` | Eq. 13 ≡ Eqs. 19-21 decomposition give identical output | max abs diff < 1e-6 on random input |
| ACGA reconstruction | `test_acga.py` | `Â = σ(ZZ^T)` shape/range | `Â ∈ [0,1]^{N×N}`, symmetric |
| Adversarial component | (same file) | discriminator output range, gradient flow to encoder | `D(z) ∈ [0,1]`; `Z.grad` is non-None after backward through `L_adv` |
| HGN-EC | `test_hgn_ec.py` | state_init/compress/restore shape round-trip | `q_final` shape matches Part 3's declared restored-state shape |
| Hamiltonian gradients | (same file) | `q̇=∂H/∂p`, `ṗ=-∂H/∂q` via autograd match finite-difference approximation | relative error < 1e-3 on a small synthetic `H_net` |
| Symplectic Euler | (same file) | Eq. 31 update arithmetic | exact match on hand-computed toy example for a given `dt` |
| Prompt insertion | `test_prompt_insertion.py` | Eq. 9/10 concatenation shapes (Blocker 2 decision); MLP-bridge round-trip shape (Blocker 3/Part 3 note) | sequence length grows by `M` per layer as expected; bridge round-trip shape matches `(L,1,d)` |
| Incremental classes | `test_incremental_sessions.py` | new node rows appended, old node rows `requires_grad=False` | node count increases by expected amount per session; old-row gradients are zero after backward |
| Loss calculation | `test_losses.py` | Eq. 34 weighted sum matches manual computation with λ1=λ2=λ3=0.04 default | exact match to float precision |
| Checkpoint loading | `test_checkpoint.py` | save→load round-trip preserves weights, optimizer state, node-freeze registry | loaded model produces identical forward-pass output to pre-save model on same input |

---

## PART 8 — REPRODUCTION SAFETY

- **Random seeds:** `utils/seed.py` sets `torch`, `numpy`, `random` seeds from
  `configs/*.yaml: seed` (default value itself is an IMPLEMENTATION-CHOICE, since
  `reproduction_protocol.md` §13 confirms the paper specifies none — recorded, not invented as fact).
- **Deterministic settings:** `torch.use_deterministic_algorithms(True)` where practical;
  `torch.backends.cudnn.benchmark=False` for reproducibility runs, with a documented performance
  trade-off noted (non-deterministic mode available via config flag for speed).
- **Logging:** structured JSONL per epoch/session (`utils/logging.py`), recording `L_total` and its
  four components plus per-session cumulative accuracy, per `reproduction_protocol.md` §14's
  reproduction-engineering convention (still flagged as convention, not paper fact).
- **Configuration saving:** the fully-resolved config (all defaults applied) is serialized into every
  `results/{run_id}/config.yaml` and every checkpoint, so any run's exact settings are recoverable.
- **Checkpoint metadata:** session index, dataset, git commit hash (if available), timestamp,
  resolved config hash.
- **Experiment IDs:** `{dataset}_{YYYYMMDD}_{HHMMSS}_{short-git-hash}` naming convention for
  `results/` subdirectories.
- **Paper-vs-choice parameter tracking:** `utils/config_tracking.py` validates, at startup, that every
  config key present in the resolved config carries one of the three provenance tags
  (`paper_fact`/`implementation_choice`/`justified_inference`); a run refuses to start if any key is
  untagged, preventing an untracked assumption from silently entering a training run.

---

## PART 9 — IMPLEMENTATION ORDER

| Stage | Files created | Dependencies | Tests | Expected result |
|---|---|---|---|---|
| 1. Config + tracking scaffolding | `configs/**`, `utils/config_tracking.py`, `utils/seed.py`, `utils/logging.py` | none | schema validation test (config loads, all keys tagged) | configs load and validate cleanly |
| 2. CLIP wrapper (frozen) | `models/clip/*.py` | Stage 1 | shape tests for `h*_T`, `h*_V` on a dummy pretrained backbone | correct `(B, d_e)` outputs; confirms zero grad on backbone params |
| 3. Prompt insertion (Blocker 2) | `models/prompts/text_prompt.py`, `vision_prompt.py` | Stage 2 | `test_prompt_insertion.py` (insertion part) | sequence length grows correctly per layer |
| 4. MLP bridge (Blocker 3) + node/graph construction (Blocker 1) | `models/prompts/mlp_bridge.py`, `models/graph/*.py` | Stage 3 | `test_graph_construction.py`, round-trip shape part of `test_prompt_insertion.py` | valid `X_node`, `A` for a synthetic `(L,1,d)` prompt tensor |
| 5. GIN | `models/gnn/gin_layer.py` | Stage 4 | `test_gin.py` | Eq.13 ≡ Eqs.19-21 |
| 6. ACGA | `models/acga/*.py`, `losses/acga_losses.py` | Stage 5 | `test_acga.py` | shapes + loss gradient flow |
| 7. HGN-EC (Blocker 4 wiring) | `models/hgn_ec/*.py`, `losses/hgn_ec_losses.py` | Stage 5 (consumes GIN's X,A directly, not ACGA's Z, per Blocker 4) | `test_hgn_ec.py` | Hamiltonian gradient correctness, Symplectic Euler correctness |
| 8. Top-level wiring (Blocker 5 per-modality loop) | `models/achg_clip.py`, `losses/contrastive_loss.py`, `losses/total_loss.py` | Stages 2-7 | `test_losses.py`, `test_tensor_shapes.py` (full pass) | full forward pass produces `L_total` |
| 9. Training loop + freezing | `training/*.py` | Stage 8 | manual short-run smoke check (Part 10) | one base-session epoch completes without error |
| 10. Incremental sessions + node freezing | extend `training/incremental_session.py`, `training/freeze.py` | Stage 9 | `test_incremental_sessions.py` | node count grows, old rows frozen |
| 11. Evaluation | `evaluation/*.py` | Stage 9 | manual check on synthetic labels | per-session accuracy computed, ΔPD/Mean correct |
| 12. Checkpointing | `utils/checkpoint.py` | Stage 9 | `test_checkpoint.py` | round-trip identical forward output |
| 13. Data pipeline (real datasets) | `data/**` | Stages 1-12 complete and tested on synthetic data | dataset-loading smoke test | correct base/incremental split sizes per `reproduction_protocol.md` |
| 14. Full experiment scripts | `scripts/*.py` | Stage 13 | end-to-end smoke test (Part 10) | full pipeline runs on tiny synthetic data |

Each stage's tests must pass before the next stage begins; this order deliberately builds and
validates the graph/ACGA/HGN-EC machinery (Stages 4–8, where all five blockers live) on **synthetic**
node tensors before any real CLIP/dataset integration, so that blocker-related reproduction risk is
isolated and testable independent of dataset/backbone concerns.

---

## PART 10 — SMOKE TEST

### Synthetic smoke test (`scripts/smoke_test.py`)
- Construct a tiny synthetic batch: `B=2` images (random tensors, shape matching config's `patch_dim`),
  `B=2` text token sequences (random ids within a small fake vocab), `L=3` (small number of layers for
  speed), `d=8`, `d_e=4`, `D=6`, `K=3` — **all IMPLEMENTATION-CHOICE synthetic values, not paper
  values**, purely to exercise shapes quickly.
- Use a randomly-initialized (not pretrained) tiny Transformer stand-in for CLIP, since the smoke test
  goal is pipeline shape/gradient-flow correctness, not accuracy.
- Run one full forward pass through the entire Part-4 data flow, for both text and vision graphs
  (Blocker 5's dual pass).
- Assert: every tensor in Part 3's contract appears with the correct shape; `L_total` is a finite
  scalar; `.backward()` succeeds; every trainable parameter (prompts, GIN, ACGA, HGN-EC, MLP bridges)
  has a non-None gradient; frozen CLIP backbone parameters have `None` or zero gradient.
- Runtime target: under 30 seconds on CPU.

### Smallest real-data experiment
- CIFAR-100, base session only, restricted to the first 5 base classes and ~20 images/class (a tiny
  slice, not the full 60-class/full-data base session), 1 epoch, to confirm real-data plumbing
  (tokenizer, image preprocessing, dataset class, DataLoader) works end-to-end before committing to a
  full run. This is explicitly a plumbing check, not a scientific result — its accuracy number is
  meaningless and must not be logged into `results/` alongside real experiment runs.

---

## PART 11 — REPRODUCTION EXPERIMENT PLAN

Order: **CIFAR-100 → miniImageNet → CUB-200-2011** (increasing image resolution / task difficulty,
matching the paper's own presentation order).

For each dataset:
```text
1. scripts/run_base_session.py --config configs/data/{dataset}.yaml
     → checkpoints/{dataset}/session_00_base.ckpt
     → results/{dataset}/{run_id}/session_00_metrics.json
2. for t in 1..T (8 for CIFAR-100/miniImageNet, 10 for CUB-200-2011):
     scripts/run_incremental_session.py --dataset {dataset} --session {t} \
         --resume checkpoints/{dataset}/session_{t-1:02d}.ckpt
         → checkpoints/{dataset}/session_{t:02d}.ckpt
         → results/{dataset}/{run_id}/session_{t:02d}_metrics.json
3. scripts/evaluate.py --dataset {dataset} --run_id {run_id}
     → results/{dataset}/{run_id}/summary.json  (Mean, ΔPD, per-session accuracy curve)
```

**Explicit non-claim:** because `dt`, `N`'s definition, the Fig.1 MLP-block role, the ACGA→HGN-EC
wiring, per-modality/joint scope, CLIP backbone variant, `K`, discriminator/H_net widths, negative-
edge sampling strategy, WGAN stabilization details, and dataset class-split identity are all
IMPLEMENTATION-CHOICEs rather than confirmed paper facts (Parts 0 and Part E of
`FINAL_RESEARCH_DECISIONS.md`), **no run of this codebase may be described as "reproducing" the
paper's reported Mean/ΔPD numbers (82.30/9.72, 85.05/8.42, 69.54/17.67) until/unless those choices
are validated against author code or errata.** Results are to be reported as "our
implementation's results under the following documented assumptions: [list]," compared against but
never conflated with `configs/targets/reported_results.yaml`.

---

## FILE CREATED: `docs/FINAL_IMPLEMENTATION_BLUEPRINT.md`
