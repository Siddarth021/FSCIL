# Ambiguity Log

Per project rule #5, every unclear, inconsistent, or underspecified point discovered while reading
the paper is recorded here. **None of these are resolved by guessing.** Each entry states exactly
what the paper says (or fails to say), why it is ambiguous, and what decision is deferred until the
project maintainer resolves it.

Status legend: `OPEN` = unresolved, blocks implementation of the affected component;
`OPEN-NONBLOCKING` = unresolved but a paper-stated default/description exists that can be
implemented literally while flagging the ambiguity in code comments.

---

## A1. ACGA vs. ARGA naming inconsistency — `OPEN-NONBLOCKING`

- The module is introduced, named, and used consistently as **ACGA** ("Adversarially Constrained
  Graph Autoencoder") in: the Abstract, Section II.C, Section IV (heading "C. ACGA"), Section IV.C
  body text, Section IV.E (loss description), Table V, Section V.F, and the Conclusion
  ("Adversarially-Constrained Graph Autoencoders (ACGA)").
- However, in **Section IV.B** (the GIN subsection), the paper states: *"Textual and visual
  learnable prompts are processed through the Graph Isomorphism Network (GIN) module to construct
  node feature matrices X ∈ R^{N×D} and adjacency matrices A ∈ R^{N×N}, respectively, which are
  passed into the following **ARGA** module."*
- Additionally, **Section IV.D** (HGN-EC) states: *"HGN-EC receives the node feature matrix X ∈
  R^{N×D} and the adjacency matrix A ∈ R^{N×N} from the **ARGA** module."*
- "ARGA" (Adversarially Regularized Graph Autoencoder) is also the name of a real, distinct prior
  method cited by the paper as reference [48] ("Adversarially regularized graph autoencoder for
  graph embedding," Pan et al. 2018), which the paper explicitly says it "draws inspiration from."
- **It is unclear whether:**
  (a) "ARGA" in Sections IV.B and IV.D is a typo for "ACGA" (the paper's own module), or
  (b) the paper intends GIN's output to route to a *different, unnamed* module modeled on the cited
  ARGA reference before or in addition to ACGA, or
  (c) ACGA and "the ARGA module" are meant to be read as the same component but the paper is
  inconsistently naming it as a nod to its inspiration.
- **Decision deferred.** For this documentation phase, all narrative text in `paper_spec.md` and
  `equation_mapping.md` treats "ARGA" as referring to the same module described in Section IV.C as
  ACGA (since that is the only graph-autoencoder module defined in the paper, and Fig. 1 shows only
  one such module positioned exactly where "ARGA" is described as feeding data), but this is flagged
  rather than silently corrected. Implementation should not merge or rename anything until this is
  confirmed by the maintainer.

## A2. HGN-EC vs. HGN-GC naming inconsistency — `OPEN-NONBLOCKING`

- The module is introduced and used as **HGN-EC** ("Hamiltonian Graph Network with Energy
  Conservation") in: the Abstract, Title's index terms, Section II.B, Section IV (heading "D.
  HGN-EC"), Section IV.D body text, Section IV.E, Fig. 1's "HGN-EC Module" label, and Sections V.C
  and V.F body text.
- However, **Table V's column header** literally reads **"HGN-GC"** (not "HGN-EC").
- Separately, the **Conclusion** section describes the same component as "Hamiltonian Graph
  Networks **with Graph Convolution** (HGN-EC)" — i.e., it keeps the abbreviation "HGN-EC" but
  expands the "-EC" as "Graph Convolution" rather than "Energy Conservation," which is internally
  inconsistent with the Abstract's expansion of the same abbreviation.
- **It is unclear** whether "HGN-GC" in Table V is a typesetting typo for "HGN-EC," or whether the
  paper intends "HGN-EC" and "HGN-GC" to refer to distinct sub-configurations of the same module
  (e.g., "energy conservation" mode vs. "graph convolution" mode) that are not otherwise described
  anywhere else in the paper.
- **Decision deferred.** Table V is transcribed in `paper_spec.md` with the literal column header
  "HGN-GC" as printed, while treating it as referring to the HGN-EC module defined in Section IV.D
  for interpretive purposes, consistent with the fact that no other component named "HGN-GC" is ever
  defined. This should not be silently normalized in code/config naming without maintainer sign-off.

## A3. Unspecified tensor dimensions — `OPEN`

The following dimensions are referenced symbolically in equations but never given numeric values
anywhere in the paper text, tables, or figure:

- `d` — CLIP token embedding dimension (used in Eqs. 1, 2, 5, 7, 9, 10).
- `d_e` — final projected text/vision embedding dimension (`W_T ∈ R^{d_e×d}`, `W_V ∈ R^{d_e×d}`,
  Eqs. 6, 8).
- `d_k` — per-head query/key dimension in multi-head attention (Eq. 5).
- `L` — number of Transformer/prompt layers (used symbolically throughout, e.g. `G ∈ R^{L×M×d}`).
- `N` — number of graph nodes (used throughout GIN/ACGA/HGN-EC; note Section IV.C explicitly says
  "N and K are independent, with no functional relationship between them" but never gives N or K a
  value).
- `D` — node feature dimension entering GIN/HGN-EC (`X ∈ R^{N×D}`).
- `K` — ACGA latent space dimension (`Z ∈ R^{N×K}`).
- Output dimensionality of the HGN-EC "Feature Compress" linear layer (Eq. 28).
- Hidden width / layer count of the discriminator's two fully-connected layers (Eq. 25) beyond
  "two fully-connected layers."
- Hidden width / layer count of `H_net` inside the Hamiltonian energy function (Eq. 29) beyond "an
  MLP consisting of GIN layers and activation functions" — it is not stated whether this reuses the
  main GIN's "4 layers, hidden dim 16" configuration or has its own separate configuration.
- **Decision deferred.** No default values will be invented. These must either be found in the
  authors' released code (https://github.com/aries-yqian/ACHG-CLIP, referenced in the Abstract but
  not fetched or inspected as part of this documentation phase per project rule #2 — the uploaded
  paper is the primary source) or explicitly supplied by the maintainer before implementation of the
  affected modules can proceed.

## A4. Unspecified time-step value `dt` — `OPEN`

- Eq. (31) (Symplectic Euler update) uses a time-step `dt`, described only as "the time step size."
- No numeric value, default, or search range for `dt` is given anywhere in the paper (not in
  Section V.B's implementation details, not in the sensitivity study of Section V.D).
- It is also not stated whether the Symplectic Euler update (Eq. 31) is applied for a single step or
  iterated for multiple steps per forward pass.
- **Decision deferred** — do not assume a default `dt`.

## A5. Prompt insertion/update mechanism is inconsistently described — `OPEN`

- Eq. (9) (text prompts): "Prompts are inserted into the text embeddings at each layer,"
  formalized as `X = [X_[CLS], g^(l), X_tokens]` — a concatenation.
- Eq. (10) (vision prompts): uses the identical concatenation notation
  `X = [X_[CLS], gV^(l), X_patches]`, but the paper then adds the sentence *"Prompts directly
  replace the input of each layer,"* immediately after Eq. (10), which contradicts a
  concatenation-style formula (a "replace" operation would not extend sequence length, whereas the
  bracket notation depicts concatenation, which does).
- It is unclear whether:
  (a) "replace" is a typo/loose wording for "are inserted into" (matching the concatenation
      formula, and matching how text prompts are described), or
  (b) vision prompts genuinely follow a different, replace-based insertion mechanism from text
      prompts (e.g., replacing certain patch tokens rather than appending new ones), which would
      require a different implementation from the text-prompt path.
- Separately, the mechanism by which the **HGN-EC module's single output `q_final`** (which appears
  to represent an updated prompt for one node, or possibly the whole graph) is redistributed back
  into the **per-layer, per-modality prompt tensors** `G ∈ R^{L×M×d}` / `GV ∈ R^{L×M×d}` is not
  described. Figure 1 shows two "MLP" blocks between the "Learnable Prompts" box and the GIN
  modules/encoders, which are never explained in the running text — it is unclear whether these
  MLPs perform the shape transformation from the graph-processed representation back to the
  `L×M×d` prompt tensor, or whether they serve a different purpose (e.g., dimensionality reduction
  before the GIN, projection between graph node space and prompt space, or both, at different
  points in the pipeline).
- **Decision deferred** — no insertion mechanism or MLP role is assumed; must be resolved before
  implementing `models/prompts/`.

## A6. Ambiguous definition of a graph "node" for the learnable-prompt graphs — `OPEN`

- Section V.D.4 states the number of learnable prompts `M = 1` is optimal and used in the final
  model.
- Section IV.B constructs a graph with node feature matrix `X ∈ R^{N×D}` from "textual and visual
  learnable prompts," implying nodes correspond to prompts (or to some structure built from
  prompts).
- With `M = 1` learnable prompt and (implicitly) `L` layers, it is unclear what `N` (number of
  graph nodes) actually equals in the final configuration:
  - If nodes = prompts within a single layer, then `N = M = 1`, which would make an adjacency
    matrix, cosine-similarity thresholding, and graph message-passing degenerate/meaningless (a
    single-node graph has no edges).
  - If nodes = one prompt-node per Transformer layer (i.e., `N = L`), the graph construction makes
    more sense, but this reading is not stated anywhere in the paper.
  - If nodes = something else entirely (e.g., per-class prototype nodes, or prompts across a batch
    of classes/sessions), this is also not described.
  - Fig. 1's toy diagram shows a 5-node graph (nodes labeled A–E) purely for illustration and does
    not indicate what a "node" corresponds to in the real prompt-graph setting.
- **Decision deferred** — this is a structurally significant ambiguity that affects the entire
  graph-construction and GIN/ACGA/HGN-EC pipeline, and must be resolved (ideally by consulting the
  authors' code, per project rule #2's primacy of the paper notwithstanding — this decision belongs
  to the maintainer, not to this documentation phase) before any node-level tensor shapes can be
  implemented.

## A7. Optional "attention re-weighting" step in adjacency construction (Eq. 18) — is it used? — `OPEN`

- Eq. (18) is introduced with: *"Optionally, an attention mechanism … can be incorporated to
  enhance the model's expressiveness, resulting in a final adjacency matrix of Ã = Ã · attention."*
- The paper never states whether this optional step is actually used in the experiments that
  produced the reported results in Tables I–V, or whether it was tested and rejected, or simply
  offered as a possible extension.
- **Decision deferred** — implementation should expose this as a togglable config option, default
  **disabled**, until confirmed.

## A8. Negative-edge sampling strategy for reconstruction loss (Eq. 24) — `OPEN`

- `L_recon` sums over `E ∪ E-`, where `E-` is "the set of negatively sampled edges."
- No sampling ratio, sampling scheme (uniform random non-edges? degree-based negative sampling?),
  or resampling frequency (per epoch/per batch/fixed) is given.
- **Decision deferred.**

## A9. Adversarial training procedure details for `L_adv` (Eq. 26) — `OPEN`

- The paper calls `L_adv` a "Wasserstein distance" form "to enhance the stability of the training
  process," but does not specify:
  - Whether weight clipping or a gradient penalty (standard WGAN / WGAN-GP stabilization
    techniques) is used.
  - The number of discriminator update steps per encoder/generator update step.
  - The discriminator's optimizer/learning rate (whether shared with the main optimizer or
    separate).
- **Decision deferred.**

## A10. Loss coefficient assignment ambiguity (`λ1, λ2, λ3`, Eq. 34) — `OPEN-NONBLOCKING`

- Section V.B states: *"The coefficients assigned to the ACGA loss and the HGN-EC loss within the
  total loss function were set to 0.04."*
- The total loss (Eq. 34) has **three** coefficients: `λ1` (for `L_recon`), `λ2` (for `L_adv`), and
  `λ3` (for `L_energy`). "The ACGA loss" plausibly refers to both `λ1` and `λ2` (since ACGA produces
  both a reconstruction loss and an adversarial loss), and "the HGN-EC loss" to `λ3`.
- It is not stated explicitly whether this means `λ1 = λ2 = λ3 = 0.04` (all three set to the same
  single value), or whether "the ACGA loss" is itself a single pre-combined quantity
  (e.g., `L_ACGA = L_recon + L_adv`, scaled once by a single coefficient before being added to
  `λ3·L_energy`), which would be a structurally different total-loss formula from Eq. (34) as
  literally written.
- **Decision deferred (non-blocking default):** for the documentation and config phase, all three
  of `λ1, λ2, λ3` are recorded as `0.04` in `reproduction_protocol.md`/configs, taking the literal
  reading of Eq. (34) with three independent coefficients all happening to share the reported value.
  This must be flagged as an assumption, not a confirmed paper detail, and revisited before
  publishing any results as "reproducing" the paper's numbers.

## A11. Table IV's `ΔA_last` vs. Tables I–III's `ΔPD` — are they the same metric? — `OPEN-NONBLOCKING`

- Tables I–III define `ΔPD` as "the difference between the classification accuracy of the model on
  the base session and the last incremental session" — an **intra-method** metric (smaller is
  better, indicates less forgetting for that method alone).
- Table IV defines `ΔA_last` as "the difference in the last incremental session accuracy between
  each method and the method at the top of the table" — an **inter-method comparative** metric
  (larger positive value = this method beats the top-of-table method by that margin at the last
  session).
- These are clearly different metrics despite superficially similar names/symbols (both involve
  `A_last`). This is recorded as a clarification rather than a blocking ambiguity, since both
  definitions are explicitly and separately stated by the paper — but the naming similarity is
  flagged to avoid accidental conflation in `evaluation/` code.

## A12. Unclear scope of "improved Transformer structures" (Section IV.A) — `OPEN-NONBLOCKING`

- The CLIP backbone's text/vision encoders are described as being "based on improved Transformer
  structures," but no citation, name, or specific modification (e.g., pre-LN vs. post-LN,
  RoPE/ALiBi positional schemes, activation function choice in the FFN) is given beyond the
  generic Eqs. (3)–(5).
- Since the CLIP backbone is frozen and (presumably) loaded from a pretrained checkpoint rather
  than trained from scratch, this ambiguity is **non-blocking** for training but **blocking** for
  selecting which pretrained CLIP checkpoint/architecture to instantiate, which is not named
  anywhere in the paper (no ViT-B/32, ViT-L/14, RN50, etc. designation is given).
- **Decision deferred** — no CLIP checkpoint variant is assumed.

## A13. Unspecified positional-encoding scheme (`P_T`, `P_V`) — `OPEN-NONBLOCKING`

- Eqs. (2) and (7) introduce positional encoding matrices `P_T`, `P_V` without specifying whether
  they are learned or fixed (e.g., sinusoidal), consistent with standard CLIP but not explicitly
  re-stated by this paper.

## A14. `n` in the energy-conservation loss (Eq. 33) — `OPEN-NONBLOCKING`

- `L_energy = (1/n) Σ_{i=1}^{n} (H_initial,i - H_final,i)^2` — `n` is used as the MSE normalization
  count but is not defined (e.g., whether it is the batch size, the number of graph nodes `N`, or
  the number of energy-scalar entries if `H` is not a single scalar per graph).
- Relatedly, it is not explicitly stated whether `H` (Eq. 29's Hamiltonian) is a single scalar for
  the whole graph or a per-node vector — the notation `(H_initial,i - H_final,i)` with an index `i`
  running to `n` suggests `H` may be a vector, which is not otherwise stated.
- **Decision deferred.**

## Summary of blocking vs. non-blocking items

| ID | Topic | Status |
|---|---|---|
| A1 | ACGA vs ARGA naming | OPEN-NONBLOCKING |
| A2 | HGN-EC vs HGN-GC naming | OPEN-NONBLOCKING |
| A3 | Unspecified tensor dimensions (d, d_e, d_k, N, D, K, compress dim, discriminator/H_net width) | OPEN — blocks module implementation |
| A4 | Unspecified `dt` (Symplectic Euler time step) | OPEN — blocks HGN-EC implementation |
| A5 | Prompt insertion mechanism / MLP role (Fig. 1) inconsistency | OPEN — blocks prompt module implementation |
| A6 | Definition of a graph "node" given M=1 learnable prompt | OPEN — blocks entire graph pipeline |
| A7 | Optional attention re-weighting (Eq. 18) usage | OPEN — config default only |
| A8 | Negative-edge sampling strategy (Eq. 24) | OPEN — blocks ACGA loss implementation |
| A9 | WGAN-style training details (Eq. 26) | OPEN — blocks ACGA adversarial training implementation |
| A10 | λ1/λ2/λ3 coefficient assignment reading | OPEN-NONBLOCKING (default recorded, flagged) |
| A11 | ΔPD vs ΔA_last metric distinction | OPEN-NONBLOCKING (clarified, not blocking) |
| A12 | CLIP backbone checkpoint/variant not named | OPEN — blocks backbone instantiation |
| A13 | Positional encoding scheme unspecified | OPEN-NONBLOCKING |
| A14 | `n` and vector-vs-scalar nature of H in energy loss | OPEN — blocks energy loss implementation |
