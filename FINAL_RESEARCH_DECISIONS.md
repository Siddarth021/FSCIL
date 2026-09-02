# FINAL RESEARCH DECISIONS — ACHG-CLIP Reproduction

Status: **Final research pass.** No code written. This document is the frozen scientific
specification handoff. It supersedes nothing in `paper_spec.md`, `equation_mapping.md`,
`ambiguity_log.md`, or `reproduction_protocol.md` — those remain the detailed record — but this
file is the single authoritative decision matrix to build from.

Evidence labels used throughout: **PAPER-FACT / REFERENCE-FACT / JUSTIFIED-INFERENCE /
IMPLEMENTATION-CHOICE / UNRESOLVED.**

---

## PART A — DECISION MATRIX (26 architectural issues)

| ID | Issue | Final Decision | Evidence | Evidence Type | Confidence | Impl. Impact |
|----|-------|-----------------|----------|----------------|------------|---------------|
| 1 | Exact CLIP backbone | **UNRESOLVED.** No ViT-B/32, ViT-L/14, RN50, or other designation anywhere in text, tables, or figure. "Improved Transformer structures" (§IV.A) is undefined. | Whole paper searched; no variant name found. | — | High confidence that it is genuinely absent (not just missed) | Non-blocking — isolate behind `configs/model/clip_backbone.yaml`, document as IMPLEMENTATION-CHOICE when set |
| 2 | All tensor dimensions | **PARTIALLY RESOLVED.** Only two numeric dims exist anywhere: GIN layers=4, GIN hidden=16 (§V.B). `d, d_e, d_k, L, N, D, K`, compress-output dim, discriminator width, `H_net` width are all symbolic-only. | §V.B (numeric); Eqs. 1–29 (symbolic-only for the rest) | PAPER-FACT (the two numbers); UNRESOLVED (everything else) | High | Non-blocking per-value — isolate each behind config, IMPLEMENTATION-CHOICE, individually flagged |
| 3 | Graph node definition | **UNRESOLVED.** M=1 optimal prompt (§V.D.4) makes "node = prompt" degenerate (1-node graph, no edges). N=L, N=classes, or another mapping is never stated. | §IV.B, §V.D.4, Fig. 1 toy 5-node diagram (illustrative only) | UNRESOLVED | — | **TRUE BLOCKER** — structural, not a single config value; see Part D |
| 4 | Graph construction | **RESOLVED** (mechanics). Cosine similarity (Eq.14) → threshold 0.8 binarize (Eq.15) → symmetrize (Eq.16) → degree-normalize (Eq.17). Optional attention reweight (Eq.18) usage in final model unstated → default OFF. | Eqs. 14–18, §V.D.2/Fig.3 for threshold=0.8 | PAPER-FACT (mechanics + threshold); OPEN-NONBLOCKING (Eq.18 usage) | High | Non-blocking |
| 5 | Prompt insertion mechanism | **UNRESOLVED / CONTRADICTION.** Eq.9 (text) and Eq.10 (vision) both write literal concatenation `[X_CLS, g, X_seq]`. Immediately after Eq.10 the prose says vision prompts "directly replace the input of each layer" — incompatible with concatenation (replace ⇒ constant seq length; concatenate ⇒ growing seq length). CPE-CLIP (REFERENCE-FACT only) supports layerwise concat AND named replace/accumulate strategies as separate configurable modes, so it does not disambiguate which ACHG-CLIP uses. | Eq.9, Eq.10 + following sentence; `research_control_log.md` §13 | PAPER-FACT (the contradiction itself is a fact); UNRESOLVED (which reading is correct) | Low (for resolution) / High (that a genuine contradiction exists) | **TRUE BLOCKER** for the vision path specifically; text path can proceed as concatenation |
| 6 | Text/vision prompt relationship | **RESOLVED (structural symmetry).** Both `G, GV ∈ R^{L×M×d}`; generated and inserted in parallel, symmetric per-modality treatment (Eqs. 9–10 mirror each other formally). §IV.B: "textual and visual learnable prompts are processed... respectively" — confirms two parallel, not merged, prompt sets at this stage. | Eqs. 9–10, §IV.B | PAPER-FACT | High | Non-blocking |
| 7 | Figure-1 MLP blocks | **UNRESOLVED.** Fig. 1 depicts two MLP boxes between "Learnable Prompts" and the GIN modules (one per modality per `paper_spec.md` §4 item 2, based on direct figure reading). Zero running-text sentence anywhere mentions these MLPs, their input/output, dimensions, activation, or layer count — the ONLY figure-only, prose-absent element in the whole architecture. | Fig. 1 (visual only); absence confirmed by full-text search of §IV | UNRESOLVED (figure-only evidence, no textual confirmation) | Low | **TRUE BLOCKER** — role not just a missing number but a missing *function* |
| 8 | G1/G2 relationship | **PARTIALLY RESOLVED.** §IV.B's "respectively" plus Fig. 1's labeled `G1`/`G'1` (text) and `G2`/`G'2` (vision) graphs confirm two separate graphs are constructed, one per modality, both individually run through GIN → ACGA encode/decode (hence the primed `G'1`, `G'2` reconstructed-graph labels). Whether they are later merged, and where, is addressed in Issue 16/18 below. | §IV.B, Fig. 1 labels | PAPER-FACT (two graphs exist) + JUSTIFIED-INFERENCE (their downstream handling) | Medium | Feeds into Issue 16 (true blocker) |
| 9 | GIN structure | **RESOLVED.** Eq.13 (compact) ≡ Eqs.19–21 (decomposed): aggregate (sum over neighbors) → combine `(1+ε)·h_v + agg_v` → MLP transform. Learnable per-layer `ε`. 4 layers, hidden dim 16 (§V.B). Internal MLP architecture (linear-layer count, activation) beyond "MLP" not stated — reused verbatim inside ACGA's encoder (Eq.22, with the added detail that ITS MLP = Linear+BatchNorm+GELU) and inside `H_net` (Eq.29). | Eqs. 13, 19–22, 29; §V.B | PAPER-FACT (formula, layer count, hidden dim); UNRESOLVED (exact per-layer MLP internals for the "main" GIN outside ACGA's encoder, since only ACGA's encoder MLP composition is explicitly given) | High | Non-blocking — reuse ACGA's stated Linear+BN+GELU composition as best-evidence default for the main GIN too, flagged as inference |
| 10 | ACGA/ARGA naming | **OPEN-NONBLOCKING, unchanged from `ambiguity_log.md` A1.** "ACGA" used everywhere except §IV.B and §IV.D, where "ARGA" appears. Reference [48] (Pan et al. 2018) is the real prior method named ARGA, cited as inspiration. No basis to conclude the paper means two different modules — Fig. 1 shows only one graph-autoencoder box. Practical reading: same module, inconsistent naming. | §IV.B, §IV.D vs. Abstract/§II.C/§IV.C/Table V/Conclusion; ref [48] | PAPER-FACT (inconsistency exists); JUSTIFIED-INFERENCE (same module) | Medium-High | Cosmetic only — use "ACGA" as canonical code/module name, preserve both spellings in comments |
| 11 | ACGA encoder/decoder | **RESOLVED (mechanics).** Encoder = GIN-based graph-conv stack (Eq.22) with Linear+BN+GELU MLP, output `Z∈R^{N×K}`. Decoder = inner-product `Â=σ(ZZ^T)` (Eq.23). | Eqs. 22–23 | PAPER-FACT | High | Non-blocking |
| 12 | ACGA latent representation | **PARTIALLY RESOLVED.** `Z∈R^{N×K}`; explicit statement "N and K are independent, with no functional relationship between them." Numeric K value: UNRESOLVED (part of Issue 2/A3). | §IV.C | PAPER-FACT (shape/independence); UNRESOLVED (value of K) | High (structure) / — (value) | Non-blocking, value isolable behind config |
| 13 | ACGA discriminator | **PARTIALLY RESOLVED.** `D(z)=Sigmoid(W2·GELU(W1 z+b1)+b2)` — exactly two FC layers, GELU hidden activation, sigmoid output (Eq.25). Hidden width UNRESOLVED. | Eq. 25 | PAPER-FACT (architecture); UNRESOLVED (width) | High / — | Non-blocking |
| 14 | Adversarial/Wasserstein formulation | **PARTIALLY RESOLVED.** Loss form fixed: `L_adv = E_{z~p_z}[D(z)] − E_{z~q(Z\|X,A)}[D(z)]` (Eq.26), called "Wasserstein distance" form. Whether a sigmoid-bounded critic output (as literally written in Eq.25, `D(z)∈[0,1]`) is even compatible with a true Wasserstein/WGAN critic (which is conventionally unbounded, no sigmoid) is itself a **paper-internal tension**, not just a missing hyperparameter: Eq.25's sigmoid output contradicts standard WGAN critic design (which drops the sigmoid specifically to allow unbounded Lipschitz critic outputs). This is a genuine, not previously logged, internal inconsistency. Weight clipping / gradient penalty / critic-update ratio: UNRESOLVED (A9). | Eqs. 25–26 | PAPER-FACT (both equations as literally written); UNRESOLVED (stabilization mechanics); **new finding**: sigmoid-bounded "Wasserstein" critic is architecturally atypical | Medium | Non-blocking for execution (implement literally as written, i.e., sigmoid-bounded D used inside the Wasserstein-labeled loss, flagged as an unusual-but-literal reading) — flag prominently in code comments |
| 15 | ACGA output → HGN-EC connection | **UNRESOLVED / CONTRADICTION.** §IV.D states verbatim: "HGN-EC receives the node feature matrix X∈R^{N×D} and the adjacency matrix A∈R^{N×N} **from the ARGA module**." These are literally the *same symbols and dimensions* as GIN's raw outputs (§IV.B: `X∈R^{N×D}`, `A∈R^{N×N}`) — **not** ACGA's own defined outputs, which are `Z∈R^{N×K}` (encoder latent) and `Â∈R^{N×N}` (decoder reconstruction). Two irreconcilable readings, neither confirmed: (a) "ARGA" here is really "GIN" (a second instance of the A1 naming slip) and ACGA/HGN-EC run in *parallel* off GIN's output, contradicting Fig.1's caption which reads serially (GIN→ACGA→"feature compression, Hamiltonian module, restoration"); or (b) ACGA passes `X,A` through unchanged as a pass-through side-channel while separately computing `Z,Â` only for its own loss terms, preserving a nominally-serial reading. | §IV.B, §IV.C, §IV.D, Fig.1 caption | PAPER-FACT (verbatim contradiction); UNRESOLVED (which reading) | Low | **TRUE BLOCKER** — determines module wiring (parallel vs. serial), not a tunable value |
| 16 | HGN-EC modality scope | **UNRESOLVED.** §IV.B explicitly uses "respectively" for GIN (dual per-modality processing confirmed, Issue 8). §IV.C (ACGA) and §IV.D (HGN-EC) never repeat "respectively," "each modality," or any dual-processing language — both sections use singular `X, A, Z, H` throughout. Fig.1 shows one ACGA-Module box and one HGN-EC-Module box (not duplicated per modality), which is evidence against two fully separate module *instances*, but is compatible with either (i) one shared-weight module applied twice (once per modality) or (ii) genuinely joint processing of a combined text+vision graph. Neither is stated. | §IV.B (has "respectively") vs. §IV.C/§IV.D (no such language); Fig. 1 (singular boxes) | PAPER-FACT (asymmetry in the paper's own language between GIN and ACGA/HGN-EC sections); UNRESOLVED (which of A/B/C/D applies — see Phase-1-style options below) | Low-Medium | **TRUE BLOCKER** — determines whether 1 or 2 forward passes occur through ACGA/HGN-EC per training step |
| 17 | HGN-EC compression | **PARTIALLY RESOLVED.** `aggregated = A·X`; `state = [X, aggregated]` (Eq.27, concatenation along feature axis); `compressed = W_compress·state + b_compress` (Eq.28, single linear layer). Output width of compression: UNRESOLVED. | Eqs. 27–28 | PAPER-FACT (structure); UNRESOLVED (output width) | High / — | Non-blocking |
| 18 | q and p | **RESOLVED.** Both initialized to the *same* value: `compressed` is assigned to both `q` and `p` — explicit, unambiguous paper statement (§IV.D.3). | §IV.D.3 | PAPER-FACT | High | Non-blocking |
| 19 | Hamiltonian H | **PARTIALLY RESOLVED.** `H = H_net(cat(q,p))` (Eq.29); `H_net` = "an MLP consisting of GIN layers and activation functions." Whether `H_net` reuses the main 4-layer/hidden-16 GIN config or has its own separate (likely smaller, since Fig.1 shows a single "Hamiltonian GIN Layer" block, singular) config is UNRESOLVED. Whether `H` is scalar-per-graph or a vector (Eq.33's energy loss indexes `H_initial,i` for `i=1..n`, suggesting possibly a vector) is UNRESOLVED (=A14). | Eq. 29, Fig.1 "Hamiltonian GIN Layer" (singular label), Eq. 33's indexed notation | PAPER-FACT (formula); UNRESOLVED (internal config, scalar-vs-vector) | Medium / Low | Non-blocking for internal config (isolate behind config); scalar-vs-vector affects `L_energy` code shape — flag |
| 20 | Hamiltonian GIN layer | **JUSTIFIED-INFERENCE.** Fig.1's singular "Hamiltonian GIN Layer" label (vs. the main pipeline's plural, 4-layer GIN block) is weak but real evidence that `H_net` may use *fewer* GIN layers than the main GIN — but this is not stated in prose and must not be treated as confirmed. | Fig. 1 label wording | JUSTIFIED-INFERENCE (weak) | Low | Non-blocking — expose `hnet_gin_layers` as a separate config from `main_gin_layers`, default value is an IMPLEMENTATION-CHOICE |
| 21 | Symplectic Euler | **RESOLVED (formula).** `p_new = p + dt·ṗ`; `q_new = q + dt·q̇` (Eq.31), gradients via `torch.autograd.grad` on Hamilton's equations (Eq.30). | Eqs. 30–31 | PAPER-FACT | High | Non-blocking |
| 22 | dt | **UNRESOLVED.** No numeric value, default, or search range anywhere (not §V.B, not the sensitivity study §V.D which sweeps GNN choice/threshold/loss-coeff/prompt-count/optimizer but never dt). | Full-text search of §V.B, §V.D | UNRESOLVED | — | Non-blocking — isolate as `configs/model/hgn_ec.yaml: dt`, IMPLEMENTATION-CHOICE |
| 23 | Number of integration steps | **UNRESOLVED.** Not stated whether Eq.31 is applied once or iterated. Eq.29–32's prose describes a single pass (compress→init q,p→H→update→restore) with no loop language ("repeat," "for each step," "iterate") anywhere in §IV.D. | §IV.D.1–.8 (no iteration language found) | JUSTIFIED-INFERENCE leaning toward single-step (absence of iteration language), but not confirmed | Low-Medium | Non-blocking — default to 1 step as IMPLEMENTATION-CHOICE, configurable |
| 24 | Energy conservation loss | **PARTIALLY RESOLVED.** `L_energy = MSE(H_initial, H_final) = (1/n)Σ(H_initial,i − H_final,i)²` (Eq.33). `n` undefined numerically (batch size? node count? see Issue 19's scalar-vs-vector question — if H is scalar-per-graph, `n`=batch size is the only sensible reading; if H is per-node, `n` could = N). | Eq. 33 | PAPER-FACT (formula); UNRESOLVED (n's exact referent) | High / Low | Non-blocking — tie `n`'s definition to whichever H-shape choice is made in Issue 19 |
| 25 | Final feature/prompt restoration | **PARTIALLY RESOLVED.** `q_final = W_restore·q_new + b_restore` (Eq.32), "restored to the original dimensionality" — a single linear layer, inverse in spirit to the Eq.28 compression. Ambiguous whether "original dimensionality" means (a) the pre-compression `state` dimensionality (= `2×D` from the Eq.27 concatenation), or (b) the original per-layer prompt dimensionality `d` needed to feed back into `G`/`GV ∈ R^{L×M×d}` — these are not obviously the same size, and the paper never states which. | Eq. 32 + surrounding prose | PAPER-FACT (formula, "single linear layer" framing); UNRESOLVED (target dimensionality identity) | Medium / Low | Ties directly into Issue 26 — non-blocking on its own but compounds Issue 7/26's blocking status |
| 26 | Feedback/update path | **PARTIALLY RESOLVED.** `q_final` "serves as the updated learnable prompt, which is passed into the vision and text encoders for subsequent learning tasks" (§IV.D.8), matching Fig.1's "Update" arrows (top/bottom, feeding back into the CLIP Text/Vision Encoder boxes). Exact reshaping from HGN-EC's single output `q_final` back into the full per-layer, per-modality prompt tensors `G∈R^{L×M×d}` / `GV∈R^{L×M×d}` is not described — this is the same gap as Issue 7 (the unexplained Fig.1 MLP blocks may be exactly this reshaping step, but that is speculative, not confirmed). | §IV.D.8, Fig. 1 "Update" arrows | PAPER-FACT (that feedback occurs, and its destination); UNRESOLVED (the reshape/redistribution mechanism) | Medium / Low | **TRUE BLOCKER**, same root cause as Issue 7 |

---

## PART B — EQUATION AUDIT (1–34)

Full per-equation tensor mapping already exists in `equation_mapping.md` and is not restated here.
This table adds only the **implementation-determinacy status** on top of that mapping.

| Eq. | Meaning | Status |
|---|---|---|
| 1–2 | Token embed + positional encoding | Formula fully determined; `d`, `n`, PE scheme (A13) undetermined numerically |
| 3–5 | Transformer block (attn+FFN) | Formula fully determined; `d_k`, head count `h`, FFN width undetermined |
| 6 | Text semantic projection | Formula fully determined; `d_e` undetermined |
| 7–8 | Vision structured input + CLS projection | Formula fully determined; `d`, `d_e`, patch size undetermined |
| 9 | Text prompt insertion | Formula stated as concatenation; **implementation path determined for text** |
| 10 | Vision prompt insertion | Formula written as concatenation but prose says "replace" — **implementation path UNDETERMINED (Issue 5)** |
| 11–12 | Cross-modal similarity + CE loss | Fully determined (standard CLIP-style contrastive loss) |
| 13, 19–21 | GIN layer (compact + decomposed) | Fully determined mechanically; internal MLP width/depth beyond "4 layers, hidden 16" for the *main* GIN instance undetermined |
| 14–17 | Adjacency construction (cosine, threshold, symmetrize, normalize) | Fully determined, threshold=0.8 numeric |
| 18 | Optional attention reweight | Formula determined; usage-in-final-model undetermined (default OFF) |
| 22 | ACGA encoder | Fully determined mechanically (reuses Eq.13, MLP=Linear+BN+GELU); `K` undetermined |
| 23 | ACGA decoder | Fully determined |
| 24 | Reconstruction loss | Formula determined; negative-edge sampling strategy undetermined |
| 25 | Discriminator | Architecture determined (2 FC + GELU + sigmoid); widths undetermined |
| 26 | Adversarial loss | Formula determined; stabilization mechanics undetermined; **sigmoid-bounded critic vs. "Wasserstein" framing tension noted (Issue 14)** |
| 27 | HGN-EC initial state | Fully determined |
| 28 | Feature compression | Formula determined; output width undetermined |
| — | q/p init | Fully determined (both = `compressed`) |
| 29 | Hamiltonian energy function | Formula determined; `H_net` internal config and H scalar/vector nature undetermined |
| 30 | Hamilton's equations (autodiff) | Fully determined |
| 31 | Symplectic Euler update | Formula determined; `dt` value and step count undetermined |
| 32 | State restoration | Formula determined; target dimensionality identity undetermined (Issue 25) |
| 33 | Energy conservation loss | Formula determined; `n` referent undetermined |
| 34 | Total loss | Formula determined structurally; whether λ1=λ2=λ3=0.04 individually or a combined-then-scaled ACGA term is meant remains an assumption (A10, non-blocking, flagged) |

**No equation is entirely undetermined** — every equation has at least a fully specified
mathematical form. What remains open in every case is either (a) a numeric hyperparameter/dimension,
or (b) for Eqs. 9–10, 26, and the Fig.1-only MLP/reshape steps that have no equation number at all,
a structural/architectural choice.

---

## PART C — TRAINING / DATASETS / LOSSES / EVALUATION / ABLATIONS / SENSITIVITY

### Training (finalized where evidence exists)
All of the following are **PAPER-FACT**, sourced from §V.B, unchanged from `paper_spec.md` §36 /
`reproduction_protocol.md` §2–3: GIN layers=4, GIN hidden=16, adjacency threshold=0.8, M=1 prompt,
optimizer=Lion, lr=0.000325, weight_decay=1e-3, scheduler=CosineAnnealingWarmupRestarts (name only —
no warmup length/restart period given, **UNRESOLVED**, non-blocking), gradient accumulation=3 steps,
gradient clipping max-norm=4.0, base batch=4/epochs=3, incremental batch=4/epochs=5, loss
coefficient=0.04 (applied to "the ACGA loss and the HGN-EC loss" — see A10).
**Frozen parameters:** CLIP backbone (explicit, §I/§IV.A). **Trainable parameters:** GIN, ACGA,
HGN-EC modules, prompt matrices `G`/`GV`, and the Fig.1 MLP blocks (if their existence is confirmed
functional per Issue 7) — PAPER-FACT for the first four, JUSTIFIED-INFERENCE for the MLP blocks
being trainable (never explicitly stated, but no mechanism is offered for them to be otherwise).
**Checkpointing / random seeds:** confirmed **UNRESOLVED** in the paper (unchanged from
`reproduction_protocol.md` §12–13) — no invented default is recorded here; a reproduction-engineering
convention was proposed in that document but is explicitly marked as non-paper-fact and is not
elevated to "resolved" by this pass.
**Evaluation timing:** end of every session (base + each incremental), cumulative over all classes
seen so far — PAPER-FACT (implied by the per-session accuracy columns of Tables I–III + §III.A's
formalization; the exact test-set file lists are not reproduced in the paper, only cited by dataset
reference — UNRESOLVED, non-blocking, standard published FSCIL splits are the only viable
IMPLEMENTATION-CHOICE fallback).

### Datasets (finalized where evidence exists)
Class/session/shot counts for CIFAR-100, miniImageNet, CUB-200-2011 are **PAPER-FACT**
(`reproduction_protocol.md` §1, §5–6, unchanged, cross-checked against `paper_spec.md` §31 —
no discrepancy found). **Exact class-identity splits** (which 60/100 classes are "base" vs. which
40/100 are "incremental," and their ordering across the 8 or 10 sessions) are **not recoverable from
the paper** — datasets are cited by reference only ([56],[57],[58]) without a supplementary split
file. **UNRESOLVED, non-blocking** — the only defensible path is adopting the standard published
FSCIL benchmark splits for these three datasets (as used by the many compared baselines in Tables
I–III, e.g. CEC/FACT/TEEN-style splits), which must be recorded as an **IMPLEMENTATION-CHOICE**, not
presented as paper-fact, when eventually adopted. **Preprocessing/augmentation:** not specified
anywhere in the paper — **UNRESOLVED**, non-blocking, deferred entirely (no default recorded here).

### Losses (finalized where evidence exists)
`L_CE` (Eq.12), `L_recon` (Eq.24), `L_adv` (Eq.26), `L_energy` (Eq.33) — all **PAPER-FACT** as
formulas (see Part B). Coefficients: `λ1=λ2=λ3=0.04` is a **JUSTIFIED-INFERENCE**, not confirmed
PAPER-FACT — the paper's own sentence ("coefficients assigned to the ACGA loss and the HGN-EC loss
... were set to 0.04") is compatible with either three independent identical coefficients or a
combined-then-scaled ACGA term (A10, unchanged, still open, still non-blocking with this default).
**No coefficient is assumed equal to another without this explicit caveat.**

### Evaluation metrics
`A_base`, `A_last`, `Mean`, `ΔPD` — **PAPER-FACT**, unambiguous (`reproduction_protocol.md` §8–11).
`ΔA_last` (Table IV) is confirmed **structurally distinct** from `ΔPD` — an inter-method comparative
metric, not an intra-method forgetting metric — **PAPER-FACT** (A11, resolved/clarified, not
blocking). These must not be conflated in `evaluation/` code, consistent with prior guidance.

### Ablations (Table V)
Four configurations (neither / ACGA alone / HGN-EC alone / both) — **PAPER-FACT**, values
transcribed unchanged from `paper_spec.md` §34. The "HGN-GC" column-header spelling is preserved
verbatim as printed (A2) — not silently corrected to "HGN-EC."

### Sensitivity experiments
GNN choice (GIN wins), adjacency threshold sweep (0.8 optimal), loss-coefficient sweep (0.04
optimal), prompt-count sweep (M=1 optimal), optimizer sweep (Lion wins) — all **PAPER-FACT**,
values unchanged from `paper_spec.md` §33 / Figs. 2–4.

---

## PART D — CONTRADICTIONS EXPLICITLY PRESERVED (not silently fixed)

| Contradiction | Resolution attempted? | Outcome |
|---|---|---|
| ACGA vs. ARGA | Yes — checked all instances, checked cited ref [48] | Terminology inconsistency confirmed; treated as same module for practical purposes (naming only), **not silently renamed in the paper record** |
| HGN-EC vs. HGN-GC | Yes — checked Table V header, Conclusion's alternate expansion | Inconsistency confirmed, no external evidence resolves it; preserved verbatim in Table V transcription |
| Wasserstein loss vs. sigmoid discriminator | Yes — checked Eq.25/26 jointly, checked WGAN convention | **Newly identified as a genuine internal architectural tension** (Issue 14) — a sigmoid-bounded critic is atypical for a Wasserstein-distance loss; the paper gives no acknowledgment of this tension; not resolved, implemented literally as written, flagged |
| Prompt replacement vs. concatenation (vision) | Yes — checked Eq.10 vs. following prose, checked CPE-CLIP | **UNRESOLVED** — genuinely irreconcilable without author code or errata (Issue 5) |
| Graph-node ambiguity | Yes — checked §IV.B, §V.D.4, Fig.1 | **UNRESOLVED** — M=1 makes the naive reading degenerate; no alternative reading is stated (Issue 3) |
| Missing dimensions (`d, d_e, d_k, N, D, K`, etc.) | Yes — exhaustive search of §IV–V | **UNRESOLVED**, confirmed genuinely absent, not merely overlooked |
| Missing `dt` | Yes — checked §V.B and the full sensitivity study §V.D | **UNRESOLVED**, confirmed absent even from the systematic hyperparameter sweeps reported |
| Missing scheduler parameters (warmup length, restart period for CosineAnnealingWarmupRestarts) | Yes — checked §V.B in full | **UNRESOLVED**, scheduler is named but not parameterized anywhere |
| ACGA output → HGN-EC input uses X/A not Z/Â | **Newly identified in this pass** (Issue 15) | **UNRESOLVED** — a real wiring contradiction between the serial architecture implied by Fig.1's caption and the literal variable reuse in §IV.D's text |
| GIN's "respectively" (dual per-modality) vs. ACGA/HGN-EC's singular language | **Newly identified in this pass** (Issue 16) | **UNRESOLVED** — the paper is internally asymmetric in how explicitly it states dual-modality processing across its three components |

---

## PART E — FROZEN SECTIONS

### FROZEN PAPER FACTS
- Full architecture skeleton: CLIP (frozen) → learnable prompts → GIN → ACGA → HGN-EC → updated
  prompts fed back (§IV, Fig. 1).
- All 34 equations' mathematical forms (Part B), including the Eq.9/Eq.10 textual contradiction and
  the Eq.25/Eq.26 sigmoid/Wasserstein tension — preserved as-written, not corrected.
- GIN: 4 layers, hidden dim 16.
- Adjacency threshold: 0.8.
- Number of learnable prompts M=1.
- Optimizer: Lion, lr=0.000325, weight_decay=1e-3, CosineAnnealingWarmupRestarts (unparameterized),
  grad accumulation=3, grad clip=4.0.
- Base session: batch=4, epochs=3. Incremental sessions: batch=4, epochs=5.
- Loss coefficient=0.04 (applied per §V.B's literal sentence; distribution across λ1/λ2/λ3 is an
  inference, see below).
- Dataset protocol: CIFAR-100 (60/40, 8 sessions, 5-way-5-shot), miniImageNet (60/40, 8 sessions,
  5-way-5-shot), CUB-200-2011 (100/100, 10 sessions, 10-way-5-shot).
- Metric definitions: A_base, A_last, Mean, ΔPD, ΔA_last (structurally distinct).
- All reported results tables (I–V) and sensitivity/ablation findings, transcribed for reference only.
- q,p both initialized to the same `compressed` vector.
- Both ACGA/ARGA and HGN-EC/HGN-GC naming inconsistencies exist in the source text.

### FROZEN REFERENCE-SUPPORTED FACTS
- CPE-CLIP (D'Alessandro et al., ICCVW 2023) implements layerwise text/vision prompt insertion with
  named concatenate/replace/accumulate strategies as configurable modes — supports that ACHG-CLIP's
  "replace" language is plausible as a real, distinct mechanism rather than necessarily a typo, but
  does not identify which mode ACHG-CLIP uses.
- ARGA (Pan et al., IJCAI 2018) is a real, distinct prior method explaining the origin of the "ARGA"
  wording, without justifying transferring its architecture wholesale onto ACGA.
- Hamiltonian Neural Networks (Greydanus et al., NeurIPS 2019) confirm the general q/p/H/symplectic
  mathematical framework ACHG-CLIP is built on, without supplying any ACHG-CLIP-specific numeric value.
- Original CLIP (Radford et al.) confirms multiple backbone variants exist in the CLIP family without
  identifying which one ACHG-CLIP uses.
- (All per `research_control_log.md` §15–17; no new reference papers were consulted in this pass —
  see output summary for justification.)

### JUSTIFIED INFERENCES
- ACGA and "ARGA" (§IV.B/§IV.D wording) most plausibly refer to the same module (Issue 10).
- The "main" GIN's internal MLP (outside ACGA's encoder) most plausibly reuses ACGA's stated
  Linear+BatchNorm+GELU composition, given no separate specification exists (Issue 9).
- Absence of any iteration language in §IV.D weakly favors a single Symplectic-Euler step over
  multiple (Issue 23).
- Fig. 1's singular "Hamiltonian GIN Layer" label weakly favors `H_net` using fewer GIN layers than
  the main 4-layer GIN, though this is not confirmed (Issue 20).

### REQUIRED IMPLEMENTATION CHOICES
*(To be exercised at actual coding time, not asserted here — this pass only enumerates what will be
required and why, per the master prompt's instruction never to hide such values.)*
1. Numeric values for `d, d_e, d_k, N, D, K`, compression output width, discriminator hidden width,
   `H_net` width/depth — necessary because no tensor can be allocated without a concrete size; each
   must be recorded individually with rationale when chosen, easily changed via config.
2. `dt` and integration-step count for Symplectic Euler — necessary to run any forward pass through
   HGN-EC.
3. CLIP backbone checkpoint/variant — necessary to instantiate any frozen encoder at all.
4. Negative-edge sampling ratio/strategy for `L_recon`.
5. WGAN-style stabilization details (clip/GP/none, critic update ratio) for `L_adv`.
6. Positional encoding scheme (learned vs. sinusoidal).
7. Scheduler warmup length / restart period for CosineAnnealingWarmupRestarts.
8. Dataset class-split identity (which specific classes are base vs. incremental, and their order) —
   proposed default: adopt standard published FSCIL benchmark splits for these three datasets.
9. Checkpoint policy and random seed policy (per `reproduction_protocol.md` §12–13, unchanged).
10. Whether λ1=λ2=λ3=0.04 independently, or a combined ACGA-then-scaled reading is used (default:
    independent, flagged).
Each of these, when actually chosen during implementation, must be logged with: exact value, why
necessary, why chosen, reproducibility risk, and how easily changed — per the master prompt's rule.
None are chosen here.

### REMAINING UNRESOLVED ITEMS (non-blocking)
Items 1, 2 (partially), 12 (value only), 13 (width only), 17 (width only), 19 (config only), 22, 23,
24 (n only) from Part A; plus scheduler parameters, dataset split identity, preprocessing/
augmentation, negative-edge sampling, WGAN stabilization details, checkpoint/seed policy. All of
these can be isolated behind configuration values and do not by themselves prevent a codebase from
being written and executed — they prevent the codebase from being *provably faithful* to the paper's
exact configuration without further evidence (author code, correspondence, or errata).

### TRUE BLOCKERS
Only items whose resolution changes the **shape of the computation graph itself**, not just a
parameter value, are classified as true blockers:
1. **Issue 3 — Graph node definition.** Determines what a "node" is; every downstream tensor shape
   in GIN/ACGA/HGN-EC depends on this.
2. **Issue 5 — Vision prompt insertion (concatenate vs. replace).** Changes whether the vision
   sequence grows or stays fixed length — a different forward-pass code path, not a config value.
3. **Issue 7 — Figure-1 MLP blocks.** An entire computational step with unknown function; cannot be
   coded as "a linear layer with configurable width" because its *role* (not just its size) in the
   pipeline is unknown.
4. **Issue 15 — ACGA→HGN-EC data connection.** Determines whether HGN-EC consumes ACGA's transformed
   latent (Z) or GIN's untransformed (X,A), i.e., whether ACGA sits serially upstream of HGN-EC or
   runs as a parallel side-branch — a wiring decision, not a value.
5. **Issue 16 — HGN-EC (and ACGA) per-modality vs. joint scope.** Determines whether 1 or 2 forward
   passes occur through these modules per step, and whether/where the two modality graphs merge —
   again a wiring decision.

Issues 7, 15, 16, and (to a lesser extent) 5 are all facets of the same underlying gap: **the paper
never explains how a graph-processed representation gets back into the L×M×d prompt tensors that
feed the CLIP encoders, nor exactly what flows between each named module.** This is the single most
consequential open question for implementation and should be the first thing sought from the
authors' code repository (linked in the Abstract but confirmed non-functional per
`research_control_log.md`'s Project Status) or from direct author contact, before any of GIN,
ACGA, or HGN-EC modules are coded.

---

## PART F — FINAL QUALITY CHECK

Every item in Part A was tested against "Can I point to evidence?" before being classified:
- Where an equation or explicit sentence exists → PAPER-FACT.
- Where a cited reference's own content (not just its name) was checked and found relevant →
  REFERENCE-FACT, explicitly distinguished from PAPER-FACT.
- Where a reading is favored by circumstantial evidence (label wording, absence of contrary
  language) but not confirmed → JUSTIFIED-INFERENCE, explicitly flagged as weak/low-confidence.
- Where nothing in the paper, cited references, or figure supports any reading → UNRESOLVED.
- No blank was filled merely because a standard/typical implementation exists elsewhere (e.g., no
  assumed CLIP variant, no assumed `dt`, no assumed dataset split, no assumed negative-sampling
  ratio).
