# ACHG-CLIP Reproduction — Research Control Log

## Project Status

This is a standalone reproduction project for:

**A Few-Shot Class Incremental Learning Method Using Graph Neural Networks**

Proposed method: **ACHG-CLIP (Adversarially Constrained Hamiltonian Graph-CLIP)**

The official repository linked by the paper was checked and does not currently provide a usable implementation. Therefore, the reproduction will be reconstructed from the paper and verified against cited/reference works where those references can resolve technical ambiguities.

---

## 1. Fixed Research Goal

Reproduce the paper as faithfully as possible.

The final project must:

- use the paper as the primary source;
- preserve the proposed architecture;
- map mathematical equations to code;
- build a modular, testable PyTorch codebase;
- use YAML configuration rather than hidden hard-coded experimental settings;
- implement unit tests for major modules;
- reproduce CIFAR-100, miniImageNet, and CUB-200-2011 protocols;
- reproduce the reported ablation and sensitivity experiments where possible;
- compare our actual results honestly against the paper's reported results;
- document every ambiguity, assumption, implementation decision, and source of evidence.

### Explicit prohibitions

Do NOT:

- redesign the proposed method;
- silently correct inconsistencies in the paper;
- invent missing parameter values and present them as paper facts;
- hard-code the paper's reported results;
- claim exact reproduction before experiments support it;
- replace an unclear paper detail with generic ML practice without documenting the decision.

---

## 2. Source Hierarchy

Use sources in this order:

1. The target paper itself.
2. The specific reference paper(s) cited by the target paper that define/explain the ambiguous component.
3. Official documentation or primary sources for libraries/backbones/datasets when needed.
4. Other authoritative research sources.
5. Independent implementations only as secondary evidence, never as proof of what the target authors did.

Whenever a reference paper is used to resolve an ambiguity:

- record the reference paper;
- record exactly what it contributes;
- distinguish it from what the target paper explicitly states;
- do not transfer an implementation detail automatically unless the evidence supports that transfer.

---

## 3. Reference-Paper Rule

The target paper contains numbered citations between sentences and around technical claims.

These references are potentially important for resolving:

- CLIP architecture/backbone details;
- FSCIL protocol and dataset splits;
- GIN formulation;
- graph construction;
- adversarial graph autoencoder formulation;
- Wasserstein/adversarial objective;
- Hamiltonian neural networks;
- Symplectic Euler integration;
- prompt learning/insertion;
- any other component whose definition is delegated to earlier work.

### Required procedure

For every unresolved technical detail:

1. Identify whether the target paper cites a reference immediately around that claim.
2. Locate the cited paper.
3. Read the relevant section/equations, not just the abstract.
4. Check whether the cited paper actually defines the missing detail.
5. Record the evidence.
6. Decide whether the reference legitimately resolves the ambiguity.
7. If it does not, keep the issue unresolved.
8. Never invent a value merely because a cited method commonly uses one.

---

## 4. Evidence Labels

Every important implementation statement should have one of these labels:

### PAPER-FACT
Explicitly stated or mathematically defined in the target paper.

### REFERENCE-FACT
Supported by a cited/reference paper, but not explicitly stated in the target paper.

### LIBRARY-FACT
Defined by the official implementation/documentation of a dependency.

### INFERENCE
Reasoned interpretation from available evidence. Must be clearly marked.

### IMPLEMENTATION-CHOICE
A choice required to make the code executable when the sources do not specify the detail. Must never be presented as an author-confirmed value.

### UNRESOLVED
Insufficient evidence. Do not guess.

---

## 5. Current Documentation Files

The project should maintain:

```text
docs/
├── paper_spec.md
├── equation_mapping.md
├── ambiguity_log.md
├── reproduction_protocol.md
└── research_control_log.md
```

### paper_spec.md
What the target paper explicitly says.

### equation_mapping.md
Equation → meaning → tensors → dimensions → code module/function.

### ambiguity_log.md
Every inconsistency, missing detail, or unclear statement.

### reproduction_protocol.md
Dataset/session/training/evaluation protocol.

### research_control_log.md
Master record of:
- assumptions;
- evidence;
- reference-paper investigations;
- implementation decisions;
- finalizations;
- changes to earlier decisions;
- unresolved items.

---

## 6. Current Confirmed Paper Settings

These are treated as PAPER-FACT only where explicitly supported by the paper:

- GIN layers: 4
- GIN hidden dimension: 16
- graph similarity: cosine similarity
- graph threshold: 0.8
- learnable prompts: 1
- optimizer: Lion
- learning rate: 0.000325
- weight decay: 1e-3
- gradient accumulation: 3 steps
- gradient clipping: 4.0
- base batch size: 4
- base epochs: 3
- incremental batch size: 4
- incremental epochs: 5
- reported loss coefficient: 0.04

Dataset protocols:

### CIFAR-100
- 100 total classes
- 60 base classes
- 40 novel classes
- 8 incremental sessions
- 5 classes per incremental session
- 5-shot

### miniImageNet
- 100 total classes
- 60 base classes
- 40 novel classes
- 8 incremental sessions
- 5 classes per incremental session
- 5-shot

### CUB-200-2011
- 200 total classes
- 100 base classes
- 100 novel classes
- 10 incremental sessions
- 10 classes per incremental session
- 5-shot

Reported paper targets:

| Dataset | Mean Accuracy | ΔPD |
|---|---:|---:|
| CIFAR-100 | 82.30% | 9.72% |
| miniImageNet | 85.05% | 8.42% |
| CUB-200 | 69.54% | 17.67% |

These are reproduction targets only. They must never be hard-coded.

---

## 7. Current Architecture Understanding

High-level architecture currently understood as:

```text
CLIP
  │
  ├── text pathway
  │      ↓
  │   learnable text prompts
  │      ↓
  │   graph processing
  │      ↓
  │   GIN
  │      ↓
  │   ACGA
  │      ↓
  │   HGN-EC
  │
  └── vision pathway
         ↓
      learnable vision prompts
         ↓
      graph processing
         ↓
      GIN
         ↓
      ACGA
         ↓
      HGN-EC
```

The exact arrow-level interpretation must continue to be checked against Figure 1, equations, prose, and relevant cited papers.

---

## 8. Current Major Ambiguities

These remain OPEN until evidence resolves them.

### A1 — ACGA vs ARGA naming
Some target-paper text uses “ARGA” while the proposed module is described as “ACGA”.

Status: OPEN.

### A2 — HGN-EC vs HGN-GC naming
The ablation table uses “HGN-GC” while the proposed method is HGN-EC; terminology in the conclusion also needs careful checking.

Status: OPEN.

### A3 — Missing dimensions
Values/dimensions such as d, d_e, N, K, and HGN-EC compression dimensions are not yet fully established.

Status: OPEN.

### A4 — Symplectic Euler dt
The target paper gives the update equations but the exact time-step value must be verified.

Status: OPEN.

### A5 — Prompt insertion mechanism
Text and vision prompt descriptions use different wording. Figure 1 also contains MLP blocks that require exact interpretation.

Status: OPEN.

### A6 — Graph node definition with M=1
The paper reports one learnable prompt in its sensitivity study, but the precise relationship between M, prompt tensors, graph nodes, and graph structure must be established before implementation.

Status: OPEN.

### A7 — ACGA graph details
Need to verify exactly which representations form nodes/edges and how the graph is passed through ACGA.

Status: OPEN.

### A8 — HGN-EC input/state dimensions
Need to establish exact tensor dimensions and layer dimensions.

Status: OPEN.

### A9 — Wasserstein/adversarial stabilization
The target paper describes a Wasserstein-style adversarial objective, but all stabilization/training details must be verified rather than assumed.

Status: OPEN.

### A10 — MLP blocks in Figure 1
The figure shows MLP components whose exact role and dimensions need to be established from the paper/reference work.

Status: OPEN.

### A11 — Prompt feedback/update mechanism
Need to establish exactly how HGN-EC output is fed back into the prompt/CLIP pathway.

Status: OPEN.

### A12 — Exact CLIP variant
The exact CLIP backbone variant must be established from the paper or a cited/primary source.

Status: OPEN.

### A13 — Dataset split identity
Need to establish the exact class split files/protocol if not explicitly specified by the target paper.

Status: OPEN.

### A14 — Seed/checkpoint protocol
Random seed and checkpoint policy are not to be invented. If absent from the paper, they must be recorded as implementation choices.

Status: OPEN.

---

## 9. Decision Protocol

For each OPEN item:

```text
OPEN issue
   ↓
Check target-paper prose
   ↓
Check target-paper equations
   ↓
Check target-paper Figure/Table
   ↓
Inspect cited reference paper(s)
   ↓
Check authoritative implementation/documentation if relevant
   ↓
Decision:
    PAPER-FACT
    REFERENCE-FACT
    INFERENCE
    IMPLEMENTATION-CHOICE
    or UNRESOLVED
```

No issue should be closed merely because a value is “typical”.

---

## 10. Change Log

### Initial baseline
- Project goal fixed.
- Official linked repository found but no usable implementation available.
- Paper established as primary source.
- Four initial documentation files established.
- Reference-paper investigation added as a mandatory ambiguity-resolution step.
- No implementation assumptions are currently finalized unless explicitly supported.

---

## 11. Finalization Rule

Once an implementation decision is made, record:

```text
Decision ID:
Issue:
Decision:
Evidence:
Source type:
Reason:
Confidence:
Date:
Impact on implementation:
```

If later evidence contradicts it:

1. Do not silently overwrite it.
2. Mark the old decision as superseded.
3. record the new evidence;
4. record why the decision changed.

---

## 12. Scientific Reproduction Standard

At the end of the project, we should be able to answer for every major implementation detail:

> “Where did this come from?”

The acceptable answers are:

- “The target paper explicitly says it.”
- “Reference [X] defines this component, and the target paper cites it here.”
- “Official library documentation requires this.”
- “The paper does not specify it; this is our explicitly documented implementation choice.”
- “This remains unresolved.”

Never:

> “This is probably what the authors meant.”

---

## 13. Reference Investigation Record — A5 Prompt Insertion

### Question
How are the text and vision learnable prompts inserted into the CLIP encoders, and can the cited CPE-CLIP work resolve the ambiguity?

### Target-paper evidence

The target paper defines text prompts as `G ∈ R^{L×M×d}` and vision prompts as `GV ∈ R^{L×M×d}`. Text Eq. (9) uses `X = [X_CLS, g^(l), X_tokens]`; vision Eq. (10) uses `X = [X_CLS, gV^(l), X_patches]`. The vision prose additionally says prompts "directly replace the input of each layer," which conflicts with the concatenation notation. Figure 1 also shows two MLP blocks whose exact role is not specified in the running text.

### Reference investigated

D'Alessandro et al., "Multimodal Parameter-Efficient Few-Shot Class Incremental Learning" (CPE-CLIP), ICCVW 2023.

Official implementation: https://github.com/neuraptic/cpe-clip

### What CPE-CLIP establishes

The official implementation supports learnable prompts in both language and vision encoders, layerwise prompt processing, text prompt insertion/concatenation, and vision prompt insertion/accumulation strategies. The paper describes language prompts as layerwise concatenation and vision prompts as concatenation with image patch embeddings.

### What this does NOT establish

CPE-CLIP does not prove that ACHG-CLIP uses exactly the same insertion implementation. Its implementation must not be presented as the authors' ACHG-CLIP implementation.

### Current conclusion

A5 remains OPEN. The evidence gives a stronger candidate interpretation: both Eq. (9) and Eq. (10) are written as concatenation, while the word "replace" is ambiguous. CPE-CLIP supports layerwise prompt insertion but does not resolve ACHG-CLIP's exact mechanism or the two MLP blocks.

### Decision

No final implementation decision yet.

Evidence types: Target paper = PAPER-FACT; CPE-CLIP = REFERENCE-FACT; exact vision "replace" interpretation = UNRESOLVED.

### Next evidence required

1. Inspect the exact CPE-CLIP sections cited by ACHG-CLIP around prompt learning.
2. Reinspect ACHG-CLIP Figure 1 for the MLP blocks and update arrows.
3. Check whether the cited surgical visual-language paper contributes a prompt/graph transformation relevant to the MLP blocks.
4. Check whether ACHG-CLIP equations elsewhere determine the MLP input/output dimensions.

Do not implement prompt injection yet.

---

## 14. Current Investigation Status

| Issue | Status | Latest evidence |
|---|---|---|
| A5 Prompt insertion / MLP role | OPEN | Target Eq. 9–10 + CPE-CLIP investigated; exact insertion and MLP role remain unresolved |
| A6 Graph node definition | OPEN | Prompt-as-node supported; exact mapping from `L×M×d` to graph nodes unresolved |


---

## 15. Batch Reference-Paper Investigation — 2026-08-28

The remaining reference chain was investigated with the rule that a cited paper may resolve an ambiguity only when it actually supports the relevant detail.

### CPE-CLIP

D'Alessandro et al., "Multimodal Parameter-Efficient Few-Shot Class Incremental Learning", ICCVW 2023.

Verified:
- learnable prompts for language and vision encoders;
- layerwise prompt processing;
- official implementation supports text and vision prompt insertion;
- replace/accumulate/accumulate_same strategies are explicitly implemented;
- CIFAR100, CUB200 and miniImageNet are supported.

Conclusion:
CPE-CLIP is useful evidence for prompt-learning mechanics, but it does NOT prove the exact ACHG-CLIP prompt strategy.

Classification: REFERENCE-FACT.

### Surgical Video Workflow Analysis via Visual-Language Learning

Verified:
- dual textual and visual graphs;
- semantic text features can be graph nodes;
- visual category representations can be constructed from average training features;
- cosine similarity is used for graph relationships;
- a textual prompt can be inserted as a new node and connected to textual/visual graphs;
- GCN integrates relational semantics.

Conclusion:
This strongly supports the general idea of prompt/semantic nodes and dual-modal graphs, but it does NOT prove the exact ACHG-CLIP graph construction.

Classification: REFERENCE-FACT.

### ARGA

Pan et al., "Adversarially Regularized Graph Autoencoder for Graph Embedding", IJCAI 2018.

Verified:
- graph encoder produces latent representation;
- decoder reconstructs graph structure;
- adversarial training matches latent representations to a prior;
- ARGA is the established name of that method.

Conclusion:
This explains why "ARGA" appears in ACHG-CLIP terminology, but does not justify renaming ACHG-CLIP's proposed ACGA module.

Classification: REFERENCE-FACT / terminology evidence.

### Wasserstein graph autoencoder

Liang & Gao, "Wasserstein Adversarially Regularized Graph Autoencoder".

Verified:
- Wasserstein distance can regularize graph node embeddings;
- weight clipping and gradient penalty are two approaches used to enforce Lipschitz continuity.

Conclusion:
These are possible standard mechanisms, but ACHG-CLIP does not currently provide enough evidence to choose either one.

Classification: REFERENCE-FACT.

### Hamiltonian Neural Networks

Greydanus et al., "Hamiltonian Neural Networks", NeurIPS 2019.

Verified:
- Hamiltonian is parameterized by a neural network;
- q and p represent canonical state variables;
- derivatives of H produce Hamiltonian dynamics;
- energy-like quantities are conserved by the formulation.

Symplectic-learning literature further confirms the role of symplectic integration.

Conclusion:
This supports the mathematical foundation of HGN-EC, but does not specify ACHG-CLIP's missing dimensions or dt.

Classification: REFERENCE-FACT.

### CLIP

Radford et al., "Learning Transferable Visual Models From Natural Language Supervision".

Verified:
- CLIP contains image and text encoders;
- image-text contrastive pretraining is the core objective;
- multiple vision backbone choices exist.

Conclusion:
Original CLIP does not identify the exact ACHG-CLIP backbone variant.

Classification: REFERENCE-FACT.

---

## 16. Batch Investigation Conclusions

### A1 — ACGA vs ARGA
Status: TERMINOLOGY DISCREPANCY CONFIRMED.

Use ACGA as the proposed-module name because that is the target paper's main terminology. Preserve ARGA occurrences as an ambiguity rather than silently changing the paper.

### A2 — HGN-EC vs HGN-GC
Status: TARGET-PAPER INCONSISTENCY CONFIRMED.

The proposed component is HGN-EC, while HGN-GC appears in the ablation terminology. No external evidence found that justifies silently changing it.

### A5 — Prompt insertion / MLP blocks
Status: OPEN.

The target equations show concatenation, while prose uses "replace" wording for vision. CPE-CLIP provides related implementation evidence but not proof of the ACHG-CLIP implementation. The exact role/dimensions of the Figure 1 MLP blocks remain unresolved.

### A6 — Graph node definition
Status: PARTIALLY RESOLVED; exact tensor mapping OPEN.

Confirmed: the target paper describes learnable prompts as graph nodes.

Not established:
- N = M;
- N = L;
- N = L × M;
- any other exact mapping.

### A9 — Wasserstein adversarial details
Status: OPEN.

Relevant Wasserstein graph-autoencoder literature was checked, but weight clipping and gradient penalty cannot be assigned to ACHG-CLIP without direct evidence.

### A12 — Exact CLIP variant
Status: OPEN.

Original CLIP permits multiple backbone choices; no verified source currently identifies the ACHG-CLIP variant.

### HGN-EC dt and dimensions
Status: OPEN.

Hamiltonian references explain the mathematical structure but do not provide ACHG-CLIP-specific missing values.

---

## 17. Explicit Negative Findings

Do NOT treat any of the following as established:

- N = M;
- N = L;
- N = L × M;
- ACHG-CLIP uses CPE-CLIP's exact prompt replacement strategy;
- ACHG-CLIP uses CPE-CLIP's exact CLIP backbone;
- ACHG-CLIP uses WARGA weight clipping;
- ACHG-CLIP uses WARGA gradient penalty;
- HGN-EC dt can be borrowed from another Hamiltonian paper;
- Figure 1 MLP dimensions can be inferred from common practice.

These remain unsupported unless direct evidence is found.

---

## 18. Investigation State

The broad reference-paper investigation is complete.

Remaining work is to build the final implementation-decision matrix and classify every remaining item as:

- PAPER-FACT
- REFERENCE-FACT
- INFERENCE
- IMPLEMENTATION-CHOICE
- UNRESOLVED

Only after that matrix is frozen should model coding begin.

**STATUS UPDATE (§19 below): the final implementation-decision matrix has now been built.**
See `docs/FINAL_RESEARCH_DECISIONS.md`. Nothing in §1–18 above is deleted or altered; superseded
readings are marked explicitly in §19/§20 rather than edited in place, per the finalization rule
in §11.

---

## 19. Final Research Pass — 2026-08-29

This section records the outcome of the final scientific-investigation pass, whose full decision
matrix lives in `docs/FINAL_RESEARCH_DECISIONS.md`. Only the headline outcomes and any changes to
prior entries are duplicated here; the matrix file is authoritative for detail.

### 19.1 Newly identified issues (not previously logged anywhere)

- **New — ACGA→HGN-EC data-flow contradiction.** §IV.D's literal statement that HGN-EC receives
  `X∈R^{N×D}` and `A∈R^{N×N}` "from the ARGA module" reuses GIN's own output symbols/dimensions, not
  ACGA's actually-defined outputs (`Z∈R^{N×K}`, `Â∈R^{N×N}`). This is a distinct issue from the A1
  ACGA/ARGA *naming* ambiguity — it is a *data-flow wiring* ambiguity that naming alone does not
  explain. Classified as a TRUE BLOCKER in `FINAL_RESEARCH_DECISIONS.md` Part A Issue 15 / Part E.
- **New — GIN vs. ACGA/HGN-EC modality-scope asymmetry.** §IV.B uses "respectively" to describe dual
  per-modality GIN processing; §IV.C (ACGA) and §IV.D (HGN-EC) never repeat equivalent language and
  use singular notation throughout, while Fig. 1 shows only one ACGA-Module box and one HGN-EC-Module
  box (not duplicated per modality). Not previously logged as its own issue (partially anticipated by
  old A7 "ACGA graph details" but not stated this precisely). Classified as a TRUE BLOCKER
  (`FINAL_RESEARCH_DECISIONS.md` Issue 16).
- **New — Sigmoid-bounded discriminator vs. "Wasserstein" loss framing tension.** Eq.25 defines the
  discriminator with a sigmoid output (`D(z)∈[0,1]`), which is architecturally atypical for a
  Wasserstein/WGAN-style critic (conventionally unbounded, sigmoid intentionally omitted). The paper
  calls Eq.26 a "Wasserstein distance" form without acknowledging this tension. Not previously logged
  (old A9 covered only the *missing* stabilization details, not this *internal* inconsistency).
  Classified UNRESOLVED, non-blocking for execution (implemented literally as written), flagged
  prominently (`FINAL_RESEARCH_DECISIONS.md` Issue 14 / Part D).
- **New — Figure-1 MLP blocks are the only architecture element with zero prose support anywhere in
  §IV.** Previously logged as part of A5 (prompt insertion); the final pass elevates this to its own
  TRUE BLOCKER (Issue 7) because its uncertainty is about *function*, not merely dimensions — no
  sentence in the entire paper mentions these MLPs.

### 19.2 Superseded / refined readings

- **A6 (graph node definition):** previous status "PARTIALLY RESOLVED; exact tensor mapping OPEN"
  (§16 above) is **not superseded** — still open, no new mapping evidence found. Re-confirmed as a
  TRUE BLOCKER in the final matrix (Issue 3), not merely OPEN, given its structural centrality.
- **A5 (prompt insertion):** previous status OPEN (§16) is **not superseded** in terms of resolution,
  but is now explicitly split into two sub-questions with different blocking status in the final
  matrix: (i) text-path concatenation — effectively usable as-is (non-blocking), vs. (ii) vision-path
  concatenate-vs-replace contradiction — TRUE BLOCKER (Issue 5). This split is new precision, not a
  reversal.
- No other prior entry (§8, §14, §16, §17) is reversed or contradicted by this pass. All remain OPEN
  as previously recorded; the final pass adds structure (blocking vs. non-blocking classification)
  rather than new resolutions to A1–A4, A7–A9, A12–A14.

### 19.3 Reference papers consulted in this pass

None beyond what §15–17 already covered. This pass worked entirely from re-reading the target
paper's own text/equations/figure more closely (yielding the three "new" findings in §19.1); no new
external reference was fetched, consistent with the source-hierarchy rule that the target paper is
checked exhaustively before further reference investigation is warranted. If Issues 3, 5, 7, 15, or
16 (the five TRUE BLOCKERS) are to be resolved at all short of the authors' own code, the most likely
next useful reference is CPE-CLIP's official implementation (already partially reviewed, §13) probed
specifically for its MLP/reshape layer between prompt space and graph space — this is a
recommendation, not an action taken in this pass.

### 19.4 What is now frozen

Per `FINAL_RESEARCH_DECISIONS.md` Part E: architecture skeleton, all 34 equations' literal
mathematical forms (including preserved contradictions), all explicitly-stated hyperparameters,
dataset/session/shot protocol, metric definitions, and all reported result tables are FROZEN as
paper-fact and will not be re-derived. Five TRUE BLOCKERS (graph node definition; vision prompt
insertion; Fig.1 MLP-block role; ACGA→HGN-EC wiring; per-modality vs. joint ACGA/HGN-EC scope) remain
open and must be resolved — via author code/contact, or via explicitly-flagged implementation
choices accepted as reproduction risk — before those specific modules can be coded. All other open
items are non-blocking and isolable behind configuration.
