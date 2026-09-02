# Reproduction Protocol

This document defines the experimental protocol for reproducing ACHG-CLIP, strictly as specified
in the paper (Section V.A–V.C). Values not stated in the paper are marked "Not specified" and are
cross-referenced to `ambiguity_log.md`; they are not filled in with assumed defaults.

## 1. Dataset Protocol

Three benchmark datasets are used, each with a fixed base/incremental class split and a fixed
few-shot incremental sampling scheme:

### 1.1 CIFAR-100
- 60,000 total images, 32×32 RGB, 100 classes.
- **Base session:** 60 classes, full training data per class (exact per-class sample count for the
  base session not stated beyond "N0 is large" — see general FSCIL formalism in Section III.A).
- **Incremental sessions:** 8 sessions total.
  - 5 new classes introduced per session ("5-way").
  - 5 labeled samples per new class per session ("5-shot").
  - Total incremental classes across all 8 sessions: 40 (8 × 5).
- Sessions are indexed 0 (base) through 8 (final incremental session) in Table I.

### 1.2 miniImageNet
- 60,000 total images, 84×84 RGB, 100 classes.
- **Base session:** 60 classes.
- **Incremental sessions:** 8 sessions, 5-way 5-shot, identical structure to CIFAR-100.
- Total incremental classes: 40.
- Sessions indexed 0 through 8 in Table II.

### 1.3 CUB-200-2011
- 11,788 total images, 224×224 RGB, 200 bird species.
- **Base session:** 100 classes.
- **Incremental sessions:** 10 sessions.
  - 10 new classes introduced per session ("10-way").
  - 5 labeled samples per new class per session ("5-shot").
  - Total incremental classes across all 10 sessions: 100 (10 × 10).
- Sessions indexed 0 through 10 in Table III.

### 1.4 Class-Incremental Constraints (all datasets)

Per Section III.A's formal FSCIL definition:
- New classes at each session are disjoint from all previously introduced classes.
- The model must classify test samples **without knowledge of which session a class was
  introduced in** (pure class-incremental setting — not task-incremental with session-ID given at
  test time).
- The model **cannot store or revisit raw data** from prior sessions (`D0, …, D_{t-1}` are not
  accessible once a new session begins). Per Section IV.B.1, only *feature-level* information (old
  graph node features, kept unchanged) persists across sessions — not raw images/labels.

## 2. Base Session

- Applies to all three datasets identically in terms of training schedule (per Section V.B):
  - Batch size: **4**
  - Training epochs: **3**
- Purpose: establish initial recognition ability over the base classes and train the CLIP-frozen,
  GIN/ACGA/HGN-EC prompt-update pipeline.

## 3. Incremental Sessions

- Applies to all three datasets identically in terms of training schedule (per Section V.B):
  - Batch size: **4**
  - Training epochs: **5**
- Each incremental session:
  1. Introduces new classes' 5-shot support data.
  2. Adds new node features to the prompt graph while keeping old node features fixed
     (Section IV.B.1).
  3. Trains for 5 epochs per session using the same total loss (Eq. 34) and optimizer settings as
     the base session (Section V.B does not state any incremental-session-specific change to
     optimizer, learning rate, or loss coefficients).

## 4. 5-Shot Setting

- All three datasets use a uniform **5-shot** incremental sampling scheme: exactly 5 labeled
  images per new class are available in the session that introduces that class.
- This applies to *every* incremental session across all datasets (CIFAR-100, miniImageNet: 5-way
  5-shot; CUB-200-2011: 10-way 5-shot).

## 5. Class Counts

| Dataset | Base classes | Incremental classes (total) | Classes/session | Total classes |
|---|---|---|---|---|
| CIFAR-100 | 60 | 40 | 5 | 100 |
| miniImageNet | 60 | 40 | 5 | 100 |
| CUB-200-2011 | 100 | 100 | 10 | 200 |

## 6. Session Counts

| Dataset | Base session | Incremental sessions | Total sessions (incl. base) |
|---|---|---|---|
| CIFAR-100 | 1 (session 0) | 8 (sessions 1–8) | 9 |
| miniImageNet | 1 (session 0) | 8 (sessions 1–8) | 9 |
| CUB-200-2011 | 1 (session 0) | 10 (sessions 1–10) | 11 |

## 7. Evaluation Procedure

- At the end of **each session** (base and every incremental session), the model is evaluated on a
  test set covering **all classes seen so far** (cumulative class-incremental evaluation), producing
  one accuracy value per session (the "Accuracy (%)" columns in Tables I–III, indexed 0 through
  the final session number).
- The paper does not explicitly restate the test-set composition/size for each dataset beyond the
  standard FSCIL cumulative-evaluation convention implied by the per-session accuracy tables and
  Section III.A's problem formalization; standard splits for CIFAR-100/miniImageNet/CUB-200-2011
  FSCIL benchmarks are referenced by dataset citation only ([56], [57], [58]) — exact test-set
  file lists/splits are not reproduced in the paper itself.
- Ablation study (Table V) uses the same cumulative per-session evaluation procedure, on CUB200
  only, across all 11 sessions.

## 8. A_base

- Defined in Table IV's caption: "A_base denotes the accuracy in the base session phase" — i.e.,
  the session-0 accuracy value from the per-session accuracy sequence (equivalent to the first
  entry in Tables I–III's "Accuracy (%)" columns for a given method).

## 9. A_last

- Defined in Table IV's caption: "A_last represents the accuracy in the last incremental session"
  — i.e., the final session's accuracy value from the per-session accuracy sequence (equivalent to
  the last entry in Tables I–III's "Accuracy (%)" columns for a given method).

## 10. Mean Accuracy

- Defined identically in Tables I–III and Table IV: "Mean is the average accuracy across the base
  session and all incremental sessions" — i.e., the arithmetic mean of all per-session accuracy
  values (base session + every incremental session), not a weighted average by class count or
  sample count.

## 11. ΔPD

- Defined in Tables I–III's captions: "ΔPD represents the difference between the classification
  accuracy of the model on the base session and the last incremental session. The smaller the ΔPD
  value, the lower the model's catastrophic forgetting of old classes."
- Formula (as stated in words): `ΔPD = A_base - A_last` for a given method, evaluated on classes
  the model has seen. Lower is better (less forgetting).
- **Note:** this is an intra-method metric, distinct from Table IV's `ΔA_last`, which is an
  inter-method comparative metric (see `ambiguity_log.md`, item A11). Do not conflate the two in
  evaluation code — implement as two separate functions.

## 12. Checkpoint Policy

- **Not specified in the paper.** No statement is made about:
  - Checkpoint save frequency (per epoch / per session / best-only).
  - Which checkpoint is used for final reported numbers (last epoch of each session vs.
    early-stopped/best-validation checkpoint).
  - Whether a single continuously-updated checkpoint carries state across all sessions (implied by
    the class-incremental setting, since old node features are explicitly kept unchanged across
    sessions per Section IV.B.1), or whether separate per-session checkpoints are also saved for
    evaluation/comparison purposes.
- **Decision deferred to maintainer.** Proposed convention for the reproduction codebase (to be
  confirmed, not assumed as paper-fact): save one checkpoint at the end of the base session and one
  at the end of each incremental session, since Tables I–III require per-session accuracy
  measurements. This is a reproduction-engineering convention only, not a claim about what the
  original authors did.

## 13. Random Seed Policy

- **Not specified in the paper.** No random seed value, seed-averaging protocol (e.g., mean over N
  seeds ± std), or statement of how many runs the reported numbers in Tables I–V represent (single
  run vs. averaged) is given anywhere in the paper.
- **Decision deferred to maintainer** before claiming any reproduction number is comparable to the
  paper's reported values.

## 14. Experiment Logging

- **Not specified in the paper** beyond the general PyTorch implementation statement (Section V.B)
  and the single-GPU compute comparison in Table IV (NVIDIA GeForce RTX 2080Ti, used only for the
  Table IV timing/params comparison — not necessarily the hardware used for Tables I–III's main
  results, which is not stated).
- No mention of logging framework (TensorBoard/W&B/etc.), logged metrics beyond final per-session
  accuracy, or intermediate training-curve logging.
- **Reproduction-engineering convention (not a paper fact):** all training runs in this
  reproduction should log, per epoch and per session: total loss and each of its four components
  (`L_CE`, `L_recon`, `L_adv`, `L_energy`), plus per-session cumulative test accuracy, to
  `logs/` in a structured (e.g., JSONL or CSV) format, to support the required `ambiguity_log.md`
  and unit-test infrastructure (project rules #6–#9).

## 15. Reproducibility Requirements

Per project rules, this reproduction must:
1. Map every implemented equation to code (`equation_mapping.md`) — done for Eqs. (1)–(34).
2. Provide unit tests for every major module (GIN layer, adjacency construction, ACGA
   encoder/decoder/discriminator, HGN-EC state init/compression/Hamiltonian net/integrator/
   restoration, all four loss terms, total loss) once implementation begins.
3. Keep experimental parameters in YAML configuration files (`configs/`), separated at minimum by:
   - `configs/model/*.yaml` — architecture hyperparameters (GIN layers=4, hidden=16, adjacency
     threshold=0.8, number of learnable prompts=1, and any values resolved from the open items in
     `ambiguity_log.md` once available).
   - `configs/optim/*.yaml` — optimizer (Lion, lr=0.000325, weight_decay=1e-3), scheduler
     (CosineAnnealingWarmupRestarts), gradient accumulation (3 steps), gradient clipping (max
     norm=4.0).
   - `configs/data/{cifar100,mini_imagenet,cub200}.yaml` — dataset-specific base/incremental class
     counts, session counts, shot count, batch sizes, epoch counts, per Sections 1–6 of this
     document.
   - `configs/loss/*.yaml` — loss coefficients (λ1, λ2, λ3 = 0.04 per the reading recorded in
     ambiguity log item A10).
4. Not hard-code the paper's reported target metrics (Mean/ΔPD values in Section 35 of
   `paper_spec.md`) anywhere in evaluation code — these exist only as target reference values for
   comparison, recorded in `configs/targets/reported_results.yaml` (read-only reference, never
   written to by an evaluation run) and never asserted as ground truth to be output by our own
   runs.
5. Record all newly-discovered ambiguities in `ambiguity_log.md` as implementation proceeds,
   rather than silently resolving them.

## 16. Reported Targets (for reference only — see Section 15.4)

| Dataset | Mean (%) | ΔPD (%) |
|---|---|---|
| CIFAR-100 | 82.30 | 9.72 |
| miniImageNet | 85.05 | 8.42 |
| CUB-200-2011 | 69.54 | 17.67 |

These are **targets for reproduction only**. They must not be hard-coded into model, training, or
evaluation logic, and no claim of having reproduced them should be made until an actual
independent training/evaluation run of this codebase produces them.
