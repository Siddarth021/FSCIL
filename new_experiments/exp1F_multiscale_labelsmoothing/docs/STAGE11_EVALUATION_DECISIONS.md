# STAGE 11 EVALUATION DECISIONS

## 1. Known Evaluation Metrics
- **Cumulative Accuracy:** The target paper specifies that per-session accuracy columns evaluate over all classes seen so far. This is established as a `PAPER-FACT`.

## 2. Unresolved Metrics
- Additional metric breakdowns (e.g., Performance Drop / PD, Forgetting Rate, Base-only accuracy, Novel-only accuracy, Harmonic Mean) are absent from the paper text and authoritative references. 
- **Decision:** These are left `UNRESOLVED`. Only cumulative accuracy will be implemented to prevent inventing unsupported evaluation protocols.

## 3. Evaluation Protocol
- The model must evaluate without updating parameters (`model.eval()` and `torch.no_grad()`).
- The evaluation must seamlessly load a Stage 9 format checkpoint.
- **Session Definitions:** Matches the strict benchmark verified in Stage 10 (CIFAR-100: 60+40/8; miniImageNet: 60+40/8; CUB-200: 100+100/10).

## 4. Result Storage
- **Format:** JSON structure containing run configuration, session metrics, sample counts, and provenance tags.
- **Location:** `results/<dataset_name>/evaluation.json`.
- **Decision:** Result files will include a timestamp explicitly for logging but will index identically by `run_id` or `seed` to maintain deterministic tracking across reproductions.
