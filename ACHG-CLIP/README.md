# ACHG-CLIP Reproduction

Reproduction codebase for "A Few-Shot Class Incremental Learning Method Using Graph Neural
Networks" (ACHG-CLIP), IEEE TIP 2026.

Frozen specification documents (do not edit without unfreezing the process):
- `docs/FINAL_RESEARCH_DECISIONS.md` — scientific investigation, frozen.
- `docs/FINAL_IMPLEMENTATION_BLUEPRINT.md` — software architecture, frozen.

Current status: **Stage 1 (Configuration + Reproducibility Infrastructure) complete.**
See `docs/implementation_progress.md` for details, test results, and the next stage.

## Stage 1 quick start

```bash
python3 -m unittest discover -s tests -v
```

```python
from utils.config_tracking import ConfigManager
from utils.seed import seed_from_config

mgr = ConfigManager(config_root="configs")
resolved = mgr.load(dataset="cifar100")
mgr.validate(resolved)                 # raises on untagged/missing keys
seed_from_config(resolved)             # seeds random/numpy/(torch)

mgr.save(resolved, "results/example/config.yaml")
```

Every scientifically relevant parameter in `configs/**/*.yaml` carries a `provenance` tag:
`PAPER_FACT`, `REFERENCE_FACT`, `JUSTIFIED_INFERENCE`, `IMPLEMENTATION_CHOICE`, or `UNRESOLVED`.
`UNRESOLVED` parameters have `value: null` and raise `UnresolvedParameterError` if read without
passing `allow_unresolved=True` — no scientifically meaningful value is ever silently invented.
