"""
models/
=======

Top-level package for ACHG-CLIP's model components.

Stage 2 populates only `models/clip/` (the frozen CLIP backbone). Later stages will add
`models/prompts/`, `models/graph/`, `models/gnn/`, `models/acga/`, `models/hgn_ec/`, and the
top-level `models/achg_clip.py` wiring module, per `docs/FINAL_IMPLEMENTATION_BLUEPRINT.md`
Part 1 / Part 9. This file intentionally stays empty of re-exports until those stages exist,
so importing `models` never implies more is available than has actually been implemented.
"""
