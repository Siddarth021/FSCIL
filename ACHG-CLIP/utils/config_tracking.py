"""
utils/config_tracking.py
=========================

Stage 1 reproducibility infrastructure for ACHG-CLIP.

This module is the single source of truth for:
  1. Loading every YAML file under `configs/` into a flat, dotted-key registry of
     `ParamEntry` objects, each carrying a mandatory provenance tag.
  2. Validating that registry (no untagged keys, no missing required keys, provenance
     values restricted to the allowed set).
  3. Guarding against silent use of UNRESOLVED parameters (project rule #3).
  4. Saving/reloading a fully-resolved configuration bundle so every experiment's exact
     settings (values + provenance + source citations) are recoverable later
     (project rule: "Configuration must be saved with every experiment" /
     "Configuration must be reloadable from a saved experiment").

Design notes
------------
Every leaf parameter in every `configs/**/*.yaml` file is written as a small mapping:

    some_key:
      value: <the actual value, or null if UNRESOLVED>
      provenance: PAPER_FACT | REFERENCE_FACT | JUSTIFIED_INFERENCE | IMPLEMENTATION_CHOICE | UNRESOLVED
      source: "<citation / evidence>"
      note: "<optional extra context>"

This keeps provenance machine-readable (no comment-parsing heuristics) while remaining a
completely ordinary, human-readable YAML file. Nothing here invents defaults: a parameter
either has a `value` recorded from the paper/reference/inference/choice, or its provenance
is `UNRESOLVED` and its value is `null`, and any attempt to *use* that value for something
that requires a real number raises `UnresolvedParameterError` unless the caller explicitly
opts in via `allow_unresolved=True`.

No architectural decision from FINAL_IMPLEMENTATION_BLUEPRINT.md (the five frozen
Blocker resolutions) is re-derived or altered here -- this module only loads and validates
whatever values are recorded in the YAML files; it does not choose them.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as _dt
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

# --------------------------------------------------------------------------------------
# Provenance vocabulary
# --------------------------------------------------------------------------------------

#: The only provenance tags this project recognizes. Matches the five categories required
#: by the project rules: PAPER-FACT, REFERENCE-FACT, JUSTIFIED-INFERENCE,
#: IMPLEMENTATION-CHOICE, UNRESOLVED.
PROVENANCE_TAGS = {
    "PAPER_FACT",
    "REFERENCE_FACT",
    "JUSTIFIED_INFERENCE",
    "IMPLEMENTATION_CHOICE",
    "UNRESOLVED",
}

#: Reserved keys that make a YAML mapping a "leaf" parameter entry rather than a nested group.
_LEAF_KEYS = {"value", "provenance"}
_OPTIONAL_LEAF_KEYS = {"source", "note"}


class ConfigError(Exception):
    """Base class for all configuration errors."""


class ConfigParseError(ConfigError):
    """Raised when a YAML file does not follow the required leaf-entry schema."""


class ConfigValidationError(ConfigError):
    """Raised when a resolved configuration fails validation (untagged/missing keys)."""


class UnresolvedParameterError(ConfigError):
    """Raised when code tries to use the value of a parameter whose provenance is UNRESOLVED."""


# --------------------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------------------


@dataclass
class ParamEntry:
    """A single scientifically-tracked configuration parameter."""

    key: str  # dotted path, e.g. "model.graph.adjacency_threshold"
    value: Any
    provenance: str
    source: str = ""
    note: str = ""
    file: str = ""  # relative path of the YAML file this entry came from

    def __post_init__(self) -> None:
        if self.provenance not in PROVENANCE_TAGS:
            raise ConfigParseError(
                f"Parameter '{self.key}' has unrecognized provenance tag "
                f"'{self.provenance}'. Must be one of {sorted(PROVENANCE_TAGS)}."
            )
        if self.provenance == "UNRESOLVED" and self.value is not None:
            raise ConfigParseError(
                f"Parameter '{self.key}' is tagged UNRESOLVED but has a non-null value "
                f"({self.value!r}). An UNRESOLVED parameter must have value: null, "
                f"never a silently-assumed default."
            )

    def is_unresolved(self) -> bool:
        return self.provenance == "UNRESOLVED"

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ResolvedConfig:
    """A fully-loaded, flattened configuration for one experiment."""

    entries: Dict[str, ParamEntry] = field(default_factory=dict)
    dataset: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- read access -------------------------------------------------------------------

    def get(self, key: str, allow_unresolved: bool = False, default: Any = "__NO_DEFAULT__") -> Any:
        """Return the value stored at `key`.

        Raises:
            KeyError: if `key` is not present in this config at all.
            UnresolvedParameterError: if the parameter's provenance is UNRESOLVED and
                `allow_unresolved` is False.
        """
        if key not in self.entries:
            if default != "__NO_DEFAULT__":
                return default
            raise KeyError(f"Config key '{key}' not found in resolved configuration.")
        entry = self.entries[key]
        if entry.is_unresolved() and not allow_unresolved:
            raise UnresolvedParameterError(
                f"Parameter '{key}' is UNRESOLVED (source: {entry.source!r}). "
                f"Refusing to silently use it. Pass allow_unresolved=True only if the "
                f"caller has an explicit, documented fallback for this specific case."
            )
        return entry.value

    def get_entry(self, key: str) -> ParamEntry:
        return self.entries[key]

    def provenance_of(self, key: str) -> str:
        return self.entries[key].provenance

    def unresolved_keys(self) -> List[str]:
        return [k for k, e in self.entries.items() if e.is_unresolved()]

    def keys(self):
        return self.entries.keys()

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def nested_values(self) -> Dict[str, Any]:
        """Return a plain nested dict of values (provenance stripped), for convenience
        when passing values into code that doesn't need provenance (e.g. a quick print).
        UNRESOLVED entries appear as None, exactly as stored -- this does NOT bypass
        `get()`'s UnresolvedParameterError guard, it is purely for inspection/display."""
        out: Dict[str, Any] = {}
        for dotted_key, entry in self.entries.items():
            parts = dotted_key.split(".")
            node = out
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = entry.value
        return out


# --------------------------------------------------------------------------------------
# YAML parsing
# --------------------------------------------------------------------------------------


def _is_leaf_mapping(node: Any) -> bool:
    return isinstance(node, dict) and _LEAF_KEYS.issubset(node.keys())


def _flatten(node: Any, prefix: str, file_rel: str, out: Dict[str, ParamEntry]) -> None:
    if _is_leaf_mapping(node):
        unexpected = set(node.keys()) - _LEAF_KEYS - _OPTIONAL_LEAF_KEYS
        if unexpected:
            raise ConfigParseError(
                f"Leaf entry '{prefix}' in {file_rel} has unexpected keys: {sorted(unexpected)}. "
                f"Allowed keys are {_LEAF_KEYS | _OPTIONAL_LEAF_KEYS}."
            )
        entry = ParamEntry(
            key=prefix,
            value=node.get("value"),
            provenance=node.get("provenance"),
            source=node.get("source", ""),
            note=node.get("note", ""),
            file=file_rel,
        )
        if prefix in out:
            raise ConfigParseError(f"Duplicate config key '{prefix}' (also defined in {out[prefix].file}).")
        out[prefix] = entry
        return

    if isinstance(node, dict):
        for k, v in node.items():
            child_prefix = f"{prefix}.{k}" if prefix else str(k)
            _flatten(v, child_prefix, file_rel, out)
        return

    raise ConfigParseError(
        f"Value at '{prefix}' in {file_rel} is neither a nested group (dict) nor a valid "
        f"leaf entry (dict with 'value'/'provenance' keys). Got: {type(node)}."
    )


def load_yaml_file(path: str, group_prefix: str, config_root: str) -> Dict[str, ParamEntry]:
    """Load a single YAML file into a flat dict of dotted-key -> ParamEntry."""
    if not os.path.isfile(path):
        raise ConfigError(f"Config file not found: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    file_rel = os.path.relpath(path, config_root)
    out: Dict[str, ParamEntry] = {}
    _flatten(raw, group_prefix, file_rel, out)
    return out


# --------------------------------------------------------------------------------------
# ConfigManager
# --------------------------------------------------------------------------------------


class ConfigManager:
    """Loads, merges, validates, and (de)serializes ACHG-CLIP configuration.

    Group files mirror the frozen project structure in FINAL_IMPLEMENTATION_BLUEPRINT.md
    Part 1. This class does not choose any scientific value; it only loads what is written
    in the YAML files under `config_root` and enforces the provenance-tracking rules.
    """

    #: (dotted key prefix, path relative to config_root) pairs, always loaded.
    GROUP_FILES = [
        ("model.clip_backbone", "model/clip_backbone.yaml"),
        ("model.prompts", "model/prompts.yaml"),
        ("model.graph", "model/graph.yaml"),
        ("model.gin", "model/gin.yaml"),
        ("model.mlp_bridge", "model/mlp_bridge.yaml"),
        ("model.acga", "model/acga.yaml"),
        ("model.hgn_ec", "model/hgn_ec.yaml"),
        ("optim", "optim/optim.yaml"),
        ("loss", "loss/loss.yaml"),
        ("experiment", "experiment.yaml"),
    ]

    #: dataset name -> path relative to config_root. Loaded on demand via `load(dataset=...)`.
    DATA_FILES = {
        "cifar100": "data/cifar100.yaml",
        "mini_imagenet": "data/mini_imagenet.yaml",
        "cub200": "data/cub200.yaml",
    }

    #: Read-only reference data. Never merged into the trainable/resolved config.
    TARGETS_FILE = "targets/reported_results.yaml"

    #: Keys that MUST exist in every resolved config for it to be considered structurally
    #: complete (their *provenance* may still be UNRESOLVED -- this only checks presence,
    #: never silently fills in a value). Kept intentionally small at Stage 1: it exercises
    #: "missing required parameter detection" without hard-coding every future stage's needs.
    REQUIRED_KEYS = [
        "model.gin.num_layers",
        "model.gin.hidden_dim",
        "model.graph.adjacency_threshold",
        "model.graph.num_nodes_mode",
        "model.prompts.num_learnable_prompts_M",
        "optim.optimizer",
        "optim.learning_rate",
        "loss.lambda1_recon",
        "loss.lambda2_adv",
        "loss.lambda3_energy",
        "experiment.seed",
    ]

    def __init__(self, config_root: str = "configs"):
        self.config_root = config_root

    # -- loading -------------------------------------------------------------------------

    def load(self, dataset: Optional[str] = None) -> ResolvedConfig:
        """Load and flatten all group config files, plus one dataset's config if given."""
        entries: Dict[str, ParamEntry] = {}
        for prefix, rel_path in self.GROUP_FILES:
            path = os.path.join(self.config_root, rel_path)
            entries.update(load_yaml_file(path, prefix, self.config_root))

        if dataset is not None:
            if dataset not in self.DATA_FILES:
                raise ConfigError(
                    f"Unknown dataset '{dataset}'. Known datasets: {sorted(self.DATA_FILES)}."
                )
            path = os.path.join(self.config_root, self.DATA_FILES[dataset])
            entries.update(load_yaml_file(path, f"data.{dataset}", self.config_root))

        return ResolvedConfig(entries=entries, dataset=dataset, meta={})

    def load_targets(self) -> Dict[str, ParamEntry]:
        """Load the read-only reported-results reference data.

        This is intentionally kept OUT of `load()`'s ResolvedConfig: it must never be
        merged into a trainable configuration or accidentally written to
        (reproduction_protocol.md Section 15.4).
        """
        path = os.path.join(self.config_root, self.TARGETS_FILE)
        return load_yaml_file(path, "targets", self.config_root)

    # -- validation ----------------------------------------------------------------------

    def validate(
        self,
        resolved: ResolvedConfig,
        required_keys: Optional[List[str]] = None,
        strict_unresolved: bool = False,
    ) -> List[str]:
        """Validate a resolved configuration.

        Args:
            resolved: the ResolvedConfig to check.
            required_keys: keys that must be present (defaults to `self.REQUIRED_KEYS`).
            strict_unresolved: if True, raise if ANY entry has provenance UNRESOLVED
                (use this before starting an actual training run; leave False when just
                inspecting/loading a config, since Stage 1 explicitly allows UNRESOLVED
                entries to exist and be visible).

        Returns:
            A list of human-readable warnings (e.g. which keys are UNRESOLVED). Warnings
            are returned rather than printed, so callers/tests can inspect them.

        Raises:
            ConfigValidationError: on any untagged key, unrecognized provenance value, or
                missing required key.
        """
        required_keys = self.REQUIRED_KEYS if required_keys is None else required_keys
        errors: List[str] = []
        warnings: List[str] = []

        for key, entry in resolved.entries.items():
            if not entry.provenance:
                errors.append(f"Key '{key}' has no provenance tag.")
            elif entry.provenance not in PROVENANCE_TAGS:
                errors.append(
                    f"Key '{key}' has invalid provenance '{entry.provenance}'. "
                    f"Must be one of {sorted(PROVENANCE_TAGS)}."
                )
            if entry.is_unresolved():
                warnings.append(f"Key '{key}' is UNRESOLVED (source: {entry.source!r}).")

        for key in required_keys:
            if key not in resolved.entries:
                errors.append(f"Required key '{key}' is missing from the resolved configuration.")

        if strict_unresolved and warnings:
            errors.append(
                "strict_unresolved=True but the following parameters are UNRESOLVED: "
                + "; ".join(w.split(" (source:")[0] for w in warnings)
            )

        if errors:
            raise ConfigValidationError(
                "Configuration validation failed with "
                f"{len(errors)} error(s):\n  - " + "\n  - ".join(errors)
            )

        return warnings

    # -- saving / reloading ----------------------------------------------------------------

    @staticmethod
    def _git_hash() -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if out.returncode == 0:
                return out.stdout.strip()
        except Exception:
            pass
        return None

    def build_meta(self, seed: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta = {
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "git_commit": self._git_hash(),
            "seed": seed,
        }
        if extra:
            meta.update(extra)
        return meta

    def save(self, resolved: ResolvedConfig, path: str) -> None:
        """Serialize a fully-resolved config (values + provenance + source + meta) to disk.

        The saved file is a complete, self-contained snapshot: reloading it via `load_saved`
        reconstructs an equivalent ResolvedConfig without needing the original `configs/`
        tree, which is what makes a `results/{run_id}/config.yaml` or checkpoint's embedded
        config independently auditable later.
        """
        bundle = {
            "dataset": resolved.dataset,
            "meta": resolved.meta,
            "entries": {k: v.as_dict() for k, v in resolved.entries.items()},
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(bundle, f, sort_keys=True, default_flow_style=False)

    @staticmethod
    def load_saved(path: str) -> ResolvedConfig:
        """Reconstruct a ResolvedConfig from a file written by `save()`."""
        if not os.path.isfile(path):
            raise ConfigError(f"Saved config file not found: {path}")
        with open(path, "r") as f:
            bundle = yaml.safe_load(f) or {}

        entries: Dict[str, ParamEntry] = {}
        for key, entry_dict in bundle.get("entries", {}).items():
            entries[key] = ParamEntry(
                key=entry_dict["key"],
                value=entry_dict["value"],
                provenance=entry_dict["provenance"],
                source=entry_dict.get("source", ""),
                note=entry_dict.get("note", ""),
                file=entry_dict.get("file", ""),
            )
        return ResolvedConfig(
            entries=entries,
            dataset=bundle.get("dataset"),
            meta=bundle.get("meta", {}),
        )


# --------------------------------------------------------------------------------------
# Convenience module-level helpers
# --------------------------------------------------------------------------------------


def load_config(dataset: Optional[str] = None, config_root: str = "configs") -> ResolvedConfig:
    """One-shot convenience loader used by scripts/tests."""
    return ConfigManager(config_root=config_root).load(dataset=dataset)


def deep_copy_resolved(resolved: ResolvedConfig) -> ResolvedConfig:
    """Return an independent deep copy (useful in tests that mutate a config to check
    validation failure paths without touching the original)."""
    return copy.deepcopy(resolved)
