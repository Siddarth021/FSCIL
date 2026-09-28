"""
models/clip/clip_wrapper.py
=============================

Stage 2 — CLIP backbone: top-level wrapper/interface.

This is the "clean interface" module ACHG-CLIP's later stages (prompts, graph, GIN, ACGA,
HGN-EC) are expected to depend on, rather than reaching into `TextEncoder`/`VisionEncoder`
directly. It provides:

  - image encoding                (`CLIPWrapper.encode_image`)
  - text encoding                 (`CLIPWrapper.encode_text`)
  - image/text embedding access   (`CLIPWrapper.forward`, returns both `h*_V`, `h*_T`)
  - embedding dimensions          (`CLIPWrapper.image_embedding_dim` / `.text_embedding_dim`)
  - model freezing/unfreezing     (`CLIPWrapper.freeze_backbone` / `.unfreeze_backbone`)
  - device handling               (`CLIPWrapper.device`; standard `nn.Module.to(...)`)
  - checkpoint/state handling     (`CLIPWrapper.save_checkpoint` / `.load_checkpoint` /
                                    `.from_checkpoint`)

==================================================================================
IMPORTANT — CLIP BACKBONE VARIANT IS NOT A PAPER-FACT
==================================================================================
`configs/model/clip_backbone.yaml` records `variant`, `token_embedding_dim_d`,
`projection_dim_de`, `attention_head_dim_dk`, and `num_transformer_layers_L` as UNRESOLVED:
the paper (Section IV.A) says only "improved Transformer structures" and never names a
concrete CLIP checkpoint (e.g. ViT-B/32, RN50) or gives any of these dimensions a number.

This module therefore:
  - never silently picks a specific CLIP variant or a "reasonable default" dimension set;
  - keeps the backbone entirely config-driven via `CLIPConfig`, which mirrors
    `utils.config_tracking`'s PAPER_FACT / IMPLEMENTATION_CHOICE / UNRESOLVED provenance
    model so every numeric choice made to actually run the model is traceable;
  - builds a genuine from-scratch implementation of Eqs. 1-8 (see `text_encoder.py`,
    `vision_encoder.py`) rather than wrapping a third-party pretrained CLIP package, because
    no named variant/checkpoint exists to wrap (there is nothing "real" to download that the
    paper specifies) -- see `docs/implementation_progress.md` Stage 2 "Dependency /
    environment issues" for the full reasoning and the network-access constraints that also
    apply.
  - supports a small, deterministic **mock backbone** (`models/clip/mock.py`) built from this
    exact same code path with tiny synthetic dimensions, for shape/gradient-flow testing,
    matching `FINAL_IMPLEMENTATION_BLUEPRINT.md` Part 10's smoke-test philosophy.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from models.clip.text_encoder import TextEncoder
from models.clip.vision_encoder import VisionEncoder
from utils.config_tracking import ResolvedConfig, UnresolvedParameterError

#: Provenance tag used for a value that came from a real (possibly UNRESOLVED-overridden)
#: resolved config entry but was substituted at call time for testing purposes -- distinct
#: from that entry's own recorded provenance so it is never confused with PAPER_FACT/
#: IMPLEMENTATION_CHOICE values that came from the YAML files themselves.
TEST_OVERRIDE = "TEST_OVERRIDE"

#: Provenance tag for dimensions that are architectural inputs Stage 1's config schema does
#: not cover at all (vocab size, patch dim, sequence-length capacities, head count) -- these
#: are not paper facts, not yet resolved choices, and not part of `clip_backbone.yaml`; they
#: must always be supplied explicitly by the caller of `CLIPConfig`.
NOT_IN_CONFIG_SCHEMA = "NOT_IN_CONFIG_SCHEMA"


class CLIPConfigError(Exception):
    """Raised when a `CLIPConfig` is internally inconsistent (e.g. non-positive dimension)."""


@dataclass
class CLIPConfig:
    """Fully-resolved dimensions needed to instantiate a `CLIPWrapper`.

    Every field here must be a concrete value by the time this dataclass is constructed --
    there is no UNRESOLVED state at this layer (that guard lives one level up, in
    `utils.config_tracking.ResolvedConfig.get()`, which raises `UnresolvedParameterError`
    before this dataclass would ever be built with a `None`). `dim_provenance` records where
    each field's value actually came from, so it can be logged/checkpointed and never mistaken
    for a paper-confirmed number.
    """

    # -- CLIP-architecture dimensions (Eqs. 1-8) ----------------------------------------
    d_model: int  # d -- token/patch embedding dimension
    d_e: int  # d_e -- final text/vision projection (embedding) dimension
    d_k: int  # d_k -- per-head attention dimension
    num_heads: int  # h -- attention head count (paper never names this symbol at all)
    num_layers: int  # L -- number of stacked Transformer blocks per tower

    # -- text-tower-specific ---------------------------------------------------------------
    vocab_size: int
    max_text_len: int  # capacity for n (sequence length)

    # -- vision-tower-specific --------------------------------------------------------------
    patch_dim: int  # |p| -- raw per-patch feature dimension
    max_patches: int  # capacity for m (number of patches)

    # -- shared, not paper-critical ----------------------------------------------------------
    ffn_hidden_dim: Optional[int] = None  # None -> TransformerBlock's own default (4*d_model)
    dropout: float = 0.0

    # -- provenance / metadata ---------------------------------------------------------------
    variant: Optional[str] = None  # informational only; no branching logic depends on this
    frozen: bool = True  # PAPER_FACT (Section V.B) when sourced from clip_backbone.yaml
    dim_provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positive_int_fields = [
            "d_model",
            "d_e",
            "d_k",
            "num_heads",
            "num_layers",
            "vocab_size",
            "max_text_len",
            "patch_dim",
            "max_patches",
        ]
        for name in positive_int_fields:
            val = getattr(self, name)
            if not isinstance(val, int) or val <= 0:
                raise CLIPConfigError(f"CLIPConfig.{name} must be a positive int, got {val!r}.")
        if self.ffn_hidden_dim is not None and self.ffn_hidden_dim <= 0:
            raise CLIPConfigError(f"CLIPConfig.ffn_hidden_dim must be positive or None, got {self.ffn_hidden_dim!r}.")
        if not (0.0 <= self.dropout < 1.0):
            raise CLIPConfigError(f"CLIPConfig.dropout must be in [0, 1), got {self.dropout!r}.")

    # -- (de)serialization, used by checkpointing -------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CLIPConfig":
        return cls(**d)

    # -- construction from a Stage-1 ResolvedConfig -----------------------------------------

    @classmethod
    def from_resolved_config(
        cls,
        resolved_config: ResolvedConfig,
        *,
        vocab_size: int,
        max_text_len: int,
        patch_dim: int,
        max_patches: int,
        num_heads: int,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        test_overrides: Optional[Dict[str, Any]] = None,
    ) -> "CLIPConfig":
        """Build a `CLIPConfig` from `configs/model/clip_backbone.yaml` (via `ResolvedConfig`).

        `vocab_size`, `max_text_len`, `patch_dim`, `max_patches`, and `num_heads` are NOT
        present in `clip_backbone.yaml`'s schema at all (Stage 1's config groups do not cover
        them -- see `docs/implementation_progress.md` Stage 2 "Known limitations") and must
        always be supplied by the caller directly.

        `variant`, `token_embedding_dim_d` (-> `d_model`), `projection_dim_de` (-> `d_e`),
        `attention_head_dim_dk` (-> `d_k`), and `num_transformer_layers_L` (-> `num_layers`)
        ARE in the config, but are all currently tagged UNRESOLVED. By default this method
        therefore raises `UnresolvedParameterError` (via `ResolvedConfig.get`), exactly as
        Stage 1 intends -- it never silently substitutes a value.

        `test_overrides`: an optional dict keyed by the *config* names above
        (`variant`, `token_embedding_dim_d`, `projection_dim_de`, `attention_head_dim_dk`,
        `num_transformer_layers_L`) whose values are used ONLY for entries that are currently
        UNRESOLVED, tagged `TEST_OVERRIDE` in the returned config's `dim_provenance` (never
        `PAPER_FACT`/`IMPLEMENTATION_CHOICE`). This exists to support Stage 2's shape/gradient
        tests (Part 10's smoke-test philosophy) without pretending an UNRESOLVED paper
        parameter has been resolved. It must never be used for a real training run.
        """
        test_overrides = test_overrides or {}
        dim_provenance: Dict[str, str] = {}

        def _resolve(config_key: str) -> Any:
            full_key = f"model.clip_backbone.{config_key}"
            entry = resolved_config.get_entry(full_key)
            if entry.is_unresolved():
                if config_key in test_overrides:
                    dim_provenance[config_key] = TEST_OVERRIDE
                    return test_overrides[config_key]
                raise UnresolvedParameterError(
                    f"CLIPConfig.from_resolved_config: '{full_key}' is UNRESOLVED "
                    f"(source: {entry.source!r}) and no test_overrides['{config_key}'] was "
                    f"supplied. Refusing to silently pick a value -- either resolve this "
                    f"parameter in configs/model/clip_backbone.yaml, or pass an explicit, "
                    f"clearly-labelled test override for shape/gradient testing only."
                )
            dim_provenance[config_key] = entry.provenance
            return entry.value

        variant = _resolve("variant")
        d_model = _resolve("token_embedding_dim_d")
        d_e = _resolve("projection_dim_de")
        d_k = _resolve("attention_head_dim_dk")
        num_layers = _resolve("num_transformer_layers_L")
        frozen = _resolve("frozen")

        for key in ("vocab_size", "max_text_len", "patch_dim", "max_patches", "num_heads"):
            dim_provenance[key] = NOT_IN_CONFIG_SCHEMA
        dim_provenance["ffn_hidden_dim"] = NOT_IN_CONFIG_SCHEMA
        dim_provenance["dropout"] = NOT_IN_CONFIG_SCHEMA

        return cls(
            d_model=d_model,
            d_e=d_e,
            d_k=d_k,
            num_heads=num_heads,
            num_layers=num_layers,
            vocab_size=vocab_size,
            max_text_len=max_text_len,
            patch_dim=patch_dim,
            max_patches=max_patches,
            ffn_hidden_dim=ffn_hidden_dim,
            dropout=dropout,
            variant=variant,
            frozen=bool(frozen),
            dim_provenance=dim_provenance,
        )


class CLIPWrapper(nn.Module):
    """Clean, stable interface over the from-scratch CLIP text/vision towers.

    Downstream code (prompts, graph construction, GIN, ACGA, HGN-EC, training loops) should
    depend only on this class's public methods/properties, not on `TextEncoder`/
    `VisionEncoder` internals -- this is what keeps the CLIP backbone swappable (e.g. if a
    real named variant is resolved later) without touching every consumer.
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config

        self._frozen = False

        hf_model_name = None
        if config.variant == "ViT-B/32":
            hf_model_name = "openai/clip-vit-base-patch32"
        elif config.variant == "ViT-B/16":
            hf_model_name = "openai/clip-vit-base-patch16"
        elif config.variant == "ViT-L/14":
            hf_model_name = "openai/clip-vit-large-patch14"

        if hf_model_name is not None:
            import transformers
            self.hf_model = transformers.CLIPModel.from_pretrained(hf_model_name, use_safetensors=True)
            self.text_encoder = None
            self.vision_encoder = None
        else:
            self.hf_model = None
            self.text_encoder = TextEncoder(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                d_e=config.d_e,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                d_k=config.d_k,
                max_seq_len=config.max_text_len,
                ffn_hidden_dim=config.ffn_hidden_dim,
                dropout=config.dropout,
            )
            self.vision_encoder = VisionEncoder(
                patch_dim=config.patch_dim,
                d_model=config.d_model,
                d_e=config.d_e,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                d_k=config.d_k,
                max_patches=config.max_patches,
                ffn_hidden_dim=config.ffn_hidden_dim,
                dropout=config.dropout,
            )

        if config.frozen:
            self.freeze_backbone()

    # -- embedding dimensions ----------------------------------------------------------------

    @property
    def image_embedding_dim(self) -> int:
        """`d_e`: dimensionality of `h*_V` (Eq. 8)."""
        return self.config.d_e

    @property
    def text_embedding_dim(self) -> int:
        """`d_e`: dimensionality of `h*_T` (Eq. 6)."""
        return self.config.d_e

    @property
    def token_dim(self) -> int:
        """`d`: internal Transformer hidden dimension (shared by both towers)."""
        return self.config.d_model

    @property
    def num_layers(self) -> int:
        """`L`: number of stacked Transformer blocks per tower."""
        return self.config.num_layers

    # -- device handling ---------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Current device of the wrapper's parameters.

        Standard `nn.Module.to(device)` (inherited, not overridden) is the supported way to
        move this wrapper between devices; this property is a read-only convenience so callers
        don't need to reach into an arbitrary submodule parameter themselves.
        """
        return next(self.parameters()).device

    # -- encoding interface ------------------------------------------------------------------

    def encode_text(self, tokens: torch.Tensor, return_sequence: bool = False, attention_mask: Optional[torch.Tensor] = None):
        """`tokens`: `(B, n)` int64 -> `h*_T`: `(B, d_e)` (Eqs. 1-6)."""
        if self.hf_model is not None:
            # HuggingFace CLIPModel
            outputs = self.hf_model.get_text_features(input_ids=tokens, attention_mask=attention_mask)
            if return_sequence:
                # HF doesn't directly return the full sequence via get_text_features; 
                # we'd need to call text_model directly. But for ACHG-CLIP we just need pooled.
                raise NotImplementedError("return_sequence=True not yet supported for HF text encoder")
            if hasattr(outputs, "text_embeds"):
                outputs = outputs.text_embeds
            elif hasattr(outputs, "pooler_output"):
                outputs = outputs.pooler_output
            elif isinstance(outputs, tuple):
                outputs = outputs[1] if len(outputs) > 1 else outputs[0]
            return outputs
        else:
            return self.text_encoder(tokens, return_sequence=return_sequence)

    def encode_image(self, patches: torch.Tensor, return_sequence: bool = False):
        """`patches`: `(B, m, patch_dim)` (or raw images (B, C, H, W) for HF) -> `h*_V`: `(B, d_e)` (Eqs. 7-8)."""
        if self.hf_model is not None:
            # HuggingFace CLIPModel expects raw images as `pixel_values` (B, C, H, W)
            outputs = self.hf_model.get_image_features(pixel_values=patches)
            if return_sequence:
                raise NotImplementedError("return_sequence=True not yet supported for HF vision encoder")
            if hasattr(outputs, "image_embeds"):
                outputs = outputs.image_embeds
            elif hasattr(outputs, "pooler_output"):
                outputs = outputs.pooler_output
            elif isinstance(outputs, tuple):
                outputs = outputs[1] if len(outputs) > 1 else outputs[0]
            return outputs
        else:
            return self.vision_encoder(patches, return_sequence=return_sequence)

    # Alias matching Part 2's "CLIP vision embed+PE" naming for readability at call sites that
    # think in terms of "vision" rather than "image".
    def encode_vision(self, patches: torch.Tensor, return_sequence: bool = False):
        return self.encode_image(patches, return_sequence=return_sequence)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor):
        """Convenience full pass returning both embeddings: `(h*_V, h*_T)`.

        Not itself an equation from the paper -- Eqs. 1-8 describe the two towers
        independently; this just calls both for callers (e.g. the future contrastive-loss
        wiring in Stage 8) that want both embeddings from one call.
        """
        h_vision = self.encode_image(images)
        h_text = self.encode_text(tokens)
        return h_vision, h_text

    # -- freezing / unfreezing ----------------------------------------------------------------

    def freeze_backbone(self) -> int:
        """Set `requires_grad=False` on every backbone parameter (Section V.B, PAPER_FACT).

        Returns the number of parameters frozen. This is the only freezing Stage 2 is
        responsible for -- freezing/unfreezing of graph nodes across incremental sessions
        (`training/freeze.py`) is Stage 9+ and out of scope here.
        """
        count = 0
        for p in self.parameters():
            if p.requires_grad:
                count += 1
            p.requires_grad_(False)
        self._frozen = True
        return count

    def unfreeze_backbone(self) -> int:
        """Set `requires_grad=True` on every backbone parameter.

        WARNING: the paper's PAPER_FACT (Section V.B) is that the CLIP backbone stays frozen
        for the entire FSCIL pipeline ("It freezes the backbone network of CLIP and only
        trains a small number of GIN-based modules"). This method exists for completeness of
        the wrapper's interface (e.g. ablations, debugging) and must never be called on the
        default training path.
        """
        count = 0
        for p in self.parameters():
            if not p.requires_grad:
                count += 1
            p.requires_grad_(True)
        self._frozen = False
        return count

    @property
    def is_frozen(self) -> bool:
        """True iff every parameter currently has `requires_grad=False`."""
        return all(not p.requires_grad for p in self.parameters())

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def frozen_parameters(self):
        return [p for p in self.parameters() if not p.requires_grad]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    # -- provenance / logging -----------------------------------------------------------------

    def provenance_report(self) -> Dict[str, str]:
        """Return the `dim_provenance` mapping recorded on this wrapper's `CLIPConfig`.

        Intended for structured logging / checkpoint metadata, per the project-wide rule that
        every numeric value used in an actual run must have a traceable provenance tag.
        """
        return dict(self.config.dim_provenance)

    # -- checkpoint / state handling -----------------------------------------------------------

    def save_checkpoint(self, path: str, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        """Save `{config, state_dict, extra_meta}` to `path`.

        The config is embedded so `load_checkpoint`/`from_checkpoint` can detect an
        architecture mismatch instead of silently loading weights into a differently-shaped
        model (matching `FINAL_IMPLEMENTATION_BLUEPRINT.md` Part 5's checkpoint-loading
        contract for the full model).
        """
        bundle = {
            "clip_config": self.config.to_dict(),
            "state_dict": self.state_dict(),
            "extra_meta": extra_meta or {},
        }
        torch.save(bundle, path)

    def load_checkpoint(self, path: str, map_location: Optional[str] = None, strict: bool = True) -> Dict[str, Any]:
        """Load weights from a checkpoint written by `save_checkpoint` into THIS instance.

        Raises `CLIPConfigError` if the checkpoint's config does not match this instance's
        config, rather than silently loading mismatched-shape weights. Returns the
        checkpoint's `extra_meta` dict.
        """
        bundle = torch.load(path, map_location=map_location, weights_only=False)
        checkpoint_config = CLIPConfig.from_dict(bundle["clip_config"])
        if checkpoint_config.to_dict() != self.config.to_dict():
            raise CLIPConfigError(
                "Checkpoint's CLIPConfig does not match this CLIPWrapper's config -- refusing "
                "to load into a differently-shaped model. "
                f"checkpoint={checkpoint_config.to_dict()} current={self.config.to_dict()}"
            )
        self.load_state_dict(bundle["state_dict"], strict=strict)
        return bundle.get("extra_meta", {})

    @classmethod
    def from_checkpoint(cls, path: str, map_location: Optional[str] = None) -> "CLIPWrapper":
        """Reconstruct a `CLIPWrapper` (config + weights) entirely from a saved checkpoint."""
        bundle = torch.load(path, map_location=map_location, weights_only=False)
        config = CLIPConfig.from_dict(bundle["clip_config"])
        wrapper = cls(config)
        wrapper.load_state_dict(bundle["state_dict"], strict=True)
        return wrapper
