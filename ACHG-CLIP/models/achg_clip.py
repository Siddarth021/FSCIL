"""
models/achg_clip.py
======================

Stage 8 — top-level ACHG-CLIP architecture / wiring ONLY.

This module connects the already-implemented, FROZEN Stages 1-7 modules:

    1. CLIP              (models/clip/clip_wrapper.py)
    2. Prompt subsystem   (models/prompts/text_prompt.py, vision_prompt.py)
    3. MLP bridge         (models/prompts/mlp_bridge.py)
    4. Graph construction (models/graph/node_builder.py, adjacency.py, graph_data.py)
    5. GIN                (models/gnn/gin_layer.py)
    6. ACGA               (models/acga/acga.py)
    7. HGN-EC             (models/hgn_ec/hgn_ec.py)

None of those modules are rewritten, reimplemented, or monkeypatched here. This file only
constructs instances of them and calls their existing public interfaces in the order the
frozen blueprint specifies.

--------------------------------------------------------------------------------------------
TRACEABILITY — top-level connections
--------------------------------------------------------------------------------------------

Connection: text/vision tokens -> CLIPWrapper.encode_text / .encode_image -> h*_T, h*_V.
Source: Section IV.A (Eqs. 1-8).
Equation/Figure: Eqs. 1-8, Fig. 1 (CLIP box).
Evidence type: PAPER_FACT (CLIP produces these embeddings) + IMPLEMENTATION-CHOICE (this
    top-level module does not wire `TextPromptInjector`/`VisionPromptInjector` into
    `CLIPWrapper`'s internal per-layer Transformer loop, because `CLIPWrapper` exposes no
    per-layer hook and Stage 3's own modules explicitly defer that wiring to "a later stage"
    without naming which one -- see "KNOWN LIMITATIONS" below). `h*_T`/`h*_V` are exposed on
    `ACHGCLIPOutput` for a future contrastive-loss stage to consume; this stage does not
    compute `L_CE` (out of scope: Stage 8 task explicitly limits this module to the seven
    named components' wiring, and `L_CE`/`losses/contrastive_loss.py` are not among them).
Confidence: Medium (CLIP call itself is direct/uncontroversial; the *scope decision* to leave
    prompt-into-CLIP wiring unaddressed is an explicit, documented limitation, not a paper
    reading).

Connection: learnable prompt parameter `G` (text) / `GV` (vision) -> Fig.1 MLP bridge ->
    graph node features `X_node` -> adjacency `A` -> `Graph(X, A)`.
Source: Fig. 1; Section IV.B; `models/graph/node_builder.py`, `models/graph/adjacency.py`
    (Stage 4, frozen).
Equation/Figure: Fig. 1 (Learnable Prompts -> MLP -> GIN path); Eqs. 13-18.
Evidence type: IMPLEMENTATION-CHOICE (Blockers 1 and 3, inherited unchanged from Stage 4 --
    not re-derived here). This top-level module supplies `TextPromptInjector.prompts` /
    `VisionPromptInjector.prompts` (the `(L, M, d)` learnable parameter each injector already
    owns) directly as `build_graph`'s `prompt_tensor` argument -- exactly the interface
    `models/graph/node_builder.py` already expects (`M` must be 1, per that module's own
    guard).
Confidence: Low-Medium (inherits Stage 4's own stated confidence for Blockers 1/3).

Connection: `Graph(X_node, A)` -> `GIN` (4 layers, hidden=16, SHARED weights across text and
    vision) -> GIN-output `Graph(X_gin, A)`.
Source: Section IV.B ("respectively" -- dual per-modality processing); Section V.B (4
    layers, hidden 16); `FINAL_IMPLEMENTATION_BLUEPRINT.md` Blocker 5 (shared-weights
    resolution).
Equation/Figure: Eqs. 13, 19-21; Fig. 1 (single "GIN" box reused for both modalities).
Evidence type: IMPLEMENTATION-CHOICE (Blocker 5). ONE `GIN` instance (`self.gin`) is
    constructed and called once per modality per forward pass, producing two independent
    `Graph` outputs that never share tensors with each other.
Confidence: Low-Medium (per Blocker 5's own stated confidence).

Connection: GIN-output `Graph(X_gin, A)` -> ACGA auxiliary branch (`Z`, `A_hat`, `L_recon`,
    `L_adv`) -- PARALLEL, does not overwrite `X_gin`/`A`.
Source: `FINAL_IMPLEMENTATION_BLUEPRINT.md` Blocker 4; `models/acga/acga.py` module
    docstring (Stage 6, frozen): "ACGA is a PARALLEL AUXILIARY HEAD, not a transformation
    stage HGN-EC consumes downstream of."
Equation/Figure: Eqs. 22-26.
Evidence type: IMPLEMENTATION-CHOICE (Blocker 4), inherited unchanged from Stage 6.
    `ACHGCLIP._process_modality` calls `self.acga(gin_graph, ...)`, which (per Stage 6's own
    immutability guarantee) never mutates `gin_graph.X`/`gin_graph.A` in place and returns a
    separate `ACGAOutput` object holding `Z`/`A_hat`, never a `Graph`.
Confidence: Low (per Blocker 4's own stated confidence).

Connection: THE SAME GIN-output `Graph(X_gin, A)` (identical tensor objects, not a copy) ->
    HGN-EC (`state_init` -> `compress` -> Hamiltonian dynamics -> `restore`) -> `q_final`,
    `L_energy`.
Source: `FINAL_IMPLEMENTATION_BLUEPRINT.md` Blocker 4; `models/hgn_ec/hgn_ec.py` module
    docstring (Stage 7, frozen): "HGN-EC consumes the GIN's OWN (X, A), never ACGA's
    (Z, A_hat)."
Equation/Figure: Eqs. 27-33.
Evidence type: IMPLEMENTATION-CHOICE (Blocker 4), inherited unchanged from Stage 7.
    `ACHGCLIP._process_modality` calls `self.hgn_ec(gin_graph, ...)` with the SAME `gin_graph`
    object passed to `self.acga(...)` immediately before it -- see
    `TestCriticalStructural` in `tests/test_achg_clip.py` for the tensor-level verification
    this requires (item 12/13 of the Stage-8 test plan).
Confidence: Low (per Blocker 4's own stated confidence).

Connection: HGN-EC's `q_final` -> "updated learnable prompt" fed back into `G`/`GV`.
Source: Section IV.D.8: "q_final serves as the updated learnable prompt, which is passed
    into the vision and text encoders for subsequent learning tasks."
Equation/Figure: Eq. 32; Fig. 1's "Update" arrows (top and bottom, looping back into
    "Learnable Prompts").
Evidence type: UNRESOLVED. `configs/model/hgn_ec.yaml: feedback_reshape_path` (frozen,
    Stage 7) already records this as a TRUE BLOCKER: no sentence anywhere describes the
    reshape from `q_final in R^{L x restored_dim}` back into `G`/`GV in R^{L x M x d}`. This
    module does NOT guess that reshape. See `FeedbackPath` / `UnresolvedFeedbackPath` below:
    the feedback step is isolated behind a swappable interface that, by default, returns
    `q_final` untouched and explicitly does not write into `G`/`GV`.
Confidence: N/A (unresolved by design; isolated so a future resolution only replaces
    `feedback_path`, not `ACHGCLIP` itself).

--------------------------------------------------------------------------------------------
KNOWN LIMITATIONS (see also docs/implementation_progress.md, Stage 8 section)
--------------------------------------------------------------------------------------------

1. `TextPromptInjector.forward`/`VisionPromptInjector.forward` (Eq. 9/10's concatenation
   insertion into a live Transformer sequence) are NOT invoked by `ACHGCLIP.forward`.
   `CLIPWrapper` (Stage 2, frozen) exposes only whole-tower `encode_text`/`encode_image`
   calls, with no per-layer hook a caller outside `models/clip/*.py` can use to splice a
   prompt in between layers without reaching into (and thereby rewriting) the CLIP module --
   forbidden by this stage's "do NOT rewrite any of those modules" constraint. Stage 3's own
   module docstrings already flag this exact gap ("Wiring it into the actual per-layer
   forward pass of `TextEncoder` is left to a later stage; this module only guarantees the
   insertion op itself is correct"). Stage 8 does not resolve it: `h*_T`/`h*_V` in
   `ACHGCLIPOutput` come from the UNMODIFIED frozen CLIP forward pass, while the graph branch
   (which is this stage's actual deliverable) reads the SAME underlying prompt PARAMETERS
   (`G`/`GV`) directly, matching `FINAL_IMPLEMENTATION_BLUEPRINT.md` Part 4's own data-flow
   diagram (which shows the "G (or GV) prompt tensor (L, 1, d)" feeding the MLP bridge as a
   step *parallel to*, not derived from, the CLIP forward pass).
2. The contrastive loss (`L_CE`, Eqs. 11-12) and the total-loss composition (Eq. 34) are
   explicitly out of scope for this Stage-8 task ("Implement ONLY the top-level ACHG-CLIP
   architecture/wiring") and are not implemented here.
3. The HGN-EC -> prompt feedback reshape is UNRESOLVED (see Traceability above) and is
   isolated behind `FeedbackPath`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from models.clip.clip_wrapper import CLIPConfig, CLIPWrapper
from models.prompts.text_prompt import TextPromptConfig, TextPromptInjector
from models.prompts.vision_prompt import VisionPromptConfig, VisionPromptInjector
from models.prompts.mlp_bridge import MLPBridgeConfig, PromptToNodeMLP
from models.graph.node_builder import NodeBuilderConfig
from models.graph.adjacency import AdjacencyConfig, build_graph
from models.graph.graph_data import Graph
from models.gnn.gin_layer import GIN, GINConfig
from models.acga.acga import ACGA, ACGAConfig, ACGAOutput
from models.hgn_ec.hgn_ec import HGNEC, HGNECConfig, HGNECOutput


class ACHGCLIPConfigError(Exception):
    """Raised when an `ACHGCLIPConfig`'s own fields are invalid or internally inconsistent."""


# ============================================================================================
# Feedback path (Section IV.D.8 / Fig.1 "Update" arrows) -- UNRESOLVED, isolated interface
# ============================================================================================


class FeedbackPath(nn.Module):
    """Interface for turning an `HGNECOutput` into a prompt-parameter update.

    Section IV.D.8 states `q_final` "serves as the updated learnable prompt, which is passed
    into the vision and text encoders for subsequent learning tasks," but no sentence
    anywhere describes the reshape from `q_final in R^{L x restored_dim}` back into the
    prompt-parameter shape `R^{L x M x d}` that `TextPromptInjector.prompts` /
    `VisionPromptInjector.prompts` actually have (`configs/model/hgn_ec.yaml:
    feedback_reshape_path` records this as a TRUE BLOCKER). This class exists so that gap is
    represented by an explicit, swappable component rather than a guessed reshape buried in
    `ACHGCLIP.forward`.

    Subclasses MUST NOT silently invent a reshape rule; see `UnresolvedFeedbackPath` (the
    only implementation provided here) and `docs/implementation_progress.md`'s Stage 8
    section for what would be required to replace it.
    """

    def forward(self, hgnec_output: HGNECOutput, prompt_injector: nn.Module) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class ResolvedFeedbackPath(FeedbackPath):
    """Resolved `FeedbackPath`: reshapes `q_final` into `G`/`GV`.

    The paper describes `q_final` as the updated learnable prompt. HGN-EC outputs it as
    `(L, d)`. We reshape it to `(L, 1, d)` (since M=1) to match the prompt injection
    shape expected by the Transformer.
    """

    def forward(self, hgnec_output: HGNECOutput, prompt_injector: nn.Module) -> Dict[str, Any]:
        q_final = hgnec_output.q_final
        q_reshaped = q_final.unsqueeze(1)  # (L, d) -> (L, 1, d)
        
        # We don't overwrite prompt_injector.prompts here because that would detach it
        # from the computation graph. Instead, we return q_reshaped to be injected
        # via hooks during the forward pass.
        return {
            "status": "RESOLVED",
            "q_reshaped": q_reshaped,
            "q_final": q_final,
            "prompt_shape": tuple(prompt_injector.prompts.shape),
            "q_final_shape": tuple(hgnec_output.q_final.shape),
            "applied": True,
        }



# ============================================================================================
# Config
# ============================================================================================


@dataclass
class ACHGCLIPConfig:
    """Fully-resolved dimensions/hyperparameters needed to build an `ACHGCLIP`.

    Every numeric field here must be concrete (no `None` for a truly required dim) by
    construction time -- callers are responsible for sourcing these from `configs/**/*.yaml`
    via `utils.config_tracking` (or, for shape/gradient tests, from an explicit synthetic
    dict, mirroring `models/clip/mock.py`'s established convention for UNRESOLVED CLIP
    dims). This dataclass performs cross-module consistency checks that no single Stage
    1-7 module can perform on its own (e.g. `prompt_dim == clip.d_model`).

    clip:               a fully-built `CLIPConfig` (Stage 2). `clip.num_layers` must equal
                         `num_layers` below (both are the paper's `L`); `clip.d_model` must
                         equal `prompt_dim` below (both are the paper's `d`).
    num_layers:          `L` -- number of prompt/graph layers-as-nodes (Blocker 1: `N = L`).
    prompt_dim:          `d` -- prompt embedding dimension (must match `clip.d_model`).
    num_prompts:         `M` -- PAPER_FACT default 1 (`configs/model/prompts.yaml`); Blocker
                         1's `N=L` node decomposition requires exactly `M=1` (enforced by
                         `models/graph/node_builder.py`, not re-checked here).
    prompt_init_std:     forwarded to `TextPromptConfig`/`VisionPromptConfig.init_std`.
    prompt_seed:         optional int; if given, BOTH text and vision prompt parameters are
                         initialized deterministically from it (text uses `prompt_seed`,
                         vision uses `prompt_seed + 1` if not `None`, so the two modalities
                         do not receive numerically identical initial prompt tensors).
    node_feature_dim:    `D` -- Fig.1 MLP bridge output width / GIN input width
                         (`configs/model/graph.yaml: node_feature_dim_D`, UNRESOLVED in the
                         paper; required here, no default).
    mlp_bridge_hidden_dim: hidden width of both (independent) `PromptToNodeMLP` instances
                         (`configs/model/mlp_bridge.yaml: hidden_dim`, UNRESOLVED; required).
    mlp_bridge_dropout: forwarded to `MLPBridgeConfig.dropout`.
    adjacency_threshold: forwarded to `AdjacencyConfig.threshold` (PAPER_FACT default 0.8).
    attention_reweight_enabled: forwarded to `AdjacencyConfig` (JUSTIFIED_INFERENCE default
                         `False`, Eq. 18 "optional").
    gin_num_layers, gin_hidden_dim, gin_mlp_hidden_dim, gin_eps_init: forwarded to the
                         SHARED `GINConfig` (Section V.B PAPER_FACT defaults: 4 layers,
                         hidden 16).
    acga_latent_dim:     `K` (`configs/model/acga.yaml: latent_dim_K`, UNRESOLVED; required).
    acga_encoder_num_layers, acga_encoder_mlp_hidden_dim, acga_discriminator_hidden_dim,
    acga_negative_sampling_ratio, acga_reduction: forwarded to the SHARED `ACGAConfig`.
    hgnec_compressed_dim: `Dc` (`configs/model/hgn_ec.yaml: compressed_dim_Dc`, UNRESOLVED;
                         required).
    hgnec_restored_dim, hgnec_hnet_gin_hidden_dim, hgnec_hnet_num_gin_layers,
    hgnec_hnet_mlp_hidden_dim, hgnec_num_steps: forwarded to the SHARED `HGNECConfig`.
    provenance:          free-form dict, mirrors every other config dataclass in this
                         codebase.
    """

    clip: CLIPConfig

    num_layers: int
    prompt_dim: int
    num_prompts: int = 1
    prompt_init_std: float = 0.02
    prompt_seed: Optional[int] = None

    node_feature_dim: int = 0  # overwritten below if not explicitly required; see __post_init__
    mlp_bridge_hidden_dim: int = 0
    mlp_bridge_dropout: float = 0.0

    adjacency_threshold: float = 0.8
    attention_reweight_enabled: bool = False

    gin_num_layers: int = 4
    gin_hidden_dim: int = 16
    gin_mlp_hidden_dim: Optional[int] = None
    gin_eps_init: float = 0.0

    acga_latent_dim: int = 0
    acga_encoder_num_layers: int = 1
    acga_encoder_mlp_hidden_dim: Optional[int] = None
    acga_discriminator_hidden_dim: Optional[int] = None
    acga_negative_sampling_ratio: Optional[float] = None
    acga_reduction: str = "mean"

    hgnec_compressed_dim: int = 0
    hgnec_restored_dim: Optional[int] = None
    hgnec_hnet_gin_hidden_dim: Optional[int] = None
    hgnec_hnet_num_gin_layers: int = 1
    hgnec_hnet_mlp_hidden_dim: Optional[int] = None
    hgnec_num_steps: int = 1

    provenance: Dict[str, str] = field(default_factory=dict)

    _REQUIRED_POSITIVE_FIELDS = (
        "num_layers",
        "prompt_dim",
        "num_prompts",
        "node_feature_dim",
        "mlp_bridge_hidden_dim",
        "gin_num_layers",
        "gin_hidden_dim",
        "acga_latent_dim",
        "hgnec_compressed_dim",
    )

    def __post_init__(self) -> None:
        for name in self._REQUIRED_POSITIVE_FIELDS:
            val = getattr(self, name)
            if not isinstance(val, int) or val <= 0:
                raise ACHGCLIPConfigError(
                    f"ACHGCLIPConfig.{name} must be a positive int (UNRESOLVED-in-the-paper "
                    f"dims must be supplied explicitly by the caller, never defaulted to 0); "
                    f"got {val!r}."
                )
        if self.num_prompts != 1:
            raise ACHGCLIPConfigError(
                "ACHGCLIPConfig.num_prompts must be 1: Blocker 1's N=L node decomposition "
                f"(models/graph/node_builder.py) is only well-defined for M=1; got "
                f"{self.num_prompts!r}."
            )
        if self.clip.num_layers != self.num_layers:
            raise ACHGCLIPConfigError(
                f"ACHGCLIPConfig: clip.num_layers ({self.clip.num_layers}) must equal "
                f"num_layers ({self.num_layers}) -- both are the paper's L."
            )
        if self.clip.d_model != self.prompt_dim:
            raise ACHGCLIPConfigError(
                f"ACHGCLIPConfig: clip.d_model ({self.clip.d_model}) must equal prompt_dim "
                f"({self.prompt_dim}) -- both are the paper's d (prompts live in CLIP's "
                f"token embedding space)."
            )
        if not (-1.0 <= self.adjacency_threshold <= 1.0):
            raise ACHGCLIPConfigError(
                f"ACHGCLIPConfig.adjacency_threshold must be in [-1, 1], got {self.adjacency_threshold!r}."
            )
        if self.acga_reduction not in ("mean", "sum"):
            raise ACHGCLIPConfigError(
                f"ACHGCLIPConfig.acga_reduction must be 'mean' or 'sum', got {self.acga_reduction!r}."
            )
        if not isinstance(self.hgnec_num_steps, int) or self.hgnec_num_steps <= 0:
            raise ACHGCLIPConfigError(
                f"ACHGCLIPConfig.hgnec_num_steps must be a positive int, got {self.hgnec_num_steps!r}."
            )
            
        # TRUE BLOCKER resolution: to use q_final as the updated prompt (d),
        # HGN-EC must restore it to the prompt dimension.
        self.hgnec_restored_dim = self.prompt_dim


# ============================================================================================
# Output contract
# ============================================================================================


@dataclass
class ModalityOutput:
    """Everything produced for ONE modality's (text or vision) graph branch.

    modality:       `"text"` or `"vision"`.
    pre_gin_graph:  `Graph(X_node, A)` -- output of the Fig.1 MLP bridge + adjacency
                    construction (Stage 4), BEFORE GIN. Exposed for debugging/inspection;
                    not itself part of the paper's stated output contract.
    gin_graph:      `Graph(X_gin, A)` -- the GIN output (Stage 5). THIS is the object passed,
                    unchanged, to both `acga` and `hgn_ec` below (Blocker 4). `gin_graph.A`
                    is the identical tensor object as `pre_gin_graph.A` (GIN never modifies
                    adjacency); `gin_graph.X` is a new tensor (GIN's own output), distinct
                    from `pre_gin_graph.X`.
    acga:           `ACGAOutput` (Stage 6) -- `Z`, `A_hat`, discriminator outputs,
                    `reconstruction_loss`, `adversarial_loss`. A PARALLEL auxiliary branch;
                    never overwrites `gin_graph`.
    hgn_ec:         `HGNECOutput` (Stage 7) -- `q_final`, `H_initial`, `H_final`,
                    `energy_loss`. Consumes `gin_graph` directly (Blocker 4), NOT `acga.Z`.
    feedback:       `FeedbackPath` output (UNRESOLVED by default -- see module docstring).
    """

    modality: str
    pre_gin_graph: Graph
    gin_graph: Graph
    acga: ACGAOutput
    hgn_ec: HGNECOutput
    feedback: Dict[str, Any]


@dataclass
class ACHGCLIPOutput:
    """Top-level structured output of `ACHGCLIP.forward` (and its text/vision-only variants).

    Deliberately NOT an unexplained tuple (Stage 8 "OUTPUT CONTRACT" requirement): every
    field is named and independently accessible, and text/vision results are never silently
    concatenated, averaged, or otherwise fused (Stage 8 "HGN-EC OUTPUT" requirement) -- a
    downstream fusion operation is not specified anywhere in the paper (Section IV.D never
    describes combining text and vision `q_final`/`H`/loss values into one representation)
    and is therefore NOT invented here; see `docs/implementation_progress.md` Stage 8
    "Implementation choices" for this being left as a documented UNRESOLVED/future
    IMPLEMENTATION-CHOICE for a training-stage that actually needs one.

    h_text, h_vision: CLIP embeddings (Eqs. 6, 8), `None` when the corresponding modality
                    was not requested (`text_only_forward`/`vision_only_forward`).
    text, vision:   `ModalityOutput` for each modality's graph branch, `None` when not
                    requested. Modality identity is preserved throughout (never merged).
    """

    h_text: Optional[torch.Tensor]
    h_vision: Optional[torch.Tensor]
    text: Optional[ModalityOutput]
    vision: Optional[ModalityOutput]


# ============================================================================================
# Top-level model
# ============================================================================================


class ACHGCLIP(nn.Module):
    """Top-level ACHG-CLIP wiring (Stage 8). See module docstring for the full data flow and
    traceability table.

    Owns exactly one instance of each Stage 1-7 component:
      - `self.clip`             : `CLIPWrapper` (frozen backbone, Stage 2)
      - `self.text_prompt` /
        `self.vision_prompt`    : `TextPromptInjector` / `VisionPromptInjector` (Stage 3) --
                                  their `.prompts` parameters (`G`, `GV`) are the graph
                                  branch's actual input (see module docstring, Traceability).
      - `self.text_mlp_bridge` /
        `self.vision_mlp_bridge`: two INDEPENDENT `PromptToNodeMLP` instances (Stage 4,
                                  Blocker 3: `shared_across_modalities = false`).
      - `self.gin`              : ONE SHARED `GIN` stack (Stage 5, Blocker 5).
      - `self.acga`             : ONE SHARED `ACGA` (Stage 6, Blocker 5).
      - `self.hgn_ec`           : ONE SHARED `HGNEC` (Stage 7, Blocker 5).
      - `self.feedback_path`    : `FeedbackPath` (default `ResolvedFeedbackPath`).

    Text and vision are processed independently end-to-end (Section IV.B "respectively";
    Frozen Decision 1 in the Stage-8 task spec) -- `_process_modality` is called once per
    modality with that modality's own prompt injector / MLP bridge, but the SAME shared
    `gin`/`acga`/`hgn_ec` module instances (Blocker 5's "independent processing, shared
    weights" resolution). No cross-modal graph edges are introduced anywhere (Frozen
    Decision 7) and the two modalities' graphs are never merged (Frozen Decision 8).
    """

    def __init__(
        self,
        config: ACHGCLIPConfig,
        *,
        clip_wrapper: Optional[CLIPWrapper] = None,
        feedback_path: Optional[FeedbackPath] = None,
    ):
        super().__init__()
        self.config = config

        # -- 1. CLIP (frozen backbone) --------------------------------------------------
        self.clip = clip_wrapper if clip_wrapper is not None else CLIPWrapper(config.clip)

        # -- 2. Prompt subsystem (one learnable G / GV per modality) ---------------------
        text_seed = config.prompt_seed
        vision_seed = None if config.prompt_seed is None else config.prompt_seed + 1
        self.text_prompt = TextPromptInjector(
            TextPromptConfig(
                num_layers=config.num_layers,
                num_prompts=config.num_prompts,
                prompt_dim=config.prompt_dim,
                init_std=config.prompt_init_std,
                seed=text_seed,
            )
        )
        self.vision_prompt = VisionPromptInjector(
            VisionPromptConfig(
                num_layers=config.num_layers,
                num_prompts=config.num_prompts,
                prompt_dim=config.prompt_dim,
                init_std=config.prompt_init_std,
                seed=vision_seed,
            )
        )

        # -- 3. MLP bridge (two independent instances, Blocker 3) ------------------------
        self.text_mlp_bridge = PromptToNodeMLP(
            MLPBridgeConfig(
                input_dim=config.prompt_dim,
                hidden_dim=config.mlp_bridge_hidden_dim,
                output_dim=config.node_feature_dim,
                dropout=config.mlp_bridge_dropout,
            )
        )
        self.vision_mlp_bridge = PromptToNodeMLP(
            MLPBridgeConfig(
                input_dim=config.prompt_dim,
                hidden_dim=config.mlp_bridge_hidden_dim,
                output_dim=config.node_feature_dim,
                dropout=config.mlp_bridge_dropout,
            )
        )

        # -- 4. Graph construction configs (no learnable parameters) ---------------------
        self.text_node_config = NodeBuilderConfig(num_nodes_mode="per_layer", modality="text")
        self.vision_node_config = NodeBuilderConfig(num_nodes_mode="per_layer", modality="vision")
        self.adjacency_config = AdjacencyConfig(
            threshold=config.adjacency_threshold,
            attention_reweight_enabled=config.attention_reweight_enabled,
        )

        # -- 5. GIN (SHARED across modalities, Blocker 5) ---------------------------------
        self.gin = GIN(
            GINConfig(
                input_dim=config.node_feature_dim,
                num_layers=config.gin_num_layers,
                hidden_dim=config.gin_hidden_dim,
                mlp_hidden_dim=config.gin_mlp_hidden_dim,
                eps_init=config.gin_eps_init,
            )
        )

        # -- 6. ACGA (SHARED, PARALLEL auxiliary head, Blocker 4/5) -----------------------
        self.acga = ACGA(
            ACGAConfig(
                input_dim=config.gin_hidden_dim,
                latent_dim=config.acga_latent_dim,
                encoder_num_layers=config.acga_encoder_num_layers,
                encoder_mlp_hidden_dim=config.acga_encoder_mlp_hidden_dim,
                discriminator_hidden_dim=config.acga_discriminator_hidden_dim,
                negative_sampling_ratio=config.acga_negative_sampling_ratio,
                reduction=config.acga_reduction,
            )
        )

        # -- 7. HGN-EC (SHARED, consumes GIN's own (X, A), Blocker 4/5) -------------------
        self.hgn_ec = HGNEC(
            HGNECConfig(
                input_dim=config.gin_hidden_dim,
                compressed_dim=config.hgnec_compressed_dim,
                restored_dim=config.hgnec_restored_dim,
                hnet_gin_hidden_dim=config.hgnec_hnet_gin_hidden_dim,
                hnet_num_gin_layers=config.hgnec_hnet_num_gin_layers,
                hnet_mlp_hidden_dim=config.hgnec_hnet_mlp_hidden_dim,
                num_steps=config.hgnec_num_steps,
            )
        )

        # -- Feedback path (Section IV.D.8, RESOLVED) -----------------------
        self.feedback_path = feedback_path if feedback_path is not None else ResolvedFeedbackPath()

        # -- HF Injection Projections ----------------------------------------
        # If using a pre-trained HF model, its internal text/vision dimensions (e.g. 512/768 for ViT-B/32)
        # will mismatch the shared `prompt_dim` (e.g. 128) used by the graph. We need to project `q_reshaped`
        # to the transformer's hidden dimension before replacing the token.
        if self.clip.hf_model is not None:
            text_hidden = self.clip.hf_model.config.text_config.hidden_size
            vision_hidden = self.clip.hf_model.config.vision_config.hidden_size
            self.text_injection_proj = nn.Linear(config.prompt_dim, text_hidden)
            self.vision_injection_proj = nn.Linear(config.prompt_dim, vision_hidden)
        else:
            self.text_injection_proj = nn.Identity()
            self.vision_injection_proj = nn.Identity()

    # -- internal: one modality's full graph branch --------------------------------------

    def _build_pre_gin_graph(self, prompt_injector: nn.Module, mlp_bridge: PromptToNodeMLP, node_config: NodeBuilderConfig) -> Graph:
        """Prompt parameter `G`/`GV` -> MLP bridge -> node features -> adjacency -> `Graph`.

        Reads `prompt_injector.prompts` (the `(L, M, d)` learnable parameter) DIRECTLY --
        does not call `prompt_injector.forward` (that method performs Eq. 9/10's
        concatenation-into-a-sequence insertion, a different operation not exercised by this
        top-level module; see "KNOWN LIMITATIONS" in the module docstring).
        """
        return build_graph(prompt_injector.prompts, mlp_bridge, node_config, self.adjacency_config)

    def _process_modality(
        self,
        *,
        modality: str,
        prompt_injector: nn.Module,
        mlp_bridge: PromptToNodeMLP,
        node_config: NodeBuilderConfig,
        dt: float,
        num_steps: Optional[int],
        generator: Optional[torch.Generator],
    ) -> ModalityOutput:
        pre_gin_graph = self._build_pre_gin_graph(prompt_injector, mlp_bridge, node_config)

        # Stage 5: GIN. `gin_graph` is THE object subsequently fed, unchanged, to BOTH the
        # ACGA auxiliary branch and HGN-EC (Blocker 4) -- not a fresh copy for each.
        gin_graph = self.gin(pre_gin_graph)

        # Stage 6: ACGA -- parallel auxiliary branch. Does not mutate `gin_graph`.
        acga_out = self.acga(gin_graph, generator=generator)

        # Stage 7: HGN-EC -- consumes the SAME `gin_graph` (its own X, A), never `acga_out.Z`.
        hgnec_out = self.hgn_ec(gin_graph, dt=dt, num_steps=num_steps)

        # Feedback path (UNRESOLVED by default; never mutates prompt_injector.prompts).
        feedback_out = self.feedback_path(hgnec_out, prompt_injector)

        return ModalityOutput(
            modality=modality,
            pre_gin_graph=pre_gin_graph,
            gin_graph=gin_graph,
            acga=acga_out,
            hgn_ec=hgnec_out,
            feedback=feedback_out,
        )

    # -- public forward passes -------------------------------------------------------------

    def _register_hooks(self, text_prompts: Optional[torch.Tensor], vision_prompts: Optional[torch.Tensor]):
        """Dynamically registers forward pre-hooks on CLIP's transformer layers to inject prompts.
        
        Uses the Replacement strategy: replaces the token immediately after [CLS] (index 1)
        with the graph-updated prompt token.
        """
        hooks = []
        
        if self.clip.hf_model is not None:
            text_layers = self.clip.hf_model.text_model.encoder.layers
            vision_layers = self.clip.hf_model.vision_model.encoder.layers
        else:
            text_layers = self.clip.text_encoder.blocks
            vision_layers = self.clip.vision_encoder.blocks
        
        if text_prompts is not None:
            # Project to text transformer dimension
            projected_text = self.text_injection_proj(text_prompts)
            for l_idx, layer in enumerate(text_layers):
                def text_pre_hook(module, args, kwargs, layer_idx=l_idx):
                    hidden_states = args[0].clone()  # Clone to avoid in-place graph modifications
                    # projected_text shape: (L, M, text_hidden) where M=1
                    prompt = projected_text[layer_idx]  # (1, text_hidden)
                    hidden_states[:, 1:2, :] = prompt
                    return (hidden_states,) + args[1:], kwargs
                
                # HuggingFace standard transformers expect a tuple of args and kwargs from a pre-hook
                # PyTorch >= 2.0 supports with_kwargs=True
                hook_handle = layer.register_forward_pre_hook(text_pre_hook, with_kwargs=True)
                hooks.append(hook_handle)
                
        if vision_prompts is not None:
            # Project to vision transformer dimension
            projected_vision = self.vision_injection_proj(vision_prompts)
            for l_idx, layer in enumerate(vision_layers):
                def vision_pre_hook(module, args, kwargs, layer_idx=l_idx):
                    hidden_states = args[0].clone()
                    prompt = projected_vision[layer_idx]
                    hidden_states[:, 1:2, :] = prompt
                    return (hidden_states,) + args[1:], kwargs
                
                hook_handle = layer.register_forward_pre_hook(vision_pre_hook, with_kwargs=True)
                hooks.append(hook_handle)
                
        return hooks

    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        *,
        dt: float,
        num_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> ACHGCLIPOutput:
        """Joint text+vision forward pass.
        
        Executes graph branch first, injects updated prompts via hooks, then runs CLIP.
        """
        # 1. Process Modalities (Graph Branch)
        text_out = self._process_modality(
            modality="text",
            prompt_injector=self.text_prompt,
            mlp_bridge=self.text_mlp_bridge,
            node_config=self.text_node_config,
            dt=dt,
            num_steps=num_steps,
            generator=generator,
        )
        vision_out = self._process_modality(
            modality="vision",
            prompt_injector=self.vision_prompt,
            mlp_bridge=self.vision_mlp_bridge,
            node_config=self.vision_node_config,
            dt=dt,
            num_steps=num_steps,
            generator=generator,
        )
        
        # 2. Register Hooks for Prompt Injection
        q_text = text_out.feedback["q_reshaped"] if text_out.feedback["applied"] else None
        q_vision = vision_out.feedback["q_reshaped"] if vision_out.feedback["applied"] else None
        hooks = self._register_hooks(q_text, q_vision)
        
        # 3. Run CLIP
        try:
            h_vision = self.clip.encode_image(images)
            h_text = self.clip.encode_text(tokens)
        finally:
            # 4. Clean up hooks
            for hook in hooks:
                hook.remove()

        return ACHGCLIPOutput(h_text=h_text, h_vision=h_vision, text=text_out, vision=vision_out)

    def text_only_forward(
        self,
        tokens: torch.Tensor,
        *,
        dt: float,
        num_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> ACHGCLIPOutput:
        """Text-only forward pass."""
        text_out = self._process_modality(
            modality="text",
            prompt_injector=self.text_prompt,
            mlp_bridge=self.text_mlp_bridge,
            node_config=self.text_node_config,
            dt=dt,
            num_steps=num_steps,
            generator=generator,
        )
        
        q_text = text_out.feedback["q_reshaped"] if text_out.feedback["applied"] else None
        hooks = self._register_hooks(text_prompts=q_text, vision_prompts=None)
        
        try:
            h_text = self.clip.encode_text(tokens)
        finally:
            for hook in hooks:
                hook.remove()
                
        return ACHGCLIPOutput(h_text=h_text, h_vision=None, text=text_out, vision=None)

    def vision_only_forward(
        self,
        images: torch.Tensor,
        *,
        dt: float,
        num_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> ACHGCLIPOutput:
        """Vision-only forward pass."""
        vision_out = self._process_modality(
            modality="vision",
            prompt_injector=self.vision_prompt,
            mlp_bridge=self.vision_mlp_bridge,
            node_config=self.vision_node_config,
            dt=dt,
            num_steps=num_steps,
            generator=generator,
        )
        
        q_vision = vision_out.feedback["q_reshaped"] if vision_out.feedback["applied"] else None
        hooks = self._register_hooks(text_prompts=None, vision_prompts=q_vision)
        
        try:
            h_vision = self.clip.encode_image(images)
        finally:
            for hook in hooks:
                hook.remove()
                
        return ACHGCLIPOutput(h_text=None, h_vision=h_vision, text=None, vision=vision_out)

    # -- introspection ---------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def extra_repr(self) -> str:
        return (
            f"num_layers={self.config.num_layers}, prompt_dim={self.config.prompt_dim}, "
            f"node_feature_dim={self.config.node_feature_dim}, "
            f"gin_hidden_dim={self.config.gin_hidden_dim}, "
            f"acga_latent_dim={self.config.acga_latent_dim}, "
            f"hgnec_compressed_dim={self.config.hgnec_compressed_dim}"
        )
