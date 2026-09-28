"""
losses/acga_losses.py
========================

Stage 6 — ACGA-specific loss terms only (Eqs. 24, 26). Per this stage's task spec ("LOSSES"
section): "Implement only ACGA-specific loss terms required for the ACGA module. Do not
implement the complete training loss for ACHG-CLIP yet." `L_CE`, `L_energy`, and `L_total`
(Eq. 34) are explicitly out of scope here and implemented nowhere in this module.

Both loss functions are returned as separate, uncombined scalars (Stage 8 composes them into
`L_total` per Eq. 34's `lambda1*L_recon + lambda2*L_adv + ...`).

--------------------------------------------------------------------------------------------
REFERENCE TRACEABILITY
--------------------------------------------------------------------------------------------

Component: Reconstruction loss formula.
Equation/reference: Eq. 24:
    L_recon = - sum_{(i,j) in E union E-} [A_ij log(A_hat_ij) + (1-A_ij) log(1-A_hat_ij)]
Source: Section IV.C.2, verbatim.
Evidence type: PAPER-FACT (the per-entry negative-log-likelihood form).
Confidence: High.

Component: Which entries `(i, j)` the sum in Eq. 24 actually ranges over (`E union E-`).
Equation/reference: Eq. 24.
Source: `configs/model/acga.yaml: negative_sampling_ratio` = UNRESOLVED -- "Eq. 24 introduces
    E^- (negatively sampled edges) but the paper never states the sampling strategy or
    ratio."
Evidence type: UNRESOLVED (which entries) -> IMPLEMENTATION-CHOICE (default behavior).
    Default (`negative_sampling_ratio=None`): sum over EVERY entry of the dense `N x N`
    adjacency (i.e. treat `E- = {all non-edges}`, the "full negative set" reading). This is
    the same simplification implicitly available whenever the graph is small and dense enough
    that exhaustive coverage is tractable (the Stage-4 `Graph` contract's `N` is small --
    `N = num_layers`, per `FINAL_IMPLEMENTATION_BLUEPRINT.md` Blocker 1 -- so no sub-sampling
    is required for tractability). When `negative_sampling_ratio` is given (a float in
    `(0, 1]`), a random subset of the NEGATIVE (non-edge, `A_ij == 0`) entries of the given
    ratio is sampled per call and combined with every positive entry -- kept isolated behind
    an explicit, documented parameter (never silently applied) so this reading can be swapped
    for a different sampling strategy later without touching `ACGA`/`ACGAEncoder`.
Confidence: Low (no paper evidence for either reading; "full dense matrix" is the more
    literal/conservative default since it discards no information rather than guessing a
    ratio).

Component: Reduction (`mean` vs. `sum`) over the `E union E-` entries.
Equation/reference: Eq. 24's `sum_{(i,j)}` notation.
Source: Not addressed anywhere in Section IV.C -- the paper's `sum` notation is standard
    NLL-loss notation and does not by itself specify whether the *reported*/*optimized*
    quantity is a raw sum (which scales with `N^2` and graph size) or a normalized mean.
Evidence type: IMPLEMENTATION-CHOICE. Default `reduction="mean"` -- avoids `L_recon`'s
    magnitude depending on `N` (keeping it comparable in scale to `L_adv`/`L_energy` for
    Eq. 34's shared `lambda` weighting), never claimed as a literal transcription of Eq. 24's
    `sum` symbol. `reduction="sum"` is available for a literal-sum reading.
Confidence: Low.

Component: Adversarial loss formula.
Equation/reference: Eq. 26: L_adv = E_{z~p_z}[D(z)] - E_{z~q(Z|X,A)}[D(z)].
Source: Section IV.C.3, verbatim.
Evidence type: PAPER-FACT.
Confidence: High.

Component: Expectations `E_{z~p_z}[.]` / `E_{z~q(Z|X,A)}[.]` realized as sample means over the
    per-node/per-graph-slot discriminator outputs of a single forward pass (rather than e.g. a
    running average across steps, or multiple Monte-Carlo prior draws per step).
Equation/reference: Eq. 26.
Source: Not specified in Section IV.C.3 -- standard practice for this loss family (a single
    mini-batch Monte-Carlo estimate of the expectation), consistent with how every other
    per-batch loss in this paper (`L_CE`, `L_recon`, `L_energy`) is computed from one forward
    pass's tensors.
Evidence type: JUSTIFIED-INFERENCE (standard single-sample Monte-Carlo estimator for a loss of
    this form; not a literal paper statement).
Confidence: Medium.
"""

from __future__ import annotations

from typing import Optional

import torch


class ACGALossError(Exception):
    """Raised on invalid inputs to an ACGA loss function."""


_EPS = 1e-7  # numerical clamp for log(.), IMPLEMENTATION-CHOICE, standard BCE stabilization.


def reconstruction_loss(
    A: torch.Tensor,
    A_hat: torch.Tensor,
    *,
    negative_sampling_ratio: Optional[float] = None,
    reduction: str = "mean",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Eq. 24: negative log-likelihood reconstruction loss.

    ``L_recon = - sum_{(i,j) in E union E-} [A_ij log(A_hat_ij) + (1-A_ij) log(1-A_hat_ij)]``

    Args:
        A: ground-truth adjacency, `(..., N, N)`, entries in `{0, 1}` (Stage-4 contract).
        A_hat: reconstructed adjacency, `(..., N, N)`, entries in `[0, 1]` (Eq. 23 output).
        negative_sampling_ratio: `None` (default) -> sum over every entry (dense "full
            negative set" reading, see module docstring). A float in `(0, 1]` -> sample that
            fraction of the `A_ij == 0` entries per call, combined with every `A_ij == 1`
            entry (IMPLEMENTATION-CHOICE, isolated, off by default).
        reduction: `"mean"` (default, IMPLEMENTATION-CHOICE) or `"sum"` (literal Eq. 24
            reading). See module docstring.
        generator: optional `torch.Generator` for deterministic negative sampling.

    Returns:
        A scalar tensor.
    """
    if A.shape != A_hat.shape:
        raise ACGALossError(f"reconstruction_loss: A.shape {tuple(A.shape)} != A_hat.shape {tuple(A_hat.shape)}.")
    if reduction not in ("mean", "sum"):
        raise ACGALossError(f"reconstruction_loss: reduction must be 'mean' or 'sum', got {reduction!r}.")
    if negative_sampling_ratio is not None and not (0.0 < negative_sampling_ratio <= 1.0):
        raise ACGALossError(
            f"reconstruction_loss: negative_sampling_ratio must be in (0, 1], got {negative_sampling_ratio!r}."
        )

    A_hat_c = A_hat.clamp(min=_EPS, max=1.0 - _EPS)
    per_entry = -(A * torch.log(A_hat_c) + (1.0 - A) * torch.log(1.0 - A_hat_c))  # (..., N, N)

    if negative_sampling_ratio is None:
        mask = torch.ones_like(A)
    else:
        pos_mask = (A != 0).to(A.dtype)
        neg_candidates = (A == 0)
        if generator is not None:
            rand = torch.empty_like(A).uniform_(0.0, 1.0, generator=generator)
        else:
            rand = torch.rand_like(A)
        sampled_neg = neg_candidates & (rand < negative_sampling_ratio)
        mask = pos_mask + sampled_neg.to(A.dtype)

    masked = per_entry * mask
    if reduction == "sum":
        return masked.sum()
    denom = mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def adversarial_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Eq. 26: ``L_adv = E_{z~p_z}[D(z)] - E_{z~q(Z|X,A)}[D(z)]``.

    Args:
        d_real: discriminator output on prior samples `z ~ N(0, I)` (Eq. 26's `p_z` term).
        d_fake: discriminator output on encoder-produced `Z` (Eq. 26's `q(Z|X,A)` term).

    Returns:
        A scalar tensor. Positive when the discriminator scores prior samples higher than
        encoded ones on average (matching Eq. 26's literal sign, `p_z` term minus
        `q(Z|X,A)` term).
    """
    if d_real.shape != d_fake.shape:
        raise ACGALossError(
            f"adversarial_loss: d_real.shape {tuple(d_real.shape)} != d_fake.shape {tuple(d_fake.shape)}."
        )
    return d_real.mean() - d_fake.mean()


# ------------------------------------------------------------------------------------------
# OPTIONAL WGAN stabilization mechanisms -- NONE of these are applied automatically anywhere
# in this stage. Per the Stage 6 task's "ADVERSARIAL / WASSERSTEIN ISSUE" section: "DO NOT
# silently invent a complete WGAN-GP/WGAN-clipping algorithm and call it the paper's method...
# Any required stabilization mechanism not established by the paper must be
# IMPLEMENTATION-CHOICE and configurable. Keep it isolated so it can later be replaced."
# `configs/model/acga.yaml: wgan_weight_clipping`, `wgan_gradient_penalty`,
# `critic_update_ratio` are all UNRESOLVED (null) -- these two helpers exist purely so a
# LATER stage can opt in explicitly; nothing in `ACGA`/`ACGAConfig` calls them.
# ------------------------------------------------------------------------------------------


def clip_discriminator_weights(discriminator: torch.nn.Module, clip_value: float) -> None:
    """WGAN-style weight clipping (IMPLEMENTATION-CHOICE, UNRESOLVED whether/how the paper
    intends any such mechanism -- `configs/model/acga.yaml: wgan_weight_clipping`). Not
    called anywhere in Stage 6; provided isolated and opt-in for a later training stage.
    """
    if clip_value <= 0:
        raise ACGALossError(f"clip_discriminator_weights: clip_value must be > 0, got {clip_value!r}.")
    with torch.no_grad():
        for p in discriminator.parameters():
            p.clamp_(-clip_value, clip_value)


def gradient_penalty(
    discriminator: torch.nn.Module,
    z_real: torch.Tensor,
    z_fake: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """WGAN-GP-style gradient penalty (IMPLEMENTATION-CHOICE, UNRESOLVED whether/how the
    paper intends any such mechanism -- `configs/model/acga.yaml: wgan_gradient_penalty`).
    Not called anywhere in Stage 6; provided isolated and opt-in for a later training stage.

    Standard formulation: penalizes `(||grad_z_hat D(z_hat)||_2 - 1)^2` at interpolated points
    `z_hat = eps*z_real + (1-eps)*z_fake`, `eps ~ U(0,1)`.
    """
    if z_real.shape != z_fake.shape:
        raise ACGALossError(
            f"gradient_penalty: z_real.shape {tuple(z_real.shape)} != z_fake.shape {tuple(z_fake.shape)}."
        )
    shape = [z_real.shape[0]] + [1] * (z_real.dim() - 1)
    if generator is not None:
        eps = torch.empty(shape, device=z_real.device, dtype=z_real.dtype).uniform_(0.0, 1.0, generator=generator)
    else:
        eps = torch.rand(shape, device=z_real.device, dtype=z_real.dtype)
    z_hat = (eps * z_real + (1.0 - eps) * z_fake).detach().requires_grad_(True)
    d_hat = discriminator(z_hat)
    grad = torch.autograd.grad(
        outputs=d_hat,
        inputs=z_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_norm = grad.reshape(grad.shape[0], -1).norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()
