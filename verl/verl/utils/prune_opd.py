from __future__ import annotations

import math
from typing import Any

import torch


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def apply_prune_opd_to_scores(
    rm_scores: torch.Tensor,
    overlap_mask: torch.Tensor | None,
    response_mask: torch.Tensor,
    config: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the overlap-based Prune-OPD weighting schedule.

    ``overlap_mask`` marks which student top-k candidates also occur in the
    teacher top-k set.  A response position whose overlap ratio is below the
    configured threshold is treated as an unreliable teacher signal.  Each
    such event decreases the weight of all subsequent positions, while
    ``w_base`` preserves a non-zero floor when desired.

    The function is intentionally independent of the trainer so it can be
    unit-tested on CPU and applied on the actor GPU worker.
    """

    if not bool(_cfg_get(config, "enable", False)):
        return rm_scores, {}
    if rm_scores.dim() != 3:
        raise ValueError(
            "Prune-OPD expects top-k reward scores with shape (batch, response, k), "
            f"got {tuple(rm_scores.shape)}"
        )
    if overlap_mask is None:
        raise ValueError("Prune-OPD overlap mode requires overlap_mask")
    if overlap_mask.shape != rm_scores.shape:
        raise ValueError(
            "Prune-OPD overlap_mask must align with rm_scores, "
            f"got {tuple(overlap_mask.shape)} and {tuple(rm_scores.shape)}"
        )
    if response_mask.shape != rm_scores.shape[:2]:
        raise ValueError(
            "Prune-OPD response_mask must align with rm_scores, "
            f"got {tuple(response_mask.shape)} and {tuple(rm_scores.shape[:2])}"
        )

    metric = str(_cfg_get(config, "metric", "overlap_ratio"))
    if metric != "overlap_ratio":
        raise NotImplementedError(
            f"Unsupported Prune-OPD metric {metric!r}; only 'overlap_ratio' is implemented"
        )

    threshold = float(_cfg_get(config, "threshold", 0.7))
    w_drop = float(_cfg_get(config, "w_drop", 0.01))
    w_base = float(_cfg_get(config, "w_base", 0.5))
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"Prune-OPD threshold must be in (0, 1], got {threshold}")
    if not math.isfinite(w_drop) or w_drop < 0.0:
        raise ValueError(f"Prune-OPD w_drop must be finite and non-negative, got {w_drop}")
    if not math.isfinite(w_base) or w_base < 0.0:
        raise ValueError(f"Prune-OPD w_base must be finite and non-negative, got {w_base}")

    response_mask = response_mask.to(device=rm_scores.device)
    overlap_ratio = overlap_mask.to(device=rm_scores.device).float().mean(dim=-1)
    valid_mask = response_mask.bool()
    bad_event = (overlap_ratio < threshold) & valid_mask

    # Pruning is causal: a low-confidence position affects later positions,
    # not the position that produced the signal itself.
    cumulative_bad_events = bad_event.to(rm_scores.dtype).cumsum(dim=-1)
    weights = torch.clamp(1.0 - w_drop * cumulative_bad_events, min=0.0, max=1.0)
    weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
    loss_weights = weights + w_base
    loss_weights = torch.where(valid_mask, loss_weights, torch.zeros_like(loss_weights))

    aux = {
        "prune_opd_overlap_ratio": overlap_ratio,
        "prune_opd_bad_event": bad_event.to(rm_scores.dtype),
        "prune_opd_weights": weights,
        "prune_opd_loss_weights": loss_weights,
        "prune_opd_effective_response_length": (
            (valid_mask & (weights > 0)).sum(dim=-1).to(torch.float32)
        ),
    }
    return rm_scores * loss_weights.unsqueeze(-1), aux
