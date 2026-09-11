"""Shared distillation method metadata.

Keep method naming and pipeline capabilities in one place so adding a new
distillation method does not require repeating the same alias checks in the
launcher, trainer, and workers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistillationMethodSpec:
    name: str
    policy_loss_mode: str | None = None
    requires_top_k: bool = False
    policy_loss_extra_keys: tuple[str, ...] = ()


_METHOD_ALIASES = {
    "prune-opd": "pruneopd",
    "prune_opd": "pruneopd",
    "pruned-opd": "pruneopd",
    "pruned_opd": "pruneopd",
}

_METHOD_SPECS = {
    "vanilla": DistillationMethodSpec("vanilla"),
    "eopd": DistillationMethodSpec(
        "eopd",
        policy_loss_mode="eopd",
        requires_top_k=True,
        policy_loss_extra_keys=("teacher_entropy",),
    ),
    "poweropd": DistillationMethodSpec("poweropd"),
    "aopd": DistillationMethodSpec(
        "aopd",
        policy_loss_mode="aopd",
        requires_top_k=True,
        policy_loss_extra_keys=("aopd_gkd_mask",),
    ),
    "pruneopd": DistillationMethodSpec("pruneopd", requires_top_k=True),
}

TOPK_POLICY_LOSS_METHODS = frozenset(
    method.name for method in _METHOD_SPECS.values() if method.policy_loss_mode is not None
)


def normalize_distillation_method(method: str) -> str:
    """Return the canonical name for a distillation method."""

    normalized = str(method).strip().lower()
    normalized = _METHOD_ALIASES.get(normalized, normalized)
    if normalized not in _METHOD_SPECS:
        supported = ", ".join(_METHOD_SPECS)
        raise ValueError(f"Unsupported distillation method {method!r}; expected one of: {supported}")
    return normalized


def get_distillation_method_spec(method: str) -> DistillationMethodSpec:
    """Return the shared capabilities for a canonical or aliased method."""

    return _METHOD_SPECS[normalize_distillation_method(method)]


def get_policy_loss_extra_keys(method: str) -> tuple[str, ...]:
    """Return batch tensors required by a method-specific actor loss."""

    return get_distillation_method_spec(method).policy_loss_extra_keys


def uses_topk_policy_loss(method: str) -> bool:
    """Whether the actor policy update consumes teacher top-k distributions."""

    return get_distillation_method_spec(method).policy_loss_mode is not None


def requires_top_k(method: str) -> bool:
    """Whether a method cannot run without teacher top-k probabilities."""

    return get_distillation_method_spec(method).requires_top_k


def should_defer_topk_reward_to_actor(method: str, top_k: int) -> bool:
    """Whether top-k reward construction needs an actor-side forward pass."""

    return top_k > 0 and not uses_topk_policy_loss(method)


def should_keep_teacher_topk_for_update(method: str) -> bool:
    """Whether teacher top-k tensors must survive until actor update."""

    return uses_topk_policy_loss(method)
