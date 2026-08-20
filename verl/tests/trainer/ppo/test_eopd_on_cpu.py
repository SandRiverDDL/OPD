# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch

from verl.trainer.ppo.core_algos import (
    compute_policy_loss_eopd,
    compute_policy_loss_vanilla,
    get_policy_loss_fn,
)
from verl.workers.config import ActorConfig, PolicyLossConfig


def _config(threshold: float = 0.8, coef: float = 1.0) -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        ppo_micro_batch_size=1,
        rollout_n=1,
        policy_loss=PolicyLossConfig(
            loss_mode="eopd",
            eopd_entropy_threshold=threshold,
            eopd_forward_kl_coef=coef,
        ),
    )


def _inputs():
    old_log_prob = torch.tensor([[-0.5, -0.5]])
    log_prob = old_log_prob.detach().clone().requires_grad_()
    advantages = torch.zeros_like(log_prob)
    response_mask = torch.ones_like(log_prob)
    teacher_entropy = torch.tensor([[0.2, 1.0]])
    teacher_topk_log_probs = torch.log(torch.tensor([[[0.8, 0.1], [0.6, 0.3]]]))
    student_topk_log_probs = torch.log(torch.tensor([[[0.7, 0.2], [0.5, 0.4]]])).requires_grad_()
    return (
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        teacher_entropy,
        teacher_topk_log_probs,
        student_topk_log_probs,
    )


def test_eopd_is_registered():
    assert get_policy_loss_fn("eopd") is compute_policy_loss_eopd


def test_eopd_only_applies_forward_kl_to_high_entropy_tokens():
    inputs = _inputs()
    loss, metrics = compute_policy_loss_eopd(
        *inputs[:4],
        config=_config(),
        teacher_entropy=inputs[4],
        teacher_topk_log_probs=inputs[5],
        student_topk_log_probs=inputs[6],
    )

    teacher_log_probs = inputs[5][:, 1] - torch.logsumexp(inputs[5][:, 1], dim=-1)
    teacher_probs = teacher_log_probs.exp()
    expected_high_entropy_kl = (
        teacher_probs * (teacher_log_probs - inputs[6][:, 1])
    ).sum()
    expected_loss = expected_high_entropy_kl / 2

    assert torch.allclose(loss, expected_loss)
    assert metrics["actor/eopd_high_entropy_tokens"] == 1
    assert abs(metrics["actor/eopd_high_entropy_ratio"] - 0.5) < 1e-6


def test_eopd_matches_sampled_token_loss_when_all_tokens_are_low_entropy():
    inputs = _inputs()
    low_entropy_inputs = (*inputs[:4], torch.tensor([[0.2, 0.7]]), *inputs[5:])
    eopd_loss, eopd_metrics = compute_policy_loss_eopd(
        *low_entropy_inputs[:4],
        config=_config(),
        teacher_entropy=low_entropy_inputs[4],
        teacher_topk_log_probs=low_entropy_inputs[5],
        student_topk_log_probs=low_entropy_inputs[6],
    )
    vanilla_loss, _ = compute_policy_loss_vanilla(
        *low_entropy_inputs[:4],
        config=_config(),
    )

    assert torch.allclose(eopd_loss, vanilla_loss)
    assert eopd_metrics["actor/eopd_high_entropy_tokens"] == 0
    assert eopd_metrics["actor/eopd_forward_kl_loss"] == 0.0
