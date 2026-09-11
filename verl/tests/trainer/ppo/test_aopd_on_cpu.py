# Copyright 2026

import torch

from verl.trainer.ppo.core_algos import (
    compute_aopd_reward,
    compute_policy_loss_aopd,
    get_policy_loss_fn,
)
from verl.workers.config import ActorConfig, PolicyLossConfig


def _config() -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        ppo_micro_batch_size=1,
        rollout_n=1,
        policy_loss=PolicyLossConfig(
            loss_mode="aopd",
            aopd_threshold=0.1,
            aopd_gkd_weight=1.0,
            aopd_jsd_beta=1.0,
        ),
    )


def test_aopd_routes_high_student_probability_tokens_to_gkd():
    teacher = torch.log(torch.tensor([[0.2, 0.1]]))
    student = torch.log(torch.tensor([[0.5, 0.05]]))

    reward, gkd_mask = compute_aopd_reward(
        teacher_log_prob=teacher,
        student_log_prob=student,
        threshold=0.1,
    )

    assert torch.equal(gkd_mask, torch.tensor([[True, False]]))
    assert torch.allclose(
        reward,
        torch.tensor([[0.0, torch.log(torch.tensor(2.0)).item()]]),
    )


def test_aopd_policy_loss_adds_gkd_only_on_selected_tokens():
    old_log_prob = torch.log(torch.tensor([[0.5, 0.5]]))
    log_prob = old_log_prob.detach().clone().requires_grad_()
    advantages = torch.zeros_like(old_log_prob)
    response_mask = torch.ones_like(old_log_prob)
    teacher_topk_log_probs = torch.log(torch.tensor([[[0.8, 0.1], [0.6, 0.3]]]))
    student_topk_log_probs = torch.log(
        torch.tensor([[[0.7, 0.2], [0.5, 0.4]]])
    ).requires_grad_()
    gkd_mask = torch.tensor([[0.0, 1.0]])

    loss, metrics = compute_policy_loss_aopd(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=_config(),
        teacher_topk_log_probs=teacher_topk_log_probs,
        student_topk_log_probs=student_topk_log_probs,
        aopd_gkd_mask=gkd_mask,
    )

    expected = (
        torch.exp(teacher_topk_log_probs[:, 1])
        * (teacher_topk_log_probs[:, 1] - student_topk_log_probs[:, 1])
    ).sum() / 2
    assert torch.allclose(loss, expected)
    assert metrics["actor/aopd_gkd_tokens"] == 1
    assert abs(metrics["actor/aopd_gkd_token_ratio"] - 0.5) < 1e-6
    assert get_policy_loss_fn("aopd") is compute_policy_loss_aopd
