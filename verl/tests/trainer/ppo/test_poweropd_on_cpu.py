# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch

from verl.trainer.ppo.core_algos import compute_poweropd_reward


def test_poweropd_matches_bounded_power_difference():
    teacher_log_prob = torch.log(torch.tensor([[0.8, 0.1, 0.01]]))
    student_log_prob = torch.log(torch.tensor([[0.2, 0.2, 0.02]]))

    reward = compute_poweropd_reward(teacher_log_prob, student_log_prob, alpha=5.0)
    expected = teacher_log_prob.exp().pow(5.0) - student_log_prob.exp().pow(5.0)

    assert torch.allclose(reward, expected)
    assert torch.all(reward <= 1.0)
    assert torch.all(reward >= -1.0)
    assert reward[0, 0] > 0
    assert reward[0, 1] < 0


def test_poweropd_preserves_zero_when_probabilities_match():
    log_prob = torch.log(torch.tensor([[0.5, 0.25]]))
    reward = compute_poweropd_reward(log_prob, log_prob, alpha=0.1)
    assert torch.allclose(reward, torch.zeros_like(reward))
    assert not reward.requires_grad


def test_poweropd_requires_positive_alpha_and_matching_shapes():
    with pytest.raises(ValueError, match="alpha must be positive"):
        compute_poweropd_reward(torch.zeros(1, 1), torch.zeros(1, 1), alpha=0.0)
    with pytest.raises(ValueError, match="same shape"):
        compute_poweropd_reward(torch.zeros(1, 1), torch.zeros(1, 2), alpha=1.0)
