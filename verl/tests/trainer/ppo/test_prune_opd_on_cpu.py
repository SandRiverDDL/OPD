# Copyright 2026

import torch

from verl.utils.prune_opd import apply_prune_opd_to_scores


def test_prune_opd_overlap_weights_decay_causally():
    scores = torch.ones(1, 3, 2)
    overlap_mask = torch.tensor([[[1, 1], [0, 1], [0, 0]]], dtype=torch.float32)
    response_mask = torch.ones(1, 3)

    weighted, aux = apply_prune_opd_to_scores(
        scores,
        overlap_mask,
        response_mask,
        {
            "enable": True,
            "metric": "overlap_ratio",
            "threshold": 0.75,
            "w_drop": 0.1,
            "w_base": 0.5,
        },
    )

    # The first position is reliable. The second and third positions are
    # unreliable and therefore reduce all following weights.
    assert torch.allclose(aux["prune_opd_weights"], torch.tensor([[1.0, 0.9, 0.8]]))
    assert torch.allclose(weighted[0, :, 0], torch.tensor([1.5, 1.4, 1.3]))
    assert torch.allclose(aux["prune_opd_bad_event"], torch.tensor([[0.0, 1.0, 1.0]]))


def test_prune_opd_disabled_is_a_noop():
    scores = torch.randn(2, 3, 4)
    weighted, aux = apply_prune_opd_to_scores(
        scores,
        torch.zeros_like(scores),
        torch.ones(2, 3),
        {"enable": False},
    )
    assert torch.equal(weighted, scores)
    assert aux == {}
