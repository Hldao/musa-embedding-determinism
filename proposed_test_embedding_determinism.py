# SPDX-License-Identifier: BSD-3-Clause
"""Proposed torch_musa regression test for the existing embedding test suite."""

import pytest
import torch

from torch_musa import testing


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(
    testing.get_musa_arch() < 22,
    reason="bf16 is not supported on architectures older than qy2",
)
def test_embedding_dense_backward_obeys_deterministic_algorithms():
    indices = torch.tensor(
        [0] * 133 + [1 + (position % 156) for position in range(2063 - 133)],
        dtype=torch.int64,
        device="musa",
    )
    positions = torch.arange(2063 * 128, dtype=torch.float32).reshape(2063, 128)
    grad = (
        torch.sin(positions * 0.00037) * 0.125
        + torch.cos(positions * 0.00011) * 0.0625
    ).to(device="musa", dtype=torch.bfloat16)

    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        outputs = [
            torch.ops.aten.embedding_dense_backward(
                grad, indices, 512, -1, False
            ).cpu()
            for _ in range(8)
        ]
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)

    assert all(torch.equal(outputs[0], output) for output in outputs[1:])
