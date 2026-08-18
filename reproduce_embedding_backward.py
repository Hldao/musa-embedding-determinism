#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Cross-process reproducer for MUSA embedding backward nondeterminism."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ATOMIC_MAX_INDICES = 3072
SEGMENTED_MIN_INDICES = ATOMIC_MAX_INDICES + 1


def make_inputs(torch):
    """Create deterministic synthetic inputs with repeated indices."""
    indices = [0] * 133
    indices.extend(1 + (position % 156) for position in range(2063 - 133))
    cpu_indices = torch.tensor(indices, dtype=torch.int64)
    positions = torch.arange(2063 * 128, dtype=torch.float32).reshape(2063, 128)
    cpu_grad = (
        torch.sin(positions * 0.00037) * 0.125
        + torch.cos(positions * 0.00011) * 0.0625
    ).to(dtype=torch.bfloat16)
    return cpu_indices, cpu_grad


def value_hash(tensor) -> str:
    """Hash tensor values after a stable FP32 CPU conversion."""
    payload = tensor.detach().float().cpu().contiguous().numpy().tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def child(mode: str) -> int:
    import torch
    import torch_musa  # noqa: F401

    torch.use_deterministic_algorithms(mode == "deterministic")
    indices, grad = make_inputs(torch)
    indices = indices.to("musa:0")
    grad = grad.to("musa:0")
    padding = 0

    if mode == "workaround":
        padding = SEGMENTED_MIN_INDICES - int(indices.numel())
        indices = torch.cat(
            (
                indices,
                torch.zeros(padding, dtype=indices.dtype, device=indices.device),
            )
        )
        grad = torch.cat(
            (
                grad,
                torch.zeros(
                    (padding, grad.shape[-1]),
                    dtype=grad.dtype,
                    device=grad.device,
                ),
            )
        )

    result = torch.ops.aten.embedding_dense_backward(
        grad, indices, 512, -1, False
    )
    torch.musa.synchronize()
    print(
        json.dumps(
            {
                "device": str(torch.musa.get_device_name(0)),
                "mode": mode,
                "padding_rows": padding,
                "sha256": value_hash(result),
                "torch": str(torch.__version__),
                "torch_musa": str(torch_musa.__version__),
            },
            sort_keys=True,
        )
    )
    return 0


def parent(runs: int) -> int:
    script = str(Path(__file__).resolve())
    report = {}
    for mode in ("native", "deterministic", "workaround"):
        values = []
        for _ in range(runs):
            completed = subprocess.run(
                [sys.executable, script, "--child", mode],
                check=True,
                text=True,
                capture_output=True,
            )
            values.append(json.loads(completed.stdout.strip().splitlines()[-1]))

        hashes = sorted({value["sha256"] for value in values})
        report[mode] = {
            "fresh_processes": runs,
            "reproducible": len(hashes) == 1,
            "runtime": {
                key: values[0][key] for key in ("torch", "torch_musa", "device")
            },
            "unique_hashes": hashes,
        }

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce MUSA embedding backward nondeterminism."
    )
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument(
        "--child", choices=("native", "deterministic", "workaround")
    )
    args = parser.parse_args()
    if args.child:
        return child(args.child)
    if args.runs < 2:
        parser.error("--runs must be at least 2")
    return parent(args.runs)


if __name__ == "__main__":
    raise SystemExit(main())
