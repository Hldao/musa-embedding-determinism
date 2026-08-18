#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Benchmark native, deterministic, and padded embedding backward paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from reproduce_embedding_backward import (
    SEGMENTED_MIN_INDICES,
    make_inputs,
    value_hash,
)


def percentile(values: list[float], fraction: float) -> float:
    """Return the nearest-rank percentile for a non-empty sample."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def child(mode: str, warmup: int, rounds: int) -> int:
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

    def invoke():
        return torch.ops.aten.embedding_dense_backward(
            grad, indices, 512, -1, False
        )

    result = None
    for _ in range(warmup):
        result = invoke()
    torch.musa.synchronize()

    samples_ms = []
    for _ in range(rounds):
        start = time.perf_counter()
        result = invoke()
        torch.musa.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    assert result is not None
    print(
        json.dumps(
            {
                "device": str(torch.musa.get_device_name(0)),
                "mean_ms": statistics.fmean(samples_ms),
                "median_ms": statistics.median(samples_ms),
                "mode": mode,
                "output_sha256": value_hash(result),
                "p95_ms": percentile(samples_ms, 0.95),
                "padding_rows": padding,
                "rounds": rounds,
                "torch": str(torch.__version__),
                "torch_musa": str(torch_musa.__version__),
                "warmup": warmup,
            },
            sort_keys=True,
        )
    )
    return 0


def parent(warmup: int, rounds: int) -> int:
    script = str(Path(__file__).resolve())
    report = {}
    for mode in ("native", "deterministic", "workaround"):
        completed = subprocess.run(
            [
                sys.executable,
                script,
                "--child",
                mode,
                "--warmup",
                str(warmup),
                "--rounds",
                str(rounds),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        report[mode] = json.loads(completed.stdout.strip().splitlines()[-1])

    native_median = report["native"]["median_ms"]
    for mode in ("deterministic", "workaround"):
        report[mode]["median_slowdown_vs_native"] = (
            report[mode]["median_ms"] / native_median
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark MUSA embedding backward paths."
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument(
        "--child", choices=("native", "deterministic", "workaround")
    )
    args = parser.parse_args()
    if args.warmup < 1:
        parser.error("--warmup must be at least 1")
    if args.rounds < 2:
        parser.error("--rounds must be at least 2")
    if args.child:
        return child(args.child, args.warmup, args.rounds)
    return parent(args.warmup, args.rounds)


if __name__ == "__main__":
    raise SystemExit(main())
