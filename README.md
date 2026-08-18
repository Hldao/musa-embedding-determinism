# Deterministic embedding backward for torch_musa

An unofficial, minimal reproducer and proposed fix for nondeterministic
`aten::embedding_dense_backward` results in torch_musa.

This project is maintained by WKK AI R&D. It is not affiliated with or
endorsed by Moore Threads or the PyTorch project.

[简体中文](README.zh-CN.md)

## Summary

On the tested MTT S4000 environment, repeated indices can make the BF16
embedding gradient differ across fresh processes even when inputs, seeds, and
PyTorch deterministic algorithms are identical. The public torch_musa v1.3.0
kernel selects an atomic fast path for small inputs without checking PyTorch's
deterministic-algorithm setting.

The proposed patch keeps the atomic path unchanged by default. When
`torch.use_deterministic_algorithms(True)` is enabled, it routes the operation
to torch_musa's existing sorted/segmented implementation:

```cpp
const bool use_atomic_fast_path =
    num_indices <= 3072 && !scale_grad_by_freq &&
    !globalContext().deterministicAlgorithms();
```

The patch affects embedding backward only. It does not change forward
inference or the mathematical definition of the gradient.

## Tested environment and result

The issue and fix were tested on:

- device: MTT S4000;
- PyTorch: 2.2.0;
- unpatched torch_musa wheel: 1.3.0+81caf0a;
- isolated validation build: 1.3.0+3d6a817;
- input: 2,063 indices, 157 unique IDs, BF16 gradients with width 128.

`1.3.0+81caf0a` is the version identifier reported by the installed vendor
wheel. The corresponding source commit is not available in the public
torch_musa repository.

| Mode | Fresh processes | Unique hashes | Result |
| --- | ---: | ---: | --- |
| Unpatched native | 4 | 4 | nondeterministic |
| Unpatched + deterministic requested | 4 | 4 | nondeterministic |
| Patched native | 8 | 8 | default fast path unchanged |
| Patched + deterministic requested | 8 | 1 | deterministic |
| Existing segmented-path workaround | 8 | 1 | deterministic |

The patched deterministic path and the workaround produced the same output
hash. The proposed regression test failed with the installed unpatched wheel
and passed with the isolated patched build (`1 passed`). See [`results/`](results/)
for the machine-readable reports.

## Reproduce the issue

Requirements:

- a supported MUSA device and driver;
- a compatible PyTorch and torch_musa installation;
- Python 3.10 or another version supported by that installation.

Run each mode in fresh child processes:

```bash
python3 reproduce_embedding_backward.py --runs 8
```

The script prints one JSON report containing native, deterministic-requested,
and segmented-workaround results. On an affected unpatched environment, the
first two modes can contain multiple hashes while the workaround contains one.
On a patched environment, deterministic mode should contain one hash and match
the workaround.

The input is generated mathematically. No model, checkpoint, user data, or
proprietary training method is included.

## Apply the proposed patch

The patch was prepared against the public torch_musa v1.3.0 source layout:

```bash
cd /path/to/torch_musa
git apply /path/to/musa-embedding-determinism/patches/embedding-determinism.patch
```

Build torch_musa using the instructions and dependency versions appropriate
for your MUSA environment. Do not install an experimental wheel over a working
environment; use a virtual environment or an isolated installation target.

The standalone regression test can then be run from this repository:

```bash
pytest -q proposed_test_embedding_determinism.py
```

The test is formatted as an upstream proposal. Maintainers may want to move it
into torch_musa's existing embedding test module and extend dtype, shape, and
device coverage.

## Benchmark

```bash
python3 benchmark_embedding_backward.py --warmup 25 --rounds 200
```

For the synthetic shape above, synchronized median operator latency was:

| Mode | Median | Relative to native |
| --- | ---: | ---: |
| Native atomic | 0.9701 ms | 1.000x |
| Patched deterministic | 1.0233 ms | 1.055x |
| Padding workaround | 1.0708 ms | 1.104x |

This is a single-device operator microbenchmark, not an end-to-end training
throughput claim. Timing should be repeated on every supported architecture and
representative workload before drawing broader conclusions.

## Evidence and provenance boundaries

- The result demonstrates the reported environment and input shape; it does
  not claim that every torch_musa version, MUSA device, dtype, or shape is
  affected.
- The validation build used the public torch_musa v1.3.0 base, the operator
  schema shipped with the installed runtime, and applicable compatibility
  patches. It is an ABI-compatible validation build, not a bit-for-bit rebuild
  of the installed vendor wheel.
- The experimental wheel is intentionally not distributed here. This
  repository publishes source, tests, the patch, and machine-readable results.
- Determinism has a performance cost for this operator; default behavior is
  deliberately unchanged unless deterministic algorithms are requested.

Detailed build provenance, including source and artifact hashes, is recorded
in [`results/patched-build-receipt.json`](results/patched-build-receipt.json).

## Repository contents

- [`reproduce_embedding_backward.py`](reproduce_embedding_backward.py):
  fresh-process reproducer;
- [`proposed_test_embedding_determinism.py`](proposed_test_embedding_determinism.py):
  proposed regression test;
- [`benchmark_embedding_backward.py`](benchmark_embedding_backward.py):
  synchronized operator microbenchmark;
- [`patches/embedding-determinism.patch`](patches/embedding-determinism.patch):
  minimal source patch;
- [`results/`](results/): unpatched, patched, and benchmark evidence.

## Upstreaming

The preferred destination is a pull request to
[`MooreThreads/torch_musa`](https://github.com/MooreThreads/torch_musa). A useful
upstream review should include the reproducer, regression test, minimal kernel
guard, exact runtime versions, and performance measurements. The current
upstream status will be linked here after submission.

## Contributing and license

Bug reports, measurements from other MUSA devices, and focused regression-test
improvements are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md)
before submitting a change.

WKK's original work in this repository is available under the
[BSD 3-Clause License](LICENSE). Limited upstream source context in the patch
retains its original attribution and license; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`AUTHORS.md`](AUTHORS.md).

## References

- [torch_musa embedding implementation](https://github.com/MooreThreads/torch_musa/blob/main/torch_musa/csrc/aten/ops/musa/Embedding.mu)
- [PyTorch CUDA embedding backward](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu)
- [PyTorch deterministic algorithms API](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)
