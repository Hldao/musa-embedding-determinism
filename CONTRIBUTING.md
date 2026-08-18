# Contributing

Thank you for helping improve deterministic behavior in the MUSA ecosystem.

## Useful contributions

- Reproduction results from other MUSA devices or torch_musa versions.
- Minimal tests that cover another dtype or input shape.
- Corrections to the patch for a supported public torch_musa release.
- Focused documentation and benchmark improvements.

For device reports, include the device model, PyTorch and torch_musa versions,
driver/runtime version, exact command, and the generated JSON report. Remove
hostnames, account names, access tokens, model data, and other private material
before posting.

## Pull requests

Keep changes small and explain their observable behavior. Run the static checks
locally where possible:

```bash
python3 -m compileall -q \
  reproduce_embedding_backward.py \
  benchmark_embedding_backward.py \
  proposed_test_embedding_determinism.py
```

GPU results are welcome but are not required for documentation-only changes.
Do not commit vendor wheels, SDK binaries, model weights, private training data,
credentials, or proprietary logs.

By submitting a contribution, you agree that it may be distributed under this
repository's BSD-3-Clause license. You retain copyright in your contribution.

## Conduct

Be respectful, technical, and precise. Reports should distinguish observed
facts from hypotheses and clearly state their validation boundary.
