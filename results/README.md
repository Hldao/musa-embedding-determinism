# Results

These files are machine-readable snapshots from the tested MTT S4000
environment. They are evidence for the exact versions and synthetic input
described in the repository README, not a claim about every MUSA release or
device.

- `unpatched-s4000.json`: four fresh processes per mode with the installed
  `torch_musa 1.3.0+81caf0a` wheel.
- `patched-s4000.json`: eight fresh processes per mode with the isolated
  validation build.
- `patched-benchmark-s4000.json`: 25 warmup calls and 200 synchronized samples
  per mode.
- `patched-build-receipt.json`: source, schema, patch, and built artifact hashes.

The experimental wheel is not distributed in this repository.

