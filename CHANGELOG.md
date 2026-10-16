# Changelog

Everything below is what the code in this repository actually does; the README is the
reference for how to use it. Dates are the commit dates.

## 2026-10-16 - the bus becomes measurable

**Why:** a ten-trial `--balance` sweep on Maya-S24 reported `bus: 0.0% of a token is
off-card time` for every configuration. The parser was reading a line the engine never
prints on the path a sweep runs.

- `transfer.py` (new) reads two real telemetry sources:
  - `STAT tok_s= ms_tok= vram_hit= ram_fetch= disk= promo= vram_used=/slots ram_used=/slots`,
    the live counter line a `--serve` engine emits every eight tokens
    (`src/program/generate.cpp:1820, 2122`) - the one a parallel sweep actually hits.
  - `glm stat:` / `glm kv:` / `glm spec:` / `glm skip:`, kept for builds that do print
    them; `measured_times` records which source a number came from.
- Unit costs come from the engine's own boot lines (`an expert 0.136 ms on the CPU vs
  0.453 ms over PCIe`, `tiers warm: 4614 experts (32.8 GB) in 18 s (1.87 GB/s)`), not from
  assumptions. The expert blob is read from the pack's `native_experts.txt` (Maya-S24:
  7,012,352 bytes median).
- Derived: `off_card_ms_per_tok`, `transfer_share` (the larger of the attributed time and
  `1 - vram_hit`), `compute_tok_s` (the ceiling if the bus were free), `headroom`,
  `disk_ms_per_read`, `disk_saturation`, `pareto_front()`, `knee()` (the 3% minimum-gain /
  70% stop rule `project-maya/tools/calibrate_glm.py` uses) and `frontier()`.
- `--balance` objective: `decode_tok_s * (1 - 0.35 * transfer_share)`. No telemetry means
  no discount, so `--balance` behaves exactly like `--decode` on a stub or an older build.
- Independent cross-checks, so the derived share can be tested rather than trusted:
  `hardware.disk_io_snapshot()` (psutil, before and after each request) and
  `hardware.pcie_throughput_kb_s()` + `PcieSampler` (NVML, polled every 250 ms during a
  request). Both are `None` where the host has no NVML or no psutil.
- `think_end` is resolved from the pack tokenizer (`</think>` = 154842 on Maya-S24,
  `<think>` = 154841) instead of the hardcoded 151667; 151667 remains the fallback for a
  pack with no tokenizer, and the sweep says which it used.
- `stub_engine.py` emits the serve-path format (STAT lines plus the boot cost lines), so
  the whole channel is testable without a GPU.
- `test_transfer.py`: 19 tests over the parser, the derived quantities and the objective.
  Suite total: 130 tests, no GPU required.

Verified against the logs of a real sweep on this machine: 9.04 tok/s at 59% off-card,
17.68 reads/tok at 3.83 ms each, the drive at 55% of its measured 1.83 GB/s, and a
card-alone ceiling of 22.07 tok/s. The objective reorders configurations: 8.90 tok/s at
55% bus outranks 8.96 tok/s at 60%.

## 2026-10-15 - Project Maya deliverables

- Deliverables are written in Maya's own run-config shape (`max_<model>_config.json`,
  `max_<model>_run.sh`, `max_<model>_run.bat`) and never overwrite the installed
  `maya-<model>.json` / `run-maya-<model>.sh`.
- Keys the installed config carries that the installer does not own are preserved.
- A flag is emitted only when this checkout's `serve/server.py` or
  `src/program/generate.cpp` can read it; `maya_env.server_flags()` and
  `maya_env.engine_flags()` read the source to find out.
- The project root, engine binary, server script and venv are resolved from where this
  repository sits, and every path written into a deliverable is absolute.

## 2026-10-13 / 2026-10-14 - the standalone sweep

- The Optuna sweep became its own repository, attachable to a Maya checkout or run beside
  it, on Linux and Windows 10/11.
- `--trials` counts unseen configurations: a suggestion matching a point already in
  history is redrawn, not counted.
- Run data is keyed to the machine fingerprint (CPU, GPU, RAM, PCIe generation), the model,
  the objective and the context, so priors from another box are not merged.
- A prior is a seed, never a winner: the configuration written into the deliverable is one
  this sweep booted on this machine.
