#!/usr/bin/env python3
"""HyperCore-Tune-Maya/stub_engine.py - a deterministic stand-in for build/strata.

Speaks the same protocol as the real engine with --serve: it prints an INFO line, says
"session is up", reads "GEN <max_new> <token ids>" on stdin, answers with a DONE line,
and stays up until QUIT. The numbers are a smooth function of the swept flags with a
single interior optimum, which is what a Bayesian sampler needs to show convergence.

    ./HyperCore-Tune-Maya/stub_engine.py --serve --max-context 131072 --kv q4_0 --pool-workers 8
"""
from __future__ import annotations

import sys
import time


def parse_flags(argv: list[str]) -> dict[str, str]:
    flags: dict[str, str] = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            flags[argv[i]] = argv[i + 1]
            i += 2
        elif argv[i].startswith("--"):
            flags[argv[i]] = "1"
            i += 1
        else:
            i += 1
    return flags


def main() -> int:
    flags = parse_flags(sys.argv[1:])
    delay = float(flags.get("--delay", "0.02"))
    context = int(flags.get("--max-context", "4096"))
    vram = int(flags.get("--vram-reserve-mib", "512"))
    pcie = float(flags.get("--pcie-frac", "0.95"))
    pcie_mode = flags.get("--pcie-mode", "auto")
    kv = flags.get("--kv", "q4_0")
    ple = flags.get("--ple-io", "direct")
    kvstream = flags.get("--kv-streaming", "on")
    # --no-host-worker present means the host thread is kept out of the drain.
    host_out = "--no-host-worker" in flags
    workers = int(flags.get("--pool-workers", "0")) or 16
    checkpoints = int(flags.get("--prompt-cache", "6"))
    draft = float(flags.get("--spec-min-p", "0.8"))

    if vram < 256 or vram > 2048:
        print("glm error: vram reserve out of range")
        return 1
    if kv not in ("q4_0", "k8v4", "int8", "fp16"):
        print(f"glm error: invalid --kv {kv}")
        return 2

    kv_gain = {"k8v4": 1.0, "q4_0": 0.85, "int8": 0.6, "fp16": 0.3}.get(kv, 0.5)
    gen = 110.0
    gen += 45.0 * max(0.0, min(1.0, pcie))
    gen += {"dma": 8.0, "kernel": 4.0, "direct": 2.0}.get(pcie_mode, 0.0)
    gen += 22.0 * kv_gain
    gen += 6.0 * min(workers, 16)
    if host_out:
        gen -= 18.0
    if kvstream == "on":
        gen += 3.0
    if vram < 512:
        gen -= 25.0
    gen += 8.0 * max(0.0, 1.0 - abs(draft - 0.5) / 0.5)
    gen = max(20.0, gen)

    prefill = 900.0 * (1.0 if ple == "direct" else 0.7 if ple == "mmap" else 1.15)
    prefill *= 1.0 + 0.05 * min(context, 131072) / 131072
    prefill *= 1.0 + 0.02 * max(0, min(checkpoints, 6))

    print(f"INFO context={context} kv={kv} kv_resident={context if kvstream == 'on' else 0} "
          f"pool_workers={workers} pcie_frac={pcie:.2f} pcie_mode={pcie_mode} "
          f"vram_free_mib={24000 - vram} engine_kind=glm-fast engine=stub")
    print("maya generate: session is up (engine stub)")
    sys.stdout.flush()

    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "QUIT":
            break
        if parts[0] != "GEN":
            continue
        max_new = int(parts[1]) if len(parts) > 1 else 256
        prompt_tokens = len(parts[2].split(",")) if len(parts) > 2 else 0
        time.sleep(delay)
        prompt_ms = 1000.0 * prompt_tokens / prefill
        decode_ms = 1000.0 * max_new / gen
        print(f"DONE {max_new} {prompt_tokens} {prompt_ms:.1f} {decode_ms:.1f} length "
              f"0 0 0 0 0")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
