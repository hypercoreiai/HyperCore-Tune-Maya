#!/usr/bin/env python3
"""HyperCore-Tune-maya/stub_engine.py - a deterministic stand-in for build/strata.

Speaks the same protocol as the real engine with --serve: an INFO line, then the serve
contract's `READY <max_context> stop pause` line, then it reads "GEN <max_new> [keys] <token ids>"
on stdin and answers with a DONE line, and stays up until QUIT. The numbers are a smooth
function of the swept flags and request keys with a single interior optimum, which is what a
Bayesian sampler needs to show convergence.

With --parallel 2..8 it is in the several-conversation mode the real engine enters at
STRATA_GLM_SEQS > 1: READY advertises `multi=<n>`, a GEN line without `rid=<id>` is refused,
and DONE lines carry the rid. --vocab-size N rejects ids outside [0, N) the way the pack's
vocabulary does; unset, every id is accepted.

A GEN line carrying `think_budget=<n> think_end=<id>` reasons for min(max_new, n) tokens and
then answers, writing a `T <token>` line per token paced so the request's wall time matches
the decode_ms it reports. That is what lets a sweep measure the reasoning block separately.
Without those keys the request is one decode phase, as the engine's contract allows.

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


def gen_rate(
    pcie: float,
    pcie_mode: str,
    kv: str,
    workers: int,
    host_out: bool,
    kvstream: str,
    vram: int,
    draft: float,
    suffix: int,
    mtp_max: int,
    parallel: int,
    spec: int = 0,
    cpu_threads: int = 0,
    read_chunks: int = 0,
    spec_n_min: int = -1,
) -> float:
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
    window = mtp_max if mtp_max > 0 else 4
    if suffix > 0:
        gen += 14.0 * min(suffix, window) / 8.0
    gen *= 1.0 - 0.06 * (parallel - 1)
    if spec > 1:
        gen += 8.0 * min(spec, 8) / 8.0
    if cpu_threads > 0:
        gen += 2.0 * min(cpu_threads, 8)
    if read_chunks > 0:
        gen += 1.0 * min(read_chunks, 8)
    if spec_n_min >= 0:
        gen += 2.0 * min(spec_n_min, 8) / 8.0
    return max(20.0, gen)


def parse_request_keys(parts: list[str]) -> tuple[dict[str, str], str]:
    keys: dict[str, str] = {}
    ids = ""
    for tok in parts[2:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            keys[k] = v
        else:
            ids = tok
    return keys, ids


def main() -> int:
    flags = parse_flags(sys.argv[1:])
    delay = float(flags.get("--delay", "0.02"))
    context = int(flags.get("--max-context", "4096"))
    vram = int(flags.get("--vram-reserve-mib", "512"))
    pcie = float(flags.get("--pcie-frac", "0.95"))
    pcie_mode = flags.get("--pcie-mode", "auto")
    kv = flags.get("--kv", "fp16")
    ple = flags.get("--ple-io", "direct")
    kvstream = flags.get("--kv-streaming", "on")
    host_out = "--no-host-worker" in flags
    workers = int(flags.get("--pool-workers", "0")) or 16
    checkpoints = int(flags.get("--prompt-cache", "6"))
    draft = float(flags.get("--spec-min-p", "0.8"))
    glm = "--glm-pack" in flags
    parallel = int(flags.get("--parallel", "1"))
    vocab = int(flags.get("--vocab-size", "0"))   # 0: accept any id, as no pack is named
    ple_inflight = int(flags.get("--ple-inflight", "64"))
    suffix = int(flags.get("--suffix-draft", "3"))
    mtp_max = int(flags.get("--mtp-max-t", "0"))

    if vram < 256 or vram > 2048:
        print("glm error: vram reserve out of range")
        return 1
    if kv not in ("q4_0", "k8v4", "int8", "fp16"):
        print(f"glm error: invalid --kv {kv}")
        return 2
    if glm and kv in ("q4_0", "k8v4"):
        print(f"glm error: --kv {kv} is not implemented by the glm runner")
        return 2
    if kv == "k8v4" and kvstream == "on":
        print("glm error: --kv k8v4 does not support --kv-resident streaming (yet)")
        return 2
    if not 1 <= parallel <= 8:
        print("glm error: --parallel takes 1..8")
        return 2
    if not 1 <= ple_inflight <= 1024:
        print("glm error: --ple-inflight takes 1..1024")
        return 2
    if suffix < 0 or mtp_max < 0:
        print("glm error: --suffix-draft and --mtp-max-t take 0 or more")
        return 2

    prefill = 900.0 * (1.0 if ple == "direct" else 0.7 if ple == "mmap" else 1.15)
    prefill *= 1.0 + 0.05 * min(context, 131072) / 131072
    prefill *= 1.0 + 0.02 * max(0, min(checkpoints, 6))
    if ple == "mmap":
        prefill *= 1.0 + 0.35 * min(ple_inflight, 128) / 128

    print(f"INFO context={context} kv={kv} kv_resident={context if kvstream == 'on' else 0} "
          f"pool_workers={workers} pcie_frac={pcie:.2f} pcie_mode={pcie_mode} "
          f"ple_inflight={ple_inflight} parallel={parallel} suffix_draft={suffix} "
          f"mtp_max_t={mtp_max} vram_free_mib={24000 - vram} engine_kind=glm-fast engine=stub")
    # The unit costs the real engine measures for this machine at boot. The tuner reads
    # these to turn the STAT counts into times, so the stub has to print them too.
    blob_bytes = 7_012_352.0        # one IQ2_XXS expert blob, the pack the stub stands in for
    warm_gb_per_s = 1.85            # what this machine pulls off the SSD during warm-up
    cpu_ms = 0.130
    pcie_ms = 0.450
    lane = [min(workers, k) for k in range(9)]   # for a layer with k RAM experts, how many the host takes
    print(f"glm fast: CUDA0 CPU lane: {workers} threads, an expert {cpu_ms:.3f} ms on the CPU "
          f"vs {pcie_ms:.3f} ms over PCIe -> of 0..8 RAM-tier experts the CPU takes "
          f"{' '.join(str(x) for x in lane)}")
    print("glm fast: CUDA0 warming the expert tiers 100% (37.3 of 37.3 GB, 20 s)")
    print(f"glm fast: CUDA0 tiers warm: {int(37.3 / (blob_bytes / 1e9))} experts "
          f"(37.3 GB) in {20} s ({warm_gb_per_s:.2f} GB/s)")
    multi = parallel > 1
    print(f"READY {context} stop pause" + (f" multi={parallel}" if multi else ""))
    print("maya generate: glm5-next serve (M3.1): GEN <max_new> [keys] <ids>; QUIT to end")
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
        keys, ids_token = parse_request_keys(parts)
        rid = keys.get("rid")
        if multi and rid is None:
            print("ERR this engine runs several conversations: GEN needs rid=<id>")
            sys.stdout.flush()
            continue
        try:
            ids = [int(x) for x in ids_token.split(",") if x]
        except ValueError:
            print(f"ERR {rid} malformed GEN line" if rid is not None else "ERR malformed GEN line")
            sys.stdout.flush()
            continue
        if not ids or max_new <= 0:
            print(f"ERR {rid} empty prompt or max_new" if rid is not None
                  else "ERR malformed GEN line")
            sys.stdout.flush()
            continue
        bad = next((t for t in ids if vocab and not 0 <= t < vocab), None)
        if bad is not None:
            print(f"ERR {rid} token {bad} out of range [0, {vocab})" if rid is not None
                  else f"ERR token {bad} out of range [0, {vocab})")
            sys.stdout.flush()
            continue
        prompt_tokens = len(ids)

        req_pcie = pcie
        req_draft = draft
        req_spec: int | None = None
        req_cpu: int | None = None
        req_chunks: int | None = None
        req_n_min: int | None = None
        req_think = 0
        req_think_end: int | None = None
        try:
            if "pcie_frac" in keys:
                req_pcie = float(keys["pcie_frac"])
            if "spec_min_p" in keys:
                req_draft = float(keys["spec_min_p"])
            if "spec" in keys:
                req_spec = int(keys["spec"])
            if "cpu_threads" in keys:
                req_cpu = int(keys["cpu_threads"])
            if "read_chunks" in keys:
                req_chunks = int(keys["read_chunks"])
            if "spec_n_min" in keys:
                req_n_min = int(keys["spec_n_min"])
            if "think_budget" in keys:
                req_think = int(keys["think_budget"])
            if "think_end" in keys:
                req_think_end = int(keys["think_end"])
        except ValueError:
            print("ERR malformed request key")
            sys.stdout.flush()
            continue

        if not 0.0 <= req_pcie <= 1.0:
            print("ERR pcie_frac out of range")
            sys.stdout.flush()
            continue
        if not 0.0 <= req_draft <= 1.0:
            print("ERR spec_min_p out of range")
            sys.stdout.flush()
            continue
        if req_spec is not None and (req_spec < 1 or req_spec > 16):
            print("ERR spec out of range")
            sys.stdout.flush()
            continue
        if req_cpu is not None and req_cpu < 0:
            print("ERR cpu_threads out of range")
            sys.stdout.flush()
            continue
        if req_chunks is not None and (req_chunks < 0 or req_chunks > 16):
            print("ERR read_chunks out of range")
            sys.stdout.flush()
            continue
        if req_n_min is not None and (req_n_min < 0 or req_n_min > 15):
            print("ERR spec_n_min out of range")
            sys.stdout.flush()
            continue
        if req_think < 0:
            print("ERR think_budget out of range")
            sys.stdout.flush()
            continue
        if req_think > 0 and req_think_end is None:
            print("ERR think_budget needs think_end=<id>")
            sys.stdout.flush()
            continue

        gen = gen_rate(
            req_pcie, pcie_mode, kv, workers, host_out, kvstream, vram,
            req_draft, suffix, mtp_max, parallel,
            req_spec if req_spec is not None else 0,
            req_cpu if req_cpu is not None else 0,
            req_chunks if req_chunks is not None else 0,
            req_n_min if req_n_min is not None else -1,
        )
        time.sleep(delay)
        prompt_ms = 1000.0 * prompt_tokens / prefill
        if req_think > 0:
            # The reasoning block is min(max_new, budget) tokens, closed by the think-end
            # token; the answer follows in the same decode. Reasoning runs slower than the
            # answer, and every token is written as it is "produced" so the wall clock the
            # tuner sees matches the decode_ms this line reports.
            thinking = min(max_new, req_think)
            reason_rate = max(1.0, gen * 0.6)
            think_ms = 1000.0 * thinking / reason_rate
            decode_ms = think_ms + 1000.0 * max(0, max_new - thinking) / gen
            for i in range(max_new):
                tok = req_think_end if i == thinking else 1000 + i
                print(f"T {rid} {tok}" if rid is not None else f"T {tok}")
                # The real serve loop runs stdout unbuffered (generate.cpp:3146), so a
                # token is on the wire the moment it is produced.
                sys.stdout.flush()
                if i < thinking:
                    time.sleep(1.0 / reason_rate)
            time.sleep(max(0.0, (decode_ms - think_ms) / 1000.0))
        else:
            decode_ms = 1000.0 * max_new / gen
        # The same telemetry the real serve engine prints, so a sweep can weigh a rate
        # against the bus it needed. A bigger reserve leaves fewer pool slots and a thinner
        # PCIe link moves fewer experts onto the card, so residency - and the traffic that
        # has to cross a bus - follows the flags directly.
        experts = 336.0                       # 8 routed experts x 42 layers, GLM-5.3-Flash
        ms_per_tok = decode_ms / max(1, max_new)
        resident = min(0.95, (24000 - vram) / 24000.0 * 0.30 + 0.55 * req_pcie
                       + 0.10 * min(workers, 16) / 16.0)
        off_card = experts * (1.0 - resident)
        share = min(0.85, (1.0 - resident) * 0.9)
        disk_ms = ms_per_tok * share * (0.62 if ple == "direct" else 0.38)
        ram_ms = ms_per_tok * share * (0.38 if ple == "direct" else 0.62)
        # Counts follow from the time and the engine's own unit costs: a read costs the blob
        # over the measured SSD rate, a RAM fetch costs the weighted CPU/PCIe lane time.
        disk_reads = disk_ms / (blob_bytes / 1e9 / warm_gb_per_s * 1000.0)
        ram_fetches = ram_ms / (0.19 if workers else pcie_ms)
        pool_used = int((24000 - vram) * resident)
        for step in range(8, max_new + 1, 8):
            done_ms = ms_per_tok * step
            print(f"STAT tok_s={step * 1000.0 / done_ms:.2f} ms_tok={ms_per_tok:.2f} "
                  f"vram_hit={resident:.4f} ram_fetch={ram_fetches:.3f} "
                  f"disk={disk_reads:.3f} promo={off_card * 0.10:.3f} "
                  f"vram_used={pool_used} vram_slots=24000 "
                  f"ram_used={int(off_card * 7)} ram_slots=1000 ram_gb=20.0 vram_gb=11.5 "
                  f"active={parallel}")
        print((f"DONE {rid} " if rid is not None else "DONE ") +
              f"{max_new} {prompt_tokens} {prompt_ms:.1f} {decode_ms:.1f} length "
              f"0 0 0 0 0")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
