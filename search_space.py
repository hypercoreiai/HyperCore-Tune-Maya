"""HyperCore-Tune-Maya/search_space.py - the parameter space the Bayesian sweep searches.

Every trial writes one row with the columns the Project Maya engine actually has:

    layer  vram  pcie  pcie_mode  ple_io  ple_inflight  kv  kvstream  hostworker  worker
           tail  draft_floor  suffix_draft  mtp_max_t  parallel  status

plus the transfer columns transfer.py appends to every row:

    decode_ms_per_tok  off_card_ms_per_tok  transfer_share  off_card_lookup_share
    compute_tok_s  headroom  disk_ms_per_read  disk_saturation  measured_times
    disk_bytes_per_s  measured_disk_bytes_per_s  disk_attribution  pcie_bytes_per_s
    measured_pcie_rx_mb_s  measured_pcie_tx_mb_s  pcie_attribution  disk_reads_per_tok
    disk_ms_per_tok  ram_fetches_per_tok  promotions_per_tok  cpu_experts_per_tok
    cpu_ms_per_tok  vram_hit_pct  kv_hit_pct  kv_mib_from_ram  spec_accepted
    spec_steps  flags

`layer` is the group a trial is perturbing. In staged mode it is the layer the study is
currently frozen on; in joint mode it is the group whose value differs from the incumbent
best ("joint" for the first trial, which has no incumbent). `status` is not a parameter:
it is what the engine did with the configuration (PASS, FAIL_BOOT, NO_TOKENS, TIMEOUT).

Each dimension maps to a flag `src/program/generate.cpp` really parses, and every range is
the bound that file enforces - a grid outside it is not a slower arm, it is an exit 2:

    vram         --vram-reserve-mib        MiB of VRAM the expert pool leaves free  (>= 0, 3431)
    pcie         --pcie-frac               share of the PCIe link the RAM tier may use (0..1, 5944)
    pcie_mode    --pcie-mode              auto|dma|kernel|direct (5561)
    ple_io       --ple-io                  n-gram table reads: direct (SSD) or mmap (3552)
    ple_inflight --ple-inflight           outstanding SSD reads; the engine rejects 0 or >1024 (3553)
    kv           --kv                      the runner's KV modes only - see kv_modes_for below
    kvstream     --kv-streaming            on keeps the KV window resident, off keeps all of it
    hostworker   --no-host-worker         1 = the host thread joins the CPU pool (the default),
                                         0 = keep it out of the drain
    worker       --pool-workers            CPU expert pool size; 0 = every core but the host's
    tail         --prompt-cache            conversation checkpoints kept between requests (>= 0, 3323)
    draft_floor  --spec-min-p             the acceptance floor for drafted tokens (clamped 0..1, 5944)
    parallel     --parallel               conversations decoded at once; 1..8 or the engine says so (3193)
    suffix_draft --suffix-draft           prompt-lookup drafting: 0 = off (clamped >= 0, 3328)
    mtp_max_t    --mtp-max-t              cap the MTP's draft windows; 0 = as --spec gives (3329)

Two dimensions are only real for some packs, and building the space for the wrong one
measures a no-op for every trial:

  * kv: the GLM runner has a latent cache that is fp16 or INT8 and nothing else
    (generate.cpp:3415 -> glm_model.cu:3073); q4_0 and k8v4 are parsed, never reached,
    and the INFO line can only report int8 or fp16 (generate.cpp:1028).
  * suffix_draft / mtp_max_t: with a --dflash drafter the decode goes through
    Glm5Model::decode_dflash (generate.cpp:1842) and the MTP/suffix path is never run.

A freeze overlay (sample(..., frozen)) holds pinned dimensions at a constant.

Determinism: every parameter is sampled from a closed grid, so a trial's configuration is
reproducible from (seed, trial index) and the signature of a configuration is a plain
tuple. Nothing here reads the environment except the CPU count, which bounds the worker
grid and is overridable with --max-workers.
"""
from __future__ import annotations

import itertools
import os
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from maya_env import abs_path, data_roots

try:  # psutil is in .venv (requirements.txt); the fallback keeps this importable anywhere.
    import psutil
except ImportError:  # pragma: no cover - only hit on a bare interpreter
    psutil = None  # type: ignore[assignment]

LAYERS: tuple[str, ...] = (
    "topology", "ple-io", "ple-read", "kv", "kv-stream", "cpu-pool", "worker", "tail",
    "draft-floor", "draft", "serve",
)

# Column order of the CSV: the engine dimensions, then what the engine did with them.
ROW_COLUMNS: tuple[str, ...] = (
    "layer", "vram", "pcie", "pcie_mode", "ple_io", "ple_inflight", "kv", "kvstream",
    "hostworker", "worker", "tail", "draft_floor", "suffix_draft", "mtp_max_t",
    "parallel", "status",
)

Config = dict[str, object]

# The KV modes each runner implements. generate.cpp parses all four (3557-3564) but only
# the qwen4exp session has four: on the --glm-pack path the call to glm_pack_generate
# (3443) happens before the k8v4 check (3569), and Glm5Model's latent cache is fp16 or
# INT8 (STRATA_GLM_KV_INT8, glm_model.cu:3073) - q4_0 and k8v4 reach the parse table and
# are then ignored, so sweeping them measures the same run twice.
GLM_KV_MODES: tuple[str, ...] = ("int8", "fp16")
QWEN_KV_MODES: tuple[str, ...] = ("fp16", "int8", "q4_0", "k8v4")
KV_MODES: tuple[str, ...] = QWEN_KV_MODES
PLE_MODES: tuple[str, ...] = ("direct", "mmap")
PCIE_MODES: tuple[str, ...] = ("auto", "dma", "kernel", "direct")

# Bounds the engine enforces; keep these in step with the line named in each comment.
PARALLEL_RANGE: tuple[int, int] = (1, 8)          # generate.cpp:3193 "--parallel takes 1..8"
PLE_INFLIGHT_RANGE: tuple[int, int] = (1, 1024)  # generate.cpp:3553
SUFFIX_DRAFT_RANGE: tuple[int, int] = (0, 8)     # generate.cpp:3328 clamps to >= 0
MTP_MAX_T_RANGE: tuple[int, int] = (0, 8)        # generate.cpp:3329 clamps to >= 0
KV_RESIDENT_MIN: int = 20480                     # generate.cpp:375, usage line

# The engine accepts these as per-request GEN keys (generate.cpp:2998-3005, 5943-5944).
# A --two-stage sweep sends them inside one running session instead of re-booting for each.
PCIE_RANGE: tuple[float, float] = (0.0, 1.0)     # generate.cpp:5943 std::clamp(..., 0.0, 1.0)
SPEC_MIN_P_RANGE: tuple[float, float] = (0.0, 1.0)  # generate.cpp:5944 std::clamp(..., 0.0, 1.0)
VRAM_RESERVE_MIN: int = 0                        # generate.cpp:3431: 0 or more MiB
PROMPT_CACHE_MIN: int = 0                        # generate.cpp:3323 std::max(0, ...)

# Keys that the engine can override for one request without restarting. Existing CLI flags
# are kept in this table so a session can omit them from the boot arguments.
REQUEST_KEYS: dict[str, str] = {
    "pcie": "pcie_frac",
    "draft_floor": "spec_min_p",
    "cpu_threads": "cpu_threads",
    "read_chunks": "read_chunks",
    "spec": "spec",
    "spec_n_min": "spec_n_min",
}

# Optional request-only dimensions. They are not CLI flags; they are GEN keys.
REQUEST_DIMS: tuple[str, ...] = ("cpu_threads", "read_chunks", "spec", "spec_n_min")
CPU_THREADS_RANGE: tuple[int, int] = (1, 64)     # engine accepts any positive int
READ_CHUNKS_RANGE: tuple[int, int] = (1, 8)      # server.py validates 1..16; keep the grid small
SPEC_RANGE: tuple[int, int] = (1, 8)             # 1 disables drafting; the engine's MTP window is bounded by 8
SPEC_N_MIN_RANGE: tuple[int, int] = (0, 15)      # server.py validates 0..15


@dataclass(frozen=True)
class Param:
    """One dimension of the search space.

    kind    "int" | "float" | "choice"
    flag    the engine flag this parameter writes
    mode    "value" -> "--flag <value>"; "on" -> bare "--flag" when the value is truthy
    invert  with mode "on": emit the flag when the value is falsy (--no-host-worker)
    step    grid step for int/float sampling (keeps the space finite and reproducible)
    """

    key: str
    layer: str
    kind: str
    flag: str
    mode: str = "value"
    low: float = 0.0
    high: float = 1.0
    step: float = 1.0
    choices: tuple[str, ...] = ()
    invert: bool = False

    def grid(self) -> list[object]:
        if self.kind == "choice":
            return list(self.choices)
        if self.kind == "int":
            return list(range(int(self.low), int(self.high) + 1, int(self.step)))
        steps = round((self.high - self.low) / self.step)
        return [round(self.low + i * self.step, 4) for i in range(int(steps) + 1)]

    def applies(self, value: object) -> bool:
        """Whether this configuration asks the engine to turn the flag on."""
        text = str(value).strip().lower()
        on = text not in ("0", "false", "off", "none", "")
        return not on if self.invert else on


def pack_kind(pack_dir: str | Path | None) -> str:
    """Which runner a pack drives: "glm" or "qwen".

    The pack index header names the GGUF it was generated from, and that is the model the
    runner for it is written for. A pack we cannot read is treated as a GLM pack: every
    deliverable this project writes goes to --glm-pack, and the GLM mode set is the
    narrower of the two, so guessing GLM can never send the engine a flag it rejects.
    """
    if not pack_dir:
        return "glm"
    try:
        head = Path(pack_dir).read_text(encoding="utf-8", errors="replace")
    except OSError:
        for name in ("index.txt", "sources.json", "compat-bf16.json"):
            try:
                head = (Path(pack_dir) / name).read_text(encoding="utf-8", errors="replace")
                break
            except OSError:
                continue
        else:
            return "glm"
    upper = head.upper()
    if "GLM" in upper:
        return "glm"
    if "QWEN" in upper:
        return "qwen"
    return "glm"


def kv_modes_for(pack_dir: str | Path | None) -> tuple[str, ...]:
    """The --kv arms this pack's runner can actually run."""
    return GLM_KV_MODES if pack_kind(pack_dir) == "glm" else QWEN_KV_MODES


def effective_kv(pack_dir: str | Path | None, kv: str) -> str:
    """The KV mode a run on this pack really used.

    generate.cpp:3444 returns glm_pack_generate(o) before the --kv validation at 3558 and
    the qsa_set_kv_* calls at 3562-3564, so on a GLM pack q4_0 and k8v4 are parsed and then
    ignored: the DSA latent cache is INT8 only when --kv int8 sets STRATA_GLM_KV_INT8
    (3415) and fp16 otherwise. A record carrying one of those modes measured the fp16 run,
    and history has to read it as that for the prior to be usable.
    """
    modes = kv_modes_for(pack_dir)
    return kv if kv in modes else "fp16"


def pack_vram_floor_mib(pack_dir: str | Path | None) -> int:
    """The MiB of VRAM a pack needs before any configuration in the grid can run.

    index.txt's header line carries the pool size - '# align 64 pool <bytes> tensors N' -
    which is dense.bin's length. The engine loads those dense weights into a device arena
    before it sizes the expert pool from whatever VRAM is left (glm_model.cu:1237), so a
    free memory below this boots nothing at any setting. 0 when the pack cannot be read:
    the sweep then says nothing rather than guessing.
    """
    if not pack_dir:
        return 0
    try:
        head = Path(pack_dir, "index.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    for line in head.splitlines():
        if "pool " not in line:
            continue
        try:
            return int(line.split("pool", 1)[1].split()[0]) // (1024 * 1024)
        except (IndexError, ValueError):
            return 0
    return 0


def physical_cores() -> int:
    """Physical cores the pool can actually be spread over; logical cores as a fallback."""
    n = 0
    if psutil is not None:
        try:
            n = int(psutil.cpu_count(logical=False) or 0)
        except Exception:  # noqa: BLE001 - a container that reports nothing must not stop the sweep
            n = 0
    if n > 0:
        return n
    return max(1, os.cpu_count() or 1)


def parse_engine_bounds(source: str) -> dict[str, tuple[float, float]]:
    """Extract the bounds the engine source enforces.

    The parser is deliberately narrow: it only reads the clamp/validation lines that are
    already the authority for a dimension. A missing pattern is simply absent, so the test
    can still run against an older checkout without inventing a bound.
    """
    bounds: dict[str, tuple[float, float]] = {}

    m = re.search(r'--parallel takes (\d+)\.\.(\d+)', source)
    if m:
        bounds["--parallel"] = (int(m.group(1)), int(m.group(2)))

    m = re.search(r"ple_inflight\s*<\s*(\d+).*?ple_inflight\s*>\s*(\d+)", source, re.DOTALL)
    if m:
        bounds["--ple-inflight"] = (int(m.group(1)), int(m.group(2)))

    for request_key, flag in (("pcie_frac", "--pcie-frac"), ("spec_min_p", "--spec-min-p")):
        m = re.search(
            r'key == "' + re.escape(request_key) + r'".*?std::clamp\(\(double\) fv, ([0-9.]+), ([0-9.]+)',
            source,
            re.DOTALL,
        )
        if m:
            bounds[flag] = (float(m.group(1)), float(m.group(2)))

    for flag in ("--suffix-draft", "--mtp-max-t", "--prompt-cache"):
        m = re.search(r'std::max\(0, std::atoi\(next\("' + re.escape(flag) + r'"\)\)', source)
        if m:
            bounds[flag] = (0.0, float("inf"))

    if re.search(r"vram_reserve_mib\s*<\s*0", source):
        bounds["--vram-reserve-mib"] = (0.0, float("inf"))

    return bounds


def assert_params_match_engine_bounds(params: Sequence[Param], bounds: dict[str, tuple[float, float]]) -> None:
    """Fail loudly when a searched grid goes outside the engine's own accepted range.

    Exact equality is only required where the engine states an exact closed interval
    (`--parallel takes 1..8`, `ple_inflight > 1024`, the request-key clamps). A grid may
    start above a permissive lower bound (for example 0.001 when the engine accepts 0.0) or
    stop below an unbounded one; that is a search choice, not a contradiction.
    """
    for p in params:
        bound = bounds.get(p.flag)
        if bound is None or p.kind == "choice":
            continue
        low, high = bound
        if float(p.low) < low:
            raise ValueError(f"{p.flag} grid starts at {p.low}, below the engine's {low}")
        if high != float("inf") and float(p.high) > high:
            raise ValueError(f"{p.flag} grid ends at {p.high}, above the engine's {high}")


def request_keys(cfg: Config) -> dict[str, object]:
    """The GEN-line keys for a configuration, in the spelling the engine parses."""
    out: dict[str, object] = {}
    for key, engine_key in REQUEST_KEYS.items():
        value = cfg.get(key)
        if value is None or value == "":
            continue
        out[engine_key] = value
    return out


def topology_signature(cfg: Config, params: Sequence[Param]) -> tuple[str, ...]:
    """Identify configurations that can share one engine session.

    Only the boot-required dimensions belong in the signature; the request keys are allowed
    to vary inside a session.
    """
    cfg = dict(DEFAULT_CONFIG) | cfg
    keys = sorted(p.key for p in params if p.key not in REQUEST_KEYS)
    return tuple(str(cfg.get(k, "")) for k in keys)


def quasi_random_configs(params: Sequence[Param], count: int, seed: int) -> list[Config]:
    """A Latin-hypercube seed for a cold study.

    TPE needs measured points before it can model the landscape. With no history, the first
    few trials should cover the space rather than cluster in one corner. This builds a
    deterministic set of points on the actual grid, one value per dimension per trial.
    """
    if count <= 0 or not params:
        return []
    grids = [p.grid() for p in params]
    total = 1
    for grid in grids:
        total *= len(grid)
    if total <= count:
        return [
            {p.key: value for p, value in zip(params, values)}
            for values in itertools.product(*grids)
        ][:count]

    rng = random.Random(seed)
    columns: list[list[object]] = []
    for grid in grids:
        n = len(grid)
        if n >= count:
            if count == 1:
                ranks = [0]
            else:
                ranks = [round(i * (n - 1) / (count - 1)) for i in range(count)]
            rng.shuffle(ranks)
            columns.append([grid[r] for r in ranks])
        else:
            values = list(grid)
            while len(values) < count:
                values.append(rng.choice(grid))
            rng.shuffle(values)
            columns.append(values)

    configs: list[Config] = []
    seen: set[tuple[object, ...]] = set()
    for i in range(count):
        cfg: Config = {p.key: columns[j][i] for j, p in enumerate(params)}
        sig = signature(cfg)
        if sig in seen:
            continue
        seen.add(sig)
        configs.append(cfg)
    return configs


def _request_param(key: str, low: float, high: float, step: float = 1.0) -> Param:
    """A per-request dimension: it is sent as a GEN key, not as a CLI flag."""
    return Param(key, "serve", "int", None, "request", low=low, high=high, step=step)


def build_space(
    vram_low: int = 256,
    vram_high: int = 1536,
    vram_step: int = 256,
    pcie_low: float = 0.0,
    pcie_high: float = 1.0,
    pcie_step: float = 0.05,
    max_workers: int | None = None,
    kv_modes: Sequence[str] | None = None,
    ple_modes: Sequence[str] = PLE_MODES,
    pack: str | Path | None = None,
    has_drafter: bool = False,
    request_dims: Sequence[str] = (),
) -> list[Param]:
    """The dimensions of the sweep, with the ranges the Maya engine accepts.

    --pool-workers 0 is not "no workers": it means "every physical core except the host"
    (src/program/generate.cpp:183), so 0 is a legitimate arm and the grid is capped at the
    physical core count rather than the logical one.

    `pack` is the pack directory the sweep will run: it decides the KV arms (and whether
    --parallel means anything). `has_drafter` is True when a --dflash GGUF was found, in
    which case the MTP/suffix drafting dimensions are not on the code path the request
    takes and are left out rather than swept as a no-op.
    """
    cores = max_workers if max_workers is not None else physical_cores()
    kind = pack_kind(pack)
    kv_choices = tuple(kv_modes) if kv_modes else kv_modes_for(pack)
    params: list[Param] = [
        Param("vram", "topology", "int", "--vram-reserve-mib", "value",
              low=vram_low, high=vram_high, step=vram_step),
        Param("pcie", "topology", "float", "--pcie-frac", "value",
              low=pcie_low, high=pcie_high, step=pcie_step),
        Param("pcie_mode", "topology", "choice", "--pcie-mode", "value", choices=tuple(PCIE_MODES)),
        Param("ple_io", "ple-io", "choice", "--ple-io", "value", choices=tuple(ple_modes)),
        Param("ple_inflight", "ple-read", "int", "--ple-inflight", "value",
              low=16, high=128, step=16),
        Param("kv", "kv", "choice", "--kv", "value", choices=kv_choices),
        Param("kvstream", "kv-stream", "choice", "--kv-streaming", "value", choices=("on", "off")),
        Param("hostworker", "cpu-pool", "choice", "--no-host-worker", "on",
             choices=("1", "0"), invert=True),
        Param("worker", "worker", "int", "--pool-workers", "value",
              low=0, high=max(0, cores - 1), step=1),
        Param("tail", "tail", "int", "--prompt-cache", "value", low=0, high=6, step=2),
        Param("draft_floor", "draft-floor", "float", "--spec-min-p", "value",
              low=0.001, high=1.0, step=0.001),
    ]
    if not has_drafter:
        params += [
            # step 1: generate.cpp:3328-3329 take any int >= 0, and the engine's own default
            # suffix_draft is 3. A coarser step cannot hold that point, so every prior written
            # before this dimension existed fails to seed the study (optuna rejects a value
            # outside the distribution) and the TPE starts cold.
            Param("suffix_draft", "draft", "int", "--suffix-draft", "value",
                  low=SUFFIX_DRAFT_RANGE[0], high=SUFFIX_DRAFT_RANGE[1], step=1),
            Param("mtp_max_t", "draft", "int", "--mtp-max-t", "value",
                  low=MTP_MAX_T_RANGE[0], high=MTP_MAX_T_RANGE[1], step=1),
        ]
    if kind == "glm":
        # STRATA_GLM_SEQS is the GLM runner's sequence-state count; on the qwen session
        # model.seq_count() is 1 and --parallel changes nothing.
        params.append(Param("parallel", "serve", "int", "--parallel", "value",
                           low=PARALLEL_RANGE[0], high=PARALLEL_RANGE[1], step=1))
    for key in request_dims:
        if key == "cpu_threads":
            params.append(_request_param("cpu_threads", CPU_THREADS_RANGE[0],
                                        max(CPU_THREADS_RANGE[1], cores), 1))
        elif key == "read_chunks":
            params.append(_request_param("read_chunks", READ_CHUNKS_RANGE[0],
                                        READ_CHUNKS_RANGE[1], 1))
        elif key == "spec":
            params.append(_request_param("spec", SPEC_RANGE[0], SPEC_RANGE[1], 1))
        elif key == "spec_n_min":
            params.append(_request_param("spec_n_min", SPEC_N_MIN_RANGE[0],
                                        SPEC_N_MIN_RANGE[1], 1))
    return params


DEFAULT_CONFIG: Config = {
    "vram": 512,
    "pcie": 0.95,
    "pcie_mode": "auto",
    "ple_io": "direct",
    # fp16 is the engine's own default and is valid on both runners; q4_0 is silently
    # ignored on a --glm-pack (see kv_modes_for above), so it is not a safe seed.
    "kv": "fp16",
    "kvstream": "on",
    "hostworker": "1",
    "worker": 0,
    "tail": 6,
    "draft_floor": 0.8,
    # These are the values src/program/generate.cpp uses when the flag is absent, which is
    # why they are here: a record written before a dimension existed then measured exactly
    # the run its defaults give, and history must still recognise it.
    "ple_inflight": 64,      # generate.cpp:148
    "suffix_draft": 3,      # generate.cpp:318
    "mtp_max_t": 0,         # generate.cpp:320
    "parallel": 1,          # generate.cpp:3191, usage at 356
}


def signature(cfg: Config) -> tuple[object, ...]:
    """A hashable, order-stable identity for a configuration, used to skip repeats."""
    return tuple(str(cfg[k]) for k in sorted(cfg))


def sample(trial, params: Sequence[Param], frozen: dict[str, object]) -> Config:
    """Ask Optuna for one point in the space, leaving `frozen` dimensions untouched."""
    cfg: Config = dict(DEFAULT_CONFIG)
    cfg.update(frozen)
    for p in params:
        if p.key in frozen:
            continue
        if p.kind == "choice":
            cfg[p.key] = trial.suggest_categorical(p.key, list(p.choices))
        elif p.kind == "int":
            cfg[p.key] = trial.suggest_int(p.key, int(p.low), int(p.high), step=int(p.step))
        else:
            cfg[p.key] = round(trial.suggest_float(p.key, float(p.low), float(p.high)), 4)
    return cfg


def engine_args(
    cfg: Config,
    paths: dict[str, str],
    context: int,
    max_new: int,
    spec_min_p: float | None = 0.8,
    request_keys: bool = False,
) -> list[str]:
    """The engine command line for one configuration.

    The base block is the one Maya's installer writes (glm-pack, max-context, expert-cache
    auto, prefill auto) plus the flags the sweep needs to measure exactly one request in
    one process; only the swept dimensions come from cfg. The space is rebuilt from the
    pack and drafter in `paths`, so a configuration never emits a flag its runner does not
    implement - a dimension frozen out of the study still writes its pinned value here.
    """
    cfg = dict(cfg)
    if cfg.get("draft_floor") is None:
        cfg["draft_floor"] = 0.8 if spec_min_p is None else spec_min_p
    # generate.cpp:3624 refuses a request that does not fit the context, and 3620 the
    # context itself if it is 2^31 or more. Catching it here is one ValueError per trial
    # rather than one dead engine boot per trial.
    if context <= 0 or context > 2**31 - 8:
        raise ValueError(f"--cntx {context} is outside the engine's 1 .. 2^31-8")
    if max_new <= 0 or max_new > context:
        raise ValueError(
            f"--max-new {max_new} must be between 1 and --cntx {context} "
            "(src/program/generate.cpp:3624)")
    # --kv k8v4 has no streaming window: generate.cpp:3569 exits 2 when a kv-resident
    # count is set with it, and --kv-streaming on sets one (3405). The arm is not on the
    # table, so write the combination the engine accepts instead of a guaranteed exit 2.
    if cfg.get("kv") == "k8v4" and cfg.get("kvstream") == "on":
        cfg["kvstream"] = "off"
    args: list[str] = ["--serve"]
    if paths.get("pack"):
        args += ["--glm-pack", paths["pack"]]
    if paths.get("dflash"):
        args += ["--dflash", paths["dflash"]]
    if paths.get("expert_profile") and Path(paths["expert_profile"]).exists():
        args += ["--expert-profile", paths["expert_profile"]]
    args += ["--expert-cache", "auto", "--prefill", "auto",
            "--max-context", str(context), "--max-new", str(max_new),
            "--expert-profile-save-every", "0"]

    params = build_space(pack=paths.get("pack"), has_drafter=bool(paths.get("dflash")))
    for p in params:
        value = cfg.get(p.key)
        if value is None:
            continue
        if p.mode == "request" or (request_keys and p.key in REQUEST_KEYS):
            continue
        if p.mode == "on":
            if p.applies(value):
                args.append(p.flag)
        else:
            args += [p.flag, str(value)]
    return args


@dataclass
class ModelPaths:
    """Where the model lives. Every field is a path string; empty means "not found"."""

    pack: str = ""
    tokenizer: str = ""
    expert_profile: str = ""
    dflash: str = ""
    prompt_tokens: str = ""
    searched: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, str]:
        return {
            "pack": self.pack, "tokenizer": self.tokenizer,
            "expert_profile": self.expert_profile, "dflash": self.dflash,
            "prompt_tokens": self.prompt_tokens,
        }


# A pack is the directory the engine's --glm-pack points at.
PACK_MARKERS: tuple[str, ...] = ("index.txt", "dense.bin", "native_experts.txt")

# native_experts.txt names the GGUF it was built from in its header comment; the engine
# opens those shards from beside the pack directory (glm_model.cu: pack_dir + "/..").
SHARD_HEADER = "absolute offsets in "


def pack_shard_names(pack_dir: Path) -> list[str]:
    """The GGUF shard file names a native pack reads, as the engine derives them.

    The header names shard 1 only; the siblings follow its -NNNNN-of-MMMMM pattern. A pack
    with no native_experts.txt (an EXL3 pack, a test fixture) names none, and the engine
    never looks for any.
    """
    try:
        lines = (pack_dir / "native_experts.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    names: list[str] = []
    for line in lines.splitlines():
        if line.startswith("#") and SHARD_HEADER in line:
            start = line.find(SHARD_HEADER) + len(SHARD_HEADER)
            end = line.find(",", start)
            names.append(line[start:end if end != -1 else len(line)].strip())
    if not names:
        return []
    first = names[0]
    of = first.find("-of-")
    if of == -1:
        return [first]
    dash = first.rfind("-", 0, of)                       # the dash starting this shard's number
    digits = "".join(ch for ch in first[of + 4:] if ch.isdigit())   # atoi stops at the extension
    total = int(digits) if digits else 0
    stem = first[:dash] if dash != -1 else first
    if total < 1:
        return [first]
    return [f"{stem}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]


def pack_missing_shards(pack_dir: Path) -> list[str]:
    """Shards a pack names that are not beside it. Empty means the engine can open it.

    A pack whose source GGUFs live in another directory cannot be booted: load_pack opens
    pack_dir + "/.." + shard and throws before the session starts. Catching it here costs
    one stat per pack instead of one failed engine boot per trial.
    """
    beside = pack_dir.parent
    return [name for name in pack_shard_names(pack_dir) if not (beside / name).is_file()]


def _model_dirs(base: Path, quant: str) -> list[Path]:
    """Directories a model's pack may live in, matched case-insensitively.

    Maya's installer writes `Maya-data/models/Maya-S24/pack`; a sweep may name the model
    `maya-s24`. Names are compared with dashes and underscores folded away, so both spell
    match. The order is the one the installer uses: models/, packs/, then a bare pack.
    """
    folded = quant.replace("-", "").replace("_", "")
    out: list[Path] = []
    for parent in (base / "models", base / "packs", base / "data" / "models",
                   base / "data" / "packs", base):
        if parent.is_dir():
            for d in sorted(parent.iterdir()):
                if not d.is_dir():
                    continue
                name = d.name.lower().replace("-", "").replace("_", "")
                if name == folded or name.startswith(folded):
                    out.append(d)
    return out


def _pack_in(directory: Path, rejected: list[tuple[str, str]] | None = None) -> Path | None:
    """A directory holding a Maya pack the engine can open, in the places a pack may sit.

    A pack whose source shards are not beside it is recorded in `rejected` and skipped, so
    a sweep never spends trials booting an engine that cannot read its pack.
    """
    for cand in (directory / "pack", directory / "packs", directory):
        if not cand.is_dir() or not any((cand / marker).exists() for marker in PACK_MARKERS):
            continue
        missing = pack_missing_shards(cand)
        if missing:
            if rejected is not None:
                rejected.append((str(cand.resolve()), ", ".join(missing)))
            continue
        return cand
    return None


def resolve_model_paths(root: Path, model: str, data_dir: str | None = None) -> ModelPaths:
    """Find a model's pack, its tokenizer, the shipped expert profile and a drafter.

    Searches the repo first, then the data folders Maya's installer uses (data_dir wins
    over them when given). A single base is preferred when it carries both the pack and a
    tokenizer, so a run never mixes a pack from one tree with a tokenizer from another.
    A pack the engine cannot open - one whose source GGUF shards are not beside it - is
    skipped and named in the error rather than swept against.
    Raises FileNotFoundError with the list of places it looked at, so a miss is a path
    problem, not a crash.
    """
    quant = model.lower()
    root = abs_path(root) or Path(root).resolve()
    extra = abs_path(data_dir)
    candidates = [str(root)] + ([str(extra)] if extra is not None else []) + list(data_roots(root))

    found: list[ModelPaths] = []
    rejected: list[tuple[str, str]] = []
    for base in candidates:
        b = Path(base)
        if not b.is_dir():
            continue
        paths = ModelPaths()
        for sub in _model_dirs(b, quant):
            pack = _pack_in(sub, rejected)
            if pack is not None:
                paths.pack = str(pack.resolve())
                break
        if paths.pack:
            tok = Path(paths.pack) / "tokenizer"
            if (tok / "vocab.json").is_file() or (tok / "tokenizer.json").is_file():
                paths.tokenizer = str(tok.resolve())
        profile = b / "data" / "expert-profile-glm.bin"
        if profile.is_file():
            paths.expert_profile = str(profile.resolve())
        elif (b / "data").is_dir():
            hit = next(iter((b / "data").glob("expert-profile*.bin")), None)
            if hit is not None:
                paths.expert_profile = str(hit.resolve())
        drafter = None
        for model_dir in _model_dirs(b, quant):
            hit = next(iter(model_dir.glob("dflash*")), None)
            if hit is not None:
                drafter = hit
                break
        if drafter is None and paths.pack:
            drafter = next(iter(Path(paths.pack).glob("dflash*")), None)
        if drafter is not None:
            paths.dflash = str(drafter.resolve())
        prompt = Path(__file__).resolve().parent / "fixtures" / "prompt.json"
        # A pack with its own tokenizer needs no prompt file: the sweep tokenises the
        # benchmark text with that tokenizer, so every id is one the engine will accept.
        # The bundled fixture is only a fallback for a pack that has no tokenizer, and a
        # base with no pack at all contributes nothing - otherwise a repository directory
        # searched for its own pack would seed the fixture into a run that has a tokenizer.
        if prompt.is_file() and paths.pack and not paths.tokenizer:
            paths.prompt_tokens = str(prompt.resolve())
        found.append(paths)

    best = max(found, key=lambda p: (bool(p.pack), bool(p.tokenizer))) if found else ModelPaths()
    paths = ModelPaths()
    for key in ("pack", "tokenizer", "expert_profile", "dflash", "prompt_tokens"):
        for p in [best] + found:
            if getattr(p, key):
                setattr(paths, key, getattr(p, key))
                break
    paths.searched = candidates

    if not paths.pack:
        detail = ""
        for pack_dir, missing in rejected:
            detail += f"\n  {pack_dir} exists but the engine cannot open it: its source "
            detail += f"GGUF shards are not beside it (missing {missing})."
        raise FileNotFoundError(
            f"HyperCore-Tune-Maya: could not find a model pack for '{model}'. "
            "Looked in: " + ", ".join(candidates) + ". Pass --pack explicitly, or set MAYA_DATA."
            + detail
        )
    return paths
