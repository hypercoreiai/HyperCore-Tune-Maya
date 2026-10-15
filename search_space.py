"""HyperCore-Tune-Maya/search_space.py - the parameter space the Bayesian sweep searches.

Every trial writes one row with the columns the Project Maya engine actually has:

    layer  vram  pcie  pcie_mode  ple_io  kv  kvstream  hostworker  worker  tail  draft_floor  status

`layer` is the group a trial is perturbing. In staged mode it is the layer the study is
currently frozen on; in joint mode it is the group whose value differs from the incumbent
best ("joint" for the first trial, which has no incumbent). `status` is not a parameter:
it is what the engine did with the configuration (PASS, FAIL_BOOT, NO_TOKENS, TIMEOUT).

Each dimension maps to a flag `src/program/generate.cpp` really parses:

    vram        --vram-reserve-mib        MiB of VRAM the expert pool leaves free
    pcie        --pcie-frac               share of the PCIe link the RAM tier may use
    pcie_mode   --pcie-mode               how those transfers are issued (auto|dma|kernel|direct)
    ple_io      --ple-io                  n-gram table reads: direct (SSD) or mmap
    kv          --kv                      q4_0 | k8v4 | int8 | fp16
    kvstream    --kv-streaming            on keeps the KV window resident, off keeps all of it
    hostworker  --no-host-worker          1 = the host thread joins the CPU pool (the default),
                                        0 = keep it out of the drain
    worker      --pool-workers            CPU expert pool size; 0 = every core but the host's
    tail        --prompt-cache            conversation checkpoints kept between requests
    draft_floor --spec-min-p              the acceptance floor for drafted tokens

A freeze overlay (sample(..., frozen)) holds pinned dimensions at a constant.

Determinism: every parameter is sampled from a closed grid, so a trial's configuration is
reproducible from (seed, trial index) and the signature of a configuration is a plain
tuple. Nothing here reads the environment except the CPU count, which bounds the worker
grid and is overridable with --max-workers.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from maya_env import abs_path, data_roots

try:  # psutil is in .venv (requirements.txt); the fallback keeps this importable anywhere.
    import psutil
except ImportError:  # pragma: no cover - only hit on a bare interpreter
    psutil = None  # type: ignore[assignment]

LAYERS: tuple[str, ...] = (
    "topology", "ple-io", "kv", "kv-stream", "cpu-pool", "worker", "tail", "draft-floor",
)

# Column order of the CSV: the engine dimensions, then what the engine did with them.
ROW_COLUMNS: tuple[str, ...] = (
    "layer", "vram", "pcie", "pcie_mode", "ple_io", "kv", "kvstream",
    "hostworker", "worker", "tail", "draft_floor", "status",
)

Config = dict[str, object]

# The modes the GLM engine accepts (src/program/generate.cpp rejects anything else).
KV_MODES: tuple[str, ...] = ("q4_0", "k8v4", "int8", "fp16")
PLE_MODES: tuple[str, ...] = ("direct", "mmap")
PCIE_MODES: tuple[str, ...] = ("auto", "dma", "kernel", "direct")


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


def build_space(
    vram_low: int = 256,
    vram_high: int = 1536,
    vram_step: int = 256,
    pcie_low: float = 0.0,
    pcie_high: float = 1.0,
    pcie_step: float = 0.05,
    max_workers: int | None = None,
    kv_modes: Sequence[str] = KV_MODES,
    ple_modes: Sequence[str] = PLE_MODES,
) -> list[Param]:
    """The dimensions of the sweep, with the ranges the Maya engine accepts.

    --pool-workers 0 is not "no workers": it means "every physical core except the host"
    (src/program/generate.cpp:183), so 0 is a legitimate arm and the grid is capped at the
    physical core count rather than the logical one.
    """
    cores = max_workers if max_workers is not None else physical_cores()
    return [
        Param("vram", "topology", "int", "--vram-reserve-mib", "value",
              low=vram_low, high=vram_high, step=vram_step),
        Param("pcie", "topology", "float", "--pcie-frac", "value",
              low=pcie_low, high=pcie_high, step=pcie_step),
        Param("pcie_mode", "topology", "choice", "--pcie-mode", "value", choices=tuple(PCIE_MODES)),
        Param("ple_io", "ple-io", "choice", "--ple-io", "value", choices=tuple(ple_modes)),
        Param("kv", "kv", "choice", "--kv", "value", choices=tuple(kv_modes)),
        Param("kvstream", "kv-stream", "choice", "--kv-streaming", "value", choices=("on", "off")),
        Param("hostworker", "cpu-pool", "choice", "--no-host-worker", "on",
             choices=("1", "0"), invert=True),
        Param("worker", "worker", "int", "--pool-workers", "value",
              low=0, high=max(0, cores - 1), step=1),
        Param("tail", "tail", "int", "--prompt-cache", "value", low=0, high=6, step=2),
        Param("draft_floor", "draft-floor", "float", "--spec-min-p", "value",
              low=0.001, high=1.0, step=0.001),
    ]


DEFAULT_CONFIG: Config = {
    "vram": 512,
    "pcie": 0.95,
    "pcie_mode": "auto",
    "ple_io": "direct",
    "kv": "q4_0",
    "kvstream": "on",
    "hostworker": "1",
    "worker": 0,
    "tail": 6,
    "draft_floor": 0.8,
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
) -> list[str]:
    """The engine command line for one configuration.

    The base block is the one Maya's installer writes (glm-pack, max-context, expert-cache
    auto, prefill auto) plus the flags the sweep needs to measure exactly one request in
    one process; only the swept dimensions come from cfg.
    """
    cfg = dict(cfg)
    if cfg.get("draft_floor") is None:
        cfg["draft_floor"] = 0.8 if spec_min_p is None else spec_min_p
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

    for p in build_space():
        value = cfg.get(p.key)
        if value is None:
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


def _pack_in(directory: Path) -> Path | None:
    """A directory holding a Maya pack, looking at the places a pack can legitimately sit."""
    for cand in (directory / "pack", directory / "packs", directory):
        if cand.is_dir() and any((cand / marker).exists() for marker in PACK_MARKERS):
            return cand
    return None


def resolve_model_paths(root: Path, model: str, data_dir: str | None = None) -> ModelPaths:
    """Find a model's pack, its tokenizer, the shipped expert profile and a drafter.

    Searches the repo first, then the data folders Maya's installer uses (data_dir wins
    over them when given). A single base is preferred when it carries both the pack and a
    tokenizer, so a run never mixes a pack from one tree with a tokenizer from another.
    Raises FileNotFoundError with the list of places it looked at, so a miss is a path
    problem, not a crash.
    """
    quant = model.lower()
    root = abs_path(root) or Path(root).resolve()
    extra = abs_path(data_dir)
    candidates = [str(root)] + ([str(extra)] if extra is not None else []) + list(data_roots(root))

    found: list[ModelPaths] = []
    for base in candidates:
        b = Path(base)
        if not b.is_dir():
            continue
        paths = ModelPaths()
        for sub in _model_dirs(b, quant):
            pack = _pack_in(sub)
            if pack is not None:
                paths.pack = str(pack.resolve())
                break
        if paths.pack:
            tok = Path(paths.pack) / "tokenizer"
            if (tok / "vocab.json").is_file():
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
        if prompt.is_file():
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
        raise FileNotFoundError(
            f"HyperCore-Tune-Maya: could not find a model pack for '{model}'. "
            "Looked in: " + ", ".join(candidates) + ". Pass --pack explicitly, or set MAYA_DATA."
        )
    return paths
