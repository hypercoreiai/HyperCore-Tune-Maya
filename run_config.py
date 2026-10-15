"""HyperCore-Tune-Maya/run_config.py - the sweep's winner as a run config Maya can read.

Deliverables, in the project's own format (the same shape maya.py writes for a model):

    maya-<model>.json   a run config: the keys maya.py owns (exe, args, cwd, tokenizer,
                           model_name, gpu, log, lib_dirs, port, host, env) with the
                           winning engine arguments in "args"
    run-maya-<model>.sh     the launcher: maya.py with that config
    run-maya-<model>.bat    the same launcher for a Windows install of Maya

Only "args" is replaced. Everything the model's existing config carries that the installer
does not own (sampling, reasoning_effort, vision, mcp_servers, ...) is kept, so the
deliverable is a drop-in startup pair rather than a tuning dump. The full trial record -
every configuration measured and its metrics - stays in
HyperCore-Tune-Maya/results/maya-<model>.record.json.

maya.py is imported when it is available so the atomic write and the defaults are the ones
the real installer uses; a stand-in is used when it cannot be imported. The project root
comes from maya_env, so every path written here is absolute and belongs to the Project
Maya checkout on the machine running the sweep.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from maya_env import (
    REPO,
    abs_path,
    engine_binary,
    engine_flags,
    entry_flags,
    entry_script,
    maya_root,
    server_engine_name,
    venv_python,
)

# Flags that exist only because the sweep measures one request in one process. They are not
# part of a production run config: the server owns the process, the request length is
# whatever the client asks for, and the profile is written by the server, not by a sweep.
# --spec-min-p, --kv-streaming and --prompt-cache stay: Maya's engine accepts those.
SWEEP_ONLY: tuple[tuple[str, int], ...] = (
    ("--serve", 0),                            # serve/server.py adds it itself
    ("--max-new", 1),                        # the request length is the client's, not the config's
    ("--expert-profile-save-every", 1),
)

# Keys a file must have to be treated as a run config rather than a tuning record.
RUN_CONFIG_MARKERS = ("exe", "args", "cwd")


def _setup_module() -> Any | None:
    """The project's maya.py (or setup.py), or None if it cannot be imported."""
    root = str(maya_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    for name in ("maya", "setup"):
        try:
            return __import__(name)  # the installer module, resolved at call time
        except Exception as exc:  # noqa: BLE001 - a partial install must not lose a finished sweep
            print(f"note: {name}.py not importable ({exc}); using built-in defaults", file=sys.stderr)
    return None


def _cuda_lib_dirs() -> list[str]:
    """The CUDA bin/lib directories actually present, the way the run configs list them."""
    out: list[str] = []
    seen: set[str] = set()

    def add(bin_dir: Path, lib_dir: Path) -> None:
        if not lib_dir.is_dir():
            return
        for p in (bin_dir, lib_dir):
            s = str(p)
            if s not in seen:
                seen.add(s)
                out.append(s)

    for cand in sorted(Path("/usr/local").glob("cuda*")):
        add(cand / "bin", cand / "lib64")
    cuda_root = abs_path(os.environ.get("CUDA_PATH"))
    if cuda_root is not None:
        add(cuda_root / "bin", cuda_root / "lib" / "x64")
        add(cuda_root / "bin", cuda_root / "lib64")
    toolkit = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if toolkit.is_dir():
        for cand in sorted(toolkit.glob("v*")):
            add(cand / "bin", cand / "lib" / "x64")
    return out


def clean_args(args: list[str]) -> list[str]:
    """The engine arguments without the sweep-only flags, order preserved.

    A flag this Maya build does not parse is dropped too: the server passes "args"
    straight to the binary, so an unknown flag would end the run at startup. The engine's
    own source (src/program/generate.cpp) is the authority; when it cannot be read the
    list is passed through untouched.
    """
    known = engine_flags(maya_root())
    out: list[str] = []
    i = 0
    while i < len(args):
        flag = args[i]
        skip = next((n for name, n in SWEEP_ONLY if name == flag), None)
        if skip is not None:
            i += 1 + skip
            continue
        if known and flag.startswith("--") and flag not in known:
            print(f"note: dropping {flag}: this Maya build does not parse that flag",
                  file=sys.stderr)
            has_value = i + 1 < len(args) and not args[i + 1].startswith("--")
            i += 2 if has_value else 1
            continue
        out.append(flag)
        i += 1
    return out


def find_base_config(model: str, root: Path) -> dict[str, Any] | None:
    """The model's current run config, if there is one.

    Preferred order: the installed Maya config (maya-<model>.json), then a deliverable
    from an earlier sweep. A file is only accepted when it has the run-config shape, so a
    tuning record is never mistaken for a config.
    """
    quant = model.lower()
    for name in (f"maya-{quant}.json", f"maya-maya-{quant}.json",
                 f"maya-{quant}.json", f"strata-{quant}.json"):
        path = root / name
        if not path.is_file():
            continue
        try:
            cfg = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if isinstance(cfg, dict) and all(k in cfg for k in RUN_CONFIG_MARKERS):
            return cfg
    return None


def build_run_config(
    model: str,
    paths: dict[str, str],
    args: list[str],
    record: dict[str, Any],
    base: dict[str, Any] | None = None,
    port: int | None = None,
) -> dict[str, Any]:
    """The run config for the winning configuration.

    Starts from the model's own config and replaces only "args", so every key the
    installer or the user set (sampling, reasoning, vision, lib_dirs, gpu, host, api_key)
    survives. Keys the local entry point cannot read are dropped rather than left in.
    """
    root = maya_root()
    cfg: dict[str, Any] = dict(base or {})
    # The engine this sweep measured is the one the deliverable runs; a base config's
    # exe is only used when the record has none.
    engine = record.get("engine") or cfg.get("exe") or engine_binary(root)
    cfg["exe"] = str(abs_path(engine) or Path(engine).expanduser().resolve())
    cfg["cwd"] = cfg.get("cwd") or str(root)
    cfg["args"] = clean_args(args)

    tokenizer = cfg.get("tokenizer") or paths.get("tokenizer") or ""
    beside_pack = Path(paths["pack"]) / "tokenizer" if paths.get("pack") else None
    if beside_pack is not None and beside_pack.is_dir():
        if tokenizer and Path(tokenizer).expanduser().resolve() != beside_pack.resolve():
            print(f"note: {tokenizer} is not beside the pack in use ({paths['pack']}); "
                  f"using {beside_pack}", file=sys.stderr)
        tokenizer = str(beside_pack.resolve())
    elif tokenizer:
        found = abs_path(tokenizer)
        tokenizer = str(found) if found is not None else tokenizer
    if tokenizer:
        cfg["tokenizer"] = tokenizer

    if not cfg.get("model_name"):
        cfg["model_name"] = "glm-5.3-flash"

    cfg["log"] = str(root / f"maya-{model.lower()}.log")

    if not cfg.get("lib_dirs"):
        libs = _cuda_lib_dirs()
        if libs:
            cfg["lib_dirs"] = libs

    if port is not None:
        cfg["port"] = int(port)

    def default(name: str, fallback: str) -> str:
        for env in (f"MAYA_DEFAULT_{name}", f"STRATA_DEFAULT_{name}", f"HYPERCORE_DEFAULT_{name}"):
            if os.environ.get(env):
                return os.environ[env]
        return fallback

    setup = _setup_module()
    if hasattr(setup, "apply_server_defaults"):
        # The installer's own precedence: what this run asked for, then the model's config,
        # then the default. A key someone set by hand is never replaced.
        setup.apply_server_defaults(cfg, base)
    else:
        fallback: dict[str, Any] = {
            "host": default("HOST", "127.0.0.1"),
            "api_key": default("API_KEY", ""),
            "port": int(default("PORT", "8080")),
        }
        for key, value in fallback.items():
            if cfg.get(key) is None:
                cfg[key] = value
    cfg["port"] = int(cfg.get("port") or 8080)
    cfg.setdefault("gpu", 0)

    cfg["tuned_by"] = {
        "schema": "hypercore.maya_max_run/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "objective": record.get("objective"),
        "context": record.get("context"),
        "hardware": record.get("hardware", {}),
        "hardware_slug": record.get("hardware_slug", ""),
        "best": record.get("best", {}),
        "verified": bool(record.get("verified", False)),
        # history | dry_run | study: who produced the winning numbers.
        "source": record.get("source", "study"),
        "metrics": record.get("metrics", {}),
        "study": record.get("study", {}),
        "history": record.get("history", {}),
        "csv": record.get("csv"),
    }
    return cfg


def _write_json(path: Path, cfg: dict[str, Any]) -> Path:
    """Whole or not at all, the way the installer writes a run config."""
    setup = _setup_module()
    write = getattr(setup, "write_config", None)
    if callable(write):
        write(path, cfg)
        return path
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


def serve_command(
    cfg_path: Path,
    engine: str,
    port: int,
    open_browser: bool,
    root: Path | None = None,
) -> list[str]:
    """The server invocation, built from absolute paths of the project in front of us.

    The interpreter is the project venv when it exists, so the launcher runs with the
    packages the installer installed rather than whatever happened to start the sweep.
    Maya's entry point is maya.py; a checkout without one is started through its server.
    A flag the entry point does not define is not passed: it would be rejected at startup.
    """
    root = root or maya_root()
    python = venv_python(root) or sys.executable
    entry = entry_script(root)
    cmd = [python, entry]
    if entry.endswith("server.py"):
        cmd += ["--engine", engine]
    cmd += ["--config", str(cfg_path), "--port", str(port)]
    if open_browser and "--open" in entry_flags(root):
        cmd.append("--open")
    return cmd


def launcher_text(
    cfg_path: Path,
    engine: str,
    port: int,
    open_browser: bool,
    note: tuple[str, ...],
) -> str:
    """The launcher body: the server with this config, exactly as the installer writes it."""
    root = maya_root()
    serve = serve_command(cfg_path, engine, port, open_browser, root)
    lines = ["#!/bin/sh"]
    lines += [f"# {line}" for line in note]
    lines.append(f'cd "{str(root)}"')
    lines.append("exec " + " ".join(f'"{x}"' for x in serve))
    return "\n".join(lines) + "\n"


def launcher_bat(
    cfg_path: Path,
    engine: str,
    port: int,
    open_browser: bool,
    note: tuple[str, ...],
) -> str:
    """The same launcher for a Windows install of Maya, in the installer's .bat shape."""
    root = maya_root()
    serve = serve_command(cfg_path, engine, port, open_browser, root)
    lines = ["@echo off", f"title Maya {cfg_path.stem}"]
    lines += [f"rem {line}" for line in note]
    lines.append('cd /d "' + str(root) + '"')
    lines.append(" ".join(f'"{x}"' for x in serve))
    lines.append("if errorlevel 1 pause")
    return "\r\n".join(lines) + "\r\n"


def _emit_pair(
    model: str,
    paths: dict[str, str],
    args: list[str],
    record: dict[str, Any],
    deliverable_dir: Path,
    port: int | None,
    open_browser: bool,
) -> tuple[Path, Path]:
    """Write the pair into deliverable_dir; raises OSError if that directory cannot be written."""
    quant = model.lower()
    root = maya_root()
    base = find_base_config(quant, root)
    cfg = build_run_config(quant, paths, args, record, base=base, port=port)
    js = _write_json(deliverable_dir / f"maya-{quant}.json", cfg)

    note = (
        (f"run-maya-{quant}.sh - generated {cfg['tuned_by']['generated_utc']} "
         f"by HyperCore-Tune-Maya/max_tune.sh"),
        (f"Bayesian (Optuna TPE) winner for the {cfg['tuned_by']['objective']} objective "
         f"at --max-context {cfg['tuned_by']['context']}"),
        (f"gen {cfg['tuned_by']['metrics'].get('gen_tok_s', 0):.2f} tok/s, "
         f"prefill {cfg['tuned_by']['metrics'].get('prefill_tok_s', 0):.2f} tok/s, "
         f"latency {cfg['tuned_by']['metrics'].get('latency_ms', 0):.1f} ms"),
        f"tuned on {cfg['tuned_by'].get('hardware_slug', 'unknown machine')}",
        (f"the trial record is "
         f"{REPO / 'results' / f'maya-{quant}.record.json'}"),
    )
    engine_name = server_engine_name(root)
    sh = deliverable_dir / f"run-maya-{quant}.sh"
    # "--engine" is only passed to a checkout whose entry point has that selector; the
    # binary the worker runs is the "exe" key in the config beside the script.
    sh.write_text(launcher_text(js, engine_name, cfg["port"], open_browser, note), encoding="utf-8")
    sh.chmod(0o755)
    bat = deliverable_dir / f"run-maya-{quant}.bat"
    bat.write_text(launcher_bat(js, engine_name, cfg["port"], open_browser, note), encoding="utf-8")
    return js, sh


def write_deliverables(
    model: str,
    paths: dict[str, str],
    args: list[str],
    record: dict[str, Any],
    deliverable_dir: Path | None = None,
    port: int | None = None,
    open_browser: bool = True,
    fallback_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write the winning run config and its two launchers; returns (script, config).

    When the requested deliverable directory cannot be written - a root-level launcher
    owned by another user, a read-only checkout - the pair is written to fallback_dir
    (this repository's results/ folder when none is given), so a finished sweep is never
    lost to a permission problem.
    """
    root = maya_root()
    wanted = abs_path(deliverable_dir) or root
    try:
        js, sh = _emit_pair(model, paths, args, record, wanted, port, open_browser)
    except OSError as exc:
        fallback = abs_path(fallback_dir) or (REPO / "results")
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"note: cannot write {wanted} ({exc}); the deliverables go to {fallback}",
              file=sys.stderr)
        js, sh = _emit_pair(model, paths, args, record, fallback, port, open_browser)
    return sh, js
