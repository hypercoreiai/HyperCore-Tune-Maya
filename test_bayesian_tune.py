"""HyperCore-Tune-Maya/test_bayesian_tune.py - the sweep, end to end, without a GPU.

    pytest HyperCore-Tune-Maya/test_bayesian_tune.py

Every test uses stub_engine.py, which speaks the real protocol (INFO, "session is up",
DONE) and computes its numbers from the flags it was given. Nothing here touches the
engine or the card, and the project root is a stand-in tree so the suite also runs from a
standalone clone of this repository.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import maya_env
from bayesian_tune import main, signed_score
from engine_runner import (
    EngineError,
    build_prompt,
    parse_done,
    parse_info,
    run_config,
)
from run_config import clean_args
from search_space import (
    ROW_COLUMNS,
    build_space,
    data_roots,
    engine_args,
    resolve_model_paths,
)
from maya_env import abs_path, candidate_roots, engine_binary, server_engine_name, maya_root

STUB = HERE / "stub_engine.py"

# A stand-in project tree: the two files a root is recognised by, and an engine binary.
# It keeps the suite runnable wherever this repository happens to be cloned.
FAKE_ROOT = Path(tempfile.mkdtemp(prefix="hypercore-tune-maya-fake-"))
(FAKE_ROOT / "serve").mkdir(parents=True, exist_ok=True)
(FAKE_ROOT / "serve" / "server.py").write_text(
    """# The flags the stand-in server accepts. Deliberately without --turn-tail-check:
# that flag belongs to an older engine and this one does not have it.
ap.add_argument("--engine", choices=["mock", "strata"], default="mock")
ap.add_argument("--config", help="the run config")
ap.add_argument("--host", default=None)
ap.add_argument("--port", type=int, default=8095)
ap.add_argument("--open", action="store_true")
ap.add_argument("--api-key", default="")
""",
    encoding="utf-8",
)
(FAKE_ROOT / "CMakeLists.txt").write_text("# stand-in for the real project\n", encoding="utf-8")
# The flags this stand-in engine parses. --prompt-cache-tail is deliberately absent: it is
# a flag the Strata tuner used and this engine does not have.
(FAKE_ROOT / "src" / "program").mkdir(parents=True, exist_ok=True)
(FAKE_ROOT / "src" / "program" / "generate.cpp").write_text(
    """else if (a == "--serve") o.serve = true;
else if (a == "--glm-pack") o.pack = next("--glm-pack");
else if (a == "--dflash") o.dflash = next("--dflash");
else if (a == "--expert-profile") o.expert_profile = next("--expert-profile");
else if (a == "--max-context") o.max_context = std::atoi(next("--max-context"));
else if (a == "--max-new") o.max_new = std::atoi(next("--max-new"));
else if (a == "--kv") o.kv = next("--kv");
else if (a == "--kv-streaming") o.kv_streaming = next("--kv-streaming");
else if (a == "--pool-workers") o.pool_workers = std::atoi(next("--pool-workers"));
else if (a == "--no-host-worker") o.no_host_worker = true;
else if (a == "--pcie-frac") o.pcie_frac = std::atof(next("--pcie-frac"));
else if (a == "--pcie-mode") o.pcie_mode = next("--pcie-mode");
else if (a == "--ple-io") o.ple_io = next("--ple-io");
else if (a == "--prompt-cache") o.prompt_cache = std::atoi(next("--prompt-cache"));
else if (a == "--spec-min-p") o.spec_min_p = std::atof(next("--spec-min-p"));
else if (a == "--vram-reserve-mib") o.vram_reserve = std::atoi(next("--vram-reserve-mib"));
else if (a == "--expert-cache") o.expert_cache = std::atoi(next("--expert-cache"));
else if (a == "--prefill") o.prefill = next("--prefill");
else if (a == "--expert-profile-save-every") o.profile_every = std::atoi(next("--expert-profile-save-every"));
""",
    encoding="utf-8",
)
(FAKE_ROOT / "maya.py").write_text(
    """# A stand-in for the installer: the sweep uses its write_config, its defaults and
# the flags its argparse actually defines.
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default=None)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--open", action="store_true")


def write_config(path, cfg):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def apply_server_defaults(cfg, base):
    cfg.setdefault("host", "0.0.0.0")
    cfg.setdefault("api_key", "0000")
    cfg.setdefault("port", 8081)
""",
    encoding="utf-8",
)
(FAKE_ROOT / "build").mkdir(parents=True, exist_ok=True)
FAKE_ENGINE = FAKE_ROOT / "build" / "strata"
FAKE_ENGINE.write_text("#!/bin/sh\n", encoding="utf-8")
FAKE_ENGINE.chmod(0o755)
FAKE_PYTHON = FAKE_ROOT / ".venv" / "bin" / "python"
FAKE_PYTHON.parent.mkdir(parents=True, exist_ok=True)
FAKE_PYTHON.symlink_to(sys.executable)
os.environ["MAYA_ROOT"] = str(FAKE_ROOT)

# A DONE line copied from the shape src/program/generate.cpp:1917 prints.
REAL_DONE = (
    "DONE 512 10240 1380.4 3050.2 length 412 512 0 0 120 300 12 41.5 10240 88\n"
)


def test_parse_done_reads_the_real_field_table():
    d = parse_done(REAL_DONE)
    assert d["generated"] == 512
    assert d["prompt_tokens"] == 10240
    assert d["prompt_ms"] == 1380.4
    assert d["decode_ms"] == 3050.2
    assert d["finish"] == "length"
    assert d["prompt_read"] == "10240"


def test_parse_done_rejects_a_truncated_line():
    with pytest.raises(EngineError):
        parse_done("DONE 512 10240 1380.4")


def test_parse_info_extracts_what_the_engine_applied():
    info = parse_info("INFO context=131072 kv=k8v4 pool_workers=16 pcie_frac=0.95")
    assert info == {"context": "131072", "kv": "k8v4", "pool_workers": "16", "pcie_frac": "0.95"}


def test_a_flag_this_build_does_not_have_is_dropped():
    """A deliverable must not carry a flag the engine in front of it would reject."""
    from run_config import clean_args
    args = clean_args(["--glm-pack", "/p", "--kv", "int8",
                       "--prompt-cache-tail", "1", "--pool-workers", "4"])
    # The stand-in project's generate.cpp parses --kv and --pool-workers, not the older
    # --prompt-cache-tail the Strata tuner used.
    assert "--prompt-cache-tail" not in args
    assert "1" not in args[args.index("--kv") + 1:args.index("--pool-workers")]
    assert "--kv" in args and "--pool-workers" in args


def test_sweep_only_flags_never_reach_the_run_config():
    from run_config import clean_args
    args = clean_args(["--serve", "--glm-pack", "/p", "--max-new", "512",
                      "--expert-profile-save-every", "0", "--kv", "int8"])
    assert "--serve" not in args and "--max-new" not in args
    assert "--expert-profile-save-every" not in args
    assert args == ["--glm-pack", "/p", "--kv", "int8"]


def test_a_pinned_dimension_is_held_constant():
    code, out = _run(["--gen", "--kv", "int8", "--worker", "4", "--draft-floor", "0.5",
                      "--trials", "6", "--seed", "3"])
    assert code == 0
    record = _record(out)
    assert record["trials"]
    for trial in record["trials"]:
        assert trial["cfg"]["kv"] == "int8"
        assert int(trial["cfg"]["worker"]) == 4
        assert abs(float(trial["cfg"]["draft_floor"]) - 0.5) < 1e-9
    cfg = json.loads((out / "maya-maya-s24.json").read_text(encoding="utf-8"))
    args = cfg["args"]
    assert args[args.index("--kv") + 1] == "int8"
    assert args[args.index("--pool-workers") + 1] == "4"
    assert args[args.index("--spec-min-p") + 1] == "0.5"
    assert args[args.index("--prompt-cache") + 1] in {"0", "2", "4", "6"}


def test_engine_args_covers_every_swept_dimension():
    cfg = {"vram": 768, "pcie": 0.9, "pcie_mode": "dma", "ple_io": "mmap", "kv": "k8v4",
           "kvstream": "on", "hostworker": "0", "worker": 12, "tail": 4,
           "draft_floor": 0.5}
    args = engine_args(cfg, {"pack": "/p"}, 131072, 512)
    for flag, value in (("--vram-reserve-mib", "768"), ("--pcie-frac", "0.9"),
                        ("--pcie-mode", "dma"), ("--ple-io", "mmap"), ("--kv", "k8v4"),
                        ("--kv-streaming", "on"), ("--pool-workers", "12"),
                        ("--prompt-cache", "4"),
                        ("--max-context", "131072"), ("--spec-min-p", "0.5")):
        assert flag in args, f"{flag} missing"
        assert args[args.index(flag) + 1] == value
    # hostworker 0 means the host thread is kept out of the drain: a bare flag, no value.
    assert "--no-host-worker" in args
    assert args[args.index("--no-host-worker") + 1] != "0"
    # The host thread joins the pool: the flag is not written at all.
    off = engine_args({**cfg, "hostworker": "1", "kvstream": "off"}, {"pack": "/p"}, 4096, 256)
    assert "--no-host-worker" not in off
    assert args[args.index("--kv-streaming") + 1] == "on" and off[args.index("--kv-streaming") + 1] == "off"


def test_the_space_is_the_columns_the_engine_has():
    params = build_space(max_workers=8)
    keys = {p.key for p in params}
    assert keys == {"vram", "pcie", "pcie_mode", "ple_io", "kv", "kvstream", "hostworker",
                    "worker", "tail", "draft_floor"}
    assert ROW_COLUMNS[:11] == ("layer", "vram", "pcie", "pcie_mode", "ple_io", "kv",
                               "kvstream", "hostworker", "worker", "tail", "draft_floor")
    assert ROW_COLUMNS[11] == "status"
    floor = next(p for p in params if p.key == "draft_floor")
    assert floor.low == 0.001 and floor.high == 1.0 and floor.flag == "--spec-min-p"
    # The worker grid is capped at physical cores, and 0 (the engine's "all cores") is a legal arm.
    worker = next(p for p in params if p.key == "worker")
    assert worker.grid()[0] == 0 and worker.grid()[-1] == 7


def test_stub_engine_measures():
    ids = build_prompt(str(HERE / "fixtures" / "prompt.json"), 1024)
    args = ["--serve", "--max-context", "131072", "--kv", "k8v4",
            "--pool-workers", "8", "--max-new", "256"]
    m = run_config(str(STUB), args, ids, 256, Path("/tmp/hc-stub-run.log"),
                   boot_timeout=20, run_timeout=20)
    if m.status == "FAIL_BOOT" and not m.effective:
        # A subprocess that never got scheduled under load: one retry, then report it.
        m = run_config(str(STUB), args, ids, 256, Path("/tmp/hc-stub-run.log"),
                       boot_timeout=20, run_timeout=20)
    assert m.status == "PASS", m.note
    assert m.gen_tok_s > 0 and m.prefill_tok_s > 0
    assert abs(m.latency_ms - (m.prompt_ms + m.decode_ms)) < 1e-6


def test_a_bad_config_is_recorded_not_crashed():
    m = run_config(str(STUB), ["--serve", "--vram-reserve-mib", "9999"], [1, 2, 3], 8,
                   Path("/tmp/hc-stub-fail.log"), boot_timeout=10, run_timeout=10)
    assert m.status in ("FAIL_BOOT", "TIMEOUT")
    assert m.note


def test_objective_direction():
    from engine_runner import Metrics
    m = Metrics(status="PASS", gen_tok_s=150.0, prefill_tok_s=900.0, latency_ms=4000.0)
    assert signed_score(m, "gen") == 150.0
    assert signed_score(m, "prefill") == 900.0
    assert signed_score(m, "latency") == -4000.0
    failed = Metrics(status="FAIL_BOOT")
    assert signed_score(failed, "gen") == 0.0
    assert signed_score(failed, "latency") < signed_score(m, "latency")


def _run(extra, out: Path = Path("/tmp/hc-optuna-test"), fresh: bool = True):
    """One sweep against the stub engine, in a directory of its own."""
    if fresh:
        shutil.rmtree(out, ignore_errors=True)
    argv = ["--model", "maya-s24", "--engine", str(STUB), "--maya-root", str(FAKE_ROOT),
            "--pack", str(HERE / "fixtures"),
            "--prompt-tokens-file", str(HERE / "fixtures" / "prompt.json"),
            "--deliverable-dir", str(out), "--results-dir", str(out / "results"),
            "--history-dir", str(out / "history"),
            "--max-workers", "8", "--trials", "12", "--seed", "7"] + extra
    return main(argv), out


def _record(out: Path) -> dict:
    """The trial record: every configuration measured, written beside the CSV."""
    return json.loads((out / "results" / "maya-maya-s24.record.json").read_text(encoding="utf-8"))


def test_the_sweep_runs_and_writes_both_deliverables():
    code, out = _run(["--gen"])
    assert code == 0
    sh = out / "run-maya-maya-s24.sh"
    js = out / "maya-maya-s24.json"
    assert sh.exists() and js.exists()
    cfg = json.loads(js.read_text(encoding="utf-8"))
    # The deliverable is a run config the server can read, not a tuning dump.
    for key in ("exe", "args", "cwd", "tokenizer", "model_name", "log", "port",
                "host", "api_key"):
        assert key in cfg, f"run config missing {key}"
    assert "--serve" not in cfg["args"] and "--max-new" not in cfg["args"]
    assert "--prompt-cache-tail" not in cfg["args"]
    assert cfg["args"].count("--max-context") == 1
    assert cfg["args"][cfg["args"].index("--max-context") + 1] == "131072"
    tuned = cfg["tuned_by"]
    assert tuned["objective"] == "gen"
    assert tuned["context"] == 131072
    for key in ("vram", "pcie", "ple_io", "kv", "kvstream", "hostworker", "pcie_mode",
                "worker", "tail", "draft_floor"):
        assert key in tuned["best"]
    args = cfg["args"]
    assert "--spec-min-p" in args
    floor = float(args[args.index("--spec-min-p") + 1])
    assert 0.001 <= floor <= 1.0
    assert tuned["metrics"]["gen_tok_s"] > 0
    record = _record(out)
    assert record["schema"] == "hypercore.max_run/1"
    assert record["study"]["measured"] == 12
    assert len(record["trials"]) == 12
    # The launcher must be valid shell and executable.
    _bash_n(sh)
    if os.name != "nt":
        assert os.access(sh, os.X_OK)
    csvs = list((out / "results").glob("bayesian-*.csv"))
    assert csvs, "no CSV written"
    header = csvs[0].read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",")[:12] == list(ROW_COLUMNS)


def test_the_search_is_deterministic_for_a_seed():
    _run(["--gen"])
    first = _record(Path("/tmp/hc-optuna-test"))
    _run(["--gen"])
    second = _record(Path("/tmp/hc-optuna-test"))
    assert first["best"] == second["best"]
    assert first["metrics"]["gen_tok_s"] == second["metrics"]["gen_tok_s"]


def test_latency_objective_picks_the_short_request():
    code, out = _run(["--latency"])
    assert code == 0
    record = _record(out)
    assert record["objective"] == "latency"
    assert record["metrics"]["latency_ms"] > 0
    rows = record["trials"]
    assert all(r["status"] == "PASS" for r in rows)
    assert abs(record["metrics"]["latency_ms"] - min(float(r["latency_ms"]) for r in rows)) < 1e-6


def test_staged_mode_labels_rows_by_layer():
    code, out = _run(["--mode", "staged", "--layers", "kv,worker"])
    assert code == 0
    record = _record(out)
    layers = {r["layer"] for r in record["trials"]}
    assert layers == {"kv", "worker"}


def test_dry_run_launches_nothing():
    code, out = _run(["--dry-run"])
    assert code == 0
    record = _record(out)
    assert record["study"]["measured"] == 0
    assert all(r["status"] == "DRY" for r in record["trials"])
    assert all("--max-context 131072" in r["note"] for r in record["trials"])


def test_a_blocked_deliverable_path_falls_back_instead_of_crashing():
    """A root-level launcher owned by another user must not lose the whole sweep."""
    blocked_dir = Path("/tmp/hc-optuna-blocked")
    blocked_dir.mkdir(parents=True, exist_ok=True)
    locked = blocked_dir / "run-maya-maya-s24.sh"
    locked.chmod(0o644) if locked.exists() else None
    locked.write_text("# an older launcher, not ours to overwrite\n", encoding="utf-8")
    locked.chmod(0o444)
    out = Path("/tmp/hc-optuna-fallback")
    code = main([
        "--model", "maya-s24", "--engine", str(STUB),
        "--pack", str(HERE / "fixtures"),
        "--prompt-tokens-file", str(HERE / "fixtures" / "prompt.json"),
        "--deliverable-dir", str(blocked_dir),
        "--results-dir", str(out / "results"),
        "--history-dir", str(out / "history"),
        "--max-workers", "8", "--trials", "4", "--seed", "3", "--gen",
    ])
    assert code == 0
    assert locked.read_text().startswith("# an older launcher")
    assert (out / "results" / "run-maya-maya-s24.sh").exists()
    assert (out / "results" / "maya-maya-s24.json").exists()


def test_a_dead_engine_leaves_no_broken_pipe_at_finalisation():
    """The engine exiting at boot used to raise BrokenPipeError when its request pipe
    was finalised. It must now be recorded as FAIL_BOOT and stay silent."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, {root!r});"
         "from pathlib import Path;"
         "from engine_runner import run_config;"
         "m = run_config({engine!r}, ['--serve'], [1, 2, 3], 8, Path('/tmp/hc-exit.log'), "
         "boot_timeout=5, run_timeout=5);"
         "print(m.status)".format(root=str(HERE), engine=str(HERE / "stub_engine_exit.py"))],
        capture_output=True, text=True, check=True,
    )
    assert proc.stdout.strip() == "FAIL_BOOT"
    assert "BrokenPipeError" not in proc.stderr
    assert "Exception ignored" not in proc.stderr


def test_ctrl_c_keeps_the_best_configuration_measured_so_far():
    out = Path("/tmp/hc-optuna-interrupt")
    argv = [
        sys.executable, str(HERE / "bayesian_tune.py"),
        "--model", "maya-s24", "--engine", str(HERE / "stub_engine_slow.py"),
        "--maya-root", str(FAKE_ROOT),
        "--pack", str(HERE / "fixtures"),
        "--prompt-tokens-file", str(HERE / "fixtures" / "prompt.json"),
        "--deliverable-dir", str(out), "--results-dir", str(out / "results"),
        "--max-workers", "8", "--trials", "20", "--seed", "5", "--gen",
        "--history-dir", str(out / "history"),
    ]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Wait until at least one trial has started, otherwise SIGINT can land in import.
    deadline = time.monotonic() + 30
    buf: list[str] = []
    while time.monotonic() < deadline:
        line = proc.stdout.readline() if proc.stdout is not None else ""
        if line:
            buf.append(line)
            if "trial" in line:
                break
        elif proc.poll() is not None:
            break
    proc.send_signal(signal.SIGINT)
    rest_out, stderr = proc.communicate(timeout=30)
    stdout = "".join(buf) + (rest_out or "")
    assert proc.returncode == 0, stdout + stderr
    assert "interrupted" in stdout
    record = _record(out)
    assert record["interrupted"] is True
    assert record["study"]["measured"] >= 1
    assert (out / "run-maya-maya-s24.sh").exists()
    _bash_n(out / "run-maya-maya-s24.sh")


def test_the_generated_launcher_is_the_project_startup_pair():
    """run-maya-<model>.sh is the startup script a Maya checkout would run."""
    out = Path("/tmp/hc-optuna-launcher")
    main([
        "--model", "maya-s24", "--engine", str(STUB), "--maya-root", str(FAKE_ROOT),
        "--pack", str(HERE / "fixtures"),
        "--prompt-tokens-file", str(HERE / "fixtures" / "prompt.json"),
        "--deliverable-dir", str(out), "--results-dir", str(out / "results"),
        "--max-workers", "8", "--trials", "3", "--seed", "2", "--gen", "--port", "8090",
    ])
    launcher = out / "run-maya-maya-s24.sh"
    cfg_path = out / "maya-maya-s24.json"
    lines = launcher.read_text().splitlines()
    assert lines[0] == "#!/bin/sh"
    assert lines[-2] == f'cd "{FAKE_ROOT}"', lines[-2]
    assert lines[-1].startswith("exec ")
    assert f'"{FAKE_PYTHON}"' in lines[-1], "the launcher must use the project venv"
    assert f'"{FAKE_ROOT / "maya.py"}"' in lines[-1], "the launcher runs Maya's entry point"
    assert f'"--config" "{cfg_path}"' in lines[-1]
    assert '"--port" "8090"' in lines[-1]
    assert "--turn-tail-check" not in lines[-1], "the stand-in entry point has no such flag"
    _bash_n(launcher)
    if os.name != "nt":
        assert os.access(launcher, os.X_OK)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["port"] == 8090
    assert cfg["exe"] == str(STUB)
    assert cfg["cwd"] == str(FAKE_ROOT)
    # A Windows install of Maya gets the same launcher in .bat form.
    bat = (out / "run-maya-maya-s24.bat").read_text(encoding="utf-8")
    assert bat.startswith("@echo off") and "cd /d " in bat and '"--config"' in bat


def test_the_launcher_is_built_from_the_entry_point_that_exists():
    """A launcher flag the local entry point does not define would be rejected at startup."""
    from maya_env import entry_flags, entry_script, server_flags

    assert entry_script(FAKE_ROOT) == str(FAKE_ROOT / "maya.py")
    assert "--open" in entry_flags(FAKE_ROOT)
    assert "--turn-tail-check" not in server_flags(FAKE_ROOT)

    out = Path("/tmp/hc-optuna-open-capable")
    code = main([
        "--model", "maya-s24", "--engine", str(STUB), "--maya-root", str(FAKE_ROOT),
        "--pack", str(HERE / "fixtures"),
        "--prompt-tokens-file", str(HERE / "fixtures" / "prompt.json"),
        "--deliverable-dir", str(out), "--results-dir", str(out / "results"),
        "--max-workers", "8", "--trials", "2", "--seed", "3", "--gen",
    ])
    assert code == 0
    cfg = json.loads((out / "maya-maya-s24.json").read_text(encoding="utf-8"))
    assert "turn_tail_check" not in cfg, "a key nobody reads is dead weight"
    launcher_path = out / "run-maya-maya-s24.sh"
    assert "--open" in launcher_path.read_text(encoding="utf-8")
    assert "--turn-tail-check" not in launcher_path.read_text(encoding="utf-8")
    _bash_n(launcher_path)


def test_an_entry_point_without_open_does_not_get_it():
    """The same code path on an entry point that does not define --open leaves it out."""
    root = Path(tempfile.mkdtemp(prefix="hc-no-open-"))
    (root / "serve").mkdir(parents=True)
    (root / "serve" / "server.py").write_text(
        'ap.add_argument("--engine", choices=["mock", "strata"], default="mock")\n',
        encoding="utf-8",
    )
    (root / "maya.py").write_text(
        'ap.add_argument("--config", type=Path)\n'
        'ap.add_argument("--port", type=int)\n',
        encoding="utf-8",
    )
    (root / "CMakeLists.txt").write_text("", encoding="utf-8")
    maya_root.cache_clear()
    try:
        out = Path("/tmp/hc-optuna-no-open")
        main([
            "--model", "maya-s24", "--engine", str(STUB), "--maya-root", str(root),
            "--pack", str(HERE / "fixtures"),
            "--prompt-tokens-file", str(HERE / "fixtures" / "prompt.json"),
            "--deliverable-dir", str(out), "--results-dir", str(out / "results"),
            "--max-workers", "8", "--trials", "2", "--seed", "4", "--gen",
        ])
        launcher = (out / "run-maya-maya-s24.sh").read_text(encoding="utf-8")
        assert "--open" not in launcher
        assert f'"{root / "maya.py"}"' in launcher
    finally:
        maya_root.cache_clear()
        os.environ["MAYA_ROOT"] = str(FAKE_ROOT)


def test_no_browser_leaves_open_out_of_the_launcher():
    out = Path("/tmp/hc-optuna-no-browser")
    main([
        "--model", "maya-s24", "--engine", str(STUB), "--maya-root", str(FAKE_ROOT),
        "--pack", str(HERE / "fixtures"),
        "--prompt-tokens-file", str(HERE / "fixtures" / "prompt.json"),
        "--deliverable-dir", str(out), "--results-dir", str(out / "results"),
        "--max-workers", "8", "--trials", "2", "--seed", "1", "--gen", "--no-browser",
    ])
    text = (out / "run-maya-maya-s24.sh").read_text()
    assert "--open" not in text


def test_the_run_config_keeps_what_the_model_config_already_has():
    """Only "args" is replaced; everything setup does not own survives the sweep."""
    from run_config import build_run_config
    base = {"exe": "old-exe", "args": ["--glm-pack", "old"], "cwd": "/old", "tokenizer": "/tok",
            "model_name": "real-name", "sampling": {"top_p": 0.95}, "mcp_servers": ["a"]}
    record = {"engine": "/engine/hypercore", "objective": "gen", "context": 8192,
              "best": {}, "metrics": {}, "study": {}, "csv": ""}
    args = ["--serve", "--glm-pack", "/pack", "--max-new", "64", "--max-context", "8192",
            "--expert-profile-save-every", "0", "--kv", "int8"]
    cfg = build_run_config("maya-s24", {"pack": "/pack"}, args, record, base=base)
    assert cfg["args"] == ["--glm-pack", "/pack", "--max-context", "8192", "--kv", "int8"]
    assert cfg["exe"] == "/engine/hypercore"
    assert cfg["model_name"] == "real-name"
    assert cfg["tokenizer"] == "/tok"
    assert cfg["sampling"] == {"top_p": 0.95}
    assert cfg["mcp_servers"] == ["a"]
    assert cfg["tuned_by"]["objective"] == "gen"


def _bash_n(path: Path) -> None:
    bash = shutil.which("bash")
    if bash:
        subprocess.run([bash, "-n", str(path)], check=True)


def test_the_entry_point_is_executable_and_the_tree_is_gitignored():
    sh = HERE / "max_tune.sh"
    bat = HERE / "max_tune.bat"
    assert sh.exists()
    assert bat.exists()
    if os.name != "nt":
        assert os.access(sh, os.X_OK)
    _bash_n(sh)
    text = bat.read_text(encoding="utf-8")
    assert "bayesian_tune.py" in text
    assert r".venv\Scripts\python.exe" in text
    own = (HERE / ".gitignore").read_text(encoding="utf-8")
    for rule in ("results/", "history/", "run-maya-*.sh", "maya-*.json", "run-maya-*.bat"):
        assert rule in own, f"missing ignore rule in the repository: {rule}"
    project = Path(os.environ["MAYA_ROOT"]) / ".gitignore"
    if project.is_file():
        gitignore = project.read_text(encoding="utf-8")
        for rule in ("/run-maya-*.sh", "/maya-*.json", "/run-maya-*.bat"):
            assert rule in gitignore, f"the project .gitignore does not ignore the deliverables: {rule}"


# ---- where the project is, and how absolute the written paths are ---------------


def _clear_root_cache() -> None:
    maya_root.cache_clear()
    os.environ["MAYA_ROOT"] = str(FAKE_ROOT)


def test_a_nested_clone_resolves_its_own_checkout(monkeypatch):
    """<Strata>/HyperCore-Tune is found by walking up, with no environment set."""
    tmp = Path(tempfile.mkdtemp(prefix="hc-nested-clone-"))
    project = tmp / "project-maya"
    (project / "serve").mkdir(parents=True)
    (project / "serve" / "server.py").write_text("", encoding="utf-8")
    (project / "CMakeLists.txt").write_text("", encoding="utf-8")
    clone = project / "HyperCore-Tune"
    clone.mkdir()

    maya_root.cache_clear()
    monkeypatch.delenv("MAYA_ROOT", raising=False)
    monkeypatch.delenv("HYPERCORE_ROOT", raising=False)
    monkeypatch.delenv("TUNE_ROOT", raising=False)
    monkeypatch.setattr(maya_env, "REPO", clone)
    try:
        assert maya_root() == project
    finally:
        maya_root.cache_clear()
        os.environ["MAYA_ROOT"] = str(FAKE_ROOT)


def test_a_sibling_checkout_is_found_when_the_clone_is_alone(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="hc-sibling-clone-"))
    project = tmp / "project-maya"
    (project / "serve").mkdir(parents=True)
    (project / "serve" / "server.py").write_text("", encoding="utf-8")
    (project / "setup.py").write_text("", encoding="utf-8")
    clone = tmp / "HyperCore-Tune"
    clone.mkdir()

    maya_root.cache_clear()
    monkeypatch.delenv("MAYA_ROOT", raising=False)
    monkeypatch.delenv("HYPERCORE_ROOT", raising=False)
    monkeypatch.delenv("TUNE_ROOT", raising=False)
    monkeypatch.setattr(maya_env, "REPO", clone)
    try:
        assert maya_root() == project
    finally:
        maya_root.cache_clear()
        os.environ["MAYA_ROOT"] = str(FAKE_ROOT)


def _make_checkout(path: Path) -> Path:
    (path / "serve").mkdir(parents=True, exist_ok=True)
    (path / "serve" / "server.py").write_text("", encoding="utf-8")
    (path / "CMakeLists.txt").write_text("", encoding="utf-8")
    return path


def _isolate_root_env(monkeypatch) -> None:
    maya_root.cache_clear()
    monkeypatch.delenv("MAYA_ROOT", raising=False)
    monkeypatch.delenv("HYPERCORE_ROOT", raising=False)
    monkeypatch.delenv("TUNE_ROOT", raising=False)


def test_a_nested_clone_walks_up_to_the_checkout(monkeypatch):
    """~/project-maya/HyperCore-Tune-Maya is this app inside a Maya tree; walk-up finds it."""
    tmp = Path(tempfile.mkdtemp(prefix="hc-nested-tune-"))
    project = _make_checkout(tmp / "project-maya")
    clone = project / "HyperCore-Tune-Maya"
    clone.mkdir()
    _isolate_root_env(monkeypatch)
    monkeypatch.setattr(maya_env, "REPO", clone)
    try:
        assert maya_root() == project.resolve()
        assert maya_root().is_absolute()
    finally:
        _clear_root_cache()


def test_a_standalone_clone_finds_the_sibling_checkout(monkeypatch):
    """~/HyperCore-Tune-Maya next to ~/project-maya: this app is standalone, the checkout is the sibling."""
    tmp = Path(tempfile.mkdtemp(prefix="hc-standalone-tune-"))
    project = _make_checkout(tmp / "project-maya")
    clone = tmp / "HyperCore-Tune-Maya"
    clone.mkdir()
    _isolate_root_env(monkeypatch)
    monkeypatch.setattr(maya_env, "REPO", clone)
    try:
        assert maya_root() == project.resolve()
        assert maya_root().is_absolute()
    finally:
        _clear_root_cache()


def test_abs_path_expands_home_and_windows_vars(monkeypatch):
    home = Path(tempfile.mkdtemp(prefix="hc-abs-home-"))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    target = abs_path("~/HyperCore-Tune-Maya")
    assert target is not None
    assert target.is_absolute()
    assert target == (home / "HyperCore-Tune-Maya").resolve()
    assert "~" not in str(target)
    via_var = abs_path("%USERPROFILE%\\HyperCore-Tune-Maya")
    assert via_var == (home / "HyperCore-Tune-Maya").resolve()


def test_a_tilde_maya_root_is_absolute(monkeypatch):
    home = Path(tempfile.mkdtemp(prefix="hc-tilde-home-"))
    project = _make_checkout(home / "project-maya")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    maya_root.cache_clear()
    monkeypatch.setenv("MAYA_ROOT", "~/project-maya")
    monkeypatch.delenv("HYPERCORE_ROOT", raising=False)
    monkeypatch.delenv("TUNE_ROOT", raising=False)
    try:
        assert maya_root() == project.resolve()
        assert maya_root().is_absolute()
    finally:
        _clear_root_cache()


def test_relative_cli_paths_become_absolute(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="hc-rel-cli-"))
    pack = tmp / "pack"
    pack.mkdir()
    (pack / "tokenizer").mkdir()
    native = tmp / "tiny.gguf"
    native.write_text("x", encoding="utf-8")
    monkeypatch.chdir(tmp)
    out = tmp / "out"
    code = main([
        "--model", "maya-s24", "--engine", os.path.relpath(STUB, tmp),
        "--maya-root", str(FAKE_ROOT),
        "--pack", "pack",
        "--prompt-tokens-file", str(HERE / "fixtures" / "prompt.json"),
        "--deliverable-dir", "out", "--results-dir", str(out / "results"),
        "--history-dir", str(out / "history"),
        "--max-workers", "8", "--trials", "3", "--seed", "7", "--gen",
    ])
    assert code == 0
    cfg = json.loads((out / "maya-maya-s24.json").read_text(encoding="utf-8"))
    assert Path(cfg["exe"]).is_absolute()
    assert Path(cfg["args"][cfg["args"].index("--glm-pack") + 1]).is_absolute()


def test_cpu_probe_survives_without_proc_or_uname(monkeypatch):
    import hardware as hardware_mod
    from hardware import detect_cpu
    monkeypatch.setattr(hardware_mod, "_read", lambda path: "")
    def _no_uname():
        raise AttributeError("uname")
    monkeypatch.setattr(os, "uname", _no_uname, raising=False)
    name, phys, logical = detect_cpu()
    assert logical >= 1
    assert isinstance(name, str) and name


def test_windows_appdata_settings_are_read(monkeypatch):
    from maya_env import settings_data_dir
    tmp = Path(tempfile.mkdtemp(prefix="hc-appdata-"))
    appdata = tmp / "AppData" / "Roaming"
    data = tmp / "Maya-data"
    data.mkdir(parents=True)
    (appdata / "project-maya").mkdir(parents=True)
    (appdata / "project-maya" / "settings.json").write_text(
        json.dumps({"data_dir": str(data)}), encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert settings_data_dir() == str(data.resolve())


def test_a_clone_with_no_project_nearby_says_so(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="hc-lone-clone-"))
    clone = tmp / "HyperCore-Tune"
    clone.mkdir()
    maya_root.cache_clear()
    monkeypatch.delenv("MAYA_ROOT", raising=False)
    monkeypatch.delenv("HYPERCORE_ROOT", raising=False)
    monkeypatch.delenv("TUNE_ROOT", raising=False)
    monkeypatch.setattr(maya_env, "REPO", clone)
    try:
        with pytest.raises(SystemExit) as exc:
            maya_root()
        assert "--maya-root" in str(exc.value)
        assert candidate_roots(), "the error should still name what was searched"
    finally:
        maya_root.cache_clear()
        os.environ["MAYA_ROOT"] = str(FAKE_ROOT)


def test_the_engine_binary_is_the_one_this_project_built():
    assert engine_binary(FAKE_ROOT) == str(FAKE_ENGINE)
    assert engine_binary(Path(tempfile.mkdtemp(prefix="hc-no-engine-"))) == ""
    assert engine_binary(FAKE_ROOT).startswith(str(FAKE_ROOT))


def test_the_launcher_runs_the_project_venv_not_whatever_started_the_sweep():
    from maya_env import venv_python

    assert venv_python(FAKE_ROOT) == str(FAKE_PYTHON)


def test_the_launcher_uses_the_backend_the_local_server_accepts():
    assert server_engine_name(FAKE_ROOT) == "strata"


def test_every_path_the_sweep_writes_is_absolute():
    """A deliverable is only drop-in if it does not depend on the current directory."""
    code, out = _run(["--gen"])
    assert code == 0
    cfg = json.loads((out / "maya-maya-s24.json").read_text(encoding="utf-8"))
    for key in ("exe", "cwd", "log", "tokenizer", "model_name"):
        assert cfg.get(key), f"{key} is empty"
    for key in ("exe", "cwd", "log", "tokenizer"):
        assert Path(cfg[key]).is_absolute(), f"{key} is not absolute: {cfg[key]}"
    for flag in ("--glm-pack", "--dflash", "--expert-profile"):
        if flag in cfg["args"]:
            value = cfg["args"][cfg["args"].index(flag) + 1]
            assert Path(value).is_absolute(), f"{flag} is relative: {value}"
    launcher = (out / "run-maya-maya-s24.sh").read_text(encoding="utf-8")
    assert f'cd "{FAKE_ROOT}"' in launcher


# ---- run data from previous sweeps, keyed to the machine ---------------------

PINNED = json.dumps({
    "cpu": "Test CPU X1", "physical_cores": 8, "logical_cores": 16, "ram_gib": 64,
    "gpus": ["Test GPU 12GB"], "gpu_mem_mib": 12288, "pci_gen": 4, "pci_width": 16,
})
PINNED_SLUG = "Test-CPU-X1_Test-GPU-12GB_64GB_PCIe4x16"
OTHER = json.dumps({
    "cpu": "Other CPU Y2", "physical_cores": 4, "logical_cores": 8, "ram_gib": 16,
    "gpus": ["Other GPU 8GB"], "gpu_mem_mib": 8192, "pci_gen": 3, "pci_width": 8,
})


def test_the_machine_slug_is_the_four_fields_the_data_file_is_keyed_on():
    from hardware import differences, hardware_from, slug
    hw = hardware_from(json.loads(PINNED))
    assert slug(hw) == PINNED_SLUG
    assert hw.machine_key() == ("test cpu x1", ("test gpu 12gb",), 64, 4)
    assert differences(hw, hw) == []
    assert differences(hw, hardware_from(json.loads(OTHER))) == [
        "cpu Test CPU X1 vs Other CPU Y2",
        "gpu Test GPU 12GB vs Other GPU 8GB",
        "ram 64 GiB vs 16 GiB",
        "PCIe gen 4 vs 3",
    ]


def test_the_host_probe_returns_a_usable_fingerprint():
    from hardware import detect
    hw = detect()
    assert hw.ram_gib > 0 and hw.slug()
    assert all(ch not in " ()" for ch in hw.slug())


def test_pci_generation_is_read_from_sysfs_when_lspci_is_absent():
    from hardware import pci_from_sysfs
    fake = Path("/tmp/hc-pci-fake/0000:01:00.0")
    fake.mkdir(parents=True, exist_ok=True)
    fake.joinpath("max_link_speed").write_text("16.0 GT/s PCIe\n", encoding="utf-8")
    fake.joinpath("max_link_width").write_text("16\n", encoding="utf-8")
    assert pci_from_sysfs(fake.parent) == (4, 16)
    assert pci_from_sysfs("/no/such/dir") == (0, 0)


def test_old_history_without_draft_floor_still_loads():
    from history import normalise_trial
    row = {"cfg": {"vram": 512, "pcie": 0.5, "ple_io": "mmap", "kv": "int8", "kvstream": "off",
                   "hostworker": "1", "pcie_mode": "auto", "worker": 4, "tail": 0},
           "score": 1.0, "status": "PASS", "spec_min_p": 0.8}
    trial = normalise_trial(row, 512, 0.8)
    assert trial is not None
    assert abs(float(trial["cfg"]["draft_floor"]) - 0.8) < 1e-9


def test_history_merge_keeps_the_best_score_for_each_configuration():
    from hardware import hardware_from
    from history import merge
    cfg = {"vram": 512, "pcie": 0.5, "ple_io": "mmap", "kv": "int8", "kvstream": "off",
           "hostworker": "1", "pcie_mode": "auto", "worker": 4, "tail": 0,
           "draft_floor": 0.8}

    def record(score):
        return {"model": "maya-s24", "objective": "gen", "context": 131072, "max_new": 512,
                "spec_min_p": 0.8, "generated_utc": "2026-10-12T00:00:00+00:00",
                "hardware": hardware_from(json.loads(PINNED)).to_dict(),
                "hardware_slug": PINNED_SLUG,
                "trials": [{"cfg": cfg, "score": score, "status": "PASS", "layer": "joint",
                            "metrics": {"gen_tok_s": score}}]}

    merged = merge(None, record(10.0))
    merged = merge(merged, record(12.0))
    merged = merge(merged, record(8.0))
    assert len(merged["trials"]) == 1
    assert merged["trials"][0]["score"] == 12.0
    assert merged["trials"][0]["measured"] == 3
    assert len(merged["runs"]) == 3


def test_a_sweep_writes_run_data_named_after_the_machine():
    out = Path("/tmp/hc-optuna-history")
    code, out = _run(["--hardware", PINNED, "--trials", "3", "--seed", "3"], out)
    assert code == 0
    files = list((out / "history").glob("*.json"))
    assert [f.name for f in files] == [f"{PINNED_SLUG}_maya-s24_gen_131072.json"]
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["schema"] == "hypercore.tuning_history/1"
    assert data["hardware_slug"] == PINNED_SLUG
    assert data["hardware"]["cpu"] == "Test CPU X1"
    assert data["context"] == 131072 and data["objective"] == "gen"
    assert len(data["trials"]) == 3
    assert all(t["cfg"].keys() == {"vram", "pcie", "ple_io", "kv", "kvstream",
                                   "hostworker", "pcie_mode", "worker", "tail", "draft_floor"}
               for t in data["trials"])
    csvs = list((out / "results").glob("bayesian-*.csv"))
    assert any(PINNED_SLUG in c.name for c in csvs)


def test_a_second_sweep_does_not_reboot_known_points_and_keeps_drawing():
    """The same seed would re-ask the first six points; those are pruned, not counted,
    and the sweep keeps drawing until six unseen configurations have been booted."""
    out = Path("/tmp/hc-optuna-reuse")
    argv = ["--hardware", PINNED, "--sampler", "random", "--trials", "6", "--seed", "11"]
    _run(argv, out)
    first_logs = {p.name for p in (out / "results").glob("run-*.log")}
    assert len(first_logs) == 6
    code, _ = _run(argv, out, fresh=False)
    assert code == 0
    record = _record(out)
    second_logs = {p.name for p in (out / "results").glob("run-*.log")}
    assert record["history"]["trials_loaded"] == 6
    assert record["history"]["duplicates_skipped"] >= 6
    assert record["study"]["measured"] == 6
    assert first_logs <= second_logs, "the original six boots must not be repeated"
    assert len(second_logs - first_logs) == 6
    assert record["best"], "the winner must still come from the stored data"
    assert (out / "run-maya-maya-s24.sh").exists() and (out / "maya-maya-s24.json").exists()


def test_run_data_from_another_machine_is_not_used():
    out = Path("/tmp/hc-optuna-mismatch")
    _run(["--hardware", PINNED, "--trials", "3", "--seed", "4"], out)
    _run(["--hardware", OTHER, "--trials", "3", "--seed", "4"], out, fresh=False)
    assert len(list((out / "history").glob("*.json"))) == 2

    _run(["--hardware", PINNED, "--sampler", "random", "--trials", "3", "--seed", "4"],
         out, fresh=False)
    assert _record(out)["history"]["trials_loaded"] == 3

    already = sum(len(json.loads(p.read_text(encoding="utf-8"))["trials"])
                  for p in (out / "history").glob("*.json"))
    _run(["--hardware", PINNED, "--sampler", "random", "--trials", "3", "--seed", "4",
          "--allow-hardware-mismatch"], out, fresh=False)
    assert _record(out)["history"]["trials_loaded"] == already


def test_a_prior_outside_the_current_grid_is_skipped_not_clamped():
    out = Path("/tmp/hc-optuna-grid")
    _run(["--hardware", PINNED, "--trials", "6", "--seed", "9", "--vram-max", "1536"], out)
    stored = json.loads((out / "history" / f"{PINNED_SLUG}_maya-s24_gen_131072.json")
                        .read_text(encoding="utf-8"))
    outside = sum(1 for t in stored["trials"] if t["cfg"]["vram"] > 256)
    assert outside > 0, "the fixture must produce a prior the narrower grid cannot hold"
    _run(["--hardware", PINNED, "--trials", "1", "--seed", "9", "--vram-max", "256",
          "--no-deliverables"], out, fresh=False)
    record = _record(out)
    assert record["history"]["trials_loaded"] == len(stored["trials"])
    assert record["history"]["priors_skipped"] == outside


def test_the_deliverable_says_which_machine_it_was_tuned_on():
    out = Path("/tmp/hc-optuna-machine")
    _run(["--hardware", PINNED, "--trials", "2", "--seed", "1"], out)
    cfg = json.loads((out / "maya-maya-s24.json").read_text(encoding="utf-8"))
    assert cfg["tuned_by"]["hardware_slug"] == PINNED_SLUG
    assert cfg["tuned_by"]["hardware"]["ram_gib"] == 64
    assert cfg["tuned_by"]["history"]["file"].endswith(f"{PINNED_SLUG}_maya-s24_gen_131072.json")
    assert PINNED_SLUG in (out / "run-maya-maya-s24.sh").read_text(encoding="utf-8")


def test_replaying_a_record_rewrites_the_deliverables_without_booting_anything():
    out = Path("/tmp/hc-optuna-replay")
    _run(["--hardware", PINNED, "--trials", "4", "--seed", "3"], out)
    record = _record(out)
    best = record["best"]
    csvs_before = len(list((out / "results").glob("bayesian-*.csv")))

    code, _ = _run(["--hardware", PINNED, "--replay", str(out / "results" / "maya-maya-s24.record.json")],
                   out, fresh=False)
    assert code == 0
    cfg = json.loads((out / "maya-maya-s24.json").read_text(encoding="utf-8"))
    sh = (out / "run-maya-maya-s24.sh").read_text(encoding="utf-8")
    assert cfg["tuned_by"]["best"] == best
    assert cfg["tuned_by"]["hardware_slug"] == PINNED_SLUG
    assert cfg["args"] == clean_args(record["args"])
    assert PINNED_SLUG in sh
    assert len(list((out / "results").glob("bayesian-*.csv"))) == csvs_before, "a replay launches nothing"


def test_replay_refuses_a_record_with_no_winner():
    out = Path("/tmp/hc-optuna-replay-bad")
    out.mkdir(parents=True, exist_ok=True)
    bad = out / "empty.record.json"
    bad.write_text(json.dumps({"schema": "hypercore.tuning_history/1", "trials": []}),
                   encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--model", "maya-s24", "--engine", str(STUB), "--pack", str(HERE / "fixtures"),
             "--prompt-tokens-file", str(HERE / "fixtures" / "prompt.json"),
             "--replay", str(bad)])


def test_a_dry_run_writes_no_deliverables():
    """Nothing booted, so there is no winner to hand to the launcher."""
    out = Path("/tmp/hc-optuna-dry")
    code, _ = _run(["--gen", "--dry-run"], out)
    assert code == 0
    assert not (out / "run-maya-maya-s24.sh").exists()
    assert not (out / "maya-maya-s24.json").exists()
    assert _record(out)["verified"] is False


def test_a_measured_run_marks_its_record_verified():
    code, out = _run(["--gen"])
    assert code == 0
    record = _record(out)
    assert record["verified"] is True
    assert record["metrics"]["gen_tok_s"] > 0
    assert record["source"] == "study"


def test_the_deliverable_says_where_the_winner_came_from():
    """A prior carried in from history is labelled as such, not as a fresh boot."""
    out = Path("/tmp/hc-optuna-source")
    assert _run(["--gen"], out)[0] == 0

    code, _ = _run(["--gen", "--dry-run"], out, fresh=False)
    assert code == 0
    record = _record(out)
    assert record["source"] == "history"
    assert record["verified"] is True
    assert record["study"]["measured"] == 0
    cfg = json.loads((out / "maya-maya-s24.json").read_text(encoding="utf-8"))
    assert cfg["tuned_by"]["source"] == "history"
    assert cfg["tuned_by"]["metrics"]["gen_tok_s"] > 0


def test_from_records_rebuilds_the_pair_from_the_best_measured_row():
    out = Path("/tmp/hc-optuna-records")
    _run(["--gen"], out)
    (out / "run-maya-maya-s24.sh").unlink()
    (out / "maya-maya-s24.json").unlink()

    rows = [r for csv_path in (out / "results").glob("bayesian-*maya-s24-gen-*.csv")
            for r in csv.DictReader(csv_path.open(encoding="utf-8"))
            if r["status"] == "PASS"]
    expected = max(rows, key=lambda r: float(r["gen_tok_s"]))

    code, _ = _run(["--gen", "--from-records"], out, fresh=False)
    assert code == 0
    cfg = json.loads((out / "maya-maya-s24.json").read_text(encoding="utf-8"))
    sh = (out / "run-maya-maya-s24.sh").read_text(encoding="utf-8")
    assert cfg["tuned_by"]["verified"] is True
    assert cfg["tuned_by"]["metrics"]["gen_tok_s"] == float(expected["gen_tok_s"])
    assert sh.startswith("#!/bin/sh") and "maya.py" in sh
    assert "--max-context" in " ".join(cfg["args"])


def test_from_records_refuses_a_model_with_no_measurement():
    out = Path("/tmp/hc-optuna-records-empty")
    (out / "results").mkdir(parents=True, exist_ok=True)
    with pytest.raises(SystemExit):
        main(["--model", "maya-s24", "--engine", str(STUB), "--pack", str(HERE / "fixtures"),
             "--prompt-tokens-file", str(HERE / "fixtures" / "prompt.json"),
             "--results-dir", str(out / "results"), "--from-records"])


def test_model_paths_resolve_to_the_account_that_owns_the_data(monkeypatch):
    """The pack, the GGUFs and the draft layer come from the data folder this user's
    installer recorded, never from a neighbour account's copy of the same model."""
    tmp_path = Path("/tmp/hc-data-roots")
    shutil.rmtree(tmp_path, ignore_errors=True)
    cfg_dir = tmp_path / "config"
    (cfg_dir / "project-maya").mkdir(parents=True)
    mine = tmp_path / "home" / "ai-user0" / "Maya-data"
    sibling = tmp_path / "home" / "ai-user0" / "Maya-data-alt"
    other = tmp_path / "home" / "ai-agent" / "Maya-data-agent"
    (cfg_dir / "project-maya" / "settings.json").write_text(
        json.dumps({"data_dir": str(mine)}), encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_dir))
    monkeypatch.delenv("MAYA_DATA", raising=False)

    for base in (other, sibling, mine):          # the same model, three places
        pack = base / "models" / "Maya-S24" / "pack"
        pack.mkdir(parents=True)
        (pack / "index.txt").write_text("x", encoding="utf-8")
        (pack / "tokenizer").mkdir()
        (pack / "tokenizer" / "vocab.json").write_text("{}", encoding="utf-8")
        (base / "data").mkdir()
        (base / "data" / "expert-profile-glm.bin").write_text("x", encoding="utf-8")
    repo = tmp_path / "home" / "ai-user0" / "HyperCore-Tune-Maya"
    repo.mkdir(parents=True)

    roots = data_roots(repo)
    assert roots[0] == str(mine), roots          # the recorded data_dir wins
    assert str(sibling) in roots                 # a sibling is a fallback, not the first
    assert str(other) not in roots               # another account is not searched

    paths = resolve_model_paths(repo, "maya-s24")
    assert paths.pack == str(mine / "models" / "Maya-S24" / "pack")
    assert paths.tokenizer == str(mine / "models" / "Maya-S24" / "pack" / "tokenizer")
    assert paths.expert_profile == str(mine / "data" / "expert-profile-glm.bin")

    # --data-dir pins it, and an empty or missing folder is dropped rather than used
    pinned = resolve_model_paths(repo, "maya-s24", str(other))
    assert pinned.pack == str(other / "models" / "Maya-S24" / "pack")
    (empty := tmp_path / "home" / "ai-user0" / "empty-data").mkdir()
    assert str(empty) not in data_roots(repo)
    with pytest.raises(FileNotFoundError):
        resolve_model_paths(repo, "q9_9")


def test_the_deliverable_tokenizer_matches_the_pack_in_use():
    """A base config naming another account's tokenizer must not survive the sweep."""
    pack = Path("/tmp/hc-token-pack")
    (pack / "tokenizer").mkdir(parents=True, exist_ok=True)
    out = Path("/tmp/hc-optuna-tokenizer")
    shutil.rmtree(out, ignore_errors=True)
    (out / "results").mkdir(parents=True)
    code = main([
        "--model", "maya-s24", "--engine", str(STUB), "--maya-root", str(FAKE_ROOT),
        "--pack", str(pack),
        "--prompt-tokens-file", str(HERE / "fixtures" / "prompt.json"),
        "--deliverable-dir", str(out), "--results-dir", str(out / "results"),
        "--history-dir", str(out / "history"),
        "--max-workers", "8", "--trials", "3", "--seed", "7", "--gen",
    ])
    assert code == 0
    cfg = json.loads((out / "maya-maya-s24.json").read_text(encoding="utf-8"))
    args = cfg["args"]
    assert args[args.index("--glm-pack") + 1] == str(pack)
    assert cfg["tokenizer"] == str(pack / "tokenizer")
