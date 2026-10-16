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
from bayesian_tune import main, parse_args, signed_score
from engine_runner import (
    EngineError,
    EngineSession,
    build_prompt,
    failure_reason,
    is_ready,
    parse_done,
    parse_info,
    parse_ready,
    render_chat_prompt,
    run_config,
    vocab_size,
)
from maya_env import (
    abs_path,
    candidate_roots,
    engine_binary,
    maya_root,
    server_engine_name,
)
from run_config import clean_args
from search_space import (
    DEFAULT_CONFIG,
    MTP_MAX_T_RANGE,
    PARALLEL_RANGE,
    PCIE_RANGE,
    PLE_INFLIGHT_RANGE,
    ROW_COLUMNS,
    SUFFIX_DRAFT_RANGE,
    assert_params_match_engine_bounds,
    build_space,
    data_roots,
    engine_args,
    kv_modes_for,
    pack_kind,
    pack_missing_shards,
    pack_shard_names,
    parse_engine_bounds,
    quasi_random_configs,
    request_keys,
    resolve_model_paths,
)

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
else if (a == "--ple-inflight") o.ple_inflight = std::atoi(next("--ple-inflight"));
else if (a == "--suffix-draft") o.suffix_draft = std::max(0, std::atoi(next("--suffix-draft")));
else if (a == "--mtp-max-t") o.mtp_max_t = std::max(0, std::atoi(next("--mtp-max-t")));
else if (a == "--parallel") o.parallel = std::atoi(next("--parallel"));
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
    cfg = json.loads((out / "max_maya-s24_config.json").read_text(encoding="utf-8"))
    args = cfg["args"]
    assert args[args.index("--kv") + 1] == "int8"
    assert args[args.index("--pool-workers") + 1] == "4"
    assert args[args.index("--spec-min-p") + 1] == "0.5"
    assert args[args.index("--prompt-cache") + 1] in {"0", "2", "4", "6"}


def test_engine_args_covers_every_swept_dimension():
    cfg = {"vram": 768, "pcie": 0.9, "pcie_mode": "dma", "ple_io": "mmap", "kv": "int8",
           "kvstream": "on", "hostworker": "0", "worker": 12, "tail": 4,
           "draft_floor": 0.5, "ple_inflight": 64, "suffix_draft": 4, "mtp_max_t": 2,
           "parallel": 3}
    args = engine_args(cfg, {"pack": "/p"}, 131072, 512)
    for flag, value in (("--vram-reserve-mib", "768"), ("--pcie-frac", "0.9"),
                        ("--pcie-mode", "dma"), ("--ple-io", "mmap"), ("--kv", "int8"),
                        ("--kv-streaming", "on"), ("--pool-workers", "12"),
                        ("--prompt-cache", "4"), ("--ple-inflight", "64"),
                        ("--suffix-draft", "4"), ("--mtp-max-t", "2"), ("--parallel", "3"),
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


def test_k8v4_is_never_asked_to_stream():
    """generate.cpp:3569 exits 2 on --kv k8v4 with a kv-resident window, and
    --kv-streaming on sets one (3405), so that pair is not an arm: it is a dead boot.
    """
    args = engine_args({"kv": "k8v4", "kvstream": "on"}, {"pack": "/p"}, 8192, 256)
    assert args[args.index("--kv-streaming") + 1] == "off"
    # An explicit off stays off; nothing is turned on behind the operator's back.
    args = engine_args({"kv": "k8v4", "kvstream": "off"}, {"pack": "/p"}, 8192, 256)
    assert args[args.index("--kv-streaming") + 1] == "off"


def test_a_request_that_does_not_fit_the_context_is_refused_before_booting():
    """generate.cpp:3624 rejects it with exit 2; raising here costs no engine boot."""
    for context, max_new in ((0, 512), (2**31, 512), (4096, 8192), (4096, 0)):
        try:
            engine_args({}, {"pack": "/p"}, context, max_new)
        except ValueError as exc:
            assert "max-new" in str(exc).lower() or "cntx" in str(exc).lower()
        else:
            raise AssertionError(f"{context}/{max_new} should not have been accepted")


def test_the_space_is_the_columns_the_engine_has():
    params = build_space(max_workers=8)
    keys = {p.key for p in params}
    assert keys == {"vram", "pcie", "pcie_mode", "ple_io", "ple_inflight", "kv", "kvstream",
                    "hostworker", "worker", "tail", "draft_floor", "suffix_draft",
                    "mtp_max_t", "parallel"}
    assert ROW_COLUMNS[0] == "layer" and ROW_COLUMNS[-1] == "status"
    assert ROW_COLUMNS[1:-1] == tuple(p.key for p in params)
    floor = next(p for p in params if p.key == "draft_floor")
    assert floor.low == 0.001 and floor.high == 1.0 and floor.flag == "--spec-min-p"
    # The worker grid is capped at physical cores, and 0 (the engine's "all cores") is a legal arm.
    worker = next(p for p in params if p.key == "worker")
    assert worker.grid()[0] == 0 and worker.grid()[-1] == 7


def _fixture_pack(tmp: Path, header: str) -> Path:
    pack = tmp / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "index.txt").write_text(header, encoding="utf-8")
    return pack


def test_the_kv_grid_is_the_runner_the_pack_drives():
    """A --glm-pack has an fp16/INT8 latent cache and nothing else; sweeping q4_0 and k8v4
    against it measures the same run twice (generate.cpp:3443 before 3557-3564).
    """
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        glm = _fixture_pack(tmp, "# strata pack index v3 -- generated by tools/iq_pack.py "
                                      "(native experts) from GLM-5.3-Flash-Maya-S24-00001-of-00003.gguf\n")
        qwen = _fixture_pack(tmp / "qwen", "# strata pack index v3 -- generated by tools/iq_pack.py "
                                           "(native experts) from Qwen3.8-Flash-Next-00001-of-00002.gguf\n")
        assert pack_kind(glm) == "glm" and pack_kind(qwen) == "qwen"
        assert kv_modes_for(glm) == ("int8", "fp16")
        assert kv_modes_for(qwen) == ("fp16", "int8", "q4_0", "k8v4")
        glm_space = {p.key: p for p in build_space(pack=glm, max_workers=4)}
        qwen_space = {p.key: p for p in build_space(pack=qwen, max_workers=4)}
        assert glm_space["kv"].choices == ("int8", "fp16")
        assert qwen_space["kv"].choices == ("fp16", "int8", "q4_0", "k8v4")
        # --parallel is the GLM runner's sequence-state count; a qwen session has one.
        assert "parallel" in glm_space and "parallel" not in qwen_space


def test_a_pack_that_cannot_be_read_is_swept_as_a_glm_pack():
    assert pack_kind(None) == "glm"
    assert pack_kind("/no/such/pack") == "glm"
    assert kv_modes_for("/no/such/pack") == ("int8", "fp16")


def test_a_drafter_leaves_the_mtp_and_suffix_arms_out():
    """With --dflash the request decodes through Glm5Model::decode_dflash
    (generate.cpp:1842); the MTP/suffix path is never reached, so those two dimensions
    would measure a no-op for every trial.
    """
    keys = {p.key for p in build_space(max_workers=4, has_drafter=True)}
    assert {"suffix_draft", "mtp_max_t"} & keys == set()
    keys = {p.key for p in build_space(max_workers=4, has_drafter=False)}
    assert {"suffix_draft", "mtp_max_t"} <= keys


def test_every_grid_point_is_inside_the_engine_bounds():
    """An arm outside generate.cpp's range is an exit 2, not a slower configuration."""
    for p in build_space(max_workers=8):
        grid = p.grid()
        assert grid, f"{p.key} has an empty grid"
        if p.kind == "int":
            assert int(p.low) <= min(int(v) for v in grid) and max(int(v) for v in grid) <= int(p.high)
        elif p.kind == "float":
            assert float(p.low) <= min(float(v) for v in grid) and max(float(v) for v in grid) <= float(p.high)
        else:
            assert set(p.choices) <= {str(c) for c in p.choices}
    assert PARALLEL_RANGE == (1, 8)          # generate.cpp:3193
    assert PLE_INFLIGHT_RANGE == (1, 1024)  # generate.cpp:3553
    assert SUFFIX_DRAFT_RANGE[0] == 0 and MTP_MAX_T_RANGE[0] == 0
    assert next(p for p in build_space() if p.key == "parallel").grid() == list(range(1, 9))
    assert next(p for p in build_space() if p.key == "ple_inflight").grid() == [
        16, 32, 48, 64, 80, 96, 112, 128]
    # generate.cpp:3328-3329 take any int >= 0 and the engine's own suffix_draft default is
    # 3, so the draft arms must step by 1: a coarser grid cannot express the run the engine
    # gives with no flags, and every prior written before the dimension existed is unseedable.
    sfx = next(p for p in build_space() if p.key == "suffix_draft")
    assert sfx.step == 1 and 3 in sfx.grid()
    assert next(p for p in build_space() if p.key == "mtp_max_t").step == 1


def test_the_engine_defaults_are_a_point_in_the_grid():
    """DEFAULT_CONFIG is what the engine does with no flags.

    history.py fills a key a record predates from it, so if the grid cannot hold that point
    every prior from that history is rejected by Optuna and the study starts cold.
    """
    from history import distributions_for

    for space in (build_space(max_workers=24), build_space(max_workers=24, has_drafter=True)):
        dists = distributions_for(space, DEFAULT_CONFIG)
        assert set(dists) == {p.key for p in space}, "a dimension is missing from the seed"


def test_a_prior_off_the_step_is_outside_the_grid_not_a_crash():
    """Inside the range is not the same as on the grid; say so before Optuna does."""
    from history import distributions_for

    space = [p for p in build_space(max_workers=4) if p.key == "tail"]
    cfg = dict(DEFAULT_CONFIG)
    cfg["tail"] = 5  # the tail arm is stepped by 2
    try:
        distributions_for(space, cfg)
    except ValueError as exc:
        assert "tail=5" in str(exc)
    else:
        raise AssertionError("a value off the step must not reach create_trial")


def _glm_pack(dir_name: str, pool_bytes: int = 943911040) -> Path:
    """A pack directory with the header a real GLM pack carries."""
    pack = Path(tempfile.mkdtemp(prefix=f"hc-{dir_name}-"))
    (pack / "index.txt").write_text(
        "# strata pack index v3 -- generated by tools/iq_pack.py (native experts)\n"
        f"# align 64 pool {pool_bytes} tensors 1257\n",
        encoding="utf-8",
    )
    return pack


def test_a_kv_the_pack_ignores_is_read_as_the_run_it_measured():
    """generate.cpp:3444 returns before the --kv arms, so on a GLM pack k8v4 ran fp16.

    A prior carrying a flag the runner ignores is still a measurement of something, and
    history has to name that something for it to seed the study.
    """
    from history import normalise_trial
    from search_space import effective_kv

    pack = _glm_pack("glm-kv")
    assert effective_kv(pack, "k8v4") == "fp16"
    assert effective_kv(pack, "q4_0") == "fp16"
    assert effective_kv(pack, "int8") == "int8"

    row = {"cfg": {k: v for k, v in DEFAULT_CONFIG.items() if k != "kv"} | {"kv": "k8v4"},
           "metrics": {"gen_tok_s": 31.0}, "score": 31.0, "status": "OK"}
    prior = normalise_trial(row, 512, 0.8, pack=pack)
    assert prior is not None
    assert prior["cfg"]["kv"] == "fp16"
    assert prior["signature"] == [str(prior["cfg"][k]) for k in sorted(prior["cfg"])]


def test_two_records_that_mean_one_configuration_seed_one_prior():
    from history import dedupe, normalise_trial

    pack = _glm_pack("glm-dedupe")
    rows = [{"cfg": {**DEFAULT_CONFIG, "kv": kv}, "metrics": {"gen_tok_s": score},
             "score": score, "status": "OK"} for kv, score in (("k8v4", 20.0), ("fp16", 25.0))]
    priors = [p for p in (normalise_trial(r, 512, 0.8, pack=pack) for r in rows) if p]
    assert len(priors) == 2
    merged = dedupe(priors)
    assert len(merged) == 1
    assert merged[0]["score"] == 25.0, "the better measurement of the same run wins"


def test_the_pack_header_states_the_vram_floor_the_gpu_must_clear():
    from search_space import pack_vram_floor_mib

    assert pack_vram_floor_mib(_glm_pack("floor", 943911040)) == 900
    assert pack_vram_floor_mib(Path(tempfile.mkdtemp())) == 0, "no index.txt, no claim"


def test_preflight_refuses_a_gpu_that_cannot_host_the_pack(monkeypatch):
    import argparse

    import bayesian_tune
    from search_space import ModelPaths

    args = argparse.Namespace(engine="/home/you/project-maya/build/strata",
                             dry_run=False, no_preflight=False)
    paths = ModelPaths(pack=str(_glm_pack("preflight")))
    hw = bayesian_tune._hardware(PINNED)

    monkeypatch.setattr(bayesian_tune, "free_vram_mib", lambda: 360)
    try:
        bayesian_tune.preflight(args, paths, hw)
    except SystemExit as exc:
        assert "900 MiB" in str(exc) and "360 MiB is free" in str(exc)
    else:
        raise AssertionError("a GPU with 360 MiB free cannot host a 900 MiB pool")

    monkeypatch.setattr(bayesian_tune, "free_vram_mib", lambda: 24000)
    bayesian_tune.preflight(args, paths, hw)  # room: nothing to say

    monkeypatch.setattr(bayesian_tune, "free_vram_mib", lambda: 0)
    bayesian_tune.preflight(args, paths, hw)  # nothing reports: do not guess
    args.dry_run = True
    monkeypatch.setattr(bayesian_tune, "free_vram_mib", lambda: 360)
    bayesian_tune.preflight(args, paths, hw)  # a dry run boots nothing


def test_a_repeated_failure_stops_the_sweep_instead_of_scoring_zeros():
    """Every unmeasurable trial scores PENALTY, so a busy GPU looks like a flat landscape.

    The same failure three times in a row is the machine, not the search; the run should
    say so and stop rather than spend the rest of the budget on identical zeros.
    """
    out = Path("/tmp/hc-fail-streak")
    engine = FAKE_ROOT / "always_fails.py"
    engine.write_text("print('maya generate: pack: out of memory')\n", encoding="utf-8")
    _run(["--trials", "12", "--seed", "4", "--engine", str(engine), "--no-deliverables"], out)
    record = _record(out)
    assert record["study"]["trials"] == 3, "stopped on the third identical failure"
    assert all(t["status"] == "FAIL_BOOT" for t in record["trials"])
    assert record["interrupted"] is False, "a stopped run is not an interrupted one"


def test_every_flag_in_the_space_is_parsed_by_the_engine_source():
    """The engine's own parse table and clamp lines are the authority.

    A flag it does not name is a typo; a bound in the search grid that crosses a clamp
    would become an exit-2 trial at runtime instead of a test failure.
    Skipped when no Maya checkout is on this machine.
    """
    import re
    src = Path("/home/ai-user0/project-maya/src/program/generate.cpp")
    if not src.is_file():
        return
    source = src.read_text(encoding="utf-8", errors="replace")
    known = set(re.findall(r'a == "(--[a-z0-9-]+)"', source))
    for p in build_space():
        if p.flag is None:
            continue
        assert p.flag in known, f"{p.flag} is not parsed by generate.cpp"
    bounds = parse_engine_bounds(source)
    assert bounds["--parallel"] == PARALLEL_RANGE
    assert bounds["--ple-inflight"] == PLE_INFLIGHT_RANGE
    assert bounds["--pcie-frac"] == PCIE_RANGE
    assert bounds["--suffix-draft"][0] == SUFFIX_DRAFT_RANGE[0]
    assert bounds["--mtp-max-t"][0] == MTP_MAX_T_RANGE[0]
    assert_params_match_engine_bounds(build_space(), bounds)


def test_the_stub_refuses_what_the_engine_refuses():
    for bad in (["--serve", "--glm-pack", "/p", "--kv", "k8v4"],
                ["--serve", "--parallel", "9"],
                ["--serve", "--ple-inflight", "0"]):
        log = Path("/tmp/hc-stub-bound.log")
        m = run_config(str(STUB), bad, [1, 2, 3], 8, log,
                       boot_timeout=10, run_timeout=10)
        assert m.status != "PASS", f"{bad} should not have measured"
        assert "glm error" in log.read_text(encoding="utf-8"), m.note


def test_stub_engine_measures():
    ids = build_prompt(str(HERE / "fixtures" / "prompt.json"), 1024)
    args = ["--serve", "--max-context", "131072", "--kv", "int8",
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


def test_ready_is_the_only_signal_that_the_engine_will_read_a_request():
    """generate.cpp opens its GEN loop only after the READY line (2830, 5877). Sending the
    request earlier is read as garbage and never answered, so a whole sweep scores 0.
    """
    assert is_ready("READY 32768 stop pause")
    assert is_ready("READY 8192 stop")
    assert is_ready("maya generate: session is up (engine 3.1)")   # pre-M3.1 wording
    for line in ("glm fast: CUDA0 tiers warm: 5332 experts (37.9 GB) in 19 s",
                 "INFO context=32768 kv=int8 pcie_share=0.17",
                 "maya serve: CUDA0: 1495 MiB of VRAM free with everything loaded"):
        assert not is_ready(line), line
    assert parse_ready("READY 32768 stop pause") == {"ready_context": "32768"}
    assert parse_ready("READY") == {}
    assert parse_ready("INFO context=32768") == {}


def test_the_request_is_sent_after_ready_and_the_context_is_recorded():
    ids = build_prompt(str(HERE / "fixtures" / "prompt.json"), 512)
    m = run_config(str(STUB), ["--serve", "--max-context", "32768", "--max-new", "64"],
                   ids, 64, Path("/tmp/hc-ready-run.log"), boot_timeout=20, run_timeout=20)
    assert m.status == "PASS", m.note
    assert m.effective.get("ready_context") == "32768", m.effective
    log = Path("/tmp/hc-ready-run.log").read_text(encoding="utf-8")
    assert log.index("READY 32768") < log.index("DONE ")


def test_an_engine_that_boots_but_answers_nothing_is_named_as_such():
    """A 0 from a sweep must not be blamed on a boot that never happened."""
    log = ("INFO context=32768 kv=int8\nREADY 32768 stop pause\n"
           "maya generate: glm5-next serve (M3.1): GEN <max_new> [keys] <ids>; QUIT to end\n")
    assert failure_reason(log) == (
        "the engine booted and answered READY, but no DONE line came back")
    assert failure_reason("glm error: vram reserve out of range") == (
        "the engine rejected its configuration")
    assert failure_reason("") == "the engine exited with no message"


def test_objective_direction():
    from engine_runner import Metrics
    m = Metrics(status="PASS", decode_tok_s=150.0, thinking_tok_s=40.0, prefill_tok_s=900.0,
                latency_ms=4000.0)
    assert signed_score(m, "decode") == 150.0
    assert signed_score(m, "think") == 40.0
    assert signed_score(m, "prefill") == 900.0
    assert signed_score(m, "latency") == -4000.0
    failed = Metrics(status="FAIL_BOOT")
    assert signed_score(failed, "decode") == 0.0
    assert signed_score(failed, "think") == 0.0
    assert signed_score(failed, "latency") < signed_score(m, "latency")


def test_the_gen_flag_is_the_decode_objective():
    """--gen is the name older scripts use; the metric is the engine's decode phase."""
    assert parse_args(["--gen"]).objective == "decode"
    assert parse_args(["--decode"]).objective == "decode"
    assert parse_args(["--objective", "decode"]).objective == "decode"
    assert parse_args(["--think", "--think-budget", "64"]).objective == "think"
    try:
        parse_args(["--think"])
    except SystemExit:
        pass
    else:
        raise AssertionError("--think without --think-budget has nothing to measure")


def test_reasoning_split_is_the_timeline_up_to_the_think_end_token():
    from engine_runner import reasoning_split
    # monotonic-clock seconds, as the session records them
    times = [(0.0, 1000), (0.010, 1001), (0.020, 151667), (0.030, 1002)]
    assert reasoning_split(times, 4, 100.0, 151667) == (3, 20.0, 1, 80.0)
    assert reasoning_split(times, 4, 100.0, None) == (0, 0.0, 0, 0.0)
    assert reasoning_split([], 4, 100.0, 151667) == (0, 0.0, 0, 0.0)
    # A timeline that runs past the engine's own decode_ms is clamped to it.
    assert reasoning_split([(0.0, 1), (0.500, 151667)], 2, 100.0, 151667) == (2, 100.0, 0, 0.0)


def test_a_request_with_a_thinking_budget_reports_its_reasoning_block():
    ids = [1, 2, 3]
    log = Path("/tmp/hc-thinking-split.log")
    with EngineSession(str(STUB), ["--serve", "--max-context", "4096", "--delay", "0.0"],
                      log, boot_timeout=10, run_timeout=20) as session:
        plain = session.request(ids, 32)
        reasoned = session.request(ids, 32, {"think_budget": 10, "think_end": 151667})
    assert plain.status == "PASS", plain.note
    assert not plain.thinking and plain.thinking_tokens == 0 and plain.thinking_tok_s == 0.0
    assert reasoned.status == "PASS", reasoned.note
    assert reasoned.thinking
    # ten reasoning tokens, then the token that closes the block, then the answer
    assert reasoned.thinking_tokens == 11
    assert reasoned.answer_tokens == 32 - reasoned.thinking_tokens
    assert 0.0 < reasoned.think_ms < reasoned.decode_ms
    assert abs(reasoned.think_ms + reasoned.answer_ms - reasoned.decode_ms) < 1.0
    assert reasoned.thinking_tok_s > 0 and reasoned.answer_tok_s > 0
    # The stub reasons at 0.6 of its answer rate, so the split has to show that.
    assert reasoned.thinking_tok_s < reasoned.answer_tok_s
    assert log.read_text(encoding="utf-8").count("\nT 151667") == 1


def test_the_think_objective_measures_the_reasoning_block_end_to_end():
    out = Path("/tmp/hc-think-sweep")
    code, out = _run(["--think", "--think-budget", "16", "--max-new", "32",
                      "--trials", "3", "--seed", "4"])
    assert code == 0
    record = _record(out)
    assert record["objective"] == "think"
    assert record["conditions"]["think_budget"] == 16
    assert record["conditions"]["think_end"] == 151667
    assert record["metrics"]["thinking_tok_s"] > 0
    winner = record["trials"][0]
    assert float(winner["thinking_tok_s"]) > 0 and float(winner["decode_tok_s"]) > 0
    # score is the signed thinking rate; the CSV rounds it to 4 dp and the rate to 2.
    assert abs(float(winner["score"]) - float(winner["thinking_tok_s"])) < 0.01


def test_a_history_written_under_the_old_gen_name_is_still_read():
    from hardware import hardware_from
    from history import find_history, normalise_trial
    hw = hardware_from(json.loads(PINNED))
    d = Path(tempfile.mkdtemp(prefix="hc-legacy-history-"))
    legacy = d / f"{PINNED_SLUG}_maya-s24_gen_131072.json"
    legacy.write_text(json.dumps({
        "schema": "hypercore.tuning_history/1",
        "model": "maya-s24", "objective": "gen", "context": 131072, "max_new": 512,
        "hardware": hw.to_dict(), "hardware_slug": PINNED_SLUG,
        "trials": [{"cfg": dict(DEFAULT_CONFIG), "score": 42.0, "status": "PASS",
                    "layer": "joint", "metrics": {"gen_tok_s": 42.0}}],
    }), encoding="utf-8")
    found = find_history(d, "maya-s24", "decode", 131072, hw)
    assert found == [legacy], "a pre-decode history file is not a prior"
    prior = json.loads(legacy.read_text(encoding="utf-8"))["trials"][0]
    trial = normalise_trial(prior, 512, 0.8)
    assert trial["metrics"]["decode_tok_s"] == 42.0


TEST_CNTX = "131072"  # pinned so a sweep never inherits the CLI default


def _run(extra, out: Path = Path("/tmp/hc-optuna-test"), fresh: bool = True):
    """One sweep against the stub engine, in a directory of its own."""
    if fresh:
        shutil.rmtree(out, ignore_errors=True)
    argv = ["--model", "maya-s24", "--engine", str(STUB), "--maya-root", str(FAKE_ROOT),
            "--cntx", TEST_CNTX,
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
    sh = out / "max_maya-s24_run.sh"
    js = out / "max_maya-s24_config.json"
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
    assert tuned["objective"] == "decode"
    assert tuned["context"] == 131072
    for key in ("vram", "pcie", "ple_io", "kv", "kvstream", "hostworker", "pcie_mode",
                "worker", "tail", "draft_floor"):
        assert key in tuned["best"]
    args = cfg["args"]
    assert "--spec-min-p" in args
    floor = float(args[args.index("--spec-min-p") + 1])
    assert 0.001 <= floor <= 1.0
    assert tuned["metrics"]["decode_tok_s"] > 0
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
    assert header.split(",")[:len(ROW_COLUMNS)] == list(ROW_COLUMNS)


def test_the_search_is_deterministic_for_a_seed():
    _run(["--gen"])
    first = _record(Path("/tmp/hc-optuna-test"))
    _run(["--gen"])
    second = _record(Path("/tmp/hc-optuna-test"))
    assert first["best"] == second["best"]
    assert first["metrics"]["decode_tok_s"] == second["metrics"]["decode_tok_s"]


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
    locked = blocked_dir / "max_maya-s24_run.sh"
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
    assert (out / "results" / "max_maya-s24_run.sh").exists()
    assert (out / "results" / "max_maya-s24_config.json").exists()


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
    assert (out / "max_maya-s24_run.sh").exists()
    _bash_n(out / "max_maya-s24_run.sh")


def test_the_generated_launcher_is_the_project_startup_pair():
    """max-<model>_run.sh is the startup script a Maya checkout would run."""
    out = Path("/tmp/hc-optuna-launcher")
    main([
        "--model", "maya-s24", "--engine", str(STUB), "--maya-root", str(FAKE_ROOT),
        "--pack", str(HERE / "fixtures"),
        "--prompt-tokens-file", str(HERE / "fixtures" / "prompt.json"),
        "--deliverable-dir", str(out), "--results-dir", str(out / "results"),
        "--max-workers", "8", "--trials", "3", "--seed", "2", "--gen", "--port", "8090",
    ])
    launcher = out / "max_maya-s24_run.sh"
    cfg_path = out / "max_maya-s24_config.json"
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
    bat = (out / "max_maya-s24_run.bat").read_text(encoding="utf-8")
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
    cfg = json.loads((out / "max_maya-s24_config.json").read_text(encoding="utf-8"))
    assert "turn_tail_check" not in cfg, "a key nobody reads is dead weight"
    launcher_path = out / "max_maya-s24_run.sh"
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
        launcher = (out / "max_maya-s24_run.sh").read_text(encoding="utf-8")
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
    text = (out / "max_maya-s24_run.sh").read_text()
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
    for rule in ("results/", "history/", "max_*_run.sh", "max_*_config.json", "max_*_run.bat"):
        assert rule in own, f"missing ignore rule in the repository: {rule}"
    project = Path(os.environ["MAYA_ROOT"]) / ".gitignore"
    if project.is_file():
        gitignore = project.read_text(encoding="utf-8")
        for rule in ("/max_*_run.sh", "/max_*_config.json", "/max_*_run.bat"):
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


def test_the_context_default_is_32768_and_can_be_overridden():
    """The sweep's default context is the one the README and max_tune.sh advertise.

    `--context` is an alias of `--cntx`; a later flag wins over an earlier one.
    """
    assert parse_args(["--gen"]).cntx == 32768
    assert parse_args(["--gen", "--cntx", "131072"]).cntx == 131072
    assert parse_args(["--gen", "--context", "8192"]).cntx == 8192
    assert parse_args(["--gen", "--cntx", "8192", "--context", "32768"]).cntx == 32768
    # both spellings land in the same destination; there is no separate args.context
    assert not hasattr(parse_args(["--gen", "--context", "8192"]), "context")


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
    cfg = json.loads((out / "max_maya-s24_config.json").read_text(encoding="utf-8"))
    assert Path(cfg["exe"]).is_absolute()
    assert Path(cfg["args"][cfg["args"].index("--glm-pack") + 1]).is_absolute()


def test_cpu_probe_survives_without_proc_or_uname(monkeypatch):
    import hardware as hardware_mod
    from hardware import detect_cpu
    monkeypatch.setattr(hardware_mod, "_read", lambda path: "")
    def _no_uname():
        raise AttributeError("uname")
    monkeypatch.setattr(os, "uname", _no_uname, raising=False)
    name, _, logical = detect_cpu()
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
    cfg = json.loads((out / "max_maya-s24_config.json").read_text(encoding="utf-8"))
    record = _record(out)
    for key in ("exe", "cwd", "log", "tokenizer", "model_name"):
        assert cfg.get(key), f"{key} is empty"
    for key in ("exe", "cwd", "log", "tokenizer"):
        assert Path(cfg[key]).is_absolute(), f"{key} is not absolute: {cfg[key]}"
    for flag in ("--glm-pack", "--dflash", "--expert-profile"):
        if flag in cfg["args"]:
            value = cfg["args"][cfg["args"].index(flag) + 1]
            assert Path(value).is_absolute(), f"{flag} is relative: {value}"
    for key in ("engine", "csv"):
        assert Path(record[key]).is_absolute(), f"{key} is not absolute: {record[key]}"
    assert Path(record["history"]["file"]).is_absolute()
    launcher = (out / "max_maya-s24_run.sh").read_text(encoding="utf-8")
    assert f'cd "{FAKE_ROOT}"' in launcher
    for line in launcher.splitlines():
        if line.startswith(("cd ", "exec ")):
            assert Path(line.split('"')[1]).is_absolute(), f"relative in launcher: {line}"


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
    assert [f.name for f in files] == [f"{PINNED_SLUG}_maya-s24_decode_131072.json"]
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["schema"] == "hypercore.tuning_history/1"
    assert data["hardware_slug"] == PINNED_SLUG
    assert data["hardware"]["cpu"] == "Test CPU X1"
    assert data["context"] == 131072 and data["objective"] == "decode"
    assert len(data["trials"]) == 3
    assert all(set(t["cfg"]) == set(ROW_COLUMNS[1:-1]) for t in data["trials"]), \
        "every stored trial carries the whole engine dimension set"
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
    assert (out / "max_maya-s24_run.sh").exists() and (out / "max_maya-s24_config.json").exists()


def test_a_point_measured_under_another_objective_is_not_booted_again():
    """A configuration is the same fourteen flags whatever score was put on it.

    History is keyed by objective, so a --balance sweep after a --decode sweep would boot
    the identical configurations a second time to measure the same thing. The other
    objective's configurations are carried into `seen`; they are not priors, because a
    decode score is not a balance score.
    """
    out = Path("/tmp/hc-optuna-cross-objective")
    base = ["--hardware", PINNED, "--sampler", "random", "--trials", "6", "--seed", "3"]
    _run(base + ["--decode"], out)
    decode_cfgs = {tuple(sorted(r["cfg"].items())) for r in _record(out)["trials"]}
    assert len(decode_cfgs) == 6

    code, _ = _run(base + ["--balance"], out, fresh=False)
    assert code == 0
    record = _record(out)
    assert record["history"]["foreign_seen"] == 6, "the decode points must be known to the study"
    assert record["study"]["measured"] == 6, "--trials still means six new boots"
    balance_cfgs = {tuple(sorted(r["cfg"].items())) for r in record["trials"]}
    assert not decode_cfgs & balance_cfgs, "the balance sweep re-measured decode's points"
    assert len({p for p in (out / "results").glob("run-*.log")}) == 12
    assert record["history"]["trials_loaded"] == 0, "balance has no priors of its own yet"

    # Turning the carry-over off puts the duplicates back: a cold objective with no history
    # of its own re-boots the points another objective already measured.
    code, _ = _run(base + ["--latency", "--no-cross-objective"], out, fresh=False)
    assert code == 0
    record = _record(out)
    assert record["history"]["foreign_seen"] == 0
    again = {tuple(sorted(r["cfg"].items())) for r in record["trials"]}
    assert again & decode_cfgs, "--no-cross-objective must allow a re-measure"


def test_a_foreign_prior_is_scored_for_the_objective_being_run():
    """The number a prior carries belongs to the objective that produced it."""
    from bayesian_tune import prior_score
    from engine_runner import Metrics

    m = Metrics(status="PASS", decode_tok_s=8.0, prefill_tok_s=300.0, latency_ms=1000.0)
    m.transfer = {"transfer_share": 0.5}
    assert prior_score(m, "decode") == 8.0
    assert prior_score(m, "balance") == 8.0 * (1 - 0.35 * 0.5)
    assert prior_score(m, "prefill") == 300.0
    assert prior_score(m, "latency") == -1000.0
    # No telemetry, no discount: a pre-transfer record behaves like --decode.
    bare = Metrics(status="PASS", decode_tok_s=8.0)
    assert prior_score(bare, "balance") == 8.0


def test_other_objective_history_skips_the_current_objective_and_other_machines():
    from hardware import hardware_from
    from history import history_path, other_objective_history

    out = Path("/tmp/hc-optuna-history-filter")
    shutil.rmtree(out, ignore_errors=True)
    (out / "history").mkdir(parents=True)
    hw = hardware_from(json.loads(PINNED))
    for objective in ("decode", "balance", "latency"):
        history_path(out / "history", "maya-s24", objective, int(TEST_CNTX), hw).write_text(
            json.dumps({
                "schema": "hypercore.tuning_history/1", "model": "maya-s24",
                "objective": objective, "context": int(TEST_CNTX),
                "hardware": hw.to_dict(), "hardware_slug": PINNED_SLUG,
                "runs": [], "trials": [],
            }), encoding="utf-8")
    found = [p.name for p in other_objective_history(out / "history", "maya-s24",
                                                     int(TEST_CNTX), hw, "decode")]
    assert len(found) == 2 and not any("decode" in n for n in found)
    assert other_objective_history(out / "history", "maya-s24", 8192, hw, "decode") == []
    other = [p for p in other_objective_history(out / "history", "maya-s24", int(TEST_CNTX),
                                               hardware_from(json.loads(OTHER)), "decode",
                                               allow_mismatch=True)]
    assert len(other) == 2, "--allow-hardware-mismatch carries the other machine too"


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
    stored = json.loads((out / "history" / f"{PINNED_SLUG}_maya-s24_decode_131072.json")
                        .read_text(encoding="utf-8"))
    outside = sum(1 for t in stored["trials"] if t["cfg"]["vram"] > 256)
    assert outside > 0, "the fixture must produce a prior the narrower grid cannot hold"
    _run(["--hardware", PINNED, "--trials", "1", "--seed", "9", "--vram-max", "256",
          "--no-deliverables"], out, fresh=False)
    record = _record(out)
    assert record["history"]["trials_loaded"] == len(stored["trials"])
    assert record["history"]["priors_skipped"] == outside
    assert record["history"]["priors_used"] == len(stored["trials"]) - outside


def test_the_best_written_out_is_inside_the_grid_this_run_searches():
    """A prior the current bounds cannot hold must not become the deliverable.

    The interrupted Maya-S24 run left a launcher with --vram-reserve-mib 1280 while the
    operator had asked for --vram-max 1024: the incumbent best came from history and was
    never checked against the space being searched.
    """
    out = Path("/tmp/hc-best-in-grid")
    _run(["--hardware", PINNED, "--trials", "6", "--seed", "9", "--vram-max", "1536"], out)
    stored = json.loads((out / "history" / f"{PINNED_SLUG}_maya-s24_decode_131072.json")
                       .read_text(encoding="utf-8"))
    assert any(t["cfg"]["vram"] > 256 for t in stored["trials"]), "need a prior outside the grid"
    _run(["--hardware", PINNED, "--trials", "1", "--seed", "9", "--vram-max", "256",
          "--dry-run"], out, fresh=False)
    cfg = json.loads((out / "max_maya-s24_config.json").read_text(encoding="utf-8"))
    args = cfg["args"]
    vram = int(args[args.index("--vram-reserve-mib") + 1])
    assert vram <= 256, f"the launcher carries vram={vram}, outside the searched grid"


def test_the_deliverable_says_which_machine_it_was_tuned_on():
    out = Path("/tmp/hc-optuna-machine")
    _run(["--hardware", PINNED, "--trials", "2", "--seed", "1"], out)
    cfg = json.loads((out / "max_maya-s24_config.json").read_text(encoding="utf-8"))
    assert cfg["tuned_by"]["hardware_slug"] == PINNED_SLUG
    assert cfg["tuned_by"]["hardware"]["ram_gib"] == 64
    assert cfg["tuned_by"]["history"]["file"].endswith(f"{PINNED_SLUG}_maya-s24_decode_131072.json")
    assert PINNED_SLUG in (out / "max_maya-s24_run.sh").read_text(encoding="utf-8")


def test_replaying_a_record_rewrites_the_deliverables_without_booting_anything():
    out = Path("/tmp/hc-optuna-replay")
    _run(["--hardware", PINNED, "--trials", "4", "--seed", "3"], out)
    record = _record(out)
    best = record["best"]
    csvs_before = len(list((out / "results").glob("bayesian-*.csv")))

    code, _ = _run(["--hardware", PINNED, "--replay", str(out / "results" / "maya-maya-s24.record.json")],
                   out, fresh=False)
    assert code == 0
    cfg = json.loads((out / "max_maya-s24_config.json").read_text(encoding="utf-8"))
    sh = (out / "max_maya-s24_run.sh").read_text(encoding="utf-8")
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
    assert not (out / "max_maya-s24_run.sh").exists()
    assert not (out / "max_maya-s24_config.json").exists()
    assert _record(out)["verified"] is False


def test_a_measured_run_marks_its_record_verified():
    code, out = _run(["--gen"])
    assert code == 0
    record = _record(out)
    assert record["verified"] is True
    assert record["metrics"]["decode_tok_s"] > 0
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
    cfg = json.loads((out / "max_maya-s24_config.json").read_text(encoding="utf-8"))
    assert cfg["tuned_by"]["source"] == "history"
    assert cfg["tuned_by"]["metrics"]["decode_tok_s"] > 0


def test_from_records_rebuilds_the_pair_from_the_best_measured_row():
    out = Path("/tmp/hc-optuna-records")
    _run(["--gen"], out)
    (out / "max_maya-s24_run.sh").unlink()
    (out / "max_maya-s24_config.json").unlink()

    rows = [r for csv_path in (out / "results").glob("bayesian-*maya-s24-decode-*.csv")
            for r in csv.DictReader(csv_path.open(encoding="utf-8"))
            if r["status"] == "PASS"]
    expected = max(rows, key=lambda r: float(r["decode_tok_s"]))

    code, _ = _run(["--gen", "--from-records"], out, fresh=False)
    assert code == 0
    cfg = json.loads((out / "max_maya-s24_config.json").read_text(encoding="utf-8"))
    sh = (out / "max_maya-s24_run.sh").read_text(encoding="utf-8")
    assert cfg["tuned_by"]["verified"] is True
    assert cfg["tuned_by"]["metrics"]["decode_tok_s"] == float(expected["decode_tok_s"])
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


def _native_pack(pack: Path, shard: str) -> Path:
    """A native pack: the markers the resolver looks for, and a header naming `shard`.

    The header is the one tools/exl3_pack.py writes; the engine reads the named GGUFs from
    beside the pack directory.
    """
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "index.txt").write_text("x", encoding="utf-8")
    (pack / "native_experts.txt").write_text(
        "# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off "
        f"down_off [shard] (n_expert 8, total 100; absolute offsets in {shard}, or in the "
        "named shard beside it)\n0 42 42 0 100 1 2 3\n", encoding="utf-8")
    (pack / "tokenizer").mkdir(exist_ok=True)
    (pack / "tokenizer" / "vocab.json").write_text("{}", encoding="utf-8")
    return pack


def test_pack_shard_names_follow_the_engine_split_pattern():
    """The header names shard 1; the engine opens every -NNNNN-of-MMMMM sibling."""
    tmp = Path(tempfile.mkdtemp(prefix="hc-shard-names-"))
    pack = _native_pack(tmp / "pack", "GLM-5.3-Flash-S24-00001-of-00003.gguf")
    assert pack_shard_names(pack) == [
        "GLM-5.3-Flash-S24-00001-of-00003.gguf",
        "GLM-5.3-Flash-S24-00002-of-00003.gguf",
        "GLM-5.3-Flash-S24-00003-of-00003.gguf",
    ]
    (pack / "native_experts.txt").write_text(
        "# absolute offsets in GLM-5.3-Flash-S24.gguf\n0 42 42 0 1 1 1 1\n", encoding="utf-8")
    assert pack_shard_names(pack) == ["GLM-5.3-Flash-S24.gguf"]
    assert pack_shard_names(tmp / "no-such-pack") == []


def test_a_pack_whose_shards_are_not_beside_it_is_not_selected():
    """A pack built from GGUFs that live in another folder cannot be booted: the engine
    opens pack_dir + "/.." + shard and throws before the session starts. Refusing it here
    is what stops a sweep from spending every trial on a boot that cannot happen."""
    tmp = Path("/tmp/hc-pack-shards")
    shutil.rmtree(tmp, ignore_errors=True)
    legacy = _native_pack(tmp / "Strata-data" / "packs" / "q2_0", "Qwen-00001-of-00002.gguf")
    (tmp / "Strata-data" / "models" / "Q2_0").mkdir(parents=True)
    (tmp / "Strata-data" / "models" / "Q2_0" / "Qwen-00001-of-00002.gguf").write_text("x")
    maya = _native_pack(tmp / "Maya-data" / "models" / "Maya-S24" / "pack",
                        "GLM-00001-of-00002.gguf")
    for shard in ("GLM-00001-of-00002.gguf", "GLM-00002-of-00002.gguf"):
        (maya.parent / shard).write_text("x", encoding="utf-8")
    repo = tmp / "HyperCore-Tune-Maya"
    repo.mkdir()

    assert pack_missing_shards(legacy) == ["Qwen-00001-of-00002.gguf", "Qwen-00002-of-00002.gguf"]
    assert pack_missing_shards(maya) == []
    assert resolve_model_paths(repo, "maya-s24").pack == str(maya)
    with pytest.raises(FileNotFoundError) as raised:
        resolve_model_paths(repo, "q2_0")
    assert str(legacy) in str(raised.value) and "not beside it" in str(raised.value)


def test_the_maya_data_folder_is_searched_before_a_legacy_one(monkeypatch):
    """The engine is Project Maya's, so Maya's own data folder is searched first even when
    an older release recorded a different one."""
    tmp = Path("/tmp/hc-data-order")
    shutil.rmtree(tmp, ignore_errors=True)
    cfg_dir = tmp / "config" / "project-maya"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "settings.json").write_text(
        json.dumps({"data_dir": str(tmp / "Strata-data")}), encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp / "config"))
    monkeypatch.delenv("MAYA_DATA", raising=False)
    (tmp / "Strata-data" / "packs").mkdir(parents=True)
    (tmp / "Maya-data" / "models").mkdir(parents=True)
    repo = tmp / "HyperCore-Tune-Maya"
    repo.mkdir()

    roots = data_roots(repo)
    assert roots[0] == str(tmp / "Maya-data"), roots
    assert str(tmp / "Strata-data") in roots, roots
    monkeypatch.setenv("MAYA_DATA", str(tmp / "Strata-data"))
    assert data_roots(repo)[0] == str(tmp / "Strata-data")


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
    cfg = json.loads((out / "max_maya-s24_config.json").read_text(encoding="utf-8"))
    args = cfg["args"]
    assert args[args.index("--glm-pack") + 1] == str(pack)
    assert cfg["tokenizer"] == str(pack / "tokenizer")

def test_bound_parser_reads_clamps_and_validation_lines():
    """The parser must see the engine's own bounds, not just the flag names."""
    src = r'''
        else if (a == "--parallel") {
            if (o.parallel < 1 || o.parallel > 8) { std::cerr << "--parallel takes 1..8\n"; return 2; }
        }
        else if (a == "--ple-inflight") {
            if (o.ple_inflight < 1 || o.ple_inflight > 1024) { std::cerr << "--ple-inflight takes 1..1024\n"; return 2; }
        }
        else if (a == "--suffix-draft") o.suffix_draft = std::max(0, std::atoi(next("--suffix-draft")));
        else if (a == "--mtp-max-t") o.mtp_max_t = std::max(0, std::atoi(next("--mtp-max-t")));
        else if (key == "pcie_frac") req_pcie_frac = std::clamp((double) fv, 0.0, 1.0);
        else if (key == "spec_min_p") req_spec_min_p = std::clamp((double) fv, 0.0, 1.0);
        if (o.vram_reserve_mib < 0) o.vram_reserve_mib = 0;
    '''
    bounds = parse_engine_bounds(src)
    assert bounds["--parallel"] == (1.0, 8.0)
    assert bounds["--ple-inflight"] == (1.0, 1024.0)
    assert bounds["--pcie-frac"] == (0.0, 1.0)
    assert bounds["--spec-min-p"] == (0.0, 1.0)
    assert bounds["--suffix-draft"] == (0.0, float("inf"))
    assert bounds["--mtp-max-t"] == (0.0, float("inf"))
    assert bounds["--vram-reserve-mib"] == (0.0, float("inf"))


def test_quasi_random_configs_stay_on_the_grid_and_cover_it():
    params = build_space(max_workers=4)
    configs = quasi_random_configs(params, 12, 1)
    assert configs, "a cold study needs seed points"
    for cfg in configs:
        for p in params:
            assert cfg[p.key] in p.grid(), f"{p.key}={cfg[p.key]} is outside its grid"
    spread = {p.key: {cfg[p.key] for cfg in configs} for p in params}
    assert any(len(values) > 1 for values in spread.values()), "the seed sweep is a single point"


def test_session_boot_args_omit_request_flags():
    cfg = {
        "vram": 512,
        "pcie": 0.5,
        "pcie_mode": "auto",
        "ple_io": "direct",
        "ple_inflight": 64,
        "kv": "int8",
        "kvstream": "off",
        "hostworker": "1",
        "worker": 4,
        "tail": 0,
        "draft_floor": 0.5,
    }
    args = engine_args(cfg, {"pack": "/p", "model": "/m", "dflash": ""}, 4096, 8, request_keys=True)
    assert "--pcie-frac" not in args
    assert "--spec-min-p" not in args
    assert "--vram-reserve-mib" in args
    assert request_keys(cfg) == {"pcie_frac": 0.5, "spec_min_p": 0.5}


def test_one_engine_session_can_answer_several_request_keys():
    ids = [1, 2, 3]
    log = Path("/tmp/hc-session-test.log")
    with EngineSession(
        str(STUB),
        ["--serve", "--max-context", "4096", "--pcie-frac", "0.0", "--spec-min-p", "0.1"],
        log,
        boot_timeout=10,
        run_timeout=10,
    ) as session:
        first = session.request(ids, 8)
        second = session.request(ids, 8, {
            "pcie_frac": 1.0,
            "spec_min_p": 0.5,
            "cpu_threads": 4,
            "read_chunks": 2,
            "spec": 4,
            "spec_n_min": 2,
        })
    assert first.status == "PASS", first.note
    assert second.status == "PASS", second.note
    assert second.decode_tok_s > first.decode_tok_s


def test_ambient_state_is_recorded_for_every_trial(monkeypatch):
    out = Path("/tmp/hc-ambient-test")
    fake = json.dumps({
        "free_vram_mib": 12345,
        "gpu_processes": [{"pid": 42, "name": "other-app", "used_memory_mib": 400}],
    })
    monkeypatch.setenv("TUNE_AMBIENT_JSON", fake)
    code, out = _run(["--gen", "--trials", "2", "--seed", "1"], out)
    assert code == 0
    record = _record(out)
    assert all(t["ambient"]["free_vram_mib"] == 12345 for t in record["trials"])
    assert all(t["ambient"]["gpu_processes"] for t in record["trials"])


def test_ambient_state_reports_whether_the_tier_can_stay_resident():
    """The `vram` tier of a trial is compared against the host RAM that is actually free.

    A pinned tier larger than the available RAM means the expert pool is being paged, so
    the measured rate is set by memory pressure rather than by the swept knobs. The
    arithmetic is checked against a pinned ambient block so the test needs no psutil.
    """
    from hardware import ambient_state

    monkeypatch_env = dict(os.environ)
    os.environ["TUNE_AMBIENT_JSON"] = json.dumps({
        "free_vram_mib": 12345,
        "gpu_processes": [],
        "free_ram_mb": 9000.0,
        "total_ram_mb": 31000.0,
        "swap_used_mb": 4200.0,
    })
    try:
        amb = ambient_state(1024)
        assert amb["tier_mib"] == 1024
        assert amb["tier_headroom_mb"] == 7976.0
        assert amb["tier_resident"] is True

        over = ambient_state(12000)
        assert over["tier_resident"] is False
        assert over["tier_headroom_mb"] == -3000.0

        # No tier asked for: nothing is claimed about residency.
        assert "tier_mib" not in ambient_state()
    finally:
        os.environ.update(monkeypatch_env)


def test_sweep_records_the_trial_tier_in_the_ambient_block(monkeypatch):
    out = Path("/tmp/hc-ambient-tier-test")
    monkeypatch.delenv("TUNE_AMBIENT_JSON", raising=False)
    code, out = _run(["--gen", "--trials", "2", "--seed", "7",
                      "--vram-min", "512", "--vram-max", "1024", "--vram-step", "512"], out)
    assert code == 0
    record = _record(out)
    assert record["trials"], "no trials measured"
    for trial in record["trials"]:
        amb = trial["ambient"]
        assert amb["tier_mib"] == trial["cfg"]["vram"]
        if "free_ram_mb" in amb:
            assert amb["tier_headroom_mb"] == round(amb["free_ram_mb"] - amb["tier_mib"], 1)
            assert amb["tier_resident"] == (amb["free_ram_mb"] >= amb["tier_mib"])


def test_two_stage_sweep_records_request_dimensions_in_the_trial():
    out = Path("/tmp/hc-two-stage-test")
    code, out = _run(["--two-stage", "--gen", "--trials", "3", "--seed", "4"], out)
    assert code == 0
    record = _record(out)
    assert all("cpu_threads" in t["cfg"] for t in record["trials"])
    assert all("spec" in t["cfg"] for t in record["trials"])


def test_text_prompt_file_is_tokenized_when_a_tokenizer_is_available():
    root = FAKE_ROOT / "prompt-test"
    tools = root / "tools"
    tokenizer = root / "tokenizer"
    prompts = root / "prompts"
    tools.mkdir(parents=True, exist_ok=True)
    tokenizer.mkdir(parents=True, exist_ok=True)
    prompts.mkdir(parents=True, exist_ok=True)
    (tokenizer / "vocab.json").write_text(json.dumps({"vocab": ["hello", "world"]}), encoding="utf-8")
    (tools / "strata_tokenizer.py").write_text(
        """
class Tokenizer:
    def __init__(self, ids):
        self.ids = ids
    @classmethod
    def from_dir(cls, path):
        return cls([1, 2, 3])
    def encode(self, text, parse_special=False):
        return [len(text), 7, 9]
""",
        encoding="utf-8",
    )
    prompt = prompts / "real.txt"
    prompt_text = "The quick brown fox asks about expert tiers."
    prompt.write_text(prompt_text, encoding="utf-8")
    ids = build_prompt(str(prompt), 10, tokenizer_dir=str(tokenizer), root=str(root))
    assert len(ids) == 10
    assert ids[0] == len(render_chat_prompt(prompt_text))
    assert ids == ([ids[0], 7, 9] * 4)[:10]


def test_ready_line_advertises_the_several_conversation_mode():
    """--parallel 2..8 sets STRATA_GLM_SEQS; READY says so as multi=<n> (generate.cpp:2830)."""
    assert parse_ready("READY 32768 stop pause multi=4") == {"ready_context": "32768", "multi": "4"}
    assert parse_ready("READY 8192 multi=2") == {"ready_context": "8192", "multi": "2"}
    assert parse_ready("READY 32768 stop pause") == {"ready_context": "32768"}
    assert parse_ready("READY bad multi=2") == {}


def test_a_multi_mode_done_line_is_read_by_its_rid():
    """A several-conversation engine puts the rid after the keyword (generate.cpp:2162)."""
    line = "DONE 7 512 10240 1380.4 2100.5 length 3 5 0 0 0"
    d = parse_done(line, rid=7)
    assert d["generated"] == 512 and d["prompt_tokens"] == 10240
    assert d["prompt_ms"] == 1380.4 and d["decode_ms"] == 2100.5
    assert d["finish"] == "length" and d["rid"] == 7
    with pytest.raises(EngineError):
        parse_done("DONE 8 512 10240 1380.4 2100.5 length 3 5 0 0 0", rid=7)
    # a single-conversation engine has no rid, and the table stays as it was
    assert parse_done("DONE 512 10240 1380.4 2100.5 length 3 5 0 0 0")["generated"] == 512


def test_a_parallel_sweep_sends_a_rid_and_measures_anyway():
    """End to end: the stub in multi mode refuses a GEN line without rid=<id>."""
    out = Path("/tmp/hc-optuna-multi")
    code, out = _run(["--parallel", "2", "--trials", "3", "--seed", "11", "--gen"], out)
    assert code == 0
    record = _record(out)
    assert record["study"]["measured"] >= 1, record["study"]
    logs = sorted((out / "results").glob("*.log"))
    assert logs, "the sweep wrote no engine log"
    text = logs[0].read_text(encoding="utf-8")
    assert "multi=2" in text, text[:200]          # the engine really was in multi mode
    assert "DONE 1 " in text, text[:200]          # ... and answered the rid-carrying request


def test_the_stub_refuses_a_gen_line_without_a_rid():
    session = EngineSession(
        str(STUB), ["--serve", "--max-context", "4096", "--parallel", "2"],
        Path("/tmp/hc-rid-refused.log"),
    )
    try:
        session._await_ready(time.monotonic() + 30)
        assert session.multi == 2
        session._send("GEN 8 1,2,3\n")
        done = session._consume_until_done(time.monotonic() + 30)
    finally:
        session.close()
    assert done == {"__err__": "ERR this engine runs several conversations: GEN needs rid=<id>"}


def test_vocab_size_reads_the_tokenizer_then_the_pack_index():
    pack = Path("/tmp/hc-vocab-pack")
    shutil.rmtree(pack, ignore_errors=True)
    (pack / "tokenizer").mkdir(parents=True)
    (pack / "tokenizer" / "tokenizer.json").write_text('{"vocab_size": 154880}', encoding="utf-8")
    (pack / "index.txt").write_text(
        "# strata pack index v3\noutput.weight 0 0 0 0 0 0 4096 154880 8 0 32 0 0 0 0 0 0 0\n",
        encoding="utf-8",
    )
    assert vocab_size(str(pack / "tokenizer"), str(pack)) == 154880
    assert vocab_size(None, str(pack)) == 154880
    assert vocab_size(None, None) is None


def test_a_prompt_outside_the_pack_vocabulary_is_replaced_before_it_is_sent():
    """One bad id fails the whole request ("ERR token N out of range"), so every trial
    would score as a zero. The ids are folded into range when no tokenizer is beside the
    pack, and the same prompt is produced every time."""
    prompt = str(HERE / "fixtures" / "prompt.json")
    ids = build_prompt(prompt, 32, vocab=154880)
    assert len(ids) == 32 and all(0 <= t < 154880 for t in ids)
    assert ids == build_prompt(prompt, 32, vocab=154880)
    assert build_prompt(prompt, 4)[0] == 248045          # no vocab given: sent as it is
    assert build_prompt(prompt, 4, vocab=300000) [0] == 248045


def test_a_pack_with_a_tokenizer_is_not_given_the_bundled_prompt_ids():
    """The bundled ids are another model's vocabulary, so a pack that carries its own
    tokenizer needs no prompt file at all: the sweep tokenises with that tokenizer."""
    tmp = Path("/tmp/hc-prompt-default")
    shutil.rmtree(tmp, ignore_errors=True)
    repo = tmp / "HyperCore-Tune-Maya"
    repo.mkdir(parents=True)

    with_tok = _native_pack(tmp / "Maya-data" / "models" / "Maya-S24" / "pack",
                            "GLM-00001-of-00002.gguf")
    without_tok = _native_pack(tmp / "Maya-data" / "models" / "Maya-S25" / "pack",
                               "GLM2-00001-of-00002.gguf")
    for pack, shards in ((with_tok, ("GLM-00001-of-00002.gguf", "GLM-00002-of-00002.gguf")),
                        (without_tok, ("GLM2-00001-of-00002.gguf", "GLM2-00002-of-00002.gguf"))):
        for shard in shards:
            (pack.parent / shard).write_text("x", encoding="utf-8")
    (without_tok / "tokenizer" / "vocab.json").unlink()

    paths = resolve_model_paths(repo, "maya-s24")
    assert paths.tokenizer and not paths.prompt_tokens, paths.as_dict()

    paths = resolve_model_paths(repo, "maya-s25")
    assert not paths.tokenizer
    assert paths.prompt_tokens == str(HERE / "fixtures" / "prompt.json")


def test_the_default_prompt_is_the_pack_tokenizer_and_is_deterministic():
    ids = build_prompt("", 64, tokenizer_dir=str(FAKE_ROOT / "prompt-test" / "tokenizer"),
                       root=str(FAKE_ROOT / "prompt-test"), vocab=154880)
    assert len(ids) == 64
    assert all(0 <= t < 154880 for t in ids)
    assert ids == build_prompt("", 64, tokenizer_dir=str(FAKE_ROOT / "prompt-test" / "tokenizer"),
                              root=str(FAKE_ROOT / "prompt-test"), vocab=154880)


def test_a_pack_without_a_tokenizer_folds_the_bundled_ids_into_its_vocabulary(capsys):
    ids = build_prompt("", 16, vocab=154880)
    assert len(ids) == 16 and all(0 <= t < 154880 for t in ids)
    assert "no tokenizer beside the pack" in capsys.readouterr().err


def test_a_sweep_never_overwrites_the_installed_run_config():
    """The model's own maya-<model>.json and run-maya-<model>.sh stay as a fallback.

    A sweep that measures a worse configuration than the installed one must not leave the
    user with no way to go back, so the deliverables are written under names the sweep
    owns and the installed pair is read only as the base for the keys it inherits.
    """
    from run_config import deliverable_names, write_deliverables

    tmp = Path(tempfile.mkdtemp(prefix="hc-original-tune-"))
    (tmp / "serve").mkdir()
    (tmp / "serve" / "server.py").write_text('ap.add_argument("--engine", choices=["strata"])\n',
                                            encoding="utf-8")
    (tmp / "CMakeLists.txt").write_text("# stand-in project\n", encoding="utf-8")
    _clear_root_cache()
    os.environ["MAYA_ROOT"] = str(tmp)

    installed = {"exe": "/old/build/strata", "args": ["--kv", "fp16"], "cwd": str(tmp),
                "sampling": {"temperature": 0.7}, "vision": True, "port": 8090}
    (tmp / "maya-maya-s24.json").write_text(json.dumps(installed), encoding="utf-8")
    (tmp / "run-maya-maya-s24.sh").write_text("#!/bin/sh\n# the tune the model shipped with\n",
                                             encoding="utf-8")
    before = {p.name: p.read_text(encoding="utf-8")
              for p in (tmp / "maya-maya-s24.json", tmp / "run-maya-maya-s24.sh")}

    record = {"objective": "decode", "context": 32768, "best": {"vram": 512},
              "metrics": {"decode_tok_s": 100.0, "prefill_tok_s": 500.0, "latency_ms": 1000.0},
              "verified": True, "source": "study", "study": {}, "history": {},
              "hardware_slug": "test-machine", "engine": str(STUB)}

    sh, js = write_deliverables("maya-s24", {"pack": str(tmp)},
                                ["--kv", "int8", "--serve", "--max-new", "512"], record,
                                deliverable_dir=tmp)

    assert deliverable_names("Maya S24") == ("max_maya-s24_config.json",
                                            "max_maya-s24_run.sh", "max_maya-s24_run.bat")
    assert js.name == "max_maya-s24_config.json" and sh.name == "max_maya-s24_run.sh"
    assert (tmp / "max_maya-s24_run.bat").is_file()
    for name, text in before.items():
        assert (tmp / name).read_text(encoding="utf-8") == text, f"{name} was overwritten"

    cfg = json.loads(js.read_text(encoding="utf-8"))
    assert cfg["sampling"] == {"temperature": 0.7} and cfg["vision"] is True
    assert cfg["port"] == 8090
    assert "--serve" not in cfg["args"] and "--max-new" not in cfg["args"]
    launcher = sh.read_text(encoding="utf-8")
    assert "run-maya-maya-s24.sh is untouched" in launcher


def _history_file(out: Path) -> Path:
    """The machine history file this test run wrote."""
    files = list((out / "history").glob("*.json"))
    assert files, "the sweep did not write a history file"
    return files[0]


def test_a_history_prior_cannot_be_reported_as_the_winner():
    """The winner is a configuration this sweep booted, not the best number in history.

    The Maya-S24 run reported "winner ... gen 220.55 tok/s" while every trial it
    measured came in under 10 tok/s: the title went to a prior carried in from a
    history file. A prior seeds the sampler and is the number to beat; it is not a
    result, and it cannot be written into a launcher as one.
    """
    out = Path("/tmp/hc-prior-not-winner")
    assert _run(["--gen", "--trials", "6", "--seed", "5"], out)[0] == 0

    hist = _history_file(out)
    data = json.loads(hist.read_text(encoding="utf-8"))
    inflated = dict(data["trials"][0])
    inflated["score"] = 9999.0
    inflated["metrics"] = {"decode_tok_s": 9999.0, "prefill_tok_s": 500.0,
                          "latency_ms": 1000.0, "prompt_ms": 1000.0, "decode_ms": 1000.0,
                          "generated": 512, "prompt_tokens": 10240}
    data["trials"].append(inflated)
    hist.write_text(json.dumps(data), encoding="utf-8")

    assert _run(["--gen", "--trials", "6", "--seed", "5"], out, fresh=False)[0] == 0
    record = _record(out)
    assert record["source"] == "study", "the winner came from history, not from this run"
    assert record["metrics"]["decode_tok_s"] < 9999.0
    measured = [r["cfg"] for r in record["trials"] if r["status"] == "PASS"]
    assert record["best"] in measured, "the winner was never booted in this run"
    assert record["prior_best"] is not None
    assert float(record["prior_best"]["score"]) == 9999.0
    assert record["prior_best"]["cfg"] != record["best"]
    cfg = json.loads((out / "max_maya-s24_config.json").read_text(encoding="utf-8"))
    assert cfg["tuned_by"]["best"] == record["best"]
    assert cfg["tuned_by"]["metrics"]["decode_tok_s"] == record["metrics"]["decode_tok_s"]


def test_a_prior_measured_by_a_different_engine_is_not_used():
    """Numbers from another binary are not comparable, so they are dropped, not merged.

    The test stub and the project engine are different programs; a history file that
    has ever held a stub run must not hand that sweep a winner it could never reproduce.
    """
    out = Path("/tmp/hc-prior-engine")
    assert _run(["--gen", "--trials", "4", "--seed", "2"], out)[0] == 0
    hist = _history_file(out)
    data = json.loads(hist.read_text(encoding="utf-8"))
    for trial in data["trials"]:
        trial["engine"] = "/home/ai-user0/some-other-checkout/build/strata"
        trial["score"] = 9000.0
    hist.write_text(json.dumps(data), encoding="utf-8")

    assert _run(["--gen", "--trials", "4", "--seed", "2"], out, fresh=False)[0] == 0
    record = _record(out)
    assert record["history"]["priors_skipped"] >= 1
    assert record["history"]["priors_used"] == 0
    assert record["prior_best"] is None
    assert record["source"] == "study"
    assert record["conditions"]["engine"] == str(STUB)
    assert record["conditions"]["prompt_tokens"] > 0


def test_from_records_finds_rows_written_under_the_models_display_name():
    """A sweep run as --model Maya-S24 writes CSVs with that name in them.

    Rebuilding the deliverable from those rows is the way to recover a winner when the
    record's best came from history, so the lookup has to match the spelling the operator
    used, not only the lowercased slug.
    """
    out = Path("/tmp/hc-optuna-from-records-name")
    # _run passes --model maya-s24 first; a later --model on the same command line wins,
    # which is exactly how the CSV gets the display name in its filename.
    assert _run(["--model", "Maya-S24", "--trials", "4", "--seed", "7"], out)[0] == 0
    assert list((out / "results").glob("bayesian-*.csv")), "the sweep wrote no CSV"

    code, _ = _run(["--model", "Maya-S24", "--gen", "--from-records"], out, fresh=False)
    assert code == 0
    cfg = json.loads((out / "max_maya-s24_config.json").read_text(encoding="utf-8"))
    assert cfg["tuned_by"]["source"].startswith("best measured row")
    assert cfg["tuned_by"]["conditions"]["engine"] == str(STUB)
    assert cfg["tuned_by"]["conditions"]["prompt_tokens"] > 0
