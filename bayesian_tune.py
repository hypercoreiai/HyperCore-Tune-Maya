#!/usr/bin/env python3
"""HyperCore-Tune-Maya/bayesian_tune.py - Bayesian (TPE) tuning of the Project Maya engine.

Searches the dimensions the GLM engine exposes - vram, pcie, pcie-mode, ple-io, kv,
kv-streaming, host-worker, pool workers, prompt-cache, draft-floor - against one of five
user-selected objectives:

    --decode    maximise decode tokens/s (the generated-token rate; --gen is its alias)
    --think     maximise thinking tokens/s, the reasoning block alone
    --prefill   maximise prompt tokens/s
    --latency   minimise prompt_ms + decode_ms for the request
    --balance   maximise decode tokens/s discounted by the share of a token the engine spent
                off the card: fast AND resident, not just fast

Two search modes:

    joint    one study over the whole space (the Bayesian sweep proper)
    staged   one study per layer, other layers frozen at the incumbent best. `layer` in the
             CSV is then the layer under test, which is how tune/max_tune.sh rows are read.

Deliverables, written at the Maya project root as requested:

    max_<model>_run.sh      the tuned launcher for the winning configuration
    max_<model>_config.json the winning configuration, its metrics, and every trial

The installed `run-maya-<model>.sh` and `maya-<model>.json` are never touched, so the tune
the model shipped with is always there as a fallback.

Run data from previous sweeps is kept in HyperCore-Tune-Maya/history/, one file per machine,
model, objective and context:

    <CPU>_<GPU>_<RAM>_<PCIe>x<width>_<model>_<objective>_<cntx>.json

A sweep on the same machine loads it, never re-boots a configuration it has already
measured - including one measured under a different objective, since the same fourteen
flags boot the same run - seeds the TPE sampler with those trials, and keeps drawing until
--trials unseen configurations have been evaluated. --no-load starts cold; a record from
different hardware is ignored unless --allow-hardware-mismatch is given.

Run it through the project venv:  ./HyperCore-Tune-Maya/max_tune.sh --model maya-s24 --gen --cntx 131072
The Project Maya root is found from this repository's location, or pinned with --maya-root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna
from optuna.samplers import RandomSampler, TPESampler

try:
    from optuna.samplers import GPSampler
except ImportError:  # GPSampler needs scipy; it is optional here.
    GPSampler = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine_runner import (
    DEFAULT_THINK_END,
    THINK_BUDGET_KEY,
    THINK_END_KEY,
    EngineSession,
    Metrics,
    build_prompt,
    run_config,
    think_end_token,
    vocab_size,
)
from hardware import Hardware, ambient_state, fingerprint, free_vram_mib, hardware_from
from history import dedupe as dedupe_priors
from history import (
    distributions_for,
    find_history,
    history_path,
    other_objective_history,
    usable_trials,
)
from history import load as load_history
from history import save as save_history
from maya_env import abs_path, engine_binary, maya_root
from run_config import write_deliverables as write_run_pair
from search_space import (
    DEFAULT_CONFIG,
    LAYERS,
    REQUEST_DIMS,
    ROW_COLUMNS,
    Config,
    ModelPaths,
    Param,
    build_space,
    engine_args,
    kv_modes_for,
    pack_kind,
    pack_vram_floor_mib,
    physical_cores,
    quasi_random_configs,
    request_keys,
    resolve_model_paths,
    sample,
    signature,
    topology_signature,
)
from transfer import TRANSFER_COLUMNS, balance_score, expert_blob_bytes, frontier

OBJECTIVES = ("decode", "think", "prefill", "latency", "balance")
# The metric name older records and CSVs use. A sweep reads them, writes only the new one.
LEGACY_OBJECTIVE = {"gen": "decode"}
# Every score is "higher is better": a configuration that cannot boot scores PENALTY, the
# same 0.0 as a real zero throughput.
PENALTY = {objective: 0.0 for objective in OBJECTIVES}
PENALTY["latency"] = -1e9
# A configuration that cannot boot scores PENALTY, the same 0.0 as a real zero throughput.
# When the same failure repeats, it is the machine, not the search: stop and say so rather
# than spend the remaining trials on identical zeros.
FAIL_STREAK = 3


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def run_log_name(results_dir: Path, layer: str, cfg: Config, index: int) -> Path:
    """The path of one engine log.

    A second-resolution stamp alone is not enough: two sweeps started in the same second
    would write over each other's log. The configuration's own digest is in the name, so
    every boot has a file of its own and a later sweep can find it again.
    """
    digest = hashlib.sha1(
        ",".join(f"{key}={cfg[key]}" for key in sorted(cfg)).encode()
    ).hexdigest()[:10]
    return results_dir / f"run-{utc_stamp()}-{layer}-{index}-{digest}.log"


def signed_score(m: Metrics, objective: str) -> float:
    """One number to maximise. Failed boots score at the bottom of the range.

    `balance` is the decode rate discounted by the share of a token the engine spent off the
    card. That share comes from the engine's own telemetry - the `STAT` counter line a
    `--serve` engine prints every eight tokens, or the per-request `glm stat` summary on the
    single-conversation path - attributed with the unit costs the same engine measured at
    boot. A configuration that wins on raw tok/s while half its token time is bus waiting
    scores below one that is a little slower and already resident. With no telemetry the
    discount is zero, so a stub engine or an older build behaves exactly like --decode.
    """
    if not m.ok():
        return PENALTY[objective]
    if objective == "decode":
        return m.decode_tok_s
    if objective == "think":
        return m.thinking_tok_s
    if objective == "prefill":
        return m.prefill_tok_s
    if objective == "balance":
        return balance_score(m.decode_tok_s, m.transfer)
    return -m.latency_ms


def prior_score(m: Metrics, objective: str) -> float:
    """What an already-measured configuration is worth under this objective.

    `signed_score` gates on status because a live trial that failed to boot has to score at
    the bottom of the range. A prior carried in from history has no status of its own, only
    the numbers a previous sweep recorded, so the same arithmetic is applied without that
    gate. A configuration measured under a different objective is re-scored here: its stored
    score belongs to that objective and cannot be compared with this one.
    """
    if objective == "decode":
        return m.decode_tok_s
    if objective == "think":
        return m.thinking_tok_s
    if objective == "prefill":
        return m.prefill_tok_s
    if objective == "balance":
        return balance_score(m.decode_tok_s, m.transfer)
    return -m.latency_ms


def _cfg_matches_frozen(cfg: dict[str, Any], frozen: dict[str, object]) -> bool:
    """True when every pinned dimension already has the value this sweep is holding."""
    return all(str(cfg.get(key)) == str(value) for key, value in frozen.items())


def frozen_from_args(args: argparse.Namespace) -> dict[str, object]:
    """Dimensions the operator pinned to a constant, keyed as in the search space."""
    frozen: dict[str, object] = {}
    if getattr(args, "pcie", None) is not None:
        frozen["pcie"] = float(args.pcie)
    if getattr(args, "ple_io", None) is not None:
        frozen["ple_io"] = str(args.ple_io)
    if getattr(args, "kv", None) is not None:
        frozen["kv"] = str(args.kv)
    if getattr(args, "kvstream", None) is not None:
        frozen["kvstream"] = str(args.kvstream)
    if getattr(args, "hostworker", None) is not None:
        frozen["hostworker"] = str(args.hostworker)
    if getattr(args, "pcie_mode", None) is not None:
        frozen["pcie_mode"] = str(args.pcie_mode)
    if getattr(args, "worker", None) is not None:
        frozen["worker"] = int(args.worker)
    if getattr(args, "tail", None) is not None:
        frozen["tail"] = str(args.tail)
    if getattr(args, "draft_floor", None) is not None:
        frozen["draft_floor"] = float(args.draft_floor)
    if getattr(args, "ple_inflight", None) is not None:
        frozen["ple_inflight"] = int(args.ple_inflight)
    if getattr(args, "suffix_draft", None) is not None:
        frozen["suffix_draft"] = int(args.suffix_draft)
    if getattr(args, "mtp_max_t", None) is not None:
        frozen["mtp_max_t"] = int(args.mtp_max_t)
    if getattr(args, "parallel", None) is not None:
        frozen["parallel"] = int(args.parallel)
    return frozen


def draft_floor_of(args: argparse.Namespace) -> float:
    """The pinned floor, or the pre-search default used to fill old records."""
    value = getattr(args, "draft_floor", None)
    return float(value) if value is not None else 0.8


def make_sampler(name: str, seed: int) -> optuna.samplers.BaseSampler:
    """TPE is the Bayesian sampler; GP is the Gaussian-process one (needs scipy)."""
    if name == "random":
        return RandomSampler(seed=seed)
    if name == "gp":
        if GPSampler is None:
            print("note: --sampler gp needs scipy; falling back to TPE", file=sys.stderr)
            return TPESampler(seed=seed, n_startup_trials=10)
        return GPSampler(seed=seed)
    return TPESampler(seed=seed, n_startup_trials=10, multivariate=True)


@dataclass
class TrialRow:
    layer: str
    cfg: Config
    metrics: Metrics
    score: float
    from_history: bool = False
    # The engine binary that produced the numbers. A prior measured by a different
    # engine - the test stub, an older build - is not comparable to this sweep and is
    # never allowed to become the winner.
    engine: str = ""

    def csv_row(self) -> dict[str, object]:
        m = self.metrics
        # Every engine column comes from the configuration; a dimension this pack's runner
        # does not have is written empty rather than guessed, so a row never claims an arm
        # the engine was never given.
        row: dict[str, object] = {"layer": self.layer}
        for column in ROW_COLUMNS[1:-1]:
            row[column] = self.cfg.get(column, "")
        row["status"] = m.status
        row.update({
            "decode_tok_s": f"{m.decode_tok_s:.2f}",
            "thinking_tok_s": f"{m.thinking_tok_s:.2f}",
            "answer_tok_s": f"{m.answer_tok_s:.2f}",
            "prefill_tok_s": f"{m.prefill_tok_s:.2f}",
            "latency_ms": f"{m.latency_ms:.1f}",
            "prompt_ms": f"{m.prompt_ms:.1f}",
            "decode_ms": f"{m.decode_ms:.1f}",
            "think_ms": f"{m.think_ms:.1f}",
            "answer_ms": f"{m.answer_ms:.1f}",
            "generated": m.generated,
            "prompt_tokens": m.prompt_tokens,
            "score": f"{self.score:.4f}",
            "note": m.note,
        })
        if m.transfer:
            row.update({key: value for key, value in m.transfer.items() if key != "flags"})
            row["flags"] = "|".join(m.transfer.get("flags", []))
        return row

    def to_dict(self) -> dict[str, Any]:
        """The CSV row plus the configuration as a dict, so a record can be replayed."""
        row = self.csv_row()
        row["cfg"] = dict(self.cfg)
        row["signature"] = list(signature(self.cfg))
        row["ambient"] = self.metrics.ambient
        if self.metrics.transfer:
            row["transfer"] = dict(self.metrics.transfer)
        row["from_history"] = self.from_history
        row["engine"] = self.engine
        return row


# One line per unseedable prior is noise when a history predates a new dimension; the
# count is what matters, so the detail stops after a handful.
_SEED_NOTE_LIMIT = 5


class Sweeper:
    """Runs the study, keeps the incumbent best, and writes the deliverables."""

    def __init__(self, args: argparse.Namespace, root: Path, params: list[Param], paths,
                 hw: Hardware, priors: list[dict[str, Any]],
                 foreign: list[dict[str, Any]] | None = None) -> None:
        self.args = args
        self.root = root
        self.params = params
        self.paths = paths
        self.hw = hw
        self.priors = priors
        # Configurations already measured on this machine under a different objective. They
        # are not priors for this study - their scores belong to another objective - but
        # they have been booted, so they must not be booted again.
        self.foreign: list[dict[str, Any]] = []
        # One expert blob, from the pack's own index, so reads/tok can be stated in bytes/s.
        self.blob_bytes = expert_blob_bytes(self.paths.pack) if self.paths.pack else None
        # The id that closes a reasoning block has to be the pack's own. A guessed id is
        # accepted by the engine and silently breaks the thinking/answer split, so it is read
        # from the tokenizer rather than assumed.
        if args.think_end is None:
            self.think_end = think_end_token(self.paths.tokenizer) or DEFAULT_THINK_END
            if self.think_end != DEFAULT_THINK_END:
                print(f"think-end: this pack closes reasoning with token {self.think_end}")
        else:
            self.think_end = args.think_end
        self.priors_loaded = len(priors)
        self.foreign_loaded = 0
        self.foreign_skipped = 0
        self.rows: list[TrialRow] = []
        self.seen: set[tuple[object, ...]] = set()
        self.prior_rows: dict[tuple[object, ...], TrialRow] = {}
        # best_row: the best configuration this run measured. incumbent: the best prior
        # carried in from history. Only a measured configuration can be the winner; a
        # prior is a seed for the sampler and a number to compare against, nothing more.
        self.best_row: TrialRow | None = None
        self.incumbent: TrialRow | None = None
        self.priors_skipped = 0
        self.priors_merged = 0
        self.seed_notes = 0
        self.duplicates_skipped = 0
        self.fail_streak = 0
        self.last_failure = ""
        self.frozen: dict[str, object] = dict(getattr(args, "frozen", {}) or {})
        # The prompt is built before the priors are filtered: a prior measured on a
        # different prompt is not a number this run could reproduce.
        self.prompt_ids = build_prompt(
            args.prompt_tokens_file,
            args.prompt_tokens,
            tokenizer_dir=self.paths.tokenizer or None,
            root=str(self.root),
            vocab=vocab_size(self.paths.tokenizer, self.paths.pack),
        )
        if self.frozen:
            self.priors = [p for p in priors if _cfg_matches_frozen(p.get("cfg", {}), self.frozen)]
            self.foreign = [p for p in (foreign or [])
                            if _cfg_matches_frozen(p.get("cfg", {}), self.frozen)]
        else:
            self.foreign = list(foreign or [])
        self.priors = self._usable_priors(self.priors)
        self.foreign = self._usable_priors(self.foreign, foreign=True)
        self.model = args.model.lower()
        self.session: EngineSession | None = None
        self.session_topology: tuple[str, ...] | None = None
        self._seed_from_history()

    def _usable_priors(self, priors: list[dict[str, Any]], *,
                       foreign: bool = False) -> list[dict[str, Any]]:
        """The priors this run's grid can hold, one per configuration.

        A prior outside the bounds asked for (--vram-max, --max-workers) cannot seed the
        study, and keeping it as the incumbent best would put a value the engine would
        reject - or a flag this pack's runner ignores - into the generated launcher.
        A prior measured by a different engine, or with a different prompt, is dropped for
        the same reason: its tok/s is not a number this sweep could reproduce, and a
        winner taken from it would be a number the user cannot check.
        Two records that normalise to the same configuration are folded into the better one.

        `foreign` marks the list that came from another objective. Those configurations are
        kept for the same reasons - they are still measurements of this machine - but a
        configuration the study already has a prior for is not carried twice.
        """
        kept: list[dict[str, Any]] = []
        engine = str(self.args.engine)
        prompt_tokens = len(self.prompt_ids)
        known = {tuple(p["signature"]) for p in self.priors}
        skipped = 0
        for prior in priors:
            if foreign and tuple(prior.get("signature", ())) in known:
                continue
            try:
                distributions_for(self.params, prior.get("cfg", {}))
            except ValueError as exc:
                skipped += 1
                self._seed_note(prior.get("cfg", {}), exc)
                continue
            recorded_engine = str(prior.get("engine", "") or "")
            if recorded_engine and recorded_engine != engine:
                skipped += 1
                self._seed_note(prior.get("cfg", {}),
                                f"measured with a different engine ({recorded_engine})")
                continue
            recorded_prompt = int(prior.get("prompt_tokens", 0) or 0)
            if recorded_prompt and recorded_prompt != prompt_tokens:
                skipped += 1
                self._seed_note(prior.get("cfg", {}),
                                f"measured on a {recorded_prompt}-token prompt, this run "
                                f"uses {prompt_tokens}")
                continue
            kept.append(prior)
        deduped = dedupe_priors(kept)
        merged = len(kept) - len(deduped)
        if foreign:
            self.foreign_skipped = skipped
            self.foreign_loaded = len(deduped)
        else:
            self.priors_skipped = skipped
            self.priors_merged = merged
        return deduped

    def _seed_from_history(self) -> None:
        """Turn every prior measured on this machine into a row that never needs a boot.

        Priors are seeds: they warm the sampler and they are the number the sweep is trying
        to beat. They are not the winner - the launcher a sweep writes out has to be a
        configuration this run actually measured on this machine.

        The `foreign` list is the same kind of measurement, taken under a different
        objective. It seeds nothing - its scores are not this objective's scores - but every
        configuration in it goes into `seen`, so the study never boots a point this machine
        has already measured. Its score is recomputed for the objective being run.
        """
        for prior, is_foreign in [(p, False) for p in self.priors] + \
                                [(p, True) for p in self.foreign]:
            stored = prior["metrics"]
            metrics = Metrics(status="CACHED")
            metrics.decode_tok_s = float(stored.get("decode_tok_s", stored.get("gen_tok_s", 0.0)) or 0.0)
            metrics.thinking_tok_s = float(stored.get("thinking_tok_s", 0.0) or 0.0)
            metrics.answer_tok_s = float(stored.get("answer_tok_s", 0.0) or 0.0)
            metrics.prefill_tok_s = float(stored.get("prefill_tok_s", 0.0) or 0.0)
            metrics.latency_ms = float(stored.get("latency_ms", 0.0) or 0.0)
            metrics.prompt_ms = float(stored.get("prompt_ms", 0.0) or 0.0)
            metrics.decode_ms = float(stored.get("decode_ms", 0.0) or 0.0)
            metrics.think_ms = float(stored.get("think_ms", 0.0) or 0.0)
            metrics.answer_ms = float(stored.get("answer_ms", 0.0) or 0.0)
            metrics.generated = int(stored.get("generated", 0) or 0)
            metrics.prompt_tokens = int(stored.get("prompt_tokens", 0) or 0)
            metrics.transfer = prior.get("transfer", {}) or {}
            score = (prior_score(metrics, self.args.objective) if is_foreign
                     else float(prior["score"]))
            row = TrialRow(layer="history", cfg=dict(prior["cfg"]),
                          metrics=metrics, score=score,
                          from_history=True, engine=str(prior.get("engine", "")))
            sig = tuple(prior["signature"])
            self.seen.add(sig)
            self.prior_rows[sig] = row
            if self.incumbent is None or row.score > self.incumbent.score:
                self.incumbent = row

    def winner(self) -> TrialRow | None:
        """The configuration to hand to the launcher: measured here, else the incumbent.

        A prior is only used when nothing in this run produced tokens (a dry run, a busy
        GPU); record() labels that case source="history" so it is never read as a new
        measurement.
        """
        return self.best_row or self.incumbent

    def _close_session(self) -> None:
        """Close the shared engine session, if one is open."""
        if self.session is not None:
            self.session.close()
            self.session = None
            self.session_topology = None

    # ---- one configuration ----------------------------------------------------
    def evaluate(self, layer: str, cfg: Config) -> TrialRow:
        sig = signature(cfg)
        if sig in self.seen:
            # Safety net: the study loop skips seen points before calling evaluate.
            # Reuse the measurement we already have rather than boot again.
            prior = self.prior_rows.get(sig) or self.best_row
            if prior is None:
                row = TrialRow(layer, cfg, Metrics(status="DUPLICATE"),
                              PENALTY[self.args.objective])
            else:
                row = TrialRow(layer, cfg, prior.metrics, prior.score,
                              from_history=prior.from_history, engine=prior.engine)
            self.rows.append(row)
            # A prior reused from history is not a new measurement: it cannot take the
            # title. Only a configuration this run booted can.
            if not row.from_history and (self.best_row is None or row.score > self.best_row.score):
                self.best_row = row
            return row
        self.seen.add(sig)

        if self.args.dry_run:
            args = engine_args(cfg, self.paths.as_dict(), self.args.cntx,
                              self.args.max_new, draft_floor_of(self.args))
            m = Metrics(status="DRY")
            m.note = " ".join(args)
            row = TrialRow(layer, cfg, m, PENALTY[self.args.objective])
            self.rows.append(row)
            # A dry run has no measurement to win with. It falls back to the incumbent
            # prior when there is one, and only holds the command line as the best when
            # nothing has ever been measured on this machine.
            if self.best_row is None and self.incumbent is None:
                self.best_row = row
            return row

        ambient = ambient_state(cfg.get("vram"))
        ambient["captured_utc"] = utc_stamp()
        boot_args = engine_args(cfg, self.paths.as_dict(), self.args.cntx,
                                self.args.max_new, draft_floor_of(self.args),
                                request_keys=bool(self.args.two_stage))
        keys = request_keys(cfg) if self.args.two_stage else {}
        if self.args.think_budget > 0:
            keys[THINK_BUDGET_KEY] = self.args.think_budget
            keys[THINK_END_KEY] = self.think_end
        top_sig = topology_signature(cfg, self.params) if self.args.two_stage else None
        runs: list[Metrics] = []

        if self.args.two_stage:
            if not self.session or not self.session.alive() or self.session_topology != top_sig:
                self._close_session()
                log = self.args.results_dir / run_log_name(self.args.results_dir, layer, cfg, 0)
                self.session = EngineSession(
                    engine=self.args.engine, args=boot_args, log_path=log,
                    boot_timeout=self.args.boot_timeout, run_timeout=self.args.run_timeout,
                    cwd=self.root, blob_bytes=self.blob_bytes,
                )
                self.session_topology = top_sig
            for _ in range(self.args.repeat):
                if not self.session or not self.session.alive():
                    self._close_session()
                    log = self.args.results_dir / run_log_name(self.args.results_dir, layer, cfg, 0)
                    self.session = EngineSession(
                        engine=self.args.engine, args=boot_args, log_path=log,
                        boot_timeout=self.args.boot_timeout, run_timeout=self.args.run_timeout,
                        cwd=self.root, blob_bytes=self.blob_bytes,
                    )
                    self.session_topology = top_sig
                runs.append(self.session.request(self.prompt_ids, self.args.max_new, keys))
        else:
            for i in range(self.args.repeat):
                log = self.args.results_dir / run_log_name(self.args.results_dir, layer, cfg, i)
                runs.append(run_config(
                    engine=self.args.engine, args=boot_args, prompt_ids=self.prompt_ids,
                    max_new=self.args.max_new, log_path=log,
                    boot_timeout=self.args.boot_timeout, run_timeout=self.args.run_timeout,
                    cwd=self.root, request_keys=keys, blob_bytes=self.blob_bytes,
                ))
        # Median of the repeats: one boot is noisy enough to flip a ranking.
        best = max(runs, key=lambda m: signed_score(m, self.args.objective))
        best.ambient = ambient
        if self.args.repeat > 1 and best.ok():
            for field_name in ("decode_tok_s", "thinking_tok_s", "answer_tok_s",
                              "prefill_tok_s", "latency_ms"):
                setattr(best, field_name, statistics.median(
                    getattr(r, field_name) for r in runs if r.ok()))
            # The bus figures move from token to token more than the rate does, so they are
            # averaged over the repeats rather than taken from whichever boot scored best.
            transfer_rows = [r.transfer for r in runs if r.transfer]
            if transfer_rows:
                for key in ("off_card_ms_per_tok", "transfer_share", "disk_reads_per_tok",
                            "disk_ms_per_tok", "ram_fetches_per_tok", "vram_hit_pct"):
                    values = [float(row.get(key, 0.0)) for row in transfer_rows]
                    best.transfer[key] = round(statistics.median(values), 4)

        if not best.ok():
            note = best.note or "the engine exited with no message"
            if note == self.last_failure:
                self.fail_streak += 1
            else:
                self.last_failure, self.fail_streak = note, 1
        else:
            self.last_failure, self.fail_streak = "", 0

        row = TrialRow(layer, cfg, best, signed_score(best, self.args.objective),
                      engine=str(self.args.engine))
        self.rows.append(row)
        if self.best_row is None or row.score > self.best_row.score:
            self.best_row = row
        m = row.metrics
        dims = " ".join(f"{k}={cfg[k]}" for k in ROW_COLUMNS[1:-1] if k in cfg)
        bus = (f" bus {m.transfer['transfer_share']:.0%} off-card" if m.transfer else "")
        print(f"  trial {len(self.rows):>3} layer={layer} {dims} -> {m.status} "
              f"decode {m.decode_tok_s:.2f} tok/s prefill {m.prefill_tok_s:.2f} tok/s "
              f"latency {m.latency_ms:.0f} ms"
              + (f" thinking {m.thinking_tok_s:.2f} tok/s" if m.thinking else "")
              + bus
              + (f"  [{m.note}]" if m.note else ""))
        sys.stdout.flush()
        return row

    # ---- the study -------------------------------------------------------------
    def run(self) -> bool:
        """Run the study. Returns True if it was interrupted by Ctrl-C."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = make_sampler(self.args.sampler, self.args.seed)

        try:
            if self.args.mode == "joint":
                study = optuna.create_study(direction="maximize", sampler=sampler)
                self._warm_study(study, self.params)
                interrupted = self._run_study(study, self.params, dict(self.frozen), self.args.trials)
                if interrupted:
                    print("\ninterrupted: keeping the best configuration measured so far")
                return interrupted

            frozen: dict[str, object] = dict(self.frozen)
            for layer in self.layers():
                layer_params = [p for p in self.params if p.layer == layer]
                if not layer_params:
                    continue
                study = optuna.create_study(direction="maximize",
                                          sampler=make_sampler(self.args.sampler, self.args.seed))
                self._warm_study(study, layer_params)
                interrupted = self._run_study(
                    study, layer_params, frozen, self.args.trials_per_layer, layer_hint=layer)
                if interrupted:
                    print(f"\ninterrupted during layer {layer}; keeping what has been measured")
                    return True
                if study.best_trial is not None and study.best_trial.value > PENALTY[self.args.objective]:
                    for p in layer_params:
                        frozen[p.key] = study.best_trial.params[p.key]
                    print(f"--- layer {layer}: locked {frozen} at {study.best_trial.value:.4f}")
            return False
        finally:
            self._close_session()

    def _run_study(
        self,
        study: optuna.Study,
        params: list[Param],
        frozen: dict[str, object],
        n_trials: int,
        layer_hint: str | None = None,
    ) -> bool:
        """Ask Optuna until `n_trials` unseen configurations have been evaluated.

        A suggestion already in history (or already tried this sweep) is pruned, not
        scored, so it does not consume the --trials budget and does not double-count
        in TPE. Returns True if the operator hit Ctrl-C.
        """
        new = 0
        asks = 0
        streak = 0
        stopped_on_failures = False
        max_asks = max(n_trials * 50, n_trials + 32)
        if self.args.quasi_trials and n_trials:
            quasi = quasi_random_configs(params, min(self.args.quasi_trials, n_trials), self.args.seed)
            quasi.sort(key=lambda cfg: topology_signature({**DEFAULT_CONFIG, **cfg}, params))
            for cfg in quasi:
                study.enqueue_trial({p.key: cfg[p.key] for p in params if p.key in cfg})
        try:
            while new < n_trials and asks < max_asks:
                trial = study.ask()
                asks += 1
                cfg = sample(trial, params, frozen)
                if signature(cfg) in self.seen:
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                    self.duplicates_skipped += 1
                    streak += 1
                    if streak >= 32:
                        print(f"note: Optuna suggested {streak} already-measured "
                              f"configurations in a row; stopping with {new} new trials",
                              file=sys.stderr)
                        break
                    continue
                streak = 0
                study.tell(trial, self._score_cfg(cfg, params, layer_hint))
                new += 1
                if self.fail_streak >= FAIL_STREAK:
                    stopped_on_failures = True
                    print(f"note: {self.fail_streak} configurations in a row failed the same "
                          f"way ({self.last_failure}); that is the machine, not the search - "
                          f"stopping with {new} of {n_trials} trials",
                          file=sys.stderr)
                    break
        except KeyboardInterrupt:
            return True
        if stopped_on_failures:
            return False
        if new < n_trials and streak < 32:
            print(f"note: reached the redraw limit with {new} of {n_trials} new trials "
                  f"({self.duplicates_skipped} duplicate suggestions skipped)",
                  file=sys.stderr)
        return False

    def _score_cfg(self, cfg: Config, params: list[Param], layer_hint: str | None) -> float:
        """Evaluate one unseen configuration and return its Optuna score."""
        if layer_hint:
            layer = layer_hint
        else:
            incumbent = self.winner().cfg if self.winner() else {}
            changed = sorted({p.layer for p in params
                             if p.key in cfg and cfg[p.key] != incumbent.get(p.key, cfg[p.key])})
            layer = "joint" if not self.winner() else (",".join(changed) or "joint")
        return self.evaluate(layer, cfg).score

    def expected_trials(self) -> int:
        if self.args.mode == "joint":
            return self.args.trials
        return self.args.trials_per_layer * len(self.layers())

    def layers(self) -> Sequence[str]:
        wanted = self.args.layers or LAYERS
        return [l for l in wanted if l in LAYERS]

    def _warm_study(self, study, params: list[Param]) -> int:
        """Hand the study the configurations already measured on this machine.

        TPE then samples from a posterior instead of a cold prior. A prior outside the
        grid the current run is searching (--vram-max, --max-workers, ...) is skipped and
        counted, never silently clamped.
        """
        added = 0
        for prior in self.priors:
            try:
                dists = distributions_for(params, prior["cfg"])
            except ValueError as exc:
                self.priors_skipped += 1
                self._seed_note(prior["cfg"], exc)
                continue
            if not dists:
                continue
            try:
                study.add_trial(optuna.trial.create_trial(
                    params={k: prior["cfg"][k] for k in dists},
                    distributions=dists,
                    value=float(prior["score"]),
                ))
            except Exception as exc:  # noqa: BLE001 - a bad prior must not stop the sweep
                self.priors_skipped += 1
                self._seed_note(prior["cfg"], exc)
                continue
            added += 1
        if self.priors_skipped > _SEED_NOTE_LIMIT:
            print(f"note: {self.priors_skipped} priors in total are outside the grid this "
                  f"run searches; only the first {self.seed_notes} are listed",
                  file=sys.stderr)
        return added

    def _seed_note(self, cfg: dict[str, object], exc: Exception) -> None:
        """Why a prior could not seed the study, at most _SEED_NOTE_LIMIT times."""
        if self.seed_notes < _SEED_NOTE_LIMIT:
            print(f"note: prior {cfg} cannot seed the study ({exc})", file=sys.stderr)
            self.seed_notes += 1

    # ---- the outputs -----------------------------------------------------------
    def write_csv(self) -> Path:
        self.args.results_dir.mkdir(parents=True, exist_ok=True)
        out = self.args.results_dir / (
            f"bayesian-{self.hw.slug()}-{self.model}-{self.args.objective}-{utc_stamp()}.csv")
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(ROW_COLUMNS) + [
                "decode_tok_s", "thinking_tok_s", "answer_tok_s", "prefill_tok_s",
                "latency_ms", "prompt_ms", "decode_ms", "think_ms", "answer_ms",
                "generated", "prompt_tokens", "score", "note"] + list(TRANSFER_COLUMNS))
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row.csv_row())
        return out

    def winner_args(self) -> list[str]:
        best = self.winner()
        if not best:
            return []
        return engine_args(best.cfg, self.paths.as_dict(), self.args.cntx,
                          self.args.max_new, draft_floor_of(self.args))

    def record(self, csv_path: Path) -> dict[str, Any]:
        best = self.winner()
        measured = sum(1 for r in self.rows if r.metrics.ok())
        failed = sum(1 for r in self.rows if not r.metrics.ok() and r.metrics.status != "DRY")
        return {
            "schema": "hypercore.max_run/1",
            "model": self.model,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "engine": str(self.args.engine),
            "hardware": self.hw.to_dict(),
            "hardware_slug": self.hw.slug(),
            "objective": self.args.objective,
            "context": self.args.cntx,
            "max_new": self.args.max_new,
            "spec_min_p": (best.cfg.get("draft_floor") if best else None) or draft_floor_of(self.args),
            "best": dict(best.cfg) if best else {},
            "metrics": {
                "decode_tok_s": best.metrics.decode_tok_s if best else 0.0,
                "thinking_tok_s": best.metrics.thinking_tok_s if best else 0.0,
                "answer_tok_s": best.metrics.answer_tok_s if best else 0.0,
                "prefill_tok_s": best.metrics.prefill_tok_s if best else 0.0,
                "latency_ms": best.metrics.latency_ms if best else 0.0,
                "prompt_ms": best.metrics.prompt_ms if best else 0.0,
                "decode_ms": best.metrics.decode_ms if best else 0.0,
                "think_ms": best.metrics.think_ms if best else 0.0,
                "answer_ms": best.metrics.answer_ms if best else 0.0,
            },
            "effective": best.metrics.effective if best else {},
            # What the winning numbers were measured with. A tok/s from one sweep is not
            # comparable with a tok/s from another: the same configuration on a different
            # prompt, context or engine binary is a different number.
            "conditions": {
                "engine": str(self.args.engine),
                "prompt_tokens": len(self.prompt_ids),
                "max_new": self.args.max_new,
                "context": self.args.cntx,
                "repeat": self.args.repeat,
                # The reasoning split only exists when the request asked for one.
                "think_budget": self.args.think_budget,
                "think_end": self.think_end if self.args.think_budget > 0 else None,
            },
            # The best prior this machine has ever recorded. Kept so a run can be compared
            # with the numbers it started from; it is never the winner.
            "prior_best": self.incumbent.to_dict() if self.incumbent else None,
            # A winner is only a winner if a boot produced tokens. A run that booted
            # nothing (GPU busy, --dry-run) has a best_row but no measurement; a prior
            # carried in from history is a measurement taken by an earlier sweep.
            "verified": bool(best and (best.metrics.ok() or best.from_history)),
            # Where the winning numbers came from: this sweep, a prior recorded on
            # this machine, or a stored record replayed with --replay/--from-records.
            "source": ("history" if best and best.from_history else
                       "dry_run" if self.args.dry_run else "study"),
            "args": self.winner_args(),
            "study": {
                "mode": self.args.mode,
                "sampler": self.args.sampler,
                "seed": self.args.seed,
                "trials": len(self.rows),
                "measured": measured,
                "failed": failed,
                "repeat": self.args.repeat,
                "complete": not self.args.trials or len(self.rows) >= self.expected_trials(),
            },
            "history": {
                "file": str(self.args.history_file or history_path(
                    self.args.history_dir, self.model, self.args.objective,
                    self.args.cntx, self.hw)),
                "trials_loaded": self.priors_loaded,
                "priors_skipped": self.priors_skipped,
                "priors_merged": self.priors_merged,
                "priors_used": len(self.priors),
                # Configurations already booted under another objective: not priors for this
                # study, but not worth booting a second time either.
                "foreign_seen": self.foreign_loaded,
                "foreign_skipped": self.foreign_skipped,
                "reused": sum(1 for r in self.rows if r.from_history),
                "duplicates_skipped": self.duplicates_skipped,
            },
            "trials": [r.to_dict() for r in self.rows],
            "csv": str(csv_path),
        }

    def write_record(self, record: dict[str, Any]) -> Path:
        """The full trial record: every configuration measured, with its metrics."""
        out = self.args.results_dir / f"maya-{self.model}.record.json"
        out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return out

    def write_deliverables(self, record: dict[str, Any]) -> tuple[Path, Path]:
        """Write max_<model>_run.sh and max_<model>_config.json in the project's run-config format.

        The model's own launcher and config are never written; a sweep only ever adds files
        under its own names. A root-level launcher left by an earlier sweep may belong to
        another user; a PermissionError there must not throw away a finished sweep, so the
        pair falls back to the results directory and the fallback is printed.
        """
        return write_run_pair(
            self.model, self.paths.as_dict(), self.winner_args(), record,
            deliverable_dir=self.args.deliverable_dir, port=self.args.port,
            open_browser=not self.args.no_browser,
            fallback_dir=self.args.results_dir,
        )

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bayesian (Optuna) tuning of the Project Maya engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="maya-s24",
                   help="the model to tune (maya-s24, maya-s, maya-m); names the deliverables")
    metric = p.add_mutually_exclusive_group()
    metric.add_argument("--decode", action="store_const", const="decode", dest="objective",
                       help="maximise decode tokens/s (every token the request generates)")
    metric.add_argument("--gen", action="store_const", const="decode", dest="objective",
                       help="alias for --decode; the name older scripts use")
    metric.add_argument("--think", action="store_const", const="think", dest="objective",
                       help="maximise thinking tokens/s - the reasoning block alone, which "
                            "needs --think-budget")
    metric.add_argument("--prefill", action="store_const", const="prefill", dest="objective",
                       help="maximise prefill tokens/s")
    metric.add_argument("--latency", action="store_const", const="latency", dest="objective",
                       help="minimise prompt_ms + decode_ms")
    metric.add_argument("--balance", action="store_const", const="balance", dest="objective",
                       help="maximise decode tokens/s discounted by the share of a token "
                            "the engine spent off the card (its STAT telemetry line). Finds "
                            "the configuration that is fast and resident, not just fast.")
    p.add_argument("--objective", choices=OBJECTIVES, default=None,
                   help="same as the metric flag above; the flag wins")
    p.add_argument("--think-budget", type=int, default=0,
                   help="tokens the request may reason before answering (GEN key "
                        "think_budget); 0 sends no thinking keys. --think needs one.")
    p.add_argument("--think-end", type=int, default=None,
                   help="the token id the engine forces when a reasoning block closes (GEN key "
                        "think_end). Defaults to the pack's own: </think> if its tokenizer "
                        "names one, otherwise its end-of-turn token. An id the model does not "
                        "have is emitted as ordinary text, so the reasoning split never fires "
                        "and the sweep reports thinking 0.0 tok/s.")
    p.add_argument("--cntx", "--context", type=int, default=32768, dest="cntx",
                   help="context length the sweep is tuned at")
    p.add_argument("--trials", type=int, default=40,
                   help="unseen configurations to evaluate; repeats of a point already "
                        "in history are redrawn and do not count")
    p.add_argument("--mode", choices=("joint", "staged"), default="joint")
    p.add_argument("--layers", default=None, help="comma separated layers for staged mode")
    p.add_argument("--sampler", choices=("tpe", "gp", "random"), default="tpe")
    p.add_argument("--seed", type=int, default=20261012, help="reproducible search")
    p.add_argument("--repeat", type=int, default=1, help="boots per configuration (median)")
    p.add_argument("--max-new", type=int, default=512, help="tokens generated per request")
    p.add_argument("--prompt-tokens", type=int, default=10240,
                   help="prompt length; >8192 is needed for the tail arm to differ")
    p.add_argument("--prompt-tokens-file", default=None,
                   help="file of token ids to cycle; without one the pack's own tokenizer "
                        "is used on the benchmark prompt")
    p.add_argument("--prompt-file", default=None,
                   help="token-id JSON, or a text prompt rendered through the model's chat "
                        "template and tokenized when a tokenizer is available")
    p.add_argument("--two-stage", action="store_true",
                   help="boot the engine once per topology and sweep per-request keys inside "
                        "that session (pcie_frac, spec_min_p, and --request-dims)")
    p.add_argument("--request-dims", default="cpu_threads,read_chunks,spec,spec_n_min",
                   help="extra per-request dimensions to sweep with --two-stage; empty string "
                        "keeps only the existing pcie/draft-floor request keys")
    p.add_argument("--quasi-trials", type=int, default=15,
                   help="Latin-hypercube trials seeded before TPE; 0 disables")
    p.add_argument("--no-quasi", action="store_true", help="do not seed a quasi-random sweep")
    p.add_argument("--vram-min", type=int, default=256)
    p.add_argument("--vram-max", type=int, default=1536)
    p.add_argument("--vram-step", type=int, default=256)
    p.add_argument("--max-workers", type=int, default=None, help="cap the pool-workers grid")
    p.add_argument("--pcie", type=float, default=None,
                   help="hold --pcie-frac at this value instead of searching it")
    p.add_argument("--ple-io", default=None, choices=("direct", "mmap"), dest="ple_io",
                   help="hold --ple-io at this value instead of searching it")
    p.add_argument("--kv", default=None, choices=("q4_0", "k8v4", "int8", "fp16"),
                   help="hold --kv at this value instead of searching it")
    p.add_argument("--kvstream", default=None, choices=("on", "off"),
                   help="hold --kv-streaming at this value instead of searching it")
    p.add_argument("--hostworker", "--host-worker", default=None, choices=("1", "0"),
                   dest="hostworker",
                   help="1 = the host thread joins the CPU pool (the engine's default), "
                        "0 = --no-host-worker; held constant instead of searched")
    p.add_argument("--pcie-mode", default=None, choices=("auto", "dma", "kernel", "direct"),
                   dest="pcie_mode",
                   help="hold --pcie-mode at this value instead of searching it")
    p.add_argument("--worker", type=int, default=None,
                   help="hold --pool-workers at this value instead of searching it")
    p.add_argument("--tail", type=int, default=None,
                   help="hold --prompt-cache at this number of conversation checkpoints "
                        "instead of searching it")
    p.add_argument("--draft-floor", "--spec-min-p", type=float, default=None,
                   dest="draft_floor",
                   help="hold --spec-min-p at this value (0.001-1) instead of searching it")
    p.add_argument("--ple-inflight", type=int, default=None, dest="ple_inflight",
                   help="hold --ple-inflight (outstanding PLE reads, 1-1024) at this value "
                        "instead of searching it")
    p.add_argument("--suffix-draft", type=int, default=None, dest="suffix_draft",
                   help="hold --suffix-draft (prompt-lookup drafting, 0 = off) at this value; "
                        "only searched when no --dflash drafter was found")
    p.add_argument("--mtp-max-t", type=int, default=None, dest="mtp_max_t",
                   help="hold --mtp-max-t (the MTP draft window cap, 0 = as --spec gives) at "
                        "this value; only searched when no --dflash drafter was found")
    p.add_argument("--parallel", type=int, default=None,
                   help="hold --parallel (conversations decoded at once, 1-8) at this value; "
                        "a GLM-pack dimension - the qwen session has one sequence state")
    p.add_argument("--engine", default=None,
                   help="the engine binary to boot; default the one this project built "
                        "(build/strata, build-hip/strata)")
    p.add_argument("--maya-root", default=None,
                   help="the Maya checkout to tune and to write the deliverables into; "
                        "default the repository this file is inside, or $MAYA_ROOT")
    p.add_argument("--pack", default=None,
                   help="the model pack to run (--glm-pack); default the one this model's "
                        "pack folder holds")
    p.add_argument("--dflash", default=None,
                   help="a DFlash2 drafter to draft with instead of the model's MTP block")
    p.add_argument("--data-dir", default=None,
                   help="folder holding the model packs (Maya-data/models, ...); default the "
                        "one the installer recorded in the project settings file, then a data "
                        "folder next to the Maya checkout")
    p.add_argument("--boot-timeout", type=int, default=300)
    p.add_argument("--run-timeout", type=int, default=180)
    p.add_argument("--results-dir", default=str(Path(__file__).resolve().parent / "results"))
    p.add_argument("--history-dir", default=str(Path(__file__).resolve().parent / "history"),
                   help="where run data keyed to this machine is kept")
    p.add_argument("--replay", default=None,
                   help="re-emit max_<model>_run.sh/_config.json from a stored record "
                        "without booting the engine "
                        "(HyperCore-Tune-Maya/results/maya-<model>.record.json)")
    p.add_argument("--from-records", action="store_true",
                   help="rebuild max_<model>_run.sh/_config.json from the best measured "
                        "trial in the CSVs under --results-dir, without running a sweep")
    p.add_argument("--history-file", default=None,
                   help="load run data from this file instead of the machine-matched one")
    p.add_argument("--no-load", action="store_true",
                   help="ignore previous runs; start the study cold")
    p.add_argument("--no-cross-objective", action="store_true",
                   help="only reuse configurations measured under this objective; a point "
                        "measured under another one would be booted again")
    p.add_argument("--allow-hardware-mismatch", action="store_true",
                   help="load run data even when it was measured on different hardware")
    p.add_argument("--hardware", default=None,
                   help="pin the machine fingerprint to this JSON object or file instead of "
                        "probing the host (for tests, containers, and a farm of boxes)")
    p.add_argument("--dry-run", action="store_true", help="print the command lines, launch nothing")
    p.add_argument("--no-deliverables", action="store_true",
                   help="write only the CSV; skip max_<model>_run.sh/_config.json")
    p.add_argument("--deliverable-dir", default=None,
                   help="where max_<model>_run.sh/_config.json go; default the Maya root")
    p.add_argument("--port", type=int, default=None,
                   help="port the generated launcher uses; default the model config's, else 8081")
    p.add_argument("--no-browser", action="store_true",
                   help="the generated launcher leaves out --open")
    p.add_argument("--no-preflight", action="store_true",
                   help="skip the free-VRAM check that runs before the first trial")
    args = p.parse_args(argv)
    if args.maya_root:
        pinned = abs_path(args.maya_root)
        os.environ["MAYA_ROOT"] = str(pinned) if pinned is not None else args.maya_root
        maya_root.cache_clear()
    for name in ("engine", "pack", "dflash", "data_dir",
                 "prompt_tokens_file", "results_dir", "history_dir", "replay",
                 "history_file", "deliverable_dir"):
        value = getattr(args, name)
        if value:
            resolved = abs_path(value)
            setattr(args, name, str(resolved) if resolved is not None else value)
    if args.hardware:
        hw_path = Path(args.hardware).expanduser()
        if hw_path.is_file():
            args.hardware = str(hw_path.resolve())
    root = maya_root()
    args.root = root
    if not args.engine:
        fallback = root / "build" / ("strata.exe" if os.name == "nt" else "strata")
        args.engine = engine_binary(root) or str(fallback)
    args.engine = str(abs_path(args.engine) or Path(args.engine).expanduser().resolve())
    if not args.objective:
        args.objective = "decode"
    args.objective = LEGACY_OBJECTIVE.get(args.objective, args.objective)
    if args.think_budget < 0:
        p.error("--think-budget cannot be negative")
    if args.think_end is not None and args.think_end < 0:
        p.error("--think-end is a token id and cannot be negative")
    if args.objective == "think" and args.think_budget == 0:
        p.error("--think measures the reasoning block, so it needs --think-budget N; "
               "without a budget the request has no reasoning block to measure")
    args.layers = [x.strip() for x in args.layers.split(",")] if args.layers else list(LAYERS)
    args.results_dir = Path(args.results_dir)
    args.history_dir = Path(args.history_dir)
    args.deliverable_dir = Path(args.deliverable_dir) if args.deliverable_dir else root
    if args.trials < 1:
        p.error("--trials must be at least 1")
    if args.no_quasi:
        args.quasi_trials = 0
    args.quasi_trials = max(0, min(args.quasi_trials, args.trials))
    args.request_dims = tuple(
        x.strip() for x in args.request_dims.split(",") if x.strip()
    ) if args.two_stage else ()
    unknown = [x for x in args.request_dims if x not in REQUEST_DIMS]
    if unknown:
        p.error(f"--request-dims accepts only {', '.join(REQUEST_DIMS)}")
    args.trials_per_layer = max(2, args.trials // max(1, len(args.layers)))
    args.frozen = frozen_from_args(args)
    return args


def _hardware(spec: str | None) -> Hardware:
    """The machine this sweep ran on: a pinned fingerprint if given, otherwise the host."""
    if not spec:
        return fingerprint()
    try:
        raw = json.loads(Path(spec).read_text(encoding="utf-8")) if Path(spec).is_file() \
            else json.loads(spec)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"HyperCore-Tune-Maya: --hardware is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise SystemExit("HyperCore-Tune-Maya: --hardware must be a JSON object")
    return hardware_from(raw)


def best_from_records(results_dir: Path, model: str, objective: str) -> TrialRow | None:
    """The highest-scoring measured row across every CSV this tool has written for a model.

    The CSV rows are the measurements; a record's `best` can come from a run that never
    booted the engine. Rebuilding from the rows is the only way to recover a winner that
    is backed by a number.
    """
    best: TrialRow | None = None
    # The CSV name carries the model slug as the sweep wrote it, which is normally
    # lowercase but is not always; match either spelling. A CSV written before the metric
    # was named decode carries "gen", so both spellings are looked for.
    needle = str(model).lower()
    names = {objective} | {alias for alias, canonical in LEGACY_OBJECTIVE.items()
                           if canonical == objective}
    for csv_path in sorted(results_dir.glob("bayesian-*.csv")):
        name = csv_path.name.lower()
        if needle not in name or not any(f"-{alias}-" in name for alias in names):
            continue
        try:
            rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        except OSError as exc:
            print(f"note: cannot read {csv_path}: {exc}", file=sys.stderr)
            continue
        for raw in rows:
            if raw.get("status") != "PASS":
                continue
            try:
                metrics = Metrics(
                    status="PASS",
                    decode_tok_s=float(raw.get("decode_tok_s") or raw.get("gen_tok_s") or 0.0),
                    thinking_tok_s=float(raw.get("thinking_tok_s") or 0.0),
                    answer_tok_s=float(raw.get("answer_tok_s") or 0.0),
                    prefill_tok_s=float(raw.get("prefill_tok_s") or 0.0),
                    latency_ms=float(raw.get("latency_ms") or 0.0),
                    prompt_ms=float(raw.get("prompt_ms") or 0.0),
                    decode_ms=float(raw.get("decode_ms") or 0.0),
                    think_ms=float(raw.get("think_ms") or 0.0),
                    answer_ms=float(raw.get("answer_ms") or 0.0),
                    generated=int(raw.get("generated") or 0),
                    prompt_tokens=int(raw.get("prompt_tokens") or 0),
                )
                cfg: Config = {
                    "vram": int(raw["vram"]),
                    "pcie": float(raw["pcie"]),
                    "ple_io": raw["ple_io"],
                    "kv": raw["kv"],
                    "kvstream": raw["kvstream"],
                    "hostworker": raw["hostworker"],
                    "pcie_mode": raw["pcie_mode"],
                    "worker": int(raw["worker"]),
                    "tail": raw["tail"],
                    "draft_floor": float(raw["draft_floor"] if raw.get("draft_floor") is not None
                                         else raw.get("spec_min_p") or 0.8),
                }
            except (KeyError, ValueError, TypeError):
                continue
            if not metrics.ok():
                continue
            row = TrialRow("records", cfg, metrics, signed_score(metrics, objective))
            if best is None or row.score > best.score:
                best = row
    return best


def rebuild_from_records(args: argparse.Namespace, paths: ModelPaths, hw: Hardware) -> int:
    """Write the deliverable pair from measured trials already on disk. No engine boots."""
    row = best_from_records(Path(args.results_dir), args.model.lower(), args.objective)
    if row is None:
        raise SystemExit(
            f"HyperCore-Tune-Maya: no measured (status PASS) trial for model {args.model} "
            f"objective {args.objective} in {args.results_dir}"
        )
    engine = engine_args(row.cfg, paths.as_dict(), args.cntx, args.max_new, draft_floor_of(args))
    record = {
        "schema": "hypercore.max_run/1",
        "model": args.model,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "engine": str(args.engine),
        "hardware": hw.to_dict(),
        "hardware_slug": hw.slug(),
        "objective": args.objective,
        "context": args.cntx,
        "max_new": args.max_new,
        "spec_min_p": row.cfg.get("draft_floor", draft_floor_of(args)),
        "best": dict(row.cfg),
        "metrics": {
            "decode_tok_s": row.metrics.decode_tok_s,
            "thinking_tok_s": row.metrics.thinking_tok_s,
            "answer_tok_s": row.metrics.answer_tok_s,
            "prefill_tok_s": row.metrics.prefill_tok_s,
            "latency_ms": row.metrics.latency_ms,
            "prompt_ms": row.metrics.prompt_ms,
            "decode_ms": row.metrics.decode_ms,
            "think_ms": row.metrics.think_ms,
            "answer_ms": row.metrics.answer_ms,
        },
        "effective": row.metrics.effective,
        "conditions": {
            "engine": str(args.engine),
            "prompt_tokens": row.metrics.prompt_tokens,
            "max_new": args.max_new,
            "context": args.cntx,
            "repeat": args.repeat,
        },
        "prior_best": None,
        "verified": True,
        "args": engine,
        "study": {"mode": "records", "sampler": "none", "seed": args.seed,
                  "trials": 0, "measured": 1, "failed": 0, "repeat": args.repeat,
                  "complete": False},
        "source": "best measured row in the CSVs under " + str(args.results_dir),
        "trials": [],
    }
    args.deliverable_dir.mkdir(parents=True, exist_ok=True)
    sh, cfg = write_run_pair(
        args.model, paths.as_dict(), engine, record,
        deliverable_dir=args.deliverable_dir, port=args.port,
        open_browser=not args.no_browser,
        fallback_dir=args.results_dir,
    )
    print(f"rebuilt from measured records -> {sh}")
    print(f"                             {cfg}")
    print(f"  winner {json.dumps(row.cfg)}")
    print(f"  decode {row.metrics.decode_tok_s:.2f} tok/s, prefill {row.metrics.prefill_tok_s:.2f} "
          f"tok/s, latency {row.metrics.latency_ms:.1f} ms"
          + (f", thinking {row.metrics.thinking_tok_s:.2f} tok/s"
             if row.metrics.thinking else "")
          + f" ({args.objective})")
    return 0


def replay(record_path: Path, args: argparse.Namespace, paths: ModelPaths,
           hw: Hardware) -> int:
    """Re-emit the deliverables from a stored record, without booting the engine.

    A sweep that finished before the run-config deliverables existed, or one whose
    launcher was written to a read-only root, can be rebuilt from its record. The
    hardware block is refreshed to the machine doing the replay, so the pair always
    states where the numbers came from.
    """
    record = load_history(record_path)
    if record is None:
        raise SystemExit(f"HyperCore-Tune-Maya: {record_path} is not a readable tuning record")
    if not record.get("best") or not record.get("args"):
        raise SystemExit(f"HyperCore-Tune-Maya: {record_path} has no winning configuration to replay")
    if not record.get("verified", any(t.get("status") == "PASS" for t in record.get("trials", []))):
        raise SystemExit(
            f"HyperCore-Tune-Maya: {record_path} has no measured trial behind its best; "
            "use --from-records to rebuild from the CSVs instead"
        )
    record.setdefault("hardware", hw.to_dict())
    record["hardware_slug"] = hw.slug()
    record["model"] = args.model
    record["objective"] = record.get("objective") or args.objective
    record["context"] = record.get("context") or args.cntx
    sh, cfg = write_run_pair(
        args.model, paths.as_dict(), list(record["args"]), record,
        deliverable_dir=args.deliverable_dir, port=args.port,
        open_browser=not args.no_browser,
        fallback_dir=args.results_dir,
    )
    print(f"replayed {record_path} -> {sh}")
    print(f"                        {cfg}")
    print(f"  objective {record['objective']} at --max-context {record['context']}, "
          f"machine {hw.slug()}")
    return 0


def preflight(args: argparse.Namespace, paths: ModelPaths, hw: Hardware) -> None:
    """Say before the first boot whether the GPU can host this pack.

    A pack that cannot load fails every trial the same way, and a failed trial scores
    PENALTY - the same 0.0 as a real zero throughput - so fifty of them teach the sampler
    nothing and cost fifty boots. The pack's dense pool is the floor: the engine loads it
    into a device arena before it sizes the expert pool from the VRAM left over
    (glm_model.cu:1237), so below it no configuration in the grid runs at any setting.
    """
    if args.dry_run or args.no_preflight:
        return
    if str(args.engine).lower().endswith(".py"):
        return  # the stand-in engine in the test suite needs no GPU
    floor = pack_vram_floor_mib(paths.pack)
    free = free_vram_mib() if floor > 0 else 0
    if floor <= 0 or free <= 0:
        return  # nothing to check: an unreadable pack, or no nvidia-smi on this box
    if free < floor:
        raise SystemExit(
            f"HyperCore-Tune-Maya: {paths.pack} needs at least {floor} MiB of VRAM for its "
            f"dense pool and only {free} MiB is free. Stop the engine holding the GPU, or "
            f"run this with --no-preflight."
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root

    if args.pack:
        # An explicit --pack wins, but the rest of the model is still looked for: a
        # tokenizer beside the pack, the shipped expert profile, a drafter.
        paths = ModelPaths(pack=args.pack)
        try:
            extra = resolve_model_paths(root, args.model, args.data_dir)
        except FileNotFoundError:
            extra = ModelPaths()
        for key in ("tokenizer", "expert_profile", "dflash", "prompt_tokens"):
            if not getattr(paths, key) and getattr(extra, key):
                setattr(paths, key, getattr(extra, key))
    else:
        paths = resolve_model_paths(root, args.model, args.data_dir)
    if args.pack:
        paths.pack = args.pack
    if args.dflash:
        paths.dflash = args.dflash
    if args.prompt_file:
        args.prompt_tokens_file = args.prompt_file
        paths.prompt_tokens = args.prompt_file
    if args.prompt_tokens_file:
        paths.prompt_tokens = args.prompt_tokens_file
    prompt_is_text = Path(args.prompt_tokens_file or "").suffix.lower() in {".txt", ".md", ".text"}
    if not paths.prompt_tokens and not prompt_is_text and not paths.tokenizer:
        raise SystemExit(
            "HyperCore-Tune-Maya: no prompt source found. A tokenizer beside the pack is "
            "enough - otherwise pass --prompt-tokens-file <json with an 'input' list> or "
            "--prompt-file <text>."
        )
    args.prompt_tokens_file = paths.prompt_tokens or args.prompt_tokens_file

    params = build_space(vram_low=args.vram_min, vram_high=args.vram_max, vram_step=args.vram_step,
                        max_workers=args.max_workers, pack=paths.pack,
                        has_drafter=bool(paths.dflash),
                        request_dims=args.request_dims)
    allowed_kv = kv_modes_for(paths.pack)
    if args.kv and args.kv not in allowed_kv:
        raise SystemExit(
            f"HyperCore-Tune-Maya: --kv {args.kv} is not implemented by the "
            f"{pack_kind(paths.pack)} runner for {paths.pack}; use one of {allowed_kv}")
    frozen = dict(args.frozen)
    for p in params:
        if p.key not in frozen:
            continue
        value = frozen[p.key]
        if p.kind == "choice" and str(value) not in {str(c) for c in p.choices}:
            raise SystemExit(f"HyperCore-Tune-Maya: frozen {p.key}={value!r} is not one of {p.choices}")
        if p.kind == "int" and not (int(p.low) <= int(value) <= int(p.high)):
            raise SystemExit(
                f"HyperCore-Tune-Maya: frozen {p.key}={value} is outside [{int(p.low)}, {int(p.high)}]")
        if p.kind == "float" and not (float(p.low) <= float(value) <= float(p.high)):
            raise SystemExit(
                f"HyperCore-Tune-Maya: frozen {p.key}={value} is outside [{p.low}, {p.high}]")
    params = [p for p in params if p.key not in frozen]
    if not params:
        raise SystemExit("HyperCore-Tune-Maya: every dimension is frozen; nothing to search")

    hw = _hardware(args.hardware)
    history_file = Path(args.history_file) if args.history_file else history_path(
        args.history_dir, args.model, args.objective, args.cntx, hw)

    if args.replay:
        return replay(Path(args.replay), args, paths, hw)

    if args.from_records:
        return rebuild_from_records(args, paths, hw)

    priors: list[dict[str, Any]] = []
    foreign: list[dict[str, Any]] = []
    sources: list[Path] = []
    if args.no_load:
        print("history: --no-load, starting cold")
    else:
        sources = [history_file] if args.history_file else find_history(
            args.history_dir, args.model, args.objective, args.cntx, hw,
            allow_mismatch=args.allow_hardware_mismatch)
        for source in sources:
            record = load_history(source)
            if record:
                fill = record.get("spec_min_p")
                priors.extend(usable_trials(
                    record, args.max_new, 0.8 if fill is None else float(fill),
                    pack=paths.pack))
        if priors:
            print(f"history: {len(priors)} configurations from "
                  f"{', '.join(str(s) for s in sources)}")
        else:
            print(f"history: nothing for {hw.slug()} / {args.model} / {args.objective} "
                  f"/ {args.cntx}; starting cold")
        # A configuration measured under another objective has already been booted on this
        # machine with these flags. Its score is not this objective's score, so it is not a
        # prior; it is a point the study must not waste a boot on.
        if not args.history_file and not args.no_cross_objective:
            foreign_sources = other_objective_history(
                args.history_dir, args.model, args.cntx, hw,
                allow_mismatch=args.allow_hardware_mismatch, objective=args.objective)
            for source in foreign_sources:
                record = load_history(source)
                if record:
                    fill = record.get("spec_min_p")
                    foreign.extend(usable_trials(
                        record, args.max_new, 0.8 if fill is None else float(fill),
                        pack=paths.pack))
            if foreign:
                print(f"history: {len(foreign)} configurations measured under another "
                      f"objective ({', '.join(str(s.name) for s in foreign_sources)}) "
                      f"will not be booted again")

    print(f"optuna {optuna.__version__} | objective {args.objective} | mode {args.mode} "
          f"| sampler {args.sampler} | seed {args.seed} | context {args.cntx} "
          f"| trials {args.trials} | physical cores {physical_cores()}")
    print(f"machine: {hw.slug()}")
    print(f"engine: {args.engine}")
    print(f"pack:   {paths.pack}")
    if frozen:
        print("frozen: " + ", ".join(f"{k}={v}" for k, v in frozen.items()))
    preflight(args, paths, hw)

    sweeper = Sweeper(args, root, params, paths, hw, priors, foreign)
    t0 = time.monotonic()
    interrupted = sweeper.run()
    elapsed = time.monotonic() - t0

    csv_path = sweeper.write_csv()
    record = sweeper.record(csv_path)
    record["elapsed_s"] = round(elapsed, 1)
    record["interrupted"] = interrupted

    record_path = sweeper.write_record(record)
    print(f"record: {record_path}")
    hist_path = save_history(history_file, record)
    print(f"history: {hist_path} ({record['history']['trials_loaded']} priors loaded, "
          f"{record['history']['reused']} configurations reused, "
          f"{record['history']['duplicates_skipped']} duplicate suggestions skipped)")

    if not args.no_deliverables:
        winner = sweeper.winner()
        if winner is None:
            print("no configuration was evaluated; nothing written", file=sys.stderr)
            return 1
        if not record["verified"]:
            if args.dry_run:
                print("dry run: nothing was measured, so no deliverable was written; "
                      "the command lines are in the CSV")
                return 0
            print("no configuration produced tokens in this run, so the deliverables were "
                  "not written. Free the GPU (./restart.sh stop) and re-run, or use "
                  "--from-records to rebuild them from measured trials.", file=sys.stderr)
            return 1
        args.deliverable_dir.mkdir(parents=True, exist_ok=True)
        sh_path, json_path = sweeper.write_deliverables(record)
        booted = sum(1 for r in sweeper.rows if not r.from_history and r.metrics.ok())
        print(f"\nwinner: {json.dumps(winner.cfg)}")
        print(f"score:  {winner.score:.4f} ({args.objective}) "
              f"decode {winner.metrics.decode_tok_s:.2f} tok/s, "
              f"prefill {winner.metrics.prefill_tok_s:.2f} tok/s, "
              f"latency {winner.metrics.latency_ms:.1f} ms"
              + (f", thinking {winner.metrics.thinking_tok_s:.2f} tok/s "
                 f"({winner.metrics.thinking_tokens} reasoning tokens over "
                 f"{winner.metrics.think_ms:.0f} ms)"
                 if winner.metrics.thinking else ""))
        if winner.metrics.transfer:
            t = winner.metrics.transfer
            print(f"bus:    {t['transfer_share']:.1%} of a token is off-card time "
                  f"({t['off_card_ms_per_tok']:.2f} ms/tok)"
                  + (f" - measured by the engine: {t['disk_ms_per_tok']:.2f} ms SSD, "
                     f"{t['cpu_ms_per_tok']:.2f} ms CPU lane" if t["measured_times"] else "")
                  + (f"; {t['disk_reads_per_tok']:.2f} reads/tok at "
                     f"{t['disk_ms_per_read']:.2f} ms each, "
                     f"{t['disk_bytes_per_s'] / 1e6:.0f} MB/s"
                     if t.get("disk_ms_per_read") else "")
                  + (f" - the card alone could do {t['compute_tok_s']:.1f} tok/s"
                     if t["compute_tok_s"] > winner.metrics.decode_tok_s else ""))
            for flag in t.get("flags", []):
                print(f"  ! {flag}")
            lines = frontier([row.csv_row() for row in sweeper.rows])
            if len(lines) > 1:
                print("frontier: configurations nothing else beats on both rate and bus")
                for line in lines:
                    print(line)
        amb = winner.metrics.ambient or {}
        if amb.get("swap_used_mb", 0) > 0 or amb.get("tier_resident") is False:
            print(f"ambient: {amb.get('free_ram_mb', 0):.0f} MB free of "
                  f"{amb.get('total_ram_mb', 0):.0f} MB host RAM, "
                  f"{amb.get('swap_used_mb', 0):.0f} MB in swap; the "
                  f"{amb.get('tier_mib', 0)} MiB tier has "
                  f"{amb.get('tier_headroom_mb', 0):.0f} MB headroom - a tier that "
                  f"cannot stay resident sets vram_hit, not the swept knobs")
        if winner.from_history:
            print("note: this configuration was carried in from history; nothing was "
                  "booted in this run, so these numbers are not from this sweep")
        else:
            print(f"measured here: {booted} configurations booted, "
                  f"prompt {len(sweeper.prompt_ids)} tokens, max_new {args.max_new}, "
                  f"context {args.cntx}, engine {args.engine}")
        prior = sweeper.incumbent
        if prior is not None and prior is not winner and prior.score > winner.score:
            print(f"prior best on this machine, not reproduced: score {prior.score:.4f} "
                  f"decode {prior.metrics.decode_tok_s:.2f} tok/s "
                  f"(measured by {prior.engine or 'an unknown engine'}; "
                  "different engine or prompt, so not comparable)")
        print(f"wrote  {sh_path}")
        print(f"wrote  {json_path}")

    print(f"csv:    {csv_path}")
    print(f"trials: {len(sweeper.rows)} evaluated, {record['study']['measured']} measured, "
          f"{record['study']['failed']} failed, in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
