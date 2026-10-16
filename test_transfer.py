"""HyperCore-Tune-Maya/test_transfer.py - the bus telemetry and the balance objective.

Runs without a GPU: the figures come from the engine lines a sweep would have read, and the
end-to-end case boots stub_engine.py, which prints the same telemetry the real engine does.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from engine_runner import think_end_token
from transfer import (
    TRANSFER_COLUMNS,
    Transfer,
    balance_score,
    expert_blob_bytes,
    knee,
    pareto_front,
)

STAT = (
    "glm stat: decode 244.11 ms/tok (4.10 tok/s) | vram hit 64.18% | ram fetches 43.34/tok | "
    "disk 46.48 reads/tok in 28.54 waits/tok, 189.65 ms/tok | promotions 57.30/tok | "
    "cpu lane 24.21/tok in 4.58 ms/tok | prompt 375.2 ms/tok | vram 2393/2461 ram 2934/3015"
)
KV = "glm kv: KV streaming: 91.20% of 8000 block reads hit VRAM, 12.5 MiB read from RAM - OVERFLOW"
SPEC = "glm spec: 12 of 20 drafts accepted (60.00%)"


def test_stat_line_is_read_field_by_field() -> None:
    parsed = Transfer.from_lines([STAT], blob_bytes=7012352.0)
    assert parsed.decode_ms_per_tok == 244.11
    assert parsed.vram_hit_pct == 64.18
    assert parsed.disk_reads_per_tok == 46.48
    assert parsed.cpu_ms_per_tok == 4.58
    assert parsed.vram_used == 2393 and parsed.vram_slots == 2461
    # 189.65 + 4.58 of a 244.11 ms token is 79.6% off-card; the rest is on the card.
    assert abs(parsed.transfer_share - 0.7957) < 1e-4
    assert abs(parsed.compute_tok_s - 20.05) < 0.01
    assert parsed.headroom > 1.0


def test_lines_the_engine_does_not_print_change_nothing() -> None:
    quiet = Transfer.from_lines(["READY 32768", "DONE 512 1024 900.0 2000.0 length 0 0 0 0 0"])
    assert not quiet.present()
    assert balance_score(186.15, quiet) == 186.15
    assert balance_score(186.15, None) == 186.15
    assert Transfer.from_lines([STAT]).blob_bytes is None
    assert Transfer.from_lines([STAT]).bus_bytes_per_tok is None


def test_kv_and_spec_lines_are_read() -> None:
    parsed = Transfer.from_lines([STAT, KV, SPEC])
    assert parsed.kv_hit_pct == 91.20
    assert parsed.kv_overflow is True
    assert parsed.spec_accepted == 12 and parsed.spec_steps == 20


def test_a_bus_bound_configuration_scores_below_a_resident_one() -> None:
    """260 tok/s with a third of the token on the SSD loses to 250 tok/s that is resident."""
    bus_bound = Transfer.from_lines([
        ("glm stat: decode 3.85 ms/tok (260.00 tok/s) | vram hit 30.00% | ram fetches 200.00/tok | "
         "disk 180.00 reads/tok in 90.00 waits/tok, 2.40 ms/tok | promotions 20.00/tok | "
         "cpu lane 40.00/tok in 0.90 ms/tok | prompt 1.0 ms/tok | vram 1000/24000 ram 500/1000")
    ])
    resident = Transfer.from_lines([
        ("glm stat: decode 4.00 ms/tok (250.00 tok/s) | vram hit 92.00% | ram fetches 10.00/tok | "
         "disk 2.00 reads/tok in 1.00 waits/tok, 0.10 ms/tok | promotions 1.00/tok | "
         "cpu lane 3.00/tok in 0.05 ms/tok | prompt 1.0 ms/tok | vram 22000/24000 ram 100/1000")
    ])
    assert balance_score(260.0, bus_bound) < balance_score(250.0, resident)
    assert "bus-bound" in " ".join(bus_bound.flags())
    assert not resident.flags()


def test_flags_name_a_full_pool_and_an_overflowing_kv() -> None:
    parsed = Transfer.from_lines([STAT, KV])
    flags = " ".join(parsed.flags())
    assert "KV streaming overflowed" in flags
    assert parsed.tier_pressure["vram_pool"] > 0.97


def test_blob_bytes_come_from_the_pack_index() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "native_experts.txt").write_text(
            "# header\n3 16 18 0 1000 1 2 3\n4 16 18 5 2000 4 5 6\n", encoding="utf-8"
        )
        assert expert_blob_bytes(tmp) == 1500.0
    assert expert_blob_bytes(None) is None
    assert expert_blob_bytes("/definitely/not/a/pack") is None


STAT_LIVE = (
    "STAT tok_s=7.66 ms_tok=130.57 vram_hit=0.7553 ram_fetch=3.934 disk=21.703 promo=19.928 "
    "vram_used=1627 vram_slots=1736 ram_used=2994 ram_slots=3012 ram_gb=20.0 vram_gb=11.5 active=1"
)
LANE = ("glm fast: CUDA0 CPU lane: 24 threads, an expert 0.136 ms on the CPU vs 0.453 ms over "
        "PCIe -> of 0..8 RAM-tier experts the CPU takes 0 1 2 3 3 4 5 6 6")
WARM = "glm fast: CUDA0 tiers warm: 4614 experts (32.8 GB) in 18 s (1.87 GB/s)"


def test_the_serve_path_stat_line_is_read() -> None:
    """A --serve engine with STRATA_GLM_SEQS > 1 never prints glm stat; STAT is the only
    telemetry it gives, so the balance channel has to work from it alone."""
    parsed = Transfer.from_lines([LANE, WARM, STAT_LIVE], blob_bytes=7012352.0)
    assert parsed.live_lines == 1 and not parsed.measured_times
    assert parsed.decode_ms_per_tok == 130.57
    assert parsed.vram_hit_pct == 75.53
    assert parsed.disk_reads_per_tok == 21.703
    assert parsed.cpu_ms_per_expert == 0.136 and parsed.pcie_ms_per_expert == 0.453
    # 30 of the 36 RAM-tier experts go to the CPU: (30*0.136 + 6*0.453) / 36
    assert abs(parsed.ram_ms_per_expert - 0.1888) < 1e-3
    # one 7 MB blob at 1.87 GB/s is 3.75 ms
    assert abs(parsed.disk_ms_per_read - 3.75) < 0.1
    assert parsed.disk_saturation is not None and parsed.disk_saturation > 0.5
    assert parsed.transfer_share > 0.2


def test_the_last_line_of_a_series_is_the_one_used() -> None:
    """STAT counters are cumulative from the start of the request, so the last line is the
    request's own total; the earlier ones are the tiers still warming up."""
    cold = "STAT tok_s=2.10 ms_tok=476.19 vram_hit=0.4554 ram_fetch=0.875 disk=121.000 promo=73.250 vram_used=1651 vram_slots=1736 ram_used=2981 ram_slots=3012"
    warm = "STAT tok_s=7.66 ms_tok=130.57 vram_hit=0.7553 ram_fetch=3.934 disk=21.703 promo=19.928 vram_used=1627 vram_slots=1736 ram_used=2994 ram_slots=3012"
    parsed = Transfer.from_lines([cold, warm], blob_bytes=7012352.0)
    assert parsed.decode_ms_per_tok == 130.57
    assert parsed.disk_reads_per_tok == 21.703
    assert parsed.raw["trend_lines"] == 2


def test_a_real_engine_log_yields_a_real_number() -> None:
    logs = sorted(Path(__file__).resolve().parent.glob("results/run-2026-10-16_00*.log"))
    if not logs:
        return
    parsed = Transfer.from_lines(
        logs[0].read_text(encoding="utf-8", errors="replace").splitlines(),
        blob_bytes=expert_blob_bytes("/home/ai-user0/Maya-data/models/Maya-S24/pack"),
    )
    assert parsed.present(), "the sweep logs contain STAT lines"
    assert parsed.decode_ms_per_tok > 0 and parsed.transfer_share > 0
    assert parsed.disk_reads_per_tok > 0


def test_the_host_counter_cross_check_is_reported() -> None:
    """psutil is the only measurement of drive traffic independent of the engine's own
    counters. The ratio against the derived figure tells a reader whether the attribution
    can be trusted, so it is recorded and never silently zeroed."""
    parsed = Transfer.from_lines([LANE, WARM, STAT_LIVE], blob_bytes=7012352.0,
                                 measured_disk_bytes_per_s=1_500_000.0)
    derived = parsed.disk_bytes_per_s
    assert derived and derived > 0
    assert abs(parsed.disk_attribution - 1_500_000.0 / derived) < 1e-6
    assert "host counters" in parsed.summary()
    assert "measured_disk_bytes_per_s" in TRANSFER_COLUMNS
    # No host counter: the ratio is absent, not zero.
    assert Transfer.from_lines([STAT_LIVE], blob_bytes=7012352.0).disk_attribution is None


def test_the_snapshot_never_raises_without_psutil_or_a_drive() -> None:
    from hardware import disk_io_snapshot
    snapshot = disk_io_snapshot()
    assert set(snapshot) == {"read_bytes", "write_bytes"}
    assert all(value >= 0.0 for value in snapshot.values())


def test_the_link_cross_check_is_reported_when_nvml_reports() -> None:
    """NVML's rx counter is the GPU receiving experts, which is what the RAM-tier fetch
    count describes, so the two can be divided. Without NVML the field is None, never 0."""
    parsed = Transfer.from_lines([LANE, WARM, STAT_LIVE], blob_bytes=7012352.0,
                                 measured_pcie_rx_mb_s=1000.0)
    derived = parsed.pcie_bytes_per_s
    assert derived and abs(parsed.pcie_attribution - 1e9 / derived) < 1e-6
    assert "NVML saw 1000 MB/s" in parsed.summary()
    assert "measured_pcie_rx_mb_s" in TRANSFER_COLUMNS
    assert Transfer.from_lines([STAT_LIVE]).pcie_attribution is None


def test_the_sampler_is_safe_when_nvml_is_absent() -> None:
    """start() and stop() agree: a host without NVML samples nothing and the derived
    figures stand alone. Nothing raises either way."""
    from engine_runner import PcieSampler
    sampler = PcieSampler(interval=0.05)
    started = sampler.start()
    result = sampler.stop()
    assert started == bool(result)
    if result:
        assert result["samples"] >= 1 and result["rx_mean_mb_s"] >= 0.0


def test_knee_stops_where_more_resource_stops_paying() -> None:    # 10 -> 13 -> 13.2 -> 11: the gain at index 2 is under 3%, and the drop at 3 is below 70%.
    assert knee([(0.0, 10.0), (0.25, 13.0), (0.5, 13.2), (0.75, 11.0)]) == 2
    assert knee([(0.0, 10.0), (0.25, 13.0), (0.5, 17.0)]) is None
    assert knee([]) is None


def test_pareto_front_drops_a_configuration_worse_on_every_axis() -> None:
    rows = [
        {"decode_tok_s": 200.0, "off_card_ms_per_tok": 1.0, "transfer_share": 0.10},
        {"decode_tok_s": 150.0, "off_card_ms_per_tok": 2.0, "transfer_share": 0.20},
        {"decode_tok_s": 210.0, "off_card_ms_per_tok": 0.5, "transfer_share": 0.05},
    ]
    front = pareto_front(rows)
    assert len(front) == 1
    assert front[0]["decode_tok_s"] == 210.0


def test_a_sweep_records_the_bus_and_balance_uses_it() -> None:
    """End to end, no GPU: the balance objective reads the stub's telemetry and the CSV
    carries it. The discounted score can never beat the plain decode score."""
    pack = Path("/home/ai-user0/Maya-data/models/Maya-S24/pack")
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, str(Path(__file__).resolve().parent / "bayesian_tune.py"),
               "--engine", "stub_engine.py", "--balance", "--trials", "3", "--seed", "5",
               "--model", "maya-s24", "--results-dir", tmp, "--deliverable-dir", tmp,
               "--max-new", "256", "--cntx", "8192", "--prompt-tokens", "1024"]
        if pack.is_dir():
            cmd += ["--pack", str(pack)]
        # test_bayesian_tune.py sets MAYA_ROOT to a fixture tree; a subprocess inheriting it
        # would resolve the model against the fixture instead of the real pack.
        env = {key: value for key, value in os.environ.items() if key != "MAYA_ROOT"}
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                            check=False, env=env)
        assert out.returncode == 0, out.stderr[-400:]
        csv_path = next(Path(tmp).glob("*.csv"))
        with open(csv_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows, "the sweep wrote no trials"
        for column in TRANSFER_COLUMNS:
            assert column in rows[0], f"{column} missing from the results CSV"
        assert float(rows[0]["transfer_share"]) > 0.0
        assert "bus" in out.stdout, "the winner report should state the bus"
        # The recorded score is the discounted rate, not the raw one.
        row = max(rows, key=lambda r: float(r["score"]))
        expected = balance_score(float(row["decode_tok_s"]), {"transfer_share": float(row["transfer_share"])})
        assert abs(float(row["score"]) - expected) < 0.01, (row["score"], expected)
        assert float(row["score"]) < float(row["decode_tok_s"])


def test_blob_bytes_come_from_the_real_pack() -> None:
    pack = Path("/home/ai-user0/Maya-data/models/Maya-S24/pack")
    if pack.is_dir():
        blob = expert_blob_bytes(pack)
        assert blob and blob > 1_000_000, "an expert blob is megabytes, not bytes"


def test_think_end_is_read_from_the_pack_that_has_one() -> None:
    """The id that closes reasoning is per-pack. Guessing one is what makes a sweep report
    thinking 0.0 tok/s: the engine accepts any id, so the split never fires."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "vocab.json").write_text(json.dumps({"hello": 1, "</think>": 4211}),
                                         encoding="utf-8")
        assert think_end_token(str(root)) == 4211
        # No </think>: fall through to the pack's end-of-turn token.
        (root / "vocab.json").write_text(json.dumps({"<|im_end|>": 7}), encoding="utf-8")
        assert think_end_token(str(root)) == 7
        # Nothing at all: None, so the caller uses the documented default and says so.
        (root / "vocab.json").write_text(json.dumps({"hello": 1}), encoding="utf-8")
        assert think_end_token(str(root)) is None
        assert think_end_token(None) is None
        assert think_end_token("/definitely/not/a/tokenizer") is None
    pack = Path("/home/ai-user0/Maya-data/models/Maya-S24/pack/tokenizer")
    if pack.is_dir():
        # Verified against pack/tokenizer/vocab.json: </think> is 154842.
        assert think_end_token(str(pack)) == 154842


def test_a_sweep_sends_the_pack_think_end_not_a_guess() -> None:
    """With --think-budget the GEN line must carry the pack's own close-of-think id."""
    pack = Path("/home/ai-user0/Maya-data/models/Maya-S24/pack")
    if not pack.is_dir():
        return
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, str(Path(__file__).resolve().parent / "bayesian_tune.py"),
               "--engine", "stub_engine.py", "--balance", "--trials", "1", "--seed", "3",
               "--model", "maya-s24", "--pack", str(pack), "--results-dir", tmp,
               "--deliverable-dir", tmp, "--max-new", "256", "--cntx", "8192",
               "--prompt-tokens", "1024", "--think-budget", "64"]
        env = {key: value for key, value in os.environ.items() if key != "MAYA_ROOT"}
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                            check=False, env=env)
        assert out.returncode == 0, out.stderr[-400:]
        assert "think-end: this pack closes reasoning with token 154842" in out.stdout
        records = list(Path(tmp).glob("*.json"))
        assert records, "the sweep wrote no record"
        conditions = None
        for path in records:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and "conditions" in data:
                conditions = data["conditions"]
                break
        assert conditions is not None, "no record carried a conditions block"
        assert conditions["think_budget"] == 64
        assert conditions["think_end"] == 154842, conditions
