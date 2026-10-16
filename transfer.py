"""HyperCore-Tune-Maya/transfer.py - the engine's transfer telemetry, and the balance it implies.

A trial is only worth keeping when its throughput is not paid for by the bus. The engine
already says what crossed one, and engine_runner reads every line the process prints (stderr
is merged into stdout at spawn):

    STAT tok_s= ms_tok= vram_hit= ram_fetch= disk= promo= vram_used=/slots ram_used=/slots
        the live tier counters, every 8 tokens of a request (generate.cpp:1820, 2122). This
        is what a --serve engine emits; it is the same source the Monitor in
        project-maya/serve/server.py uses.
    glm stat: decode ... ms/tok | ... | disk ... ms/tok | cpu lane ... ms/tok
        the per-request summary, printed only on the single-conversation path
        (generate.cpp:1925). With STRATA_GLM_SEQS > 1 it never runs, which is why a parallel
        sweep sees no per-request ms figures.
    glm kv: / glm spec: / glm skip:                                    generate.cpp:1940-1951
    glm fast: CUDA0 CPU lane: an expert 0.136 ms on the CPU vs 0.453 ms over PCIe ...
    glm fast: CUDA0 tiers warm: 4614 experts (32.8 GB) in 18 s (1.87 GB/s)
        the machine's own measured unit costs, taken at boot: how long one expert takes on
        the host, over PCIe, and off the SSD.

The quantities that matter for a performance/resource balance:

    decode_ms_per_tok        the whole token, as the engine times it
    off_card_ms_per_tok      the part of it spent on the SSD and on the RAM/CPU lane
    transfer_share           how transfer-bound a token is, in [0, 1]
    compute_tok_s            what the same token would cost if the bus were free
    bus_bytes_per_tok        experts moved per token times their blob size
    disk_saturation          how close the drive is to the throughput this machine measured

`balance_score` is the rate the search maximises under `--balance`: throughput discounted by
how transfer-bound the configuration is. With no telemetry it returns the plain decode rate,
so a sweep never penalises a machine for not reporting.

Where the times come from matters. If the engine printed them (`glm stat`), they are used as
measured. If it only printed the counts - the serve path with more than one conversation -
the times are attributed with the unit costs the same engine measured at boot. That sum is
an upper bound on wall time, because the engine pipelines SSD reads (ple_inflight), so
`transfer_share` is clamped to 1 and the raw counts are kept alongside it. The floor is the
engine's own hit ratio: `1 - vram_hit` is the share of expert lookups that had to come from
off the card, which needs no modelling at all.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

# The exact formats, from project-maya/src/program/generate.cpp:1928, :1940, :1944, :1951.
STAT_RE = re.compile(
    r"glm stat: decode (?P<decode_ms>[\d.]+) ms/tok \((?P<tok_s>[\d.]+) tok/s\) \| "
    r"vram hit (?P<vram_hit_pct>[\d.]+)% \| ram fetches (?P<ram_fetches>[\d.]+)/tok \| "
    r"disk (?P<disk_reads>[\d.]+) reads/tok in (?P<disk_waits>[\d.]+) waits/tok, "
    r"(?P<disk_ms>[\d.]+) ms/tok \| promotions (?P<promotions>[\d.]+)/tok \| "
    r"cpu lane (?P<cpu_experts>[\d.]+)/tok in (?P<cpu_ms>[\d.]+) ms/tok \| "
    r"prompt (?P<prompt_ms>[\d.]+) ms/tok \| "
    r"vram (?P<vram_used>\d+)/(?P<vram_slots>\d+) ram (?P<ram_used>\d+)/(?P<ram_slots>\d+)"
)
KV_RE = re.compile(
    r"glm kv: KV streaming: (?P<kv_hit_pct>[\d.]+)% of (?P<kv_lookups>\d+) block reads "
    r"hit VRAM, (?P<kv_mib>[\d.]+) MiB read from RAM(?P<overflow> - OVERFLOW)?"
)
SPEC_RE = re.compile(r"glm spec: (?P<accepted>\d+) of (?P<steps>\d+) drafts accepted")
SKIP_RE = re.compile(r"glm skip: (?P<skips>[\d.]+) disk-only experts/tok left out")
TIMING_RE = re.compile(r"glm fast timing: (?P<tokens>\d+) tokens, (?P<ms_per_tok>[\d.]+) ms/token")

# The live per-request counters a --serve engine prints every 8 tokens (generate.cpp:1820).
# vram_hit is a fraction, not a percentage.
LIVE_RE = re.compile(
    r"STAT tok_s=(?P<tok_s>[\d.]+) ms_tok=(?P<decode_ms>[\d.]+) vram_hit=(?P<vram_hit>[\d.]+) "
    r"ram_fetch=(?P<ram_fetches>[\d.]+) disk=(?P<disk_reads>[\d.]+) promo=(?P<promotions>[\d.]+) "
    r"vram_used=(?P<vram_used>\d+) vram_slots=(?P<vram_slots>\d+) "
    r"ram_used=(?P<ram_used>\d+) ram_slots=(?P<ram_slots>\d+)"
)
# "an expert 0.136 ms on the CPU vs 0.453 ms over PCIe -> of 0..8 RAM-tier experts the CPU takes 0 1 2 3 3 4 5 6 6"
LANE_RE = re.compile(
    r"an expert (?P<cpu_ms>[\d.]+) ms on the CPU vs (?P<pcie_ms>[\d.]+) ms over PCIe"
    r" -> of 0\.\.(?P<max_experts>\d+) RAM-tier experts the CPU takes (?P<lane>[\d ]+)"
)
# "tiers warm: 4614 experts (32.8 GB) in 18 s (1.87 GB/s)" - this machine's SSD throughput.
WARM_RE = re.compile(r"tiers warm: (?P<experts>\d+) experts \((?P<gb>[\d.]+) GB\) in (?P<seconds>\d+) s")

# An expert blob the engine moves as one unit. None when the pack has not been read.
DEFAULT_BLOB_BYTES: float | None = None

# How much of the discount --balance applies to the transfer share of a token.
TRANSFER_WEIGHT = 0.35


# The columns a trial record gains from the telemetry, in the order as_dict() writes them.
TRANSFER_COLUMNS: tuple[str, ...] = (
    "decode_ms_per_tok", "off_card_ms_per_tok", "transfer_share", "off_card_lookup_share",
    "compute_tok_s", "headroom", "disk_ms_per_read", "disk_saturation", "measured_times",
    "disk_bytes_per_s", "pcie_bytes_per_s", "disk_reads_per_tok", "disk_ms_per_tok",
    "ram_fetches_per_tok", "promotions_per_tok", "cpu_experts_per_tok", "cpu_ms_per_tok",
    "vram_hit_pct", "kv_hit_pct", "kv_mib_from_ram", "spec_accepted", "spec_steps", "flags",
)


def expert_blob_bytes(pack_dir: str | Path | None) -> float | None:
    """The size of one expert blob, from the pack's own index.

    `pack/native_experts.txt` is a header line followed by one line per layer:
    `layer gu_type d_type offset blob_bytes gate_off up_off down_off ...`. The blob is the
    unit the engine moves when it promotes or reads an expert, so it is the multiplier that
    turns reads/tok into bytes/s. Returns None when the pack has no such file - the counts
    are still usable, only not in bytes.
    """
    if pack_dir is None:
        return None
    path = Path(pack_dir).expanduser() / "native_experts.txt"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    sizes: list[float] = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 5:
            try:
                sizes.append(float(parts[4]))
            except ValueError:
                continue
    return median(sizes) if sizes else None


def parse_live_stat(line: str) -> dict[str, float] | None:
    """The `STAT key=value` line a --serve engine emits during a request.

    vram_hit comes back as a fraction; callers convert it to a percentage.
    """
    match = LIVE_RE.search(line)
    if match is None:
        return None
    out = {key: float(value) for key, value in match.groupdict().items() if value is not None}
    out["vram_hit_pct"] = 100.0 * out.pop("vram_hit", 0.0)
    return out


def parse_unit_costs(line: str) -> dict[str, float] | None:
    """The engine's measured cost of one expert, on the CPU and over PCIe.

    The CPU-lane vector says, for a layer with k experts in the RAM tier, how many the host
    thread takes. Averaged over the k the engine reported, that is the real cost of a RAM
    fetch on this machine - cheaper than a PCIe round trip, which is why the lane exists.
    """
    match = LANE_RE.search(line)
    if match is None:
        return None
    cpu_ms = float(match.group("cpu_ms"))
    pcie_ms = float(match.group("pcie_ms"))
    lane = [int(x) for x in match.group("lane").split()]
    taken = sum(lane)
    total = sum(range(len(lane)))
    out = {"cpu_ms_per_expert": cpu_ms, "pcie_ms_per_expert": pcie_ms}
    if total:
        out["ram_ms_per_expert"] = (taken * cpu_ms + (total - taken) * pcie_ms) / total
    else:
        out["ram_ms_per_expert"] = pcie_ms
    return out


def parse_warm_line(line: str) -> dict[str, float] | None:
    """The warm-up line: how fast this machine can pull experts off the SSD."""
    match = WARM_RE.search(line)
    if match is None:
        return None
    gb = float(match.group("gb"))
    seconds = float(match.group("seconds"))
    experts = int(match.group("experts"))
    if seconds <= 0 or experts <= 0:
        return None
    return {"warm_gb_per_s": gb / seconds, "warm_experts_per_s": experts / seconds}


def parse_stat_line(line: str) -> dict[str, float] | None:
    """The per-request `glm stat` line as numbers, or None when it is not one."""
    match = STAT_RE.search(line)
    if match is None:
        return None
    return {key: float(value) for key, value in match.groupdict().items() if value is not None}


def parse_kv_line(line: str) -> dict[str, Any] | None:
    """The KV-streaming line: hit rate, block reads, MiB from RAM, and the overflow flag."""
    match = KV_RE.search(line)
    if match is None:
        return None
    out: dict[str, Any] = {
        "kv_hit_pct": float(match.group("kv_hit_pct")),
        "kv_lookups": int(match.group("kv_lookups")),
        "kv_mib_from_ram": float(match.group("kv_mib")),
        "kv_overflow": match.group("overflow") is not None,
    }
    return out


def parse_spec_line(line: str) -> dict[str, float] | None:
    """The drafting line: how many drafts were offered and accepted."""
    match = SPEC_RE.search(line)
    if match is None:
        return None
    steps = int(match.group("steps"))
    accepted = int(match.group("accepted"))
    return {"spec_accepted": accepted, "spec_steps": steps,
            "spec_hit_pct": 100.0 * accepted / steps if steps else 0.0}


def parse_skip_line(line: str) -> dict[str, float] | None:
    """Experts the engine chose never to read at all."""
    match = SKIP_RE.search(line)
    if match is None:
        return None
    return {"disk_skips_per_tok": float(match.group("skips"))}


@dataclass
class Transfer:
    """What crossed a bus while one request decoded, per token.

    Every field is a per-token figure taken straight from the engine; nothing is estimated.
    Fields the engine did not print stay 0.0 and are reported as absent by `present()`.
    """

    decode_ms_per_tok: float = 0.0
    vram_hit_pct: float = 0.0
    ram_fetches_per_tok: float = 0.0
    disk_reads_per_tok: float = 0.0
    disk_waits_per_tok: float = 0.0
    disk_ms_per_tok: float = 0.0
    promotions_per_tok: float = 0.0
    cpu_experts_per_tok: float = 0.0
    cpu_ms_per_tok: float = 0.0
    prompt_ms_per_tok: float = 0.0
    vram_used: int = 0
    vram_slots: int = 0
    ram_used: int = 0
    ram_slots: int = 0
    kv_hit_pct: float = 0.0
    kv_lookups: int = 0
    kv_mib_from_ram: float = 0.0
    kv_overflow: bool = False
    spec_accepted: int = 0
    spec_steps: int = 0
    disk_skips_per_tok: float = 0.0
    blob_bytes: float | None = DEFAULT_BLOB_BYTES
    # The engine's own unit costs, from its boot lines. None when it printed none.
    cpu_ms_per_expert: float | None = None
    pcie_ms_per_expert: float | None = None
    ram_ms_per_expert: float | None = None
    disk_gb_per_s: float | None = None
    stat_lines: int = 0
    live_lines: int = 0
    measured_times: bool = False   # the glm stat line gave the ms figures directly
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_lines(cls, lines: Iterable[str], blob_bytes: float | None = None) -> Transfer:
        """Fold every telemetry line in an engine log into one record.

        The per-request summary (`glm stat`) wins where it exists: it is the engine's own
        timing of the request. A parallel --serve run never prints it, so the live `STAT`
        counters are used instead, and the times are attributed with the unit costs the same
        engine measured at boot.
        """
        stats: list[dict[str, float]] = []
        live: list[dict[str, float]] = []
        kv: dict[str, Any] | None = None
        spec: dict[str, float] | None = None
        skip: dict[str, float] | None = None
        costs: dict[str, float] = {}
        warm: dict[str, float] | None = None
        for line in lines:
            if line.startswith("glm stat:"):
                parsed = parse_stat_line(line)
                if parsed:
                    stats.append(parsed)
            elif line.startswith("STAT "):
                parsed = parse_live_stat(line)
                if parsed:
                    live.append(parsed)
            elif line.startswith("glm kv:"):
                kv = parse_kv_line(line) or kv
            elif line.startswith("glm spec:"):
                spec = parse_spec_line(line) or spec
            elif line.startswith("glm skip:"):
                skip = parse_skip_line(line) or skip
            elif "ms on the CPU vs" in line:
                costs = parse_unit_costs(line) or costs
            elif "tiers warm:" in line:
                warm = parse_warm_line(line) or warm
        if not stats and not live and not kv and not spec and not skip:
            return Transfer(blob_bytes=blob_bytes)
        out = Transfer(blob_bytes=blob_bytes, stat_lines=len(stats), live_lines=len(live))
        out.cpu_ms_per_expert = costs.get("cpu_ms_per_expert")
        out.pcie_ms_per_expert = costs.get("pcie_ms_per_expert")
        out.ram_ms_per_expert = costs.get("ram_ms_per_expert")
        if warm and warm["warm_experts_per_s"] > 0 and blob_bytes:
            out.disk_gb_per_s = warm["warm_gb_per_s"]
        source = stats or live
        if stats:
            out.measured_times = True
        keys = ("decode_ms_per_tok", "vram_hit_pct", "ram_fetches_per_tok",
               "disk_reads_per_tok", "disk_waits_per_tok", "disk_ms_per_tok",
               "promotions_per_tok", "cpu_experts_per_tok", "cpu_ms_per_tok",
               "prompt_ms_per_tok")
        names = ("decode_ms", "vram_hit_pct", "ram_fetches", "disk_reads", "disk_waits",
                 "disk_ms", "promotions", "cpu_experts", "cpu_ms", "prompt_ms")
        # The last line of a kind is the request's own total: the counters are cumulative
        # from the start of the request, and the last STAT's ms_tok is exactly the DONE
        # line's decode_ms / generated. Averaging the whole series would report the warm-up.
        last = source[-1]
        for field_name, name in zip(keys, names):
            if name in last:
                setattr(out, field_name, last[name])
        out.vram_used = int(last.get("vram_used", 0))
        out.vram_slots = int(last.get("vram_slots", 0))
        out.ram_used = int(last.get("ram_used", 0))
        out.ram_slots = int(last.get("ram_slots", 0))
        if len(source) > 1:
            # The series is the tier warming up: the first reading is the cold path, the
            # last the settled one. Both are kept so a reader can see the climb.
            out.raw = {"first": dict(source[0]), "trend_lines": len(source)}
        if kv:
            out.kv_hit_pct = kv["kv_hit_pct"]
            out.kv_lookups = kv["kv_lookups"]
            out.kv_mib_from_ram = kv["kv_mib_from_ram"]
            out.kv_overflow = kv["kv_overflow"]
        if spec:
            out.spec_accepted = spec["spec_accepted"]
            out.spec_steps = spec["spec_steps"]
        if skip:
            out.disk_skips_per_tok = skip["disk_skips_per_tok"]
        return out

    # ---- derived quantities -------------------------------------------------------------

    def present(self) -> bool:
        """True when the engine actually reported anything."""
        return (self.stat_lines > 0 or self.live_lines > 0
                or self.kv_lookups > 0 or self.spec_steps > 0)

    @property
    def disk_ms_per_read(self) -> float | None:
        """How long one expert read from the SSD takes on this machine.

        From the engine's own warm-up line: it moved N experts of blob_bytes in T seconds.
        None when the pack's blob size or the warm-up line is unavailable.
        """
        if not self.disk_gb_per_s or not self.blob_bytes:
            return None
        return (self.blob_bytes / 1e9) / self.disk_gb_per_s * 1000.0

    @property
    def off_card_ms_per_tok(self) -> float:
        """The part of a token spent off the card: SSD reads plus the RAM/CPU lane.

        Measured when the engine printed the times; otherwise attributed from the counts
        using the unit costs the engine measured for this machine. The attributed sum is an
        upper bound - reads are pipelined (ple_inflight) - which is why transfer_share is
        clamped and the counts are kept alongside it.
        """
        if self.measured_times:
            return max(0.0, self.disk_ms_per_tok + self.cpu_ms_per_tok)
        total = 0.0
        disk_ms = self.disk_ms_per_read
        if disk_ms is not None:
            total += self.disk_reads_per_tok * disk_ms
        if self.ram_ms_per_expert is not None:
            total += self.ram_fetches_per_tok * self.ram_ms_per_expert
        elif self.pcie_ms_per_expert is not None:
            total += self.ram_fetches_per_tok * self.pcie_ms_per_expert
        return max(0.0, total)

    @property
    def off_card_lookup_share(self) -> float:
        """The fraction of expert lookups that had to come from off the card.

        Straight from the engine's own hit counter (`STAT vram_hit=` / `glm stat vram hit`),
        so it needs no unit-cost modelling and is always in [0, 1]. A token whose experts are
        95% resident is not bus-bound however slow it is; one at 40% is. Nothing reported
        means nothing known: 0.0.
        """
        if not self.present() or self.vram_hit_pct <= 0.0:
            return 0.0
        return min(1.0, max(0.0, 1.0 - self.vram_hit_pct / 100.0))

    @property
    def transfer_share(self) -> float:
        """How transfer-bound a token is, in [0, 1].

        The larger of the two the engine supports: the off-card *time* as a fraction of the
        token, and the off-card *lookup* fraction. Time is clamped because the engine's
        totals overlap (reads are pipelined, a disk read also wakes the link), so the time
        sum can exceed the token; the lookup share is the floor that is always measured.
        """
        share = self.off_card_lookup_share
        if self.decode_ms_per_tok > 0:
            share = max(share, min(1.0, self.off_card_ms_per_tok / self.decode_ms_per_tok))
        return share

    @property
    def disk_saturation(self) -> float | None:
        """How close the SSD is to the throughput this machine was measured at.

        None when the warm-up line or the blob size is unavailable. 1.0 means the decode is
        asking the drive for exactly what it was seen to deliver.
        """
        rate = self.disk_bytes_per_s
        if rate is None or not self.disk_gb_per_s:
            return None
        return rate / (self.disk_gb_per_s * 1e9)

    @property
    def compute_ms_per_tok(self) -> float:
        """What the token costs on the card alone."""
        return max(0.0, self.decode_ms_per_tok - self.off_card_ms_per_tok)

    @property
    def compute_tok_s(self) -> float:
        """The ceiling this configuration could reach if the bus cost nothing."""
        if self.compute_ms_per_tok <= 0:
            return 0.0
        return 1000.0 / self.compute_ms_per_tok

    @property
    def headroom(self) -> float:
        """compute_tok_s / decode_tok_s: 1.0 means the transfers are already free."""
        if self.decode_ms_per_tok <= 0 or self.compute_ms_per_tok <= 0:
            return 1.0
        return self.decode_ms_per_tok / self.compute_ms_per_tok

    @property
    def bus_bytes_per_tok(self) -> float | None:
        """Bytes moved per token, or None when the blob size is unknown."""
        if self.blob_bytes is None:
            return None
        return (self.disk_reads_per_tok + self.ram_fetches_per_tok
                + self.promotions_per_tok) * float(self.blob_bytes)

    @property
    def disk_bytes_per_s(self) -> float | None:
        bytes_per_tok = None
        if self.blob_bytes is not None:
            bytes_per_tok = self.disk_reads_per_tok * float(self.blob_bytes)
        if bytes_per_tok is None or self.decode_ms_per_tok <= 0:
            return None
        return bytes_per_tok / (self.decode_ms_per_tok / 1000.0)

    @property
    def pcie_bytes_per_s(self) -> float | None:
        if self.blob_bytes is None or self.decode_ms_per_tok <= 0:
            return None
        moved = (self.ram_fetches_per_tok + self.promotions_per_tok) * float(self.blob_bytes)
        return moved / (self.decode_ms_per_tok / 1000.0)

    @property
    def tier_pressure(self) -> dict[str, float]:
        """How full each tier is. A pool at 100% is not a balance, it is a ceiling."""
        out: dict[str, float] = {}
        if self.vram_slots > 0:
            out["vram_pool"] = self.vram_used / self.vram_slots
        if self.ram_slots > 0:
            out["ram_pool"] = self.ram_used / self.ram_slots
        return out

    def flags(self) -> list[str]:
        """What the telemetry says is wrong with this configuration, in plain words."""
        out: list[str] = []
        if not self.present():
            return out
        if self.transfer_share >= 0.5:
            out.append(f"bus-bound: {self.transfer_share:.0%} of a token is off-card time")
        saturation = self.disk_saturation
        if saturation is not None and saturation >= 0.8:
            out.append(f"the SSD is at {saturation:.0%} of the throughput this machine measured")
        if self.disk_ms_per_tok > 0 and self.disk_ms_per_tok >= self.decode_ms_per_tok:
            out.append("the SSD alone takes longer than a token")
        for name, ratio in self.tier_pressure.items():
            if ratio >= 0.98:
                out.append(f"the {name} is full: no room to promote anything")
        if self.kv_overflow:
            out.append("KV streaming overflowed: too few positions resident")
        if self.vram_hit_pct < 20.0 and self.ram_fetches_per_tok > 0:
            out.append("almost nothing is resident on the card")
        return out

    def summary(self) -> str:
        """One line: what the bus is doing for this configuration, and how it was worked out.

        The engine's own numbers where it gives them; its measured unit costs otherwise. The
        attribution is stated so a reader can see where the figure came from.
        """
        if not self.present():
            return "bus:    no transfer telemetry from this engine"
        parts = [(f"{self.transfer_share:.1%} of a token is off-card "
                       f"({self.off_card_ms_per_tok:.2f} ms/tok)")]
        if self.measured_times:
            parts.append(f"measured: {self.disk_ms_per_tok:.2f} ms SSD, {self.cpu_ms_per_tok:.2f} ms CPU lane")
        else:
            disk_ms = self.disk_ms_per_read
            if disk_ms is not None:
                parts.append(f"{self.disk_reads_per_tok:.2f} SSD reads/tok at {disk_ms:.2f} ms each")
            if self.ram_ms_per_expert is not None:
                parts.append(f"{self.ram_fetches_per_tok:.2f} RAM fetches/tok at "
                            f"{self.ram_ms_per_expert:.3f} ms each")
        saturation = self.disk_saturation
        if saturation is not None:
            parts.append(f"the drive at {saturation:.0%} of its measured {self.disk_gb_per_s:.2f} GB/s")
        ceiling = self.compute_tok_s
        headroom = ""
        if ceiling > 0 and self.decode_ms_per_tok > 0 and ceiling > 1000.0 / self.decode_ms_per_tok:
            headroom = f" - the card alone could do {ceiling:.1f} tok/s"
        return "bus:    " + "; ".join(parts) + headroom

    def as_dict(self) -> dict[str, Any]:
        """A flat record for the CSV and the JSON history."""
        return {
            "decode_ms_per_tok": round(self.decode_ms_per_tok, 3),
            "off_card_ms_per_tok": round(self.off_card_ms_per_tok, 3),
            "transfer_share": round(self.transfer_share, 4),
            "off_card_lookup_share": round(self.off_card_lookup_share, 4),
            "disk_ms_per_read": (round(self.disk_ms_per_read, 3)
                                 if self.disk_ms_per_read is not None else None),
            "disk_saturation": (round(self.disk_saturation, 3)
                                if self.disk_saturation is not None else None),
            "measured_times": self.measured_times,
            "compute_tok_s": round(self.compute_tok_s, 2),
            "headroom": round(self.headroom, 3),
            "disk_bytes_per_s": (round(self.disk_bytes_per_s, 1)
                                 if self.disk_bytes_per_s is not None else None),
            "pcie_bytes_per_s": (round(self.pcie_bytes_per_s, 1)
                                 if self.pcie_bytes_per_s is not None else None),
            "disk_reads_per_tok": round(self.disk_reads_per_tok, 2),
            "disk_ms_per_tok": round(self.disk_ms_per_tok, 2),
            "ram_fetches_per_tok": round(self.ram_fetches_per_tok, 2),
            "promotions_per_tok": round(self.promotions_per_tok, 2),
            "cpu_experts_per_tok": round(self.cpu_experts_per_tok, 2),
            "cpu_ms_per_tok": round(self.cpu_ms_per_tok, 2),
            "vram_hit_pct": round(self.vram_hit_pct, 2),
            "kv_hit_pct": round(self.kv_hit_pct, 2),
            "kv_mib_from_ram": round(self.kv_mib_from_ram, 2),
            "spec_accepted": self.spec_accepted,
            "spec_steps": self.spec_steps,
            "flags": self.flags(),
        }


def balance_score(decode_tok_s: float, transfer: Transfer | dict[str, Any] | None,
                  weight: float = TRANSFER_WEIGHT) -> float:
    """Throughput discounted by how transfer-bound the token is.

    Accepts a Transfer or the flat dict stored in a trial record. No telemetry returns the
    plain rate: the discount is only ever applied to a number the engine measured.
    """
    if not transfer:
        return decode_tok_s
    if isinstance(transfer, Transfer):
        share = transfer.transfer_share
    else:
        share = float(transfer.get("transfer_share", 0.0) or 0.0)
    return decode_tok_s * (1.0 - weight * min(1.0, max(0.0, share)))


def pareto_front(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The non-dominated trials: throughput up, off-card time and pressure down.

    A row dominated by another is a configuration that is worse on every axis at once, so
    it cannot be the balance point.
    """
    axes = ("decode_tok_s", "off_card_ms_per_tok", "transfer_share")
    front: list[dict[str, Any]] = []
    for row in rows:
        if any(_dominates(other, row, axes) for other in front):
            continue
        front = [other for other in front if not _dominates(row, other, axes)]
        front.append(row)
    return front


def _dominates(a: dict[str, Any], b: dict[str, Any], axes: tuple[str, ...]) -> bool:
    """a is at least as good on every axis and strictly better on one."""
    not_worse = all(
        (a.get(axis, 0.0) >= b.get(axis, 0.0)) if axis == "decode_tok_s"
        else (a.get(axis, 0.0) <= b.get(axis, 0.0))
        for axis in axes
    )
    strictly = any(
        (a.get(axis, 0.0) > b.get(axis, 0.0)) if axis == "decode_tok_s"
        else (a.get(axis, 0.0) < b.get(axis, 0.0))
        for axis in axes
    )
    return not_worse and strictly


def frontier(rows: Sequence[dict[str, Any]]) -> list[str]:
    """The configurations that are not dominated, as readable lines.

    Each line is throughput, how much of the token crossed a bus, and what the card could
    do if that traffic were free - the three numbers that decide whether a configuration is
    at the balance point or just fast for now. Rows come from a CSV record or a trial dict,
    so every value is coerced.
    """
    def num(row: dict[str, Any], key: str) -> float:
        try:
            return float(row.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    measured = [row for row in rows if num(row, "decode_tok_s") > 0.0]
    if not measured:
        return []
    for row in measured:
        row["decode_tok_s"] = num(row, "decode_tok_s")
        row["transfer_share"] = num(row, "transfer_share")
        row["off_card_ms_per_tok"] = num(row, "off_card_ms_per_tok")
    lines: list[str] = []
    for row in sorted(pareto_front(measured), key=lambda r: -r["decode_tok_s"]):
        ceiling = num(row, "compute_tok_s")
        lines.append(
            f"  {row['decode_tok_s']:>8.2f} tok/s  bus {row['transfer_share']:.0%}  "
            f"{num(row, 'disk_reads_per_tok'):>7.2f} reads/tok  "
            f"{num(row, 'disk_bytes_per_s') / 1e6:>8.0f} MB/s  "
            f"card alone {ceiling:>8.2f} tok/s"
        )
    return lines


def knee(points: Sequence[tuple[float, float]], min_gain: float = 0.03,
         stop_below: float = 0.70) -> int | None:
    """The index where spending more of a resource stops buying throughput.

    `points` is (resource, tok_s) in increasing resource order, the same shape as the
    sweeps in project-maya/tools/calibrate_glm.py. Returns the first index whose gain over
    the best so far is under `min_gain` (3%, the value that file uses), or the index where
    the rate has fallen to `stop_below` of the best seen - after which the sweep is
    measuring noise. None when the curve is still climbing.
    """
    if len(points) < 2:
        return None
    best = points[0][1]
    for index, (_resource, rate) in enumerate(points[1:], start=1):
        if best > 0 and rate < best * stop_below:
            return index
        if best > 0 and (rate - best) / best < min_gain:
            return index
        best = max(best, rate)
    return None
