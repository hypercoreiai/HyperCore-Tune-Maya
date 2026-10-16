"""HyperCore-Tune-Maya/engine_runner.py - boot the engine once, measure one request.

The engine is not an HTTP server: with --serve it reads "GEN <max_new> <token ids>" on
stdin and answers on stdout (src/program/generate.cpp). The telemetry we need is in two
lines it prints:

    READY <max_context> [stop] [pause] [multi=n]                          generate.cpp:2830, 5877
    INFO context=... kv=... kv_resident=... expert_slots=... pcie_share=...   what it applied
    T <token>                                                             every written token, 1763
    DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish> <drafts accepted>
         <drafts offered> <reused> <hits> <lookups>                        src/program/generate.cpp:1917

A finished request also prints, on stderr - which this module reads because the process is
started with stderr merged into stdout:

    glm stat: decode ... ms/tok | vram hit ...% | ram fetches .../tok | disk ... reads/tok ...
    glm kv: / glm spec: / glm skip:                                         generate.cpp:1928-1951

Those are the transfer figures. transfer.py parses them and turns them into the resource side
of a trial: how much of a token is off-card time, and what the configuration would run at if
the bus cost nothing.

READY is the go signal: the engine only reads GEN lines after it. An engine from before
the M3.1 GLM path said "session is up" in its place; both are accepted, and nothing is
sent until one of them appears - a request written before the model is loaded is read as
garbage and never answered.

With --parallel 2..8 (STRATA_GLM_SEQS > 1) the engine runs several conversations at once
and every line a request causes carries the request id after its keyword: GEN lines must
name one (`rid=<id>`) and the answer is `DONE <rid> <generated> ...` (generate.cpp:1961,
2162). READY advertises the mode as `multi=<n>`, which is what this module reads to know
whether a rid is required.

Everything is parsed with an explicit field table rather than a loose regex, so a change in
the DONE line raises instead of silently returning a wrong number.

The `T` lines are what make the decode phase measurable in parts: the engine stamps them in
order as it writes each token, so the wall time between the first one and the think-end token
is the reasoning block and the rest of `decode_ms` is the answer. A request that carries no
`think_budget`/`think_end` key (generate.cpp:1156) has no reasoning block to split.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transfer import Transfer

DONE_MIN_FIELDS = 6

# index -> name, for the fields we read. Positions past 15 are the --lookup-chain tail.
DONE_FIELDS: dict[int, str] = {
    1: "generated", 2: "prompt_tokens", 3: "prompt_ms", 4: "decode_ms", 5: "finish",
    6: "drafts_accepted", 7: "drafts_offered", 8: "reused", 14: "prompt_read",
}


class EngineError(RuntimeError):
    """The engine could not be started, or produced nothing we can measure."""


# The GEN keys that open a reasoning block (generate.cpp:1156): the budget is how many
# tokens the model may reason before the answer, think_end is the token the engine forces
# when it closes that block. Without them the decode phase is one undifferentiated run.
THINK_BUDGET_KEY = "think_budget"
THINK_END_KEY = "think_end"


@dataclass
class Metrics:
    """One measured configuration.

    Rates are tok/s and latency is the request wall time. `decode_tok_s` is the whole decode
    phase the engine reports (`generated / decode_ms`); `thinking_tok_s` and `answer_tok_s`
    split it at the reasoning block, which the engine closes with its think-end token. That
    split needs a thinking budget on the request, so a sweep without one leaves them zero
    rather than inventing them.
    """

    status: str = "FAIL"
    decode_tok_s: float = 0.0     # every generated token: reasoning and answer together
    thinking_tok_s: float = 0.0   # the reasoning block alone
    answer_tok_s: float = 0.0     # the tokens after the reasoning closes
    prefill_tok_s: float = 0.0
    latency_ms: float = 0.0
    prompt_ms: float = 0.0
    decode_ms: float = 0.0
    think_ms: float = 0.0
    answer_ms: float = 0.0
    thinking_tokens: int = 0
    answer_tokens: int = 0
    thinking: bool = False        # the request carried a thinking budget
    prompt_tokens: int = 0
    prompt_read: int = 0
    generated: int = 0
    finish: str = ""
    effective: dict[str, str] = field(default_factory=dict)
    ambient: dict[str, Any] = field(default_factory=dict)
    transfer: dict[str, Any] = field(default_factory=dict)
    log_path: str = ""
    wall_ms: float = 0.0
    note: str = ""

    @property
    def gen_tok_s(self) -> float:
        """The decode rate under the name older records use. Same number, no second copy."""
        return self.decode_tok_s

    def ok(self) -> bool:
        return self.status == "PASS"


def parse_done(line: str, rid: int | None = None) -> dict[str, Any]:
    """Split a DONE line into named fields. Raises EngineError on a malformed line.

    `rid` is the request id this session sent. A several-conversation engine puts it after
    the keyword (`DONE <rid> <generated> ...`, generate.cpp:2162), so the table shifts by
    one and a DONE belonging to another conversation is not this measurement.
    """
    parts = line.split()
    if not parts or parts[0] != "DONE":
        raise EngineError(f"not a DONE line: {line.strip()}")
    if rid is not None:
        if len(parts) < 2 or parts[1] != str(rid):
            other = parts[1] if len(parts) > 1 else "no rid"
            raise EngineError(f"DONE line names {other}, not rid={rid}: {line.strip()}")
        parts = parts[:1] + parts[2:]
    if len(parts) < DONE_MIN_FIELDS:
        raise EngineError(f"malformed DONE line ({len(parts)} fields): {line.strip()}")
    out: dict[str, Any] = {}
    for idx, name in DONE_FIELDS.items():
        if idx < len(parts):
            out[name] = parts[idx]
    try:
        out["generated"] = int(out["generated"])
        out["prompt_tokens"] = int(out["prompt_tokens"])
        out["prompt_ms"] = float(out["prompt_ms"])
        out["decode_ms"] = float(out["decode_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EngineError(f"unreadable DONE line: {line.strip()}") from exc
    if rid is not None:
        out["rid"] = rid
    return out


def reasoning_split(times: Sequence[tuple[float, int]], generated: int, decode_ms: float,
                    think_end: int | None) -> tuple[int, float, int, float]:
    """Split one decode phase into its reasoning block and its answer.

    `times` is the (monotonic clock, token id) series the engine's `T <tok>` lines produced
    for this request; the reasoning block is everything up to and including the think-end
    token the request asked for. Returns (thinking_tokens, think_ms, answer_tokens,
    answer_ms), all zero when there was no thinking request or the model answered without
    reasoning - a split that was never asked for is not measured, not zero-rate.
    """
    if think_end is None or not times or decode_ms <= 0 or generated <= 0:
        return 0, 0.0, 0, 0.0
    idx = next((i for i, (_, tok) in enumerate(times) if tok == think_end), None)
    if idx is None:
        return 0, 0.0, 0, 0.0
    # The timeline is in seconds (time.monotonic); every metric here is in milliseconds.
    think_ms = (times[idx][0] - times[0][0]) * 1000.0
    # The engine's own decode_ms is the authority: the timeline can run past it by the
    # pipe's buffering, never by more than the phase itself.
    think_ms = min(think_ms, decode_ms)
    answer_tokens = max(0, generated - (idx + 1))
    return idx + 1, think_ms, answer_tokens, max(0.0, decode_ms - think_ms)


def parse_token_line(line: str, rid: int | None = None) -> int | None:
    """The token id a `T <tok>` / `T <rid> <tok>` line carries, or None.

    Every written token gets one of these (generate.cpp:1763, and 2058 with a rid when
    several conversations run), which is how a reasoning block is timed: the tokens up to
    the think-end id are the thinking, the rest are the answer. A token from another
    conversation is not this request's, so a rid-carrying engine's line is only read when
    the rid matches.
    """
    parts = line.split()
    if not parts or parts[0] != "T":
        return None
    if rid is not None:
        if len(parts) < 3 or parts[1] != str(rid):
            return None
        parts = parts[2:]
    else:
        parts = parts[1:]
    if not parts:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


def parse_info(line: str) -> dict[str, str]:
    """key=value pairs from the INFO line, so a degraded arm is visible in the record."""
    return {k.split("=")[0]: k.split("=", 1)[1] for k in line.split()[1:] if "=" in k}


def is_ready(line: str) -> bool:
    """True when this line is the engine saying it will now accept GEN lines.

    The serve contract is `READY <max_context> [stop] [pause] [multi=n]`; the pre-M3.1
    engines said "session is up" instead. Anything before one of these is load telemetry:
    the engine is still warming its expert tiers and has not opened its request loop.
    """
    return line.startswith("READY") or "session is up" in line


def parse_ready(line: str) -> dict[str, str]:
    """What a READY line advertises: the context it will serve, and `multi`.

    `multi=<n>` is the several-conversation mode (generate.cpp:2830); without it a GEN
    line needs no rid. Empty for anything that is not a READY line.
    """
    parts = line.split()
    if not parts or parts[0] != "READY" or len(parts) < 2:
        return {}
    try:
        out: dict[str, str] = {"ready_context": str(int(parts[1]))}
    except ValueError:
        return {}
    for token in parts[2:]:
        if token.startswith("multi="):
            out["multi"] = token.split("=", 1)[1]
    return out


def failure_reason(log_text: str) -> str:
    """The engine's own last complaint, used to label a failed row."""
    for needle, label in (
        ("does not fit", "NO_VRAM_FOR_MTP: the draft head does not fit at this --vram-reserve-mib"),
        ("out of memory", "NO_VRAM: a device allocation failed"),
        ("mlock failed", "PLE table could not be pinned; the mmap arm is all that is left"),
        ("glm error", "the engine rejected its configuration"),
    ):
        if needle in log_text:
            return label
    for line in reversed(log_text.splitlines()):
        if line.startswith(("glm error", "maya generate: invalid", "error")):
            return line.strip()
    if any(is_ready(line) for line in log_text.splitlines()):
        return "the engine booted and answered READY, but no DONE line came back"
    return "the engine exited with no message"


TEXT_PROMPT_SUFFIXES = {".txt", ".md", ".text"}

# The prompt a sweep runs when the pack's own tokenizer is available. It is fixed text, so
# two trials of the same configuration measure the same thing.
BENCH_PROMPT = (
    "Explain what a speculative draft head does in a hybrid expert model, and name the two "
    "settings that decide whether a draft is worth running."
)


def _chat_template_path(root: str | None) -> str | None:
    """The model's chat template, in the order the serve side looks for one."""
    if not root:
        return None
    for candidate in (
        Path(root) / "serve" / "chat_template.jinja",
        Path(root) / "tools" / "glm_chat_template.jinja",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def vocab_size(tokenizer_dir: str | None, pack: str | None) -> int | None:
    """How many token ids this pack accepts, or None when nothing states it.

    The tokenizer beside a pack says so directly (`tokenizer.json`'s `vocab_size`); a pack
    without one states it in `index.txt`, whose `output.weight` row is (n_embd, n_vocab).
    """
    if tokenizer_dir:
        try:
            meta = json.loads(Path(tokenizer_dir).joinpath("tokenizer.json").read_text(encoding="utf-8"))
            n = int(meta.get("vocab_size", 0))
            if n > 0:
                return n
        except (OSError, ValueError, TypeError):
            pass
    if pack:
        try:
            for line in (Path(pack) / "index.txt").read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) >= 9 and fields[0] == "output.weight":
                    n = int(fields[8])
                    if n > 0:
                        return n
        except (OSError, ValueError, IndexError):
            pass
    return None


def render_chat_prompt(text: str, template_path: str | None = None) -> str:
    """Render a user prompt through the model's chat template when one is available.

    The real engine reads token ids on stdin, so the text has to be tokenized before it
    reaches `GEN`. A missing Jinja template falls back to the ChatML framing the GLM engine
    already understands; it is not as faithful as the GGUF template, but it is better than a
    synthetic numeric id block.
    """
    if template_path and Path(template_path).is_file():
        try:
            from jinja2 import Environment
            template = Path(template_path).read_text(encoding="utf-8")
            return Environment().from_string(template).render(
                messages=[{"role": "user", "content": text}],
            )
        except Exception:  # noqa: BLE001, S110 - a template is optional; the fallback keeps the sweep running
            pass
    return f"<|start|>user<|message|>{text}<|end|><|start|>assistant<|message|>"


def _fallback_token_ids(text: str, fallback_file: str) -> list[int]:
    try:
        ids = json.loads(Path(fallback_file).read_text(encoding="utf-8"))["input"]
    except (OSError, ValueError):
        ids = [1, 2, 3]
    if not ids:
        ids = [1, 2, 3]
    pieces = re.findall(r"\w+|[^\w\s]", text) or [text or "empty"]
    return [
        ids[int(hashlib.sha256(piece.encode("utf-8", "replace")).hexdigest(), 16) % len(ids)]
        for piece in pieces
    ]


def _tokenize_text(prompt: str, tokenizer_dir: str | None, root: str | None) -> list[int] | None:
    if root:
        tools = Path(root) / "tools"
        if (tools / "strata_tokenizer.py").is_file():
            sys.path.insert(0, str(tools))
            try:
                from strata_tokenizer import Tokenizer
                if tokenizer_dir:
                    return Tokenizer.from_dir(tokenizer_dir).encode(prompt, parse_special=True)
            except Exception:  # noqa: BLE001, S110 - a tokenizer is optional; the fallback is deterministic
                pass
    return None


DEFAULT_PROMPT_FILE = Path(__file__).resolve().parent / "fixtures" / "prompt.json"


def _pack_prompt(tokenizer_dir: str | None, root: str | None) -> list[int] | None:
    """The benchmark text through the pack's own tokenizer, or None when there is none.

    This is the default prompt. A pack's tokenizer is the only one whose ids the engine
    will accept, so a sweep that uses it never trips the vocabulary check at all.
    """
    return _tokenize_text(
        render_chat_prompt(BENCH_PROMPT, _chat_template_path(root)), tokenizer_dir, root
    )


def _prompt_for_vocab(vocab: int, ids: list[int], tokenizer_dir: str | None,
                     root: str | None, prompt_file: str) -> list[int]:
    """A prompt this pack's vocabulary will accept, for a file the caller named.

    Ids from another model's tokenizer are refused token by token ("ERR token N out of
    range [0, V)", generate.cpp:1721), and every trial then scores as a zero - the search
    learns nothing. The pack's own tokenizer over the benchmark text is the faithful
    replacement; without one the ids are folded into range deterministically, which is a
    valid prompt if not a meaningful one.
    """
    tokenized = _pack_prompt(tokenizer_dir, root)
    if tokenized:
        print(
            f"note: {prompt_file} has ids outside this pack's vocabulary ({vocab}); "
            "using the pack's own tokenizer instead",
            file=sys.stderr,
        )
        return tokenized
    print(
        f"note: {prompt_file} has ids outside this pack's vocabulary ({vocab}) and no "
        "tokenizer is beside the pack; the ids are folded into range",
        file=sys.stderr,
    )
    return [t % vocab for t in ids]


def build_prompt(
    prompt_file: str = "",
    tokens: int = 10240,
    tokenizer_dir: str | None = None,
    root: str | None = None,
    vocab: int | None = None,
) -> list[int]:
    """A deterministic prompt of `tokens` ids.

    No `prompt_file` - the usual case - means the pack's own tokenizer over `BENCH_PROMPT`.
    A JSON file of token ids is cycled. A text file is rendered through the model's chat
    template and tokenized when `tools/strata_tokenizer.py` and a tokenizer directory are
    available. Only when no tokenizer is beside the pack does the sweep fall back to the
    bundled `fixtures/prompt.json`, and it says so.

    `vocab` is the pack's token count. When given, ids outside it are replaced rather than
    sent - the engine rejects the whole request for one bad id.
    """
    if not prompt_file:
        ids = _pack_prompt(tokenizer_dir, root)
        if ids is None:
            print(
                "note: no tokenizer beside the pack; cycling fixtures/prompt.json, folded "
                f"into its vocabulary ({vocab})",
                file=sys.stderr,
            )
            ids = json.loads(DEFAULT_PROMPT_FILE.read_text(encoding="utf-8"))["input"]
            if vocab:
                ids = [int(t) % int(vocab) for t in ids]
        if not ids:
            raise EngineError("the default prompt produced no token ids")
    else:
        path = Path(prompt_file)
        if path.suffix.lower() in TEXT_PROMPT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            rendered = render_chat_prompt(text, _chat_template_path(root))
            ids = _tokenize_text(rendered, tokenizer_dir, root)
            if ids is None:
                print(
                    "note: no tokenizer found for --prompt-file text; using a deterministic fallback prompt",
                    file=sys.stderr,
                )
                ids = _fallback_token_ids(rendered, str(DEFAULT_PROMPT_FILE))
            if not ids:
                raise EngineError(f"prompt file {prompt_file} produced no token ids")
        else:
            ids = json.loads(path.read_text(encoding="utf-8"))["input"]

        if not ids:
            raise EngineError(f"prompt file {prompt_file} carries no token ids")
        if vocab and any(int(t) < 0 or int(t) >= vocab for t in ids):
            ids = _prompt_for_vocab(int(vocab), [int(t) for t in ids], tokenizer_dir, root, prompt_file)
    out: list[int] = []
    while len(out) < tokens:
        out.extend(ids)
    return out[:tokens]



class EngineSession:
    """One running engine that can answer several GEN requests.

    The engine's `--serve` contract is a line protocol: READY, then GEN lines, DONE lines.
    Keeping the process open lets a two-stage sweep change only the request keys. A single
    `run_config()` is just a one-request session.
    """

    def __init__(
        self,
        engine: str,
        args: list[str],
        log_path: Path,
        boot_timeout: int = 300,
        run_timeout: int = 180,
        extra_env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        blob_bytes: float | None = None,
    ) -> None:
        self.engine = str(Path(engine).expanduser().resolve())
        self.args = list(args)
        self.log_path = Path(log_path).expanduser().resolve()
        self.boot_timeout = int(boot_timeout)
        self.run_timeout = int(run_timeout)
        self.extra_env = dict(extra_env or {})
        self.cwd = cwd
        self.blob_bytes = blob_bytes

        self.lines: list[str] = []
        self.pending = b""
        self.token_times: list[tuple[float, int]] = []   # this request's T lines, in order
        self.think_end: int | None = None                # the think-end id this request asked for
        self.chunks: queue.Queue[bytes | None] = queue.Queue()
        self.ready = False
        self.multi = 0          # the READY line's multi=<n>: conversations at once
        self.rid = 0            # the id the next GEN request will carry
        self.effective: dict[str, str] = {}
        self.info: dict[str, str] = {}
        self.eof = False
        self.start_error = ""
        self.proc: subprocess.Popen | None = None
        self.raw_out = None
        self.raw_in = None
        self.thread: threading.Thread | None = None

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")
        self._start()

    def _start(self) -> None:
        cmd = [self.engine] + self.args
        if self.engine.lower().endswith(".py"):
            cmd = [sys.executable, self.engine] + self.args
        popen_kw: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "bufsize": 0,
            "env": self.extra_env or None,
        }
        if self.cwd is not None:
            popen_kw["cwd"] = str(Path(self.cwd).expanduser().resolve())
        try:
            proc = subprocess.Popen(cmd, **popen_kw)
        except OSError as exc:
            self.start_error = f"could not start {self.engine}: {exc}"
            self.eof = True
            return

        self.proc = proc
        self.raw_out = proc.stdout
        self.raw_in = proc.stdin
        if self.raw_out is None:
            self.start_error = "the engine has no stdout to read"
            self.eof = True
            return

        fd = self.raw_out.fileno()

        def reader() -> None:
            try:
                while True:
                    chunk = os.read(fd, 65536)
                    if not chunk:
                        break
                    self.chunks.put(chunk)
            except OSError:
                pass
            finally:
                self.chunks.put(None)

        self.thread = threading.Thread(target=reader, name="engine-stdout", daemon=True)
        self.thread.start()

    def _send(self, line: str) -> bool:
        if self.raw_in is None:
            return False
        try:
            self.raw_in.write(line.encode("utf-8"))
            self.raw_in.flush()
            return True
        except (BrokenPipeError, OSError, ValueError):
            return False

    def _note_line(self, text: str) -> None:
        """Record one engine line and keep whatever state it changes.

        The READY line is where the several-conversation mode is announced, and that is
        what decides whether a GEN line has to carry a rid.
        """
        self.lines.append(text)
        if text.startswith("T "):
            tok = parse_token_line(text, self.rid if self.multi else None)
            if tok is not None:
                self.token_times.append((time.monotonic(), tok))
        elif text.startswith("INFO "):
            self.info.update(parse_info(text))
        elif is_ready(text):
            self.ready = True
            advertised = parse_ready(text)
            self.effective.update(advertised)
            try:
                self.multi = max(0, int(advertised.get("multi", 0)))
            except ValueError:
                self.multi = 0

    def _await_ready(self, deadline: float) -> None:
        """Read until the engine says READY, so the request knows the mode it is in."""
        while time.monotonic() < deadline and not self.ready and not self.eof:
            try:
                chunk = self.chunks.get(timeout=0.5)
            except queue.Empty:
                continue
            if chunk is None:
                self.eof = True
                break
            self.pending += chunk
            while b"\n" in self.pending:
                raw_line, self.pending = self.pending.split(b"\n", 1)
                self._note_line(raw_line.decode("utf-8", "replace"))

    def _consume_until_done(self, deadline: float, rid: int | None = None) -> dict[str, Any] | None:
        while time.monotonic() < deadline and not self.eof:
            try:
                chunk = self.chunks.get(timeout=0.5)
            except queue.Empty:
                continue
            if chunk is None:
                self.eof = True
                break
            self.pending += chunk
            while b"\n" in self.pending:
                raw_line, self.pending = self.pending.split(b"\n", 1)
                text = raw_line.decode("utf-8", "replace")
                self._note_line(text)
                if text.startswith("ERR "):
                    return {"__err__": text}
                if text.startswith("DONE "):
                    return parse_done(text, rid)
        return None

    def request(
        self,
        prompt_ids: list[int],
        max_new: int,
        request_keys: dict[str, object] | None = None,
    ) -> Metrics:
        """Send one GEN request to the already-running engine."""
        metrics = Metrics(log_path=str(self.log_path), effective=dict(self.effective or self.info))
        t_start = time.monotonic()

        if self.start_error:
            metrics.status = "FAIL_EXEC"
            metrics.note = self.start_error
            return metrics
        if self.proc is None or self.proc.poll() is not None or self.eof:
            metrics.status = "FAIL_BOOT"
            metrics.note = failure_reason("\n".join(self.lines))
            return metrics

        # The rid a GEN line needs is advertised on the READY line: a several-conversation
        # engine (STRATA_GLM_SEQS > 1) refuses a request that does not name one, so the
        # mode has to be known before the first request is written.
        if not self.ready:
            self._await_ready(time.monotonic() + self.boot_timeout)
        if self.eof:
            metrics.status = "FAIL_BOOT"
            metrics.note = failure_reason("\n".join(self.lines))
            return metrics

        keys = dict(request_keys or {})
        rid: int | None = None
        if self.multi:
            # generate.cpp:3095 - with several conversations a GEN line without a rid is
            # refused outright, so the request is never admitted.
            self.rid += 1
            rid = self.rid
            keys["rid"] = rid
        # The token timeline is per request: a session that answers several GEN lines must
        # not carry one request's tokens into the next request's split.
        self.token_times = []
        try:
            budget = int(keys.get(THINK_BUDGET_KEY, 0) or 0)
        except (TypeError, ValueError):
            budget = 0
        self.think_end = int(keys[THINK_END_KEY]) if keys.get(THINK_END_KEY) is not None else None
        if budget > 0 and self.think_end is not None:
            metrics.thinking = True
        key_text = " ".join(f"{k}={v}" for k, v in keys.items())
        line = f"GEN {max_new} " + (key_text + " " if key_text else "") + ",".join(map(str, prompt_ids)) + "\n"
        if not self._send(line):
            metrics.status = "FAIL_EXEC"
            metrics.note = "the engine closed its stdin"
            return metrics

        deadline = time.monotonic() + (self.boot_timeout if not self.ready else self.run_timeout)
        try:
            done = self._consume_until_done(deadline, rid)
        except (BrokenPipeError, ValueError) as exc:
            metrics.note = f"the engine closed its output: {exc}"
            done = None

        metrics.wall_ms = (time.monotonic() - t_start) * 1000.0
        log_text = "\n".join(self.lines)
        metrics.effective = dict(self.effective or self.info)

        if done is None:
            metrics.status = "FAIL_BOOT" if not self.ready else "TIMEOUT"
            metrics.note = failure_reason(log_text)
            return metrics
        if "__err__" in done:
            metrics.status = "ERR"
            metrics.note = str(done["__err__"])
            return metrics

        try:
            prompt_ms = float(done["prompt_ms"])
            decode_ms = float(done["decode_ms"])
            generated = int(done["generated"])
            prompt_tokens = int(done["prompt_tokens"])
        except (KeyError, TypeError, ValueError) as exc:
            metrics.status = "BAD_DONE"
            metrics.note = f"malformed DONE line: {exc}"
            return metrics

        read_n = int(done.get("prompt_read") or 0)
        if read_n <= 0:
            read_n = max(0, prompt_tokens - int(done.get("reused") or 0))

        if generated <= 0:
            metrics.status = "NO_TOKENS"
            metrics.note = failure_reason(log_text)
            metrics.finish = str(done.get("finish", ""))
            return metrics

        metrics.status = "PASS"
        metrics.generated = generated
        metrics.prompt_tokens = prompt_tokens
        metrics.prompt_read = read_n
        metrics.prompt_ms = prompt_ms
        metrics.decode_ms = decode_ms
        metrics.finish = str(done.get("finish", ""))
        metrics.decode_tok_s = 1000.0 * generated / decode_ms if decode_ms > 0 else 0.0
        metrics.prefill_tok_s = 1000.0 * read_n / prompt_ms if prompt_ms > 0 else 0.0
        metrics.latency_ms = prompt_ms + decode_ms

        thinking_tokens, think_ms, answer_tokens, answer_ms = reasoning_split(
            self.token_times, generated, decode_ms, self.think_end)
        metrics.thinking_tokens = thinking_tokens
        metrics.think_ms = think_ms
        metrics.answer_tokens = answer_tokens
        metrics.answer_ms = answer_ms
        metrics.thinking_tok_s = 1000.0 * thinking_tokens / think_ms if think_ms > 0 else 0.0
        metrics.answer_tok_s = 1000.0 * answer_tokens / answer_ms if answer_ms > 0 else 0.0
        if metrics.thinking and thinking_tokens == 0:
            metrics.note = ("the request carried a thinking budget but the model answered "
                           "without a reasoning block, so thinking is not measured")
        metrics.transfer = Transfer.from_lines(self.lines, blob_bytes=self.blob_bytes).as_dict()
        return metrics

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and not self.eof

    def close(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self._send("QUIT\n")
        for pipe in (self.raw_out, self.raw_in):
            if pipe is not None:
                try:
                    pipe.close()
                except (BrokenPipeError, ValueError, OSError):
                    pass
        if self.proc.poll() is None:
            self.proc.terminate()
            deadline = time.monotonic() + 15
            while self.proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.5)
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()
        if self.thread is not None:
            self.thread.join(timeout=1)
        self.log_path.write_text("\n".join(self.lines), encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def run_config(
    engine: str,
    args: list[str],
    prompt_ids: list[int],
    max_new: int,
    log_path: Path,
    boot_timeout: int = 300,
    run_timeout: int = 180,
    extra_env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    request_keys: dict[str, object] | None = None,
    blob_bytes: float | None = None,
) -> Metrics:
    """Boot the engine with `args`, send one GEN request, return its metrics.

    A boot that never reaches "session is up", a process that exits, a request that produces
    no tokens, and a timeout are each recorded as their own status rather than as a crash:
    the optimizer needs a value for every point it tried. `engine` and `log_path` are
    resolved against the current directory before the process starts, so a relative value
    still works when `cwd` is the Maya root.
    """
    with EngineSession(
        engine=engine,
        args=args,
        log_path=log_path,
        boot_timeout=boot_timeout,
        run_timeout=run_timeout,
        extra_env=extra_env,
        cwd=cwd,
        blob_bytes=blob_bytes,
    ) as session:
        return session.request(prompt_ids, max_new, request_keys)
