# HyperCore-Tune-Maya

<p align="center"><img src="docs/media/HyperCore-Tune.jpeg" width="420" alt="HyperCore-Tune-Maya"></p>

Bayesian (Optuna TPE) tuning of the Project Maya engine, as a standalone repository that
attaches to a Maya checkout. It has no other association with Project Maya than
compatibility: clone it inside a checkout, or next to one as its own app.

It searches the engine dimensions on a given machine, measures one request per
configuration by booting the engine, and writes back a run config and a launcher in
exactly the format `maya.py` writes for an installed model:

```
max_<model>_config.json  the winning run config (exe, args, cwd, tokenizer, port, ...)
max_<model>_run.sh       the launcher: maya.py with that config
max_<model>_run.bat      the same launcher for a Windows install
HyperCore-Tune-Maya/results/maya-<model>.record.json   every trial and its metrics
HyperCore-Tune-Maya/results/bayesian-<machine>-<model>-<objective>-<stamp>.csv
```

A sweep never writes over the model's own `maya-<model>.json` or `run-maya-<model>.sh`: the
tune it shipped with stays in place beside the new pair, so if the winner is worse the
original launcher is still there to run.

Nothing in the sweep is hardcoded to a machine. The project root, the engine binary, the
server script and the venv are resolved from where this repository sits, and every path
written into a deliverable is absolute. Linux and Windows 10/11 are both supported.

## Install

Clone it inside a Project Maya checkout, or next to one (`~/project-maya` or
`%USERPROFILE%\project-maya` on Windows). Both layouts work: every path is absolute.

```bash
cd ~/project-maya
git clone https://github.com/hypercoreiai/HyperCore-Tune-Maya.git
```

```bash
cd ~
git clone https://github.com/hypercoreiai/HyperCore-Tune-Maya.git
```

Then install the sweep's own packages (`optuna`, `psutil`) into the project venv:

```bash
./HyperCore-Tune-Maya/max_tune.sh --install          # Linux
max_tune.bat --install                               # Windows 10/11
```

`--install` uses the Maya `.venv` if it exists (`maya.py` creates it), otherwise the
interpreter on the PATH (`python.exe` / `py -3` on Windows, including
`.venv\Scripts\python.exe`).

## Run

```bash
./HyperCore-Tune-Maya/max_tune.sh --model maya-s24 --decode   --cntx 131072
./HyperCore-Tune-Maya/max_tune.sh --model maya-s24 --prefill  --cntx 131072 --trials 60
./HyperCore-Tune-Maya/max_tune.sh --model maya-s24 --think    --think-budget 512 --max-new 1024
./HyperCore-Tune-Maya/max_tune.sh --model maya-s24 --latency  --mode staged --dry-run
./HyperCore-Tune-Maya/max_tune.sh --show-project
./HyperCore-Tune-Maya/max_tune.sh --help
```

On Windows 10/11 the same flags go to `max_tune.bat` from this folder:

```bat
max_tune.bat --model maya-s24 --decode --cntx 131072
max_tune.bat --show-project
```

Objectives: `--decode` maximises decode tokens/s, `--think` maximises thinking tokens/s,
`--prefill` maximises prompt tokens/s, `--latency` minimises `prompt_ms + decode_ms`, and
`--balance` maximises decode tokens/s discounted by the share of a token the engine spent
off the card. `--gen` is the older spelling of `--decode` and still works.

### What a trial measures

One boot, one GEN request, and the engine's own numbers:

| Measurement | Where it comes from |
|---|---|
| `prefill_tok_s` | prompt tokens read (the DONE line's prompt count minus what the prompt cache reused) over `prompt_ms` |
| `decode_tok_s` | every token the request generated over `decode_ms` |
| `thinking_tok_s` | the reasoning block alone: the tokens up to the think-end token, timed from the engine's `T <token>` lines |
| `answer_tok_s` | the tokens after the reasoning closes, over the rest of `decode_ms` |
| `latency_ms` | `prompt_ms + decode_ms`, the wall time of the request |
| `transfer_share` | the part of a token spent on the SSD and the CPU lane, from the engine's `glm stat` line |
| `compute_tok_s` | what the same configuration would run at if the bus cost nothing |
| `disk_reads_per_tok`, `ram_fetches_per_tok`, `promotions_per_tok`, `cpu_experts_per_tok` | the counters on that same line, per token |
| `disk_bytes_per_s`, `pcie_bytes_per_s` | those counts times the expert blob, read from the pack's `native_experts.txt` |

`--balance` is the objective that answers "fast *and* resident": a configuration at 260 tok/s
with a third of its token time waiting on the SSD scores below one at 250 tok/s that is
already on the card. `transfer.py` parses the telemetry, `pareto_front()` drops configurations
that are worse on every axis at once, and `knee()` finds where spending more of a resource
stops buying throughput - the same 3% / 70% rule `project-maya/tools/calibrate_glm.py` uses.
With no telemetry (a stub engine, or a build that prints nothing) the discount is zero and
`--balance` behaves exactly like `--decode`.

### Where the bus numbers come from

A `--serve` engine with `STRATA_GLM_SEQS > 1` never prints the per-request `glm stat:` line
(that block, generate.cpp:1925, is only reached on the single-conversation path). What it does
print, every eight tokens, is the live counter line the Monitor in `serve/server.py` uses:

```text
STAT tok_s=7.66 ms_tok=130.57 vram_hit=0.7553 ram_fetch=3.934 disk=21.703 promo=19.928 \
  vram_used=1627 vram_slots=1736 ram_used=2994 ram_slots=3012 ram_gb=20.0 vram_gb=11.5 active=1
```

Those are counts, not times, so `transfer.py` attributes the time with the unit costs the same
engine measured for this machine at boot:

```text
glm fast: CUDA0 CPU lane: 24 threads, an expert 0.136 ms on the CPU vs 0.453 ms over PCIe -> of 0..8 RAM-tier experts the CPU takes 0 1 2 3 3 4 5 6 6
glm fast: CUDA0 tiers warm: 4614 experts (32.8 GB) in 18 s (1.87 GB/s)
```

One SSD read therefore costs the expert blob (7,012,352 bytes, read from the pack's
`native_experts.txt`) over the measured 1.87 GB/s = 3.75 ms, and one RAM fetch costs the
weighted host/PCIe time from the lane vector. The sum is an upper bound on wall time, because
the engine pipelines reads with `ple_inflight`, so `transfer_share` is clamped to 1 and is the
larger of that and `1 - vram_hit`, which the engine measures directly.

`disk_saturation` is the same traffic against the throughput the drive was actually seen to
deliver: 0.6 means the decode is asking for 60% of what the SSD can do.

If a build does print `glm stat:` (single-conversation runs), its `ms/tok` figures are used as
measured and `measured_times` is set in the record, so the provenance of every number is visible.

### The independent cross-check

The engine's counters are the only view of what it moved, so the tuner also samples the host's
block-device counters with psutil before and after every request (`hardware.disk_io_snapshot`).
`measured_disk_bytes_per_s` is what the drive actually served during that request, and
`disk_attribution` is that divided by the derived figure. A ratio near 1.0 means the attribution
is sound; far from 1.0 means the derived share is a bound, not a measurement - reads are
pipelined, or the host is doing other I/O.

PCIe traffic cannot be cross-checked here: NVML is not installed in the tuning venv
(`import nvml` fails), so `pcie_bytes_per_s` stays a derived figure. Installing `nvidia-ml-py`
would let the same pattern apply to the link.

Re-running the parser over the logs from a real sweep on this machine:

```text
      9.04 tok/s  bus 59%    17.68 reads/tok      1121 MB/s  card alone    22.07 tok/s
      8.78 tok/s  bus 55%    16.40 reads/tok      1010 MB/s  card alone    19.57 tok/s
      7.61 tok/s  bus 54%    18.78 reads/tok      1002 MB/s  card alone    16.50 tok/s
      7.44 tok/s  bus 51%    18.59 reads/tok      970 MB/s  card alone    15.28 tok/s
```

The four fastest configurations are all on the frontier: none of them is beaten on both rate
and bus at once. That is the honest answer - at 9 tok/s on this model, roughly six of every
ten milliseconds of a token are spent moving experts, and the search's job is to find the
corner of that trade where the next millisecond of bus traffic stops buying a token.

The reasoning split only exists when the request asks for one. `--think-budget N` puts
`think_budget=N think_end=<id>` on the GEN line (generate.cpp:1156); without it a
`--think` sweep has nothing to measure and `--think` is refused at the command line. The
think-end token is the pack's `<|im_start|>` (151667), which the engine forces when a
reasoning block reaches its budget or starts repeating itself; pass `--think-end` for a
pack that uses a different id.

`--cntx` (also spelled `--context`) defaults to **32768**; pass it explicitly for any other
context. Run data is keyed by it, so a sweep at 32768 and one at 131072 keep separate
histories.

`--trials` is the number of **unseen** configurations to evaluate. A suggestion that
matches a point already in history (or already measured in this sweep) is pruned, not
counted, and the sampler keeps drawing until that many new points have been booted.

Before a real sweep, stop the running server (Ctrl-C on `./maya.sh` or on the
`run-maya-<model>.sh` launcher) so the engine can hold the card for the boot. The sweep
checks this before the first trial: a pack's dense pool is the floor, and a GPU with less
free than that cannot host it at any setting, so the run says so and stops rather than
spending `--trials` boots on the same failure. `--no-preflight` skips the check.

Three configurations in a row that fail the same way stop the sweep too. An unmeasurable
trial scores the same penalty as a real zero throughput, so a busy GPU looks to the sampler
like a flat landscape; a repeated failure is the machine, not the search.

## Search space

The dimensions searched are the columns every row is written with:

```
layer  vram  pcie  pcie_mode  ple_io  ple_inflight  kv  kvstream  hostworker  worker  tail
       draft_floor  suffix_draft  mtp_max_t  parallel  status
```

`status` is not a parameter: it is what the engine did with the configuration
(`PASS`, `FAIL_BOOT`, `NO_TOKENS`, `TIMEOUT`).

| Tuner flag | Engine flag | Default search | Hold constant |
|---|---|---|---|
| `--vram-min` / `--vram-max` / `--vram-step` | `--vram-reserve-mib` | 256–1536 step 256 | (range flags, not a freeze) |
| `--pcie` | `--pcie-frac` | 0–1 | `--pcie 0.95` |
| `--pcie-mode` | `--pcie-mode` | `auto`, `dma`, `kernel`, `direct` | `--pcie-mode auto` |
| `--ple-io` | `--ple-io` | `direct`, `mmap` | `--ple-io mmap` |
| `--ple-inflight` | `--ple-inflight` | 8–128 step 16 (the engine accepts 1–1024) | `--ple-inflight 64` |
| `--kv` | `--kv` | only what the pack's runner implements — `int8`, `fp16` for a `--glm-pack`; `fp16`, `int8`, `q4_0`, `k8v4` for a qwen pack | `--kv int8` |
| `--kvstream` | `--kv-streaming` | on / off | `--kvstream on` |
| `--hostworker` / `--host-worker` | `--no-host-worker` | 1 (host joins the pool) / 0 | `--hostworker 0` |
| `--worker` | `--pool-workers` | 0 … physical cores − 1 | `--worker 8` |
| `--tail` | `--prompt-cache` | 0–6 conversation checkpoints | `--tail 0` |
| `--draft-floor` / `--spec-min-p` | `--spec-min-p` | 0.001–1 | `--draft-floor 0.5` |
| `--suffix-draft` | `--suffix-draft` | 0–8 step 2, only when no `--dflash` drafter was found | `--suffix-draft 0` |
| `--mtp-max-t` | `--mtp-max-t` | 0–8 step 2, same condition | `--mtp-max-t 0` |
| `--parallel` | `--parallel` | 1–8, GLM packs only | `--parallel 1` |

Every range is the bound `src/program/generate.cpp` enforces, and the line that enforces
it is in the comment next to it in `search_space.py`. Three of them are not a matter of
speed but of what the runner can do at all:

* `--kv q4_0` and `--kv k8v4` are parsed and then ignored on a `--glm-pack`: the GLM
  latent cache is fp16 or INT8 (`generate.cpp:3443` hands the pack to `glm_pack_generate`
  before the mode handling at 3557-3564), so those two arms measure the same run twice.
* `--kv k8v4` cannot be streamed (`generate.cpp:3569` exits 2 with a kv-resident window),
  and `--kv-streaming on` sets one, so the sweep never asks for that pair.
* With a `--dflash` drafter the request decodes through `Glm5Model::decode_dflash`
  (`generate.cpp:1842`) and the MTP/suffix path is never reached, so `--suffix-draft` and
  `--mtp-max-t` are left out of the space rather than swept as a no-op.

A freeze flag holds that dimension at a constant instead of searching it. History rows
that do not match a freeze are not used as priors.

The bounds in `search_space.py` are checked against the engine source in the test suite:
`parse_engine_bounds()` extracts the clamp and validation lines from `generate.cpp` and
asserts that the searched grid does not cross them. If the engine changes a bound, the
test fails instead of producing exit-2 trials at runtime.

## Cold starts and sessions

A cold study with no history is seeded with a deterministic Latin-hypercube sweep before
TPE sees any measured points. `--quasi-trials 15` is the default; `--no-quasi` turns it
off. The point is coverage, not a better first winner: TPE needs a spread of measured
points before it can model the landscape.

`--two-stage` keeps one engine process open for a topology and changes only the per-request
keys inside it: `pcie_frac`, `spec_min_p`, and, if requested, `cpu_threads`, `read_chunks`,
`spec`, and `spec_n_min`. Use it with `--request-dims cpu_threads,read_chunks,spec,spec_n_min`
to sweep those extra keys as well.

Every real trial records the ambient GPU state at measurement time: free VRAM and the
compute processes holding the card. A history entry without that context is not reproducible;
"78 priors outside the grid" becomes diagnosable when the record says what the machine was
actually doing.

With `--parallel 2..8` the engine runs several conversations at once and refuses a `GEN` line
that does not name one: `this engine runs several conversations: GEN needs rid=<id>`. The
session reads `multi=<n>` off the `READY` line and carries a rid on every request, and reads
the rid-prefixed `DONE <rid> ...` answer back — a multi-sequence arm is measured, not failed.

## Prompt realism

The default prompt is the pack's **own** tokenizer over a fixed benchmark text (`BENCH_PROMPT`
in `engine_runner.py`), cycled to `--prompt-tokens` length. That is the only prompt a pack
can be guaranteed to accept, so a plain `./max_tune.sh --model maya-s24 --gen` needs no
prompt file at all. It is still synthetic: useful for a steady-state probe, but it exercises
a different expert-usage profile than real chat. `--calibrate` uses real prompts for that
reason, and `--prompt-file <text>` renders a text prompt through the model's chat template
and tokenizes it with the pack's tokenizer. Treat the synthetic sweep numbers as comparable
only to other synthetic sweeps, not to `--calibrate`.

`fixtures/prompt.json` is the fallback for a pack that ships no tokenizer, and the note says
so when it is used. Its ids belong to a different vocabulary - a prompt built for another
model's tokenizer is refused token by token (`ERR token 248045 out of range [0, 154880)`),
and one bad id fails the whole request, so every trial in the sweep scores as a zero. The
pack's token count is read from `tokenizer.json`'s `vocab_size`, or from `index.txt`'s
`output.weight` row, and any ids an explicitly named prompt file carries are checked against
it before the first request is sent.

Every flag written into a deliverable is one this build's `src/program/generate.cpp`
actually parses: `maya_env.engine_flags()` reads the engine source, and a dimension the
build does not have is left out of the space rather than passed to a binary that would
reject it at startup.

`--pool-workers 0` is a legal arm: the engine treats it as "every physical core except
the host".

Example, pin KV and the draft floor, search the rest:

```bash
./HyperCore-Tune-Maya/max_tune.sh --model maya-s24 --gen --cntx 131072 --kv int8 --draft-floor 0.5
```

## How the project is found

`maya_env.py` resolves the Project Maya root in this order, and every result is an
absolute path (`~`, `%USERPROFILE%` and `%VAR%` are expanded):

1. `--maya-root`, then `$MAYA_ROOT`, `$PROJECT_MAYA_ROOT`, `$TUNE_ROOT`
2. the nearest ancestor of this repository that is a Maya checkout
   (`~/project-maya/HyperCore-Tune-Maya` walks up to `~/project-maya`)
3. a sibling directory named `project-maya`, `maya` or `Maya`
   (`~/HyperCore-Tune-Maya` next to `~/project-maya`)
4. any other sibling that looks like a Maya checkout

A directory counts as a root when it has `serve/server.py` and one of `maya.py`,
`setup.py`, `CMakeLists.txt` or `src/program/generate.cpp`. `HyperCore-Tune-Maya` is this
app, not a Maya root. `--show-project` prints what was resolved:

```json
{
  "repo": "/home/you/project-maya/HyperCore-Tune-Maya",
  "root": "/home/you/project-maya",
  "engine": "/home/you/project-maya/build/strata",
  "server": "/home/you/project-maya/serve/server.py",
  "venv": "/home/you/project-maya/.venv/bin/python",
  "server_engine": "strata"
}
```

The engine binary is looked for at `build/strata`, `build-hip/strata`,
`build-release/strata` and `dist/strata` (CUDA in `build/`, HIP in `build-hip/`), and
can be pinned with `--engine` or `$MAYA_ENGINE`. Model packs, drafters and tokenizers are
searched in this order: `$MAYA_DATA`, the data folder beside the checkout named after the
project (`Maya-data/models/<model>/pack`), the folder the installer recorded
(`~/.config/project-maya/settings.json`, or `%APPDATA%\project-maya\settings.json` on
Windows), then a legacy folder (`strata-data`, `hypercore-data`).

A pack is only used when the engine can open it. A native pack reads its source GGUFs
from **beside the pack directory** - `native_experts.txt` names shard 1 and the engine
follows its `-NNNNN-of-MMMMM` pattern - so a pack left over from an older install whose
shards sit in a different folder is skipped, and the message names it. Put the shards next
to the pack, or pass `--pack` at a pack that has them.

## Reusing previous sweeps

Run data is kept per machine in `HyperCore-Tune-Maya/history/`, named

```
<CPU>_<GPU>_<RAM>_<PCIe gen>x<width>_<model>_<objective>_<cntx>.json
```

A sweep on the same box loads it by default, never re-boots a configuration it already
measured, seeds the TPE sampler with those trials, and keeps drawing until `--trials`
unseen configurations have been evaluated. `--no-load` starts cold; a record from
different hardware is ignored unless `--allow-hardware-mismatch` is given. Pin the
fingerprint with `--hardware '<json>'` or `$TUNE_HARDWARE` for containers and test farms.

`--replay <record.json>` and `--from-records` rewrite the deliverables from measurements
that already exist, without booting the engine.

A prior outside the bounds this run searches (`--vram-max`, `--max-workers`) is skipped and
counted, never clamped, and it cannot become the best written into `max_<model>_config.json`:
the launcher a sweep emits only ever carries flags this pack's runner acts on. A `--kv` the
runner ignores is read as the mode that actually ran — on a GLM pack `q4_0` and `k8v4` are
parsed and then dropped at `generate.cpp:3444`, so those records measure the fp16 latent
cache and are seeded as fp16 rather than discarded.

### A prior is a seed, never a winner

The configuration written into the deliverable is one this sweep booted on this machine.
A prior from history warms the sampler and is printed as "prior best on this machine, not
reproduced", but it does not take the title: a tok/s measured by another engine binary, or
on a different prompt, is not a number this run could produce, and a launcher built from it
would carry numbers the user cannot check. Each record carries the conditions the winner was
measured under - engine, prompt tokens, `--max-new`, context - and every stored trial keeps
the engine that produced it, so priors from a different binary are dropped on load.

This matters for the test suite too: `stub_engine.py` runs against the same machine
fingerprint, and its synthetic timings used to be merged into the real history file. They
are now tagged with the engine that measured them and skipped by any sweep running a
different one.

## Compatibility with Project Maya

- deliverables are named `max_<model>_run.sh` / `max_<model>_config.json`, never the
  installed `run-maya-<model>.sh` or `maya-<model>.json`: the original tune is the fallback
- the deliverable is a run config in `maya.py`'s shape; keys the model's own config
  carries that the installer does not own (`sampling`, `reasoning_effort`, `vision`,
  `mcp_servers`, ...) are kept
- a key or a flag is only written when this checkout can read it:
  `maya_env.server_flags()` reads the flags `serve/server.py` defines and
  `maya_env.engine_flags()` reads the ones `src/program/generate.cpp` parses, so a flag
  from an older release is left out rather than rejected at startup
- the launcher runs `maya.py` when the checkout has one (that is how Maya's own
  `run-maya-<model>.sh` starts the server), and `serve/server.py` when it does not
- `--serve`, `--max-new` and `--expert-profile-save-every` are stripped from the
  production args: they exist only so the sweep can measure one request in one process
- the launcher uses the backend name the local `serve/server.py` accepts, read from its
  own `--engine` choices
- the model's existing `maya-<model>.json` is the base for the new config and is never
  modified; a later sweep inherits `max_<model>_config.json` only when there is no
  installed config to inherit from
- the trial record and the CSV stay inside this repository; only the launcher pair is
  written at the project root
- a deliverable directory that cannot be written (another user's launcher, a read-only
  root) falls back to `HyperCore-Tune-Maya/results/` rather than losing a finished sweep

## Tests

The tests never touch a GPU: `stub_engine.py` speaks the real protocol (`INFO`,
`session is up`, `DONE`) and computes its numbers from the flags it was given.

```bash
python -m pip install -r HyperCore-Tune-Maya/requirements.txt
pytest HyperCore-Tune-Maya/test_bayesian_tune.py
```

## License

MIT. See `LICENSE`.
