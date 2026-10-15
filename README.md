# HyperCore-Tune-Maya

<p align="center"><img src="docs/media/HyperCore-Tune.jpeg" width="420" alt="HyperCore-Tune-Maya"></p>

Bayesian (Optuna TPE) tuning of the Project Maya engine, as a standalone repository that
attaches to a Maya checkout. It has no other association with Project Maya than
compatibility: clone it inside a checkout, or next to one as its own app.

It searches the engine dimensions on a given machine, measures one request per
configuration by booting the engine, and writes back a run config and a launcher in
exactly the format `maya.py` writes for an installed model:

```
maya-<model>.json    the winning run config (exe, args, cwd, tokenizer, port, ...)
run-maya-<model>.sh      the launcher: maya.py with that config
run-maya-<model>.bat     the same launcher for a Windows install
HyperCore-Tune-Maya/results/maya-<model>.record.json   every trial and its metrics
HyperCore-Tune-Maya/results/bayesian-<machine>-<model>-<objective>-<stamp>.csv
```

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
./HyperCore-Tune-Maya/max_tune.sh --model maya-s24 --gen      --cntx 131072
./HyperCore-Tune-Maya/max_tune.sh --model maya-s24 --prefill  --cntx 131072 --trials 60
./HyperCore-Tune-Maya/max_tune.sh --model maya-s24 --latency  --mode staged --dry-run
./HyperCore-Tune-Maya/max_tune.sh --show-project
./HyperCore-Tune-Maya/max_tune.sh --help
```

On Windows 10/11 the same flags go to `max_tune.bat` from this folder:

```bat
max_tune.bat --model maya-s24 --gen --cntx 131072
max_tune.bat --show-project
```

Objectives: `--gen` maximises generated tokens/s, `--prefill` maximises prompt tokens/s,
`--latency` minimises `prompt_ms + decode_ms`.

`--trials` is the number of **unseen** configurations to evaluate. A suggestion that
matches a point already in history (or already measured in this sweep) is pruned, not
counted, and the sampler keeps drawing until that many new points have been booted.

Before a real sweep, stop the running server (Ctrl-C on `./maya.sh` or on the
`run-maya-<model>.sh` launcher) so the engine can hold the card for the boot.

## Search space

The dimensions searched are the columns every row is written with:

```
layer  vram  pcie  pcie_mode  ple_io  kv  kvstream  hostworker  worker  tail  draft-floor  status
```

`status` is not a parameter: it is what the engine did with the configuration
(`PASS`, `FAIL_BOOT`, `NO_TOKENS`, `TIMEOUT`).

| Tuner flag | Engine flag | Default search | Hold constant |
|---|---|---|---|
| `--vram-min` / `--vram-max` / `--vram-step` | `--vram-reserve-mib` | 256–1536 step 256 | (range flags, not a freeze) |
| `--pcie` | `--pcie-frac` | 0–1 | `--pcie 0.95` |
| `--pcie-mode` | `--pcie-mode` | `auto`, `dma`, `kernel`, `direct` | `--pcie-mode auto` |
| `--ple-io` | `--ple-io` | `direct`, `mmap` | `--ple-io mmap` |
| `--kv` | `--kv` | `q4_0`, `k8v4`, `int8`, `fp16` | `--kv int8` |
| `--kvstream` | `--kv-streaming` | on / off | `--kvstream on` |
| `--hostworker` / `--host-worker` | `--no-host-worker` | 1 (host joins the pool) / 0 | `--hostworker 0` |
| `--worker` | `--pool-workers` | 0 … physical cores − 1 | `--worker 8` |
| `--tail` | `--prompt-cache` | 0–6 conversation checkpoints | `--tail 0` |
| `--draft-floor` / `--spec-min-p` | `--spec-min-p` | 0.001–1 | `--draft-floor 0.5` |

A freeze flag holds that dimension at a constant instead of searching it. History rows
that do not match a freeze are not used as priors.

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
found in the data folder the installer recorded
(`~/.config/project-maya/settings.json`, or `%APPDATA%\project-maya\settings.json` on
Windows, or `$MAYA_DATA`), then in a data folder next to the checkout
(`Maya-data/models/<model>/pack`).

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

## Compatibility with Project Maya

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
- the model's existing `maya-<model>.json` is used as the base for the new config
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
