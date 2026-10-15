#!/usr/bin/env python3
"""HyperCore-Tune-Maya/maya_env.py - where this sweep finds the Project Maya it tunes.

HyperCore-Tune-Maya is a standalone app that is compatible with Project Maya and has no
other association with it. It is cloned either inside a Maya checkout
(<project-maya>/HyperCore-Tune-Maya) or on its own (<home>/HyperCore-Tune-Maya beside
<home>/project-maya). Every path handed to the engine, the server or the deliverables is
absolute, so a sweep resolves the same tree on Windows 10/11 and on Linux, whatever the
working directory was when it started. ~ and %VAR% in a user path are expanded first.

Resolution order for the Project Maya checkout this app talks to:

    1. --maya-root, $MAYA_ROOT, $PROJECT_MAYA_ROOT, $TUNE_ROOT
    2. the nearest ancestor of this repository that is a Maya checkout
    3. a sibling of this repository named project-maya / maya / Maya
    4. any other sibling that looks like a Maya checkout

A directory counts as a Maya root when it has serve/server.py and one of maya.py,
CMakeLists.txt or src/program/generate.cpp. Nothing here probes the network or the GPU.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parent

# Names a Maya checkout uses. A checkout is only ever a root when the server and the
# engine source are both in it, so a look-alike folder cannot be mistaken for one.
ROOT_NAMES: tuple[str, ...] = ("project-maya", "maya", "Maya", "Project-Maya", "maya-cpp")
ROOT_MARKERS: tuple[str, ...] = ("serve/server.py",)
ROOT_HINTS: tuple[str, ...] = ("maya.py", "setup.py", "CMakeLists.txt", "src/program/generate.cpp")

# Engine binaries, as Maya's installer builds them: CUDA in build/, HIP in build-hip/.
# A build directory is only a target when it holds an executable.
ENGINE_CANDIDATES: tuple[str, ...] = (
    "build/strata", "build/strata.exe",
    "build-hip/strata", "build-hip/strata.exe",
    "build-release/strata", "build-release/strata.exe",
    "dist/strata", "dist/strata.exe",
)
_VENV_PYTHONS: tuple[str, ...] = (
    ".venv/Scripts/python.exe", "venv/Scripts/python.exe",
    ".venv/bin/python", ".venv/bin/python3", "venv/bin/python",
)

# Maya keeps its model packs beside the checkout (Maya-data/models/<Quant>/pack) and its
# shipped expert profile in the checkout's own data/ folder.
# The settings folder is named after the project; the older names are still read so a
# machine configured against an earlier release is not re-tuned blind.
SETTINGS_DIRS: tuple[str, ...] = ("project-maya", "maya", "strata", "hypercore")
DATA_ENV: tuple[str, ...] = ("MAYA_DATA", "PROJECT_MAYA_DATA")
SERVER_ENGINE_ENV: tuple[str, ...] = ("MAYA_SERVER_ENGINE",)
ROOT_ENV: tuple[str, ...] = ("MAYA_ROOT", "PROJECT_MAYA_ROOT", "TUNE_ROOT")
ENGINE_ENV: tuple[str, ...] = ("MAYA_ENGINE", "HC_ENGINE")
HARDWARE_ENV: tuple[str, ...] = ("TUNE_HARDWARE", "HC_TUNE_HARDWARE")


def abs_path(path: str | Path | None) -> Path | None:
    """An absolute path, with ~ and %VAR% expanded. None when the value is missing or blank.

    Path.resolve() does not expand ~, so '~/project-maya' would otherwise be read as a
    directory named '~' under the current working directory. On Windows, %USERPROFILE%
    and %MAYA_ROOT% are expanded the same way. Every caller that takes a user, env or
    settings path goes through here.
    """
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    text = os.path.expandvars(text)
    text = re.sub(r"%([A-Za-z0-9_]+)%",
                  lambda m: os.environ.get(m.group(1), m.group(0)), text)
    # A Windows path copied into a Linux test (or a .bat) uses backslashes.
    text = text.replace("\\", "/")
    return Path(text).expanduser().resolve()


def _is_runnable(path: Path) -> bool:
    """True when path is a file this process can launch. Windows has no Unix execute bit."""
    if not path.is_file():
        return False
    if os.name == "nt":
        return True
    return os.access(path, os.X_OK)


def config_bases() -> list[Path]:
    """Folders that may hold project-maya/settings.json: $XDG_CONFIG_HOME, %APPDATA%, ~/.config."""
    ordered: list[Path] = []
    for raw in (os.environ.get("XDG_CONFIG_HOME"), os.environ.get("APPDATA")):
        p = abs_path(raw)
        if p is not None and p not in ordered:
            ordered.append(p)
    fallback = Path.home() / ".config"
    if fallback not in ordered:
        ordered.append(fallback)
    return ordered


def _is_maya_root(path: Path) -> bool:
    """A directory is the project root when the server is in it and the engine source is beside it."""
    if not path.is_dir():
        return False
    if not any((path / marker).is_file() for marker in ROOT_MARKERS):
        return False
    return any((path / hint).is_file() for hint in ROOT_HINTS)


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p is not None and _is_maya_root(p):
            return p
    return None


def _sibling_dirs(base: Path) -> list[Path]:
    try:
        return sorted(p for p in base.iterdir() if p.is_dir())
    except OSError:
        return []


def candidate_roots() -> list[Path]:
    """Every root this repository could reasonably mean, best first. Never raises."""
    pinned = [abs_path(os.environ.get(name)) for name in ROOT_ENV]

    ancestors: list[Path] = list(REPO.parents)  # <project-maya>/HyperCore-Tune-Maya -> <project-maya>

    named: list[Path] = []
    markers: list[Path] = []
    names = {n.lower() for n in ROOT_NAMES}
    parent, grand = REPO.parent, REPO.parent.parent
    for d in _sibling_dirs(parent):
        if d.resolve() == REPO:
            continue
        if d.name.lower() in names:
            named.append(d)
        elif _is_maya_root(d):
            markers.append(d)
    for d in _sibling_dirs(grand):
        if d.name.lower() in names:
            named.append(d)

    ordered: list[Path] = []
    for p in pinned + ancestors + named + markers:
        if p is None:
            continue
        p = abs_path(p) or p.resolve()
        if p not in ordered:
            ordered.append(p)
    return ordered


@lru_cache(maxsize=1)
def maya_root() -> Path:
    """The absolute Project Maya root this sweep runs against. Raises with what it looked for."""
    root = _first_existing(candidate_roots())
    if root is None:
        looked = ", ".join(str(p) for p in candidate_roots()) or "nothing"
        raise SystemExit(
            "HyperCore-Tune-Maya: no Project Maya checkout found. Clone this repository "
            "inside a Maya tree, or point at one with --maya-root or MAYA_ROOT. "
            f"Looked in: {looked}"
        )
    return root


# Back-compatible alias: a sweep written against the Strata name still resolves.
strata_root = maya_root


def engine_binary(root: Path | None = None) -> str:
    """The absolute path of the engine Maya built. "" when there is no engine yet."""
    root = root or maya_root()
    pinned = next((os.environ.get(name) for name in ENGINE_ENV if os.environ.get(name)), "")
    if pinned:
        found = abs_path(pinned)
        return str(found) if found is not None else str(Path(os.path.expandvars(pinned)).expanduser())
    for rel in ENGINE_CANDIDATES:
        cand = root / rel
        if _is_runnable(cand):
            return str(cand)
    return ""


def maya_script(root: Path | None = None) -> str:
    """maya.py of the project, as an absolute path - the entry point Maya's launchers use."""
    root = root or maya_root()
    return str(root / "maya.py")


def server_script(root: Path | None = None) -> str:
    """serve/server.py of the project, as an absolute path."""
    root = root or maya_root()
    return str(root / "serve" / "server.py")


def venv_python(root: Path | None = None) -> str:
    """The project venv's interpreter, or "" if it has not been created."""
    root = root or maya_root()
    for rel in _VENV_PYTHONS:
        cand = root / rel
        if _is_runnable(cand):
            return str(cand)
    return ""


@lru_cache(maxsize=8)
def server_flags(root: Path) -> frozenset[str]:
    """The --flags serve/server.py defines, read from its own argparse block.

    Empty when the file cannot be read. Cached per root; the server is not going to
    change while a sweep is running.
    """
    try:
        text = (root / "serve" / "server.py").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return frozenset()
    return frozenset(re.findall(r'add_argument\(\s*"(--[A-Za-z0-9-]+)"', text))


def entry_script(root: Path | None = None) -> str:
    """The script a launcher should run: maya.py, or serve/server.py if there is no maya.py.

    Maya's own launchers (run-maya-<model>.sh, START-MAYA.bat) go through maya.py, which
    sets the library path and the restart-on-update environment before the server starts.
    A checkout that only has the server is started through the server.
    """
    root = root or maya_root()
    maya = root / "maya.py"
    if maya.is_file():
        return str(maya)
    return str(root / "serve" / "server.py")


@lru_cache(maxsize=8)
def entry_flags(root: Path) -> frozenset[str]:
    """The --flags this project's entry point defines, read from its own argparse block."""
    try:
        text = Path(entry_script(root)).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return frozenset()
    return frozenset(re.findall(r'add_argument\(\s*"(--[A-Za-z0-9-]+)"', text))


@lru_cache(maxsize=8)
def engine_flags(root: Path) -> frozenset[str]:
    """The --flags this Maya build's engine accepts, read from src/program/generate.cpp.

    A deliverable must not carry a flag the engine would reject at startup. The source is
    the only reliable answer: feature names move between releases, and a build directory
    does not tell you what the binary understands. Empty when the source is not there.
    """
    try:
        text = (root / "src" / "program" / "generate.cpp").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return frozenset()
    return frozenset(re.findall(r'a == "(--[A-Za-z0-9-]+)"', text))


def server_supports(flag: str, root: Path | None = None) -> bool:
    """True when this project's server accepts the flag.

    A deliverable must not carry a key or a flag the local server does not know: the
    launcher would fail at startup. Feature names move between releases, so the server's
    own source is the only reliable answer.
    """
    root = root or maya_root()
    return flag in server_flags(root)


def engine_supports(flag: str, root: Path | None = None) -> bool:
    """True when the engine in front of us has this flag; unknown flags are assumed absent."""
    root = root or maya_root()
    flags = engine_flags(root)
    return bool(flags) and flag in flags


def server_engine_name(root: Path | None = None) -> str:
    """The backend name serve/server.py accepts for the real engine.

    Read from the server's own --engine choices, so a project that renames its backend is
    followed rather than assumed. Falls back to "strata" when the server cannot be read.
    """
    root = root or maya_root()
    pinned = next((os.environ.get(name) for name in SERVER_ENGINE_ENV if os.environ.get(name)), "")
    if pinned:
        return pinned
    try:
        text = (root / "serve" / "server.py").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "strata"
    for line in text.splitlines():
        if '"--engine"' in line and "choices" in line:
            names = re.findall(r'["\']([^"\']+)["\']', line.split("choices", 1)[1])
            real = [n for n in names if n and n != "mock"]
            if real:
                return real[0]
    return "strata"


def settings_path() -> Path:
    """The installer's settings file.

    Linux: ~/.config/project-maya/settings.json (or $XDG_CONFIG_HOME). Windows: the same
    folder under %APPDATA%, then ~/.config. The older maya name is tried second.
    """
    bases = config_bases()
    for base in bases:
        for name in SETTINGS_DIRS:
            p = base / name / "settings.json"
            if p.is_file():
                return p
    return bases[0] / SETTINGS_DIRS[0] / "settings.json"


def describe() -> dict[str, str]:
    """A small, printable summary of what was resolved. Used by --show-project and the tests."""
    root = maya_root()
    return {
        "repo": str(REPO),
        "root": str(root),
        "engine": engine_binary(root),
        "maya": maya_script(root),
        "server": server_script(root),
        "server_engine": server_engine_name(root),
        "venv": venv_python(root),
        "data": ", ".join(data_roots(root)) if data_roots(root) else "",
    }


def data_roots(root: Path | None = None) -> tuple[str, ...]:
    """Model data folders in the order Maya's installer would use them.

    Order: $MAYA_DATA, the `data_dir` recorded in the user's settings, a data folder next
    to the checkout (Maya-data, strata-data, ...), then any sibling data folder. Only
    folders that exist and hold a models/ or packs/ directory are returned. Folders
    inside the running user's home come first. Never raises.
    """
    root = root or maya_root()
    pinned = [os.environ.get(name, "") for name in DATA_ENV] + [settings_data_dir()]
    discovered: list[str] = []
    base = abs_path(root) or Path(root).resolve()
    for parent in dict.fromkeys((base.parent, base.parent.parent)):
        try:
            entries = list(parent.iterdir())
        except OSError:
            continue
        for d in entries:
            name = d.name.lower()
            if d.is_dir() and (name.startswith(("maya-data", "strata-data", "hypercore-data"))
                               or name.endswith("-data")):
                discovered.append(str(d.resolve()))
    home = str(Path.home())
    discovered.sort(key=lambda r: (not r.startswith(home), r))

    ordered: list[str] = []
    for raw in pinned + discovered:
        if not raw:
            continue
        p = abs_path(raw)
        if p is None or not p.is_dir():
            continue
        if not any((p / item).is_dir() for item in ("models", "packs", "data")):
            continue
        if str(p) not in ordered:
            ordered.append(str(p))
    return tuple(ordered)


def settings_data_dir() -> str:
    """The data folder the installer remembered in the user's settings, or "".

    Mirrors Maya's own settings handling: `~/.config/project-maya/settings.json` (or
    `$XDG_CONFIG_HOME`), and on Windows `%APPDATA%\\project-maya\\settings.json`. A
    missing or unreadable file is not an error, it just means no remembered folder.
    """
    path = settings_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if isinstance(data, dict) and data.get("data_dir"):
        found = abs_path(data["data_dir"])
        if found is not None:
            return str(found)
    return ""


if __name__ == "__main__":  # pragma: no cover - a small self-check
    print(json.dumps(describe(), indent=1))
    sys.exit(0)
