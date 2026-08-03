"""Config files for reproducible runs.

A run is reproducible when every input to it is written down: architecture,
training settings, seed, and the data file. That's what this module carries.

    python qwen_next_nano.py --config configs/kimi_like.json

Precedence is defaults < config file < flags you actually typed. The last part
matters: a flag only overrides the config if it appears in argv, so a config
setting is never silently clobbered by an argparse default it happens to differ
from. That's what `overrider()` is for.

Every run writes `config.resolved.json` next to its checkpoints — the fully
merged config plus the git commit, torch version and device. Feed that file
straight back in with --config to rerun the same thing. JSON both directions is
the reason: no dependency, and the dump is itself a valid input.

    python qwen_next_nano.py --config checkpoints/config.resolved.json
"""

import json
import os
import subprocess
import sys


def load(path):
    """Read a config file. Returns {} when no path is given.

    Layout (every key optional):
        {"model": {...}, "train": {...}, "seed": 42, "size": "nano", "file": "..."}
    """
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    unknown = set(cfg) - {"model", "train", "seed", "size", "file", "env", "_comment"}
    if unknown:
        raise ValueError(f"Unknown top-level keys in {path}: {sorted(unknown)}")
    return cfg


def overrider(argv=None):
    """Return `ov(target, key, flag, value)` that applies `value` only if `flag`
    was actually typed on the command line.

    Without this, an argparse default always looks like a user choice and would
    overwrite whatever the config file said.
    """
    argv = sys.argv[1:] if argv is None else argv
    typed = {a.split("=", 1)[0] for a in argv if a.startswith("-")}

    def ov(target, key, flag, value):
        if flag in typed:
            target[key] = value
        return target

    return ov


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=os.path.dirname(os.path.abspath(__file__)),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def snapshot(out_dir, payload, device=None):
    """Write the resolved config to `out_dir/config.resolved.json`.

    Records the git commit and torch version alongside it, because the same
    config against different code is a different experiment.
    """
    import torch

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "config.resolved.json")
    payload = {**payload, "env": {
        "git_commit": _git_commit(),
        "torch": torch.__version__,
        "device": str(device) if device else None,
    }}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def add_argument(parser):
    """Add --config to a parser. Kept here so every model spells it the same."""
    parser.add_argument("--config", type=str, default=None, metavar="PATH",
                        help="JSON config file; typed flags still override it")
