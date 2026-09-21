# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What this repo is

A **learning and experiment repo** for modern LLM architectures — not a
library, not a product. Every model is a single readable file you can follow
top to bottom. There is no `setup.py` and no CI; `nano/` is a plain package,
not something to install.

```
nano/
  attention_zoo.py   interchangeable attention and state-space mixers
  components.py      mHC, PLE, Engram, Mixture-of-Depths — the hybrid's swappable parts
  optim.py           Muon (+ AdamW for the rest), one optimizer object
  config.py          config loading, precedence, run snapshots, ROOT
  models/            one readable file per architecture
configs/             worked example configs
docs/                ARCHITECTURES.md
```

Anything path-based resolves against `config.ROOT` (the repo root), so
`the-verdict.txt` and `checkpoints/` stay put no matter where you run from.

Two jobs, deliberately kept apart:

- **Reading.** `nano/models/gpt_nano.py`, `nano/models/qwen_nano.py`, `nano/models/deepseek_nano.py`,
  `nano/models/gemma_nano.py`, `nano/models/mamba_nano.py`,
  `nano/models/qwen_next_nano.py` each teach one architecture end to end.
- **Experimenting.** `nano/attention_zoo.py` and `nano/components.py`, driven by the
  component flags on `nano/models/qwen_next_nano.py`, let you swap pieces without touching
  the teaching files. The hybrid file holds the *layout*; the mechanisms live in the two modules.

Do not merge those two jobs. A generic framework that can express every model
is exactly what makes this kind of repo unreadable.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install torch tiktoken
```

`.venv/` is gitignored. Everything runs on CPU, Apple MPS or CUDA; multi-GPU is
`torchrun` with DDP. Training data defaults to `the-verdict.txt`, downloaded on
first run (also gitignored — `.gitignore` excludes `*.txt`); `--file` or
`--dataset` take a local corpus or any Hugging Face dataset instead, resolved by
`nano/data.py`.

`accelerate` is a training dependency: `nano/accel.py` owns device selection,
the process group and mixed precision, so `python`, `accelerate launch` and
`torchrun` share one path. It is imported lazily, so the architectures and every
self-check still run on torch alone.

Other optional extras, also lazy and never at module import time: `datasets`
(only for `--dataset`), `transformers` and `trl` (only for `nano/hf.py` and
`examples/`).

Three things to know before touching the loops. `accelerator()` is a singleton
because Accelerate keeps global state — a second one with different settings
silently ignores them. Do not add a `DistributedSampler`: `prepare()` shards the
dataloaders already, and doing both gives each rank a slice of a slice. And keep
the world-size guard in `accel.wait()`: `wait_for_everyone()` calls
`c10d::barrier` even at world size 1, which is unimplemented for MPS, so a
single-process `torchrun` on a Mac trains fine and then dies at the epoch
boundary.

## Running the checks

There is no pytest suite. Each file self-tests:

```bash
python -m nano.attention_zoo                  # all attention variants (~1 min)
python -m nano.models.gpt_nano --self-check           # block wiring, cached decode
python -m nano.models.qwen_next_nano --self-check     # hybrid components, init properties
python -m nano.models.qwen_next_nano --train-check    # what only gradients reveal (~30s)
python -m nano.models.deepseek_nano --self-check      # MoE routing and load balancing
python -m nano.models.gemma_nano --self-check         # window boundary, dual RoPE, K-as-V, softcap
python -m nano.models.mamba_nano --self-check         # constant-memory state, no attention, Mamba-3 dials
```

**Run the relevant ones after any change to a model or the zoo.** They are fast
and they have caught real bugs repeatedly.

Three kinds of assertion, in increasing order of what they catch:

1. **Shape and wiring** — cheap, catches typos.
2. **Incremental decode** — prefill *n* tokens, then decode one at a time, and
   require the result to equal a full forward pass. This is what catches a
   broken KV cache, recurrent state, or convolution state.
3. **Causality** — perturb token *t*, require no output before *t* to move.
   Nothing else catches a future leak, and a model that reads the future trains
   happily with a loss curve that looks unusually *good*, not broken.

Some properties only appear after gradients have flowed — an off-by-one in a
prediction target, a matrix drifting off its manifold, a router collapsing.
Those live in `--train-check` and overfit a tiny model to assert them directly.

## Adding an attention mechanism

One decorator. Nothing else to edit:

```python
from nano.attention_zoo import register

@register("myattn", "My attention (Paper, 2026)")
class MyAttention(nn.Module):
    def __init__(self, cfg): ...
    def forward(self, x, use_cache=False): ...   # -> (B, T, emb_dim)
    def reset_cache(self): ...
```

Registering picks it up everywhere: `python -m nano.models.gpt_nano --attention myattn`,
`--attention all` benchmarks it, and `python -m nano.attention_zoo` tests it for
shape, incremental decode and causality automatically.

Read new options off `cfg` with `cfg.get("my_option", default)` so existing
configs keep working.

## Adding a component to the hybrid model

Components in `nano/models/qwen_next_nano.py` follow a consistent shape:

- Read from `cfg` with a **default that disables it**, so every existing config
  and checkpoint behaves as before.
- Add a CLI flag and thread it into `cfg` in `main()`.
- Add an assertion to `self_check()` proving it is *not a no-op* — the most
  common failure is a component that silently does nothing.
- Initialise to be equivalent to the model without it, where possible. It makes
  the no-op claim exactly testable.

## Config files and reproducibility

Runs are config-driven; flags still work:

```bash
python -m nano.models.qwen_next_nano --config configs/kimi_like.json
```

Precedence is **defaults < config file < flags actually typed**. A flag only
overrides the config if it appears in `argv` — see `config.overrider()`. Do not
replace that with a comparison against argparse defaults; a config value would
then be silently clobbered whenever it happened to differ from a default.

Every run writes `checkpoints/config.resolved.json`, the fully merged config
plus git commit and torch version. That file is itself a valid `--config`
input, which is what makes a result reproducible. JSON both directions is the
reason — no dependency, and the dump round-trips.

## Conventions

- **No new dependencies.** torch and tiktoken only. The stdlib is why config is
  JSON rather than YAML.
- **Single-file models.** `nano/models/qwen_next_nano.py` imports the training loop from
  `nano/models/qwen_nano.py` rather than copying it, but its *architecture* is all local.
  That is the line: duplicate boilerplate is worse than an import, duplicated
  architecture defeats the point of the file.
- **Comment the surprise, not the syntax.** Docstrings here explain why a
  design exists and what breaks without it. Several record bugs that took real
  time to find — do not delete those.
- **`ponytail:` comments** mark deliberate simplifications and name their
  ceiling. Several components demonstrate a mechanism without its performance
  benefit (DSA and CSA/HCA mask dense tensors instead of gathering). Keep those
  notes honest; do not quietly upgrade the claim.
- **Auxiliary losses go in the gradient, not the metric.** Reported train and
  val losses are pure LM loss so runs stay comparable.

## Known gaps

- Nothing is tested under DDP, on CUDA, or above the `nano` size.
- No component is verified to *improve* anything. A 3.5M model on a tiny corpus
  cannot separate them; the differences you see are noise. Do not present them
  as results.
- `docs/ARCHITECTURES.md` documents every component, its paper, and the
  implementation gotchas. Update it when adding one.
- `docs/TIMELINE.md` orders every component by the month its paper appeared,
  with the arXiv id and where it lives here. Add a row when adding a component.
- `docs/ROADMAP.md` lists open work, what blocks each item, and what would
  count as done. Check it before proposing what to build next.

## Hard rules

- **Never `git commit` or `push` unless asked.** The user manages git here, and
  has rewritten history more than once.
- **Do not add a test framework, packaging, or CI** without being asked.
- **Do not delete the gotcha notes** in docstrings or `docs/ARCHITECTURES.md` § 6.
  They are the most expensive knowledge in the repo.
