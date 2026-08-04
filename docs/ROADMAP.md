# Roadmap

Open work, ordered by value. Each entry says what blocks it and what would prove
it done, because the hard part is usually a detail that isn't obvious until you
start.

---

## 1. Run an actual experiment

This is an experiment repo that has not yet run an experiment.

Everything here is verified *mechanically* — shapes, cache and state
equivalence, causality, gradients reaching the right parameters, auxiliary
objectives actually training. Nothing is verified to **help**. At `nano` size on
`the-verdict.txt` the differences between components are noise, and they should
never be quoted as results.

The two pieces that were missing are now in place: hub datasets
(`--dataset roneneldan/TinyStories`) and config-file reproducibility, so a run
records its architecture, seed, corpus and git commit and can be replayed from
`config.resolved.json` alone.

**Good first questions**, each a single flag apart:

| Question | Command |
|---|---|
| Does per-channel decay beat per-head? | `--linear kda` vs `--linear deltanet` |
| Does ShortConv earn its parameters? | `--short-conv 4` vs `0` |
| Do hyper-connections converge faster? | `--residual mhc` vs `plain` |
| What does MTP cost and buy? | `--mtp-weight 0.3` vs `0` |
| How much recall does 3:1 give up? | `--ratio 3` vs `--ratio 1` |

**Needs:** a GPU and `--size small` or `medium`. The differences will not
separate below that.

**Done when:** two configs, identical but for one flag, produce a loss gap
larger than seed variance — which means running each seed more than once.

---

## 2. Multi-GPU, CUDA, and scale

Nothing in this repo has ever run under DDP, on CUDA, or above `nano`.

The specific worry: `qwen_nano.train()` passes targets to models that set
`needs_targets` so their auxiliary loss is computed *inside* the DDP-wrapped
forward. That is the whole reason the plumbing is shaped that way — computed
outside, MTP's gradients would never sync across ranks. **That code path has
never executed.** It is either correct or quietly wrong, and one two-GPU run
settles it.

```bash
accelerate launch --num_processes 2 -m nano.models.qwen_next_nano --size small --mtp-weight 0.3
```

Accelerate now owns this path, which removes the hand-rolled DDP code but does
not remove the doubt — the sync itself is still unproven. Note that it cannot be
checked on an Apple machine: `accelerate launch --cpu --num_processes 2` reports
`world=1, DistributedType.NO` rather than forming a group, and `torchrun` on CPU
hangs in gloo rendezvous. This needs real GPUs.

**Done when:** a two-rank run matches a single-rank run's loss curve at the same
effective batch size, with MTP on.

---

## 3. Attention masks in the zoo

Every variant is causal-only. `NanoForCausalLM` raises on a padded batch rather
than let real tokens attend to padding, which rules out chat-style SFT, unpacked
datasets, and TRL's default path. This is the change that unlocks the most.

**The hard part is not softmax attention.** Masking a score matrix is a line.
The linear layers are the problem: Gated DeltaNet and KDA *consume* each token
into a recurrent state, so masking means suppressing the state update at padded
positions, not masking a score. Same for ShortConv's rolling window and the
compressed variants' group boundaries — a padded position must not close a
group.

**Done when:** `python -m nano.attention_zoo` passes a new assertion that a
sequence padded on the right produces identical outputs at unpadded positions to
the same sequence unpadded, for every variant. Then flip
`NanoConfig(allow_padding=True)` to the default.

---

## 4. Generation for DPO and GRPO

`.generate()` is not wired to `past_key_values`: the KV cache here is internal
module state (`use_cache=True` plus `reset_cache()`), not HF's `Cache` objects.
`SFTTrainer` never calls generate, so SFT is unaffected; the preference trainers
do.

Two steps: a `Cache` adapter over the existing per-module state, and
`prepare_inputs_for_generation`. Speculative decoding with the MTP head is a
third — and it needs the linear layers' recurrent state to roll back on a
rejected token, which it cannot currently do without snapshotting `S` per step.

**Done when:** greedy `.generate()` matches the repo's own `generate_cached()`
token for token.

---

## 5. `--attn mla` for the hybrid

The one architecture piece still missing. Kimi Linear pairs its KDA layers with
gated **MLA**, while `qwen_next_nano` uses gated GQA in the full-attention slots.
MLA exists in the zoo but with the `(x, use_cache)` signature and no RoPE, so it
cannot drop into the hybrid block as-is.

A RoPE-aware gated-MLA adapter is roughly 30 lines and makes GQA-vs-MLA a
one-flag experiment. With it plus a `kimi` preset
(`linear=kda, short_conv=4, ratio=3, attn=mla`), the repo covers Kimi Linear
faithfully without a separate model file — it is a configuration of one that
already exists, not a new architecture.

---

## Smaller

- **28 `ruff check` findings**, none applied. Mostly `import torch.nn as nn` →
  `from torch import nn`. Behaviourally free, diff-heavy.
- **Gradient checkpointing** is unsupported by the HF wrapper. It would mean
  threading `torch.utils.checkpoint` through each architecture's block loop —
  only worth it above `medium`.
- **Streaming corpora.** `nano/data.py` joins the whole corpus into one string,
  which is what the native `TextDataset` consumes. Fine here, wrong for a real
  pre-training run — that wants tokenize-to-disk and a streaming dataset.
- **DSA and CSA/HCA demonstrate their mechanism, not their speedup.** Both mask
  dense tensors instead of gathering. Real gains need a sparse kernel, and the
  `ponytail:` notes in the code say so; keep them honest.

---

## Where the bodies are buried

Before changing anything, read § 6 of [ARCHITECTURES.md](./ARCHITECTURES.md).
Nine bugs found while building this, every one of which produced a healthy
loss curve while being wrong. The pattern is consistent enough to plan around:
**a loss curve cannot tell you a component works.** Each of those was caught by
a test that asked the component directly whether it was doing its job, and the
three tiers of check (shape, incremental decode, causality) exist because of
them.
