# nano-models

Minimal, from-scratch PyTorch implementations of modern LLM architectures, built for learning. Each model is a single self-contained file: model code, training loop (single-GPU + multi-GPU via `torchrun`), checkpointing, and a generation utility.

The goal is not to be fast or competitive — it's to make each architectural idea (attention variant, normalization, position encoding, MoE routing, …) easy to read end-to-end in one file.

**[docs/ARCHITECTURES.md](./docs/ARCHITECTURES.md)** is the guide to every component: the problem it was invented to solve, how it solves it, its paper, and the implementation gotchas that aren't in any paper. **[docs/ROADMAP.md](./docs/ROADMAP.md)** is what's open next and what blocks it.

---

## Models

| Model | File | Sizes | Headline ideas |
|---|---|---|---|
| GPT Nano | [`nano/models/gpt_nano.py`](./nano/models/gpt_nano.py) | nano (3M) → xl (770M) | Decoder-only transformer, learned positional embeddings, LayerNorm, GELU MLP. Pluggable attention via `attention_zoo`. |
| Qwen Nano | [`nano/models/qwen_nano.py`](./nano/models/qwen_nano.py) | nano (5M) → qwen-0.6B (620M) | RMSNorm, SwiGLU, RoPE, Grouped-Query Attention with QK-norm, no bias. |
| DeepSeek Nano | [`nano/models/deepseek_nano.py`](./nano/models/deepseek_nano.py) | nano (5M) → large (500M) | Multi-Head Latent Attention (MLA), Mixture of Experts with shared + routed experts, aux-loss-free load balancing, LatentMoE, RMSNorm + SwiGLU + RoPE. |
| Qwen-Next Nano | [`nano/models/qwen_next_nano.py`](./nano/models/qwen_next_nano.py) | nano (5M) → large (350M) | Hybrid 3:1 linear/full attention (Qwen3-Next, Qwen3.5, Kimi Linear), gated attention, Gated DeltaNet or KDA, multi-token prediction, mHC hyper-connections, per-layer embeddings. Only 1-in-4 layers holds a KV cache. |

Shared building blocks live in [`nano/attention_zoo.py`](./nano/attention_zoo.py): `mha`, `gqa`, `gated`, `mla`, `swa`, `deltanet`, `kda`, `dsa`, `csa`, `hca`, all with the same `(cfg) → forward(x, use_cache)` interface and a KV cache.

Training data is [`the-verdict.txt`](./the-verdict.txt) — a tiny corpus that fits in memory and lets every model overfit fast enough to verify the architecture is wired correctly.

---

## Setup

```bash
pip install -r requirements.txt   # torch, tiktoken
```

Works on CPU, Apple MPS, and CUDA. Multi-GPU uses `torchrun` with PyTorch DDP.

## Quick start

```bash
# Train the smallest variant of each model on the bundled corpus
python -m nano.models.gpt_nano
python -m nano.models.qwen_nano
python -m nano.models.deepseek_nano
python -m nano.models.qwen_next_nano

# Pick a size and (for GPT) an attention variant
python -m nano.models.gpt_nano --size small --attention gqa --epochs 5
python -m nano.models.gpt_nano --attention all         # benchmark all ten attention types

# Swap components on the hybrid model
python -m nano.models.qwen_next_nano --linear kda --short-conv 4 --residual mhc

# Train on a Hugging Face dataset, or from a config file
python -m nano.models.qwen_next_nano --dataset roneneldan/TinyStories --dataset-limit 5000
python -m nano.models.qwen_next_nano --config configs/kimi_like.json

# Resume from a checkpoint
python -m nano.models.gpt_nano --resume checkpoints/ckpt_step_500.pt

# Multi-GPU — Accelerate handles device, process group and mixed precision
accelerate launch -m nano.models.qwen_nano --size qwen-0.6B --batch-size 32 --grad-accum 4
torchrun --nproc_per_node=8 -m nano.models.qwen_nano --size qwen-0.6B    # still works
```

Launching with `python`, `accelerate launch` or `torchrun` all go through the same code path — `nano/accel.py` wraps Accelerate, so the training loops carry no `init_process_group`, DDP wrapper, `DistributedSampler` or autocast context. `accelerate config` unlocks FSDP and DeepSpeed for the larger sizes.

Each script accepts `--size`, `--epochs`, `--batch-size`, `--grad-accum`, `--resume`, `--config`, `--seed` and the dataset flags. See the docstring at the top of each file for the full list.

Every file self-tests — no pytest, no CI:

```bash
python -m nano.attention_zoo                        # all ten attention variants
python -m nano.models.qwen_next_nano --self-check   # hybrid components
python -m nano.models.qwen_next_nano --train-check  # what only gradients reveal
python -m nano.models.deepseek_nano --self-check    # MoE routing and load balancing
python -m nano.hf                                   # HF adapter vs the native loop
```

## Training data

Every model and the TRL example take the same flags — a local file, the bundled sample, or any Hugging Face dataset with a text column:

```bash
python -m nano.models.qwen_next_nano                                   # bundled the-verdict.txt
python -m nano.models.qwen_next_nano --file mycorpus.txt
python -m nano.models.qwen_next_nano --dataset roneneldan/TinyStories --dataset-limit 5000
python -m nano.models.qwen_next_nano --dataset Salesforce/wikitext \
    --dataset-config wikitext-2-raw-v1 --split "train[:2000]"
python examples/sft_trl.py --dataset Salesforce/wikitext --dataset-config wikitext-2-raw-v1
```

`--text-field` picks the column if it isn't called `text`; a wrong name lists the columns that do exist. The same settings work as a `data` block in a config file (see [configs/wikitext.json](./configs/wikitext.json)), and the resolved data source is recorded in `config.resolved.json`, so a run reproduces its corpus as well as its architecture.

`datasets` is imported lazily — it's only needed if you actually pass `--dataset`. Note the corpus is joined into a single string, which is what the native `TextDataset` consumes; that's fine at this scale and wrong for a real pre-training corpus.

## Reproducible runs

Runs are config-driven; flags still work and take precedence over the file.

```bash
python -m nano.models.qwen_next_nano --config configs/kimi_like.json
python -m nano.models.deepseek_nano  --config configs/latent_moe.json --epochs 5   # flag wins
```

Precedence is **defaults < config file < flags you actually typed** — a flag only overrides the config if it appears on the command line, so a config value is never clobbered by an argparse default it happens to differ from.

Every run writes `checkpoints/config.resolved.json`: the fully merged config plus seed, git commit and torch version. That file is itself a valid `--config` input, so reproducing a result is:

```bash
python -m nano.models.qwen_next_nano --config checkpoints/config.resolved.json
```

Verified round-tripping to identical loss on all four models.

## Training with Hugging Face / TRL

`nano/hf.py` wraps any of these architectures as a `PreTrainedModel`, so they work with TRL's trainers, `save_pretrained`, and the Hub:

```python
from nano.hf import NanoConfig, NanoForCausalLM

model = NanoForCausalLM(NanoConfig(arch="qwen_next", size="nano"))
```

```bash
python -m nano.hf                  # check the wrapper's loss matches the native loop
python examples/sft_trl.py --arch qwen_next --steps 20
```

Three constraints, all of which fail silently if ignored — see the module docstring for why:

- **No attention masks.** Every variant is causal-only, so a padded batch would attend to padding. Use equal-length samples; the wrapper raises rather than let it slide. TRL's `packing=True` is *not* a fix — it flattens the batch and needs a FlashAttention varlen kernel.
- **`loss_type="nll"`.** TRL's default chunked loss bypasses the wrapper and calls the backbone directly.
- **`gradient_checkpointing=False`.** Unsupported, and irrelevant at these sizes.

Verified: the wrapper's loss matches the native training loop exactly for all four architectures, including the MTP and DSA auxiliary-loss paths, and all four train under `SFTTrainer`.

## Adding an attention mechanism

One decorator, nothing else to edit:

```python
from nano.attention_zoo import register

@register("myattn", "My attention (Paper, 2026)")
class MyAttention(nn.Module):
    def __init__(self, cfg): ...
    def forward(self, x, use_cache=False): ...   # -> (B, T, emb_dim)
    def reset_cache(self): ...
```

Registering picks it up everywhere: `gpt_nano.py --attention myattn`, `--attention all` benchmarks it against the other ten, and `python -m nano.attention_zoo` tests it for shape, incremental-decode equivalence and causality automatically. Read your own options off `cfg` with `cfg.get("my_option", default)` so existing configs keep working.

---

## Architecture notes

The sections below are my own notes from reading and re-implementing each architecture. They are intentionally informal — what stuck, what surprised me, what was non-obvious from the paper.

### GPT Nano

Reference: GPT-2 (Radford et al., 2019).

<!-- NOTES: GPT — paste your learnings here.
Suggested structure:
- Why decoder-only?
- LayerNorm placement (pre-LN vs post-LN) and why pre-LN won
- Learned positional embeddings: limitations
- Causal mask, KV cache mechanics
- Weight tying (embedding ↔ output projection)
- Anything that surprised you while implementing it
-->

### Qwen Nano

Reference: Qwen3 (Alibaba, 2024–2025).

What changed vs GPT:
- **RMSNorm** replaces LayerNorm (no mean subtraction, no bias).
- **SwiGLU** MLP replaces GELU — gated activation, ~3× wider hidden dim, but with a 2/3 factor to keep param count comparable.
- **RoPE** replaces learned positional embeddings — relative position encoded by rotating Q/K in 2D pairs.
- **Grouped-Query Attention (GQA)** — multiple Q heads share K/V heads, cutting KV-cache size at long context.
- **QK-norm** — RMSNorm on Q and K before the attention dot product, stabilizes training at large scale.
- **No bias** in any linear layer.

<!-- NOTES: Qwen — paste your learnings here.
Suggested structure:
- RMSNorm vs LayerNorm: what changes practically
- Why SwiGLU over GELU
- RoPE intuition: how rotation = relative position
- GQA: KV-cache memory math (heads → groups)
- QK-norm: what it stabilizes and why
- Anything that surprised you
-->

### DeepSeek Nano

Reference: DeepSeek-V2 / V3 (DeepSeek, 2024).

What's new vs Qwen:
- **Multi-Head Latent Attention (MLA)** — instead of caching full K, V per head, project tokens into a small latent vector and reconstruct K, V from it on the fly. Drastically smaller KV cache.
- **Mixture of Experts (MoE)** — replace the dense MLP with `N` experts; a router picks top-k per token. Total params grow, active params per token stay roughly constant.
- **Shared expert** — one expert is always active alongside the top-k routed ones, to capture knowledge that every token needs.
- **Aux-loss-free load balancing (`--balance-speed`)** — routers collapse: a few experts win early, get all the gradient, and win harder. The classic fix is an auxiliary balancing loss, which works but competes with the language-modelling loss for the same parameters. DeepSeek-V3 drops the extra loss and keeps a per-expert bias added to the router scores *for selection only*, nudged after each step toward under-used experts. It decides who runs, never how much their output counts, so no gradient is distorted — and it's updated by rule, not backprop. `0` disables.
- **LatentMoE (`--moe-latent-dim D`)** — Nemotron 3 Super (2026): project the token down to `D` dims, route and run the experts entirely in there, project back up. Each expert costs a fraction of a full-width one, so the same budget buys many more. At nano scale, 8 experts drop from 221,696 to 57,472 params (26%).
- Keeps Qwen's RMSNorm + SwiGLU + RoPE + no-bias.

`python -m nano.models.deepseek_nano --self-check` checks routing shapes, that LatentMoE really shrinks the experts, that the balancing bias steers selection while staying out of the gating weights, and — by training — that it actually evens the load (spread 0.31 → 0.02). Router collapse is silent: a model using 2 of its 8 experts has a perfectly healthy-looking loss curve.

<!-- NOTES: DeepSeek — paste your learnings here.
Suggested structure:
- MLA: the latent compression trick, decoupled RoPE, why it shrinks the KV cache
- MoE routing: top-k, gating, load balancing loss
- Shared expert: motivation
- Active vs total params: how to read the param count
- Anything that surprised you
-->

### Qwen-Next Nano

Reference: Qwen3-Next / Qwen3.5 (Alibaba, 2025-26), Kimi Linear (Moonshot, 2025).

What's new vs Qwen:
- **Hybrid 3:1 attention** — three Gated DeltaNet layers, then one full-attention layer, repeated. Linear layers carry a fixed-size recurrent state (no KV cache, O(n) compute); the periodic attention layer restores exact token lookup. Qwen3-Next and Qwen3.5 use 3:1, Kimi Linear ~3:1, Ling 2.5 7:1. `--ratio` changes it.
- **Gated attention** — a sigmoid gate on the attention output before `out_proj`. Softmax forces every query to put its mass somewhere, so idle heads dump it on token 0 (the "attention sink") and produce massive activations that break quantization. The gate lets a head emit zero instead. It is two lines in `GroupedQueryAttention`.
- **KDA (`--linear kda`)** — Kimi Linear's refinement: Gated DeltaNet's decay gate is one scalar per head, so a head is either a fast local buffer or a slow long-range memory. KDA gives each key channel its own decay rate, so one head can be both.
- **ShortConv (`--short-conv 4`)** — a depthwise causal 1D conv on the linear layers' Q/K/V (Kimi Linear, LFM2.5, Inkling all use kernel 4). Linear attention compresses the whole past into one fixed state, so it has no cheap way to look at "the last three tokens"; this buys that local bias back for `dim × kernel` params. Carries a rolling `kernel-1` state so cached decoding stays exact.
- **KV sharing (`--kv-share N`)** — the last *N* attention layers drop their K/V projections entirely and reuse the most recent earlier layer's (Gemma 4 E2B/E4B, ~50% cache cut). They still compute their own queries, so they can attend differently. Composes with the 3:1 hybrid: two independent ways of shrinking the same cache.
- **NoPE (`--posenc nope`)** — skip RoPE on the attention layers. A causal mask alone already leaks position, and dropping the rotation often extrapolates better past the training context.
- **PLE (`--ple-dim D`)** — Gemma 4's per-layer embeddings. A normal model looks a token up once, at the bottom; PLE gives every layer its own slice of embedding for that token, projected up and added after the block. The table is pure memory — never multiplied against anything large, so it can live in slower storage. This is how Gemma 4 E2B carries 5.1B parameters while activating 2.3B: a different way of separating stored knowledge from active compute than MoE routing. Zero-init scale, so it starts as an exact no-op.
- **Multi-token prediction (MTP)** — a second loss that predicts token *t+2* from the main model's hidden state at *t* plus the embedding of *t+1*. Teacher forcing only ever asks "what's next", so nothing pressures the hidden state to plan further ahead; MTP does. One extra block, embedding and output head shared, `--mtp-weight 0.3` (DeepSeek-V3's λ) by default. Here it's training-only — using it as a speculative-decoding draft head would need to roll back the linear layers' recurrent state on a rejected token, which it can't do without snapshotting.
- Training loop, dataset, DDP and generation are imported from `nano/models/qwen_nano.py` — the file contains only the architecture.

- **mHC (`--residual mhc`)** — DeepSeek V4's manifold-constrained hyper-connections. A single residual stream makes every layer read and write the same vector; mHC runs *n* parallel streams with a learned read, a learned write, and a learned n×n mixing matrix between them, widening the residual pathway without widening any layer. The matrix is projected onto the doubly stochastic manifold by Sinkhorn-Knopp, so every stream emits exactly what it receives and the widened pathway still behaves like an identity globally — that constraint is what makes plain hyper-connections trainable at scale.

  Two things this implementation learned the hard way. The Sinkhorn loop must *end* on the row normalisation: finite iterations only approach the manifold, whichever axis is normalised last is the exact one, and row sums are what conserve signal across streams. And the streams must start distinguishable — identical streams under a doubly stochastic matrix stay identical forever, and receive identical gradients, so the symmetry is a saddle point training cannot escape. Hence the one-hot init plus a little jitter (`mhc_noise`).

`python -m nano.models.qwen_next_nano --self-check` asserts the layer pattern, the gate, and that incremental decoding (KV cache + recurrent state) matches a full forward pass. `--train-check` overfits tiny models to assert the things only gradients reveal: that the MTP head really predicts *t+2* and not *t+1*, and that mHC stays on-manifold with all *n* streams differentiated.

<!-- NOTES: Qwen-Next — paste your learnings here.
Suggested structure:
- Why 3:1 and not pure-linear: what recall actually costs
- Attention sinks / massive activations: what the gate is really fixing
- Per-head vs per-channel decay: what KDA buys
- KV cache size at long context vs qwen_nano
-->

---

## Attention zoo

[`nano/attention_zoo.py`](./nano/attention_zoo.py) holds the attention variants used by `nano/models/gpt_nano.py` (and reference implementations of ones used by the other models). All share the interface:

```python
attn = get_attention("gqa", cfg)
y = attn(x, use_cache=False)
attn.reset_cache()
```

| Key | Name | Used by |
|---|---|---|
| `mha` | Multi-Head Attention | GPT-2 |
| `gqa` | Grouped-Query Attention | Llama 3, Qwen 3 |
| `gated` | Gated Attention (GQA + output gate, kills attention sinks) | Qwen3-Next, Qwen3.5 |
| `mla` | Multi-Head Latent Attention | DeepSeek V2/V3 |
| `swa` | Sliding Window Attention | Mistral, Gemma |
| `deltanet` | Gated DeltaNet (linear attention, per-head decay) | Qwen3-Next |
| `kda` | Kimi Delta Attention (DeltaNet, per-channel decay) | Kimi Linear |
| `dsa` | DeepSeek Sparse Attention (MLA + lightning indexer, top-k) | DeepSeek-V3.2 |
| `csa` | Compressed Sparse Attention (compress 4:1, then top-k) | DeepSeek-V4 |
| `hca` | Heavily Compressed Attention (compress 128:1, attend densely) | DeepSeek-V4 |

`csa` and `hca` compress the *sequence* rather than the per-token cache: merge every `m` tokens into one KV entry via a learned per-dimension softmax over the group, and attend to those, plus a sliding-window branch of recent raw tokens. CSA compresses 4:1 and top-k selects; HCA compresses 128:1 — few enough entries at 1M context (~7,800) to attend densely, so every layer keeps a global view. DeepSeek-V4 alternates them by layer and gets the KV cache to ~2% of a standard transformer.

The delicate part is causality: a compressed entry summarising tokens `[s, s+m)` may only be read by queries at position `>= s+m-1`, once the group has closed. Getting that wrong leaks the future through the compressor while every loss curve still looks healthy.

`python -m nano.attention_zoo` self-tests every variant: output shape, that prefill-then-decode-one-token reproduces a full forward pass exactly, and that changing token *t* moves no output before *t*. That last one matters because nothing else catches a causality leak — a model reading the future trains happily and its loss curve looks unusually *good*, not broken. It's the check the compressed variants most need, since a compressed entry is only legal once its group has closed. It also trains DSA's indexer in isolation and asserts its recall against the true attention top-k improves (~41% → ~82%), because DSA's failure mode is silent — top-k is not differentiable, so a broken indexer objective leaves selection random while the loss curve looks healthy.

Two things DSA needs that aren't obvious from the paper. Its indexer objective must be written as a cross-entropy rather than `F.kl_div`, since the attention target is exactly zero at masked positions and `0·log 0` is NaN (dropping the target's entropy is a constant, so gradients are unchanged). And top-k needs a deterministic tie-break: the indexer's ReLU makes exact-zero scores the common case, and `torch.topk` breaks ties by memory order, which differs between a prefill and a one-token decode step — without it, cached generation silently selects different tokens than uncached.

<!-- NOTES: Attention — paste your learnings here.
Suggested structure:
- The shared interface and why it matters for plugging variants in
- MHA → GQA: what you actually save
- SWA: when sliding helps, when it hurts
- Linear attention (DeltaNet): trade-off vs softmax
-->

---

## Adding a new model

The repo follows a simple convention so new architectures slot in cleanly.

1. **Create `<name>_nano.py`** at the repo root. Copy `nano/models/qwen_nano.py` as a template — it has the cleanest separation of model / dataset / training loop / generation.
2. **Define `MODEL_SIZES`** — a dict of size presets (`nano`, `small`, `medium`, …). Keep `nano` cheap enough to overfit `the-verdict.txt` in minutes.
3. **Define `TRAIN_SETTINGS`** — per-size training defaults (lr, batch size, epochs, …).
4. **Reuse `nano/attention_zoo.py`** if your attention is one of the existing variants. Otherwise add a new class there with the same `(cfg) → forward(x, use_cache) + reset_cache()` interface.
5. **Match the CLI**: `--size`, `--epochs`, `--batch-size`, `--grad-accum`, `--resume`. This keeps multi-GPU launches uniform across models.
6. **Add a row to the Models table** above and a new `### <Name> Nano` section under [Architecture notes](#architecture-notes), following the same pattern (reference paper → diff vs the closest existing model → notes block).

That's it — no shared base class, no framework. Each file should still read top-to-bottom.

---

## Repo layout

```
nano-models/
├── nano/
│   ├── attention_zoo.py    # mha, gqa, gated, mla, swa, deltanet, kda, dsa, csa, hca
│   ├── config.py           # config files, precedence, run snapshots
│   └── models/
│       ├── gpt_nano.py         # GPT-2 style
│       ├── qwen_nano.py        # Qwen3 style
│       ├── deepseek_nano.py    # DeepSeek-V3 style (MLA + MoE)
│       └── qwen_next_nano.py   # Qwen3-Next / Kimi Linear / DeepSeek-V4 / Gemma 4
├── configs/                # worked example configs
├── docs/ARCHITECTURES.md   # every component: problem, solution, paper
├── greek_demo/             # separate corpus-building demo
└── the-verdict.txt         # tiny training corpus (downloaded on first run)
```

Both invocations work:

```bash
python -m nano.models.gpt_nano          # as a module
python nano/models/gpt_nano.py          # as a script
```


## Roadmap

[docs/ROADMAP.md](./docs/ROADMAP.md) has the open work in full — what blocks each item and what would count as done. In short:

1. **Run an actual experiment.** Everything here is verified mechanically; nothing is verified to *help*. Hub datasets and reproducible configs were the missing pieces, and they now exist.
2. **Multi-GPU and CUDA.** Never run. The auxiliary-loss plumbing exists specifically so MTP's gradients sync inside the DDP-wrapped forward, and that path has never executed.
3. **Attention masks.** Would unlock padded and chat-style SFT. The hard part is the linear layers, where masking means suppressing a recurrent state update rather than masking a score matrix.
4. **Generation wired to `past_key_values`,** for DPO and GRPO.
5. **`--attn mla` for the hybrid,** which is what makes a faithful Kimi Linear without a separate model file.

## License

Personal learning repo — code is provided as-is, no warranty. Architectures are reimplementations from publicly described designs; original credit belongs to the respective papers and labs.
