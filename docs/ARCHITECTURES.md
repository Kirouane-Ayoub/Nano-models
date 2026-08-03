# Architecture guide

Every component in this repo: the problem it was invented to solve, how it
solves it, where it lives here, and the paper it came from.

Ordered roughly by what part of the model it touches. Each entry is
independent — read the ones you need.

> Paper links are given as arXiv IDs only where verified. Older entries cite
> title and origin instead, which is enough to find them unambiguously.

---

## Quick reference

| Component | Flag | File |
|---|---|---|
| MHA, GQA, gated, MLA, SWA | `--attention <name>` | `nano/attention_zoo.py` |
| Gated DeltaNet, KDA | `--attention deltanet\|kda`, `--linear` | `nano/attention_zoo.py` |
| DSA, CSA, HCA | `--attention dsa\|csa\|hca` | `nano/attention_zoo.py` |
| Hybrid linear:full ratio | `--ratio N` | `nano/models/qwen_next_nano.py` |
| ShortConv | `--short-conv 4` | `nano/attention_zoo.py` |
| KV sharing | `--kv-share N` | `nano/models/qwen_next_nano.py` |
| NoPE | `--posenc nope` | `nano/models/qwen_nano.py` |
| mHC hyper-connections | `--residual mhc` | `nano/models/qwen_next_nano.py` |
| Per-layer embeddings | `--ple-dim 16` | `nano/models/qwen_next_nano.py` |
| Multi-token prediction | `--mtp-weight 0.3` | `nano/models/qwen_next_nano.py` |
| MoE + shared expert | `num_experts` in config | `nano/models/deepseek_nano.py` |
| Aux-loss-free balancing | `--balance-speed 1e-3` | `nano/models/deepseek_nano.py` |
| LatentMoE | `--moe-latent-dim D` | `nano/models/deepseek_nano.py` |

---

## 1. Attention

### MHA — Multi-Head Attention
**Paper:** *Attention Is All You Need* (Google, 2017); GPT-2 (OpenAI, 2019).

The baseline. Every query attends to every previous token, and each head keeps
its own K and V. Costs O(n²) compute and a KV cache that grows linearly in
tokens × layers × heads. Everything below is an attempt to pay less than this
without losing what it does.

### GQA — Grouped-Query Attention
**Paper:** *GQA: Training Generalized Multi-Query Transformer Models* (Google, 2023), arXiv 2305.13245.

**Problem.** At long context the KV cache, not the weights, dominates memory,
and decoding becomes bandwidth-bound — the GPU spends its time reading K and V
rather than multiplying.

**Solution.** Several query heads share one K/V head. With 16 Q heads and 4 KV
groups the cache shrinks 4×, and quality barely moves. Multi-query attention
(one KV head) was the extreme version and lost too much; GQA is the compromise
that everyone now ships.

### Gated Attention
**Paper:** Qwen3-Next / Qwen3.5 (Alibaba, 2025–26).

**Problem.** Softmax must sum to 1, so a head with nothing worth retrieving is
still forced to put its probability mass *somewhere*. It dumps it on token 0.
That's the "attention sink", and the massive activations that come with it are
what break low-bit quantization.

**Solution.** A sigmoid gate on the attention output before the output
projection. A head can now emit ≈0, so no sink is needed. Two lines of code.

### MLA — Multi-Head Latent Attention
**Paper:** DeepSeek-V2 (DeepSeek, 2024).

**Problem.** Same as GQA — cache size — but GQA pays for it by removing heads.

**Solution.** Compress K and V into a small shared latent vector, cache *that*,
and reconstruct per-head K/V on the fly. You cache one small vector per token
instead of one K and V per head. Keeps head diversity, which GQA sacrifices.

### SWA — Sliding Window Attention
**Paper:** Longformer (AI2, 2020); Mistral 7B (Mistral, 2023).

**Problem.** Most attention weight is local, but you pay for global attention
at every layer.

**Solution.** Attend only to the last *w* tokens. Cache becomes constant in
sequence length. The cost: anything beyond the window is reachable only
indirectly, by information hopping up through layers. Modern models interleave
sliding and full layers rather than going all-in.

### DSA — DeepSeek Sparse Attention
**Paper:** DeepSeek-V3.2 (DeepSeek, 2025).

**Problem.** SWA picks tokens by *position* — always the most recent — whether
or not they matter. The relevant token might be 100k back.

**Solution.** A "lightning indexer": a few low-dimensional heads and a ReLU
cheaply score every previous token, the top-k are kept, and full attention runs
over just those. O(L²) → O(kL), selected by content instead of position.

**The catch.** Top-k isn't differentiable, so no gradient reaches the indexer
from the language-modelling loss. It's trained separately against the dense
attention distribution. Skip that and you have a fixed random sparsity pattern
that trains without complaint.

### CSA / HCA — Compressed Attention
**Paper:** DeepSeek-V4 (DeepSeek, 2026), arXiv 2606.19348.

**Problem.** Every method above shrinks the cache *per token*. At a million
tokens, per-token savings still leave you with a million entries.

**Solution.** Compress the sequence itself. Merge every *m* tokens into one KV
entry using a learned per-dimension softmax over the group, so each channel
keeps whichever token dominates it rather than averaging into mush. Two
flavours on alternating layers:

- **CSA** — m=4, then top-k selection. Fine-grained, keeps detail, still sparse.
- **HCA** — m=128, no selection. At 1M tokens that's ~7,800 entries: few enough
  to attend to *densely*, so every layer keeps a genuinely global view.

Both add a sliding-window branch over recent raw tokens, because compression
blurs exactly the local detail that matters most. Together: KV cache down to
roughly 2% of a standard transformer at 1M context, 27% of the per-token FLOPs.

---

## 2. Linear attention and the hybrid layout

### Gated DeltaNet
**Paper:** *Gated Delta Networks* (NVIDIA/MIT, 2024); shipped in Qwen3-Next (2025).

**Problem.** Softmax attention's cost is inherent: it keeps every token to
compare against. Linear attention replaces that with a fixed-size recurrent
state — O(n) compute, no KV cache — but early versions couldn't forget, so the
state saturated and recall collapsed.

**Solution.** Combine a *gated decay* (Mamba-2's forgetting) with the *delta
rule* (write the difference between what's stored and what's new, rather than
appending). The state stays bounded and stays useful.

### KDA — Kimi Delta Attention
**Paper:** Kimi Linear (Moonshot AI, 2025).

**Problem.** Gated DeltaNet's decay is one scalar per head, so a head is either
a fast local buffer or a slow long-range memory. It can't be both.

**Solution.** Give every key channel its own decay rate. One head can then hold
a quickly-forgotten feature and a slowly-decaying one at the same time. In this
repo the entire difference from the parent class is the shape of one tensor.

### ShortConv
**Paper:** Kimi Linear (2025), LFM2.5; *Dynamic Short Convolutions Improve
Transformers* (2026), arXiv 2606.03825.

**Problem.** Linear attention compresses all history into one state, so it has
no cheap way to ask "what were the last three tokens" — a bias softmax
attention gets for free.

**Solution.** A depthwise causal 1D convolution (kernel 4) on Q, K and V. Costs
`dim × kernel` parameters. There is a live 2026 argument that short sliding-
window attention should replace it, which makes it a good experiment rather
than a settled answer.

### Hybrid attention (the 3:1 layout)
**Papers:** Qwen3-Next, Qwen3.5 (Alibaba); Kimi Linear (Moonshot); Ling 2.5;
Nemotron 3 (NVIDIA), arXiv 2604.12374.

**Problem.** Pure linear attention loses too much recall. Pure softmax
attention pays for exact lookup in every single layer.

**Solution.** Stop choosing. Three linear layers, then one full-attention
layer, repeated. Recall degrades gracefully when only 1-in-4 layers can do
exact lookup, while KV cache and prefill cost fall ~4×. Qwen3-Next, Qwen3.5 and
Kimi Linear all landed on ~3:1; Ling 2.5 uses 7:1. This convergence across
independent labs is the strongest single signal in 2026 architecture work.

---

## 3. Normalization and position

### RMSNorm
**Paper:** *Root Mean Square Layer Normalization* (2019).

LayerNorm without mean subtraction or bias. Same stabilization, fewer
operations. Universal now.

### QK-norm
**Paper:** *Scaling Vision Transformers to 22B* (Google, 2023); Qwen3.

**Problem.** Attention logits grow with scale until softmax saturates and
training diverges.

**Solution.** Normalize Q and K before the dot product. Cheap insurance that
became standard once models got big enough to need it.

### RoPE
**Paper:** *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021), arXiv 2104.09864.

**Problem.** Learned positional embeddings don't extend past the positions seen
in training, and encode absolute rather than relative position.

**Solution.** Rotate Q and K by an angle proportional to position. The dot
product then depends only on *relative* distance. Extending context becomes a
matter of adjusting the rotation base rather than retraining embeddings.

### NoPE
**Paper:** *Transformer Language Models without Positional Encodings Still
Learn Positional Information* (2022); *The Impact of Positional Encoding on
Length Generalization* (2023).

**Problem.** Even RoPE degrades past its training length.

**Solution.** Remove positional encoding entirely from some layers. A causal
mask already leaks position — token *t* can see exactly *t* predecessors, which
is enough for the model to infer where it is. Often extrapolates better.
Hybrid models typically use no positional encoding in their linear layers,
since the recurrence is inherently ordered.

---

## 4. Feed-forward and routing

### MoE — Mixture of Experts
**Papers:** *Switch Transformer* (Google, 2021); DeepSeek-V3 (2024).

**Problem.** Capacity requires parameters, but parameters cost compute on every
token.

**Solution.** Many expert MLPs, a router picking top-k per token. Total
parameters grow; active parameters per token stay flat. By 2026 this is the
default for every serious open-weight release.

### Shared expert
**Paper:** DeepSeek-V2 / V3.

**Problem.** Some knowledge every token needs. Making each routed expert learn
it separately wastes capacity.

**Solution.** One expert always active alongside the routed ones, so the routed
experts can specialize.

### Aux-loss-free load balancing
**Paper:** DeepSeek-V3 (2024).

**Problem.** Routers collapse. A few experts win early, get all the gradient,
and win harder. The standard fix — an auxiliary balancing loss — works but
competes with the language-modelling loss over the same parameters.

**Solution.** Drop the extra loss. Keep a per-expert bias added to the router
scores *for selection only*, nudged after each step toward under-used experts.
It decides who runs, never how much their output counts, so no gradient signal
is distorted. Updated by rule, not backprop.

### LatentMoE
**Paper:** Nemotron 3 Super (NVIDIA, 2026), arXiv 2604.12374.

**Problem.** Each expert is a full-width MLP, so expert count is expensive.

**Solution.** Project the token down to a compressed space, route and run every
expert in there, project back up. Each expert costs a fraction of a full-width
one, so the same budget buys many more. Better accuracy per parameter and per
FLOP than a regular MoE.

---

## 5. Structural

### mHC — Manifold-Constrained Hyper-Connections
**Papers:** *Hyper-Connections* (ByteDance, ICLR 2025), arXiv 2409.19606;
*mHC* (DeepSeek, Dec 2025), arXiv 2512.24880; *mHC-lite* (2026), arXiv 2601.05732.

**Problem.** One residual stream means every layer reads from and writes to the
same vector. Depth and width compete for a single channel.

**Solution.** *n* parallel residual streams, with a learned read, a learned
write, and a learned n×n mixing matrix between them. Widens the residual
*pathway* without widening any layer.

The manifold constraint is what makes it trainable: the mixing matrix is
projected onto the doubly stochastic manifold (the Birkhoff polytope) by
Sinkhorn-Knopp, so every stream emits exactly as much as it receives and the
widened pathway still behaves like an identity globally. ~6.7% training
overhead for four streams; earlier hyper-connection work reached baseline
quality in roughly half the tokens.

### PLE — Per-Layer Embeddings
**Paper:** Gemma 4 (Google, 2026), arXiv 2607.02770.

**Problem.** A model looks a token up once, at the bottom, and every layer has
to reconstruct what it needs from that one vector.

**Solution.** A second embedding table giving every layer its own slice for the
current token. The PLE dimension is small and the table is never multiplied
against anything large, so it's close to pure storage — it can live in slower
memory. This is how Gemma 4 E2B carries 5.1B parameters while activating 2.3B:
separating stored knowledge from active compute by a different route than MoE.

### KV sharing (cross-layer attention)
**Paper:** Gemma 4 (Google, 2026).

**Problem.** Every layer keeps its own KV cache. Depth multiplies memory.

**Solution.** Later layers skip their K/V projections entirely and reuse an
earlier layer's. They still compute their own queries, so they can attend
differently. ~50% cache reduction — 2.7 GB at 128k context for Gemma 4 E2B.

### MTP — Multi-Token Prediction
**Paper:** *Better & Faster Large Language Models via Multi-token Prediction*
(Meta, 2024); DeepSeek-V3; Qwen3.5; Nemotron 3.

**Problem.** Teacher forcing only ever asks "what comes next", so nothing
pressures the hidden state to plan further than one token ahead.

**Solution.** A second objective predicting token *t+2* from the hidden state
at *t* plus the embedding of *t+1*. Improves the main model, and the extra head
doubles as a draft model for speculative decoding — which is why every 2026
release ships one.

---

## 6. Implementation gotchas

Things that cost real time while building this, none of which are in any paper.
Every one of them trains happily while being wrong.

**A loss curve cannot tell you a component works.** Five of the bugs below
produced healthy-looking losses. Each needed a test that asked the component
directly whether it was doing its job.

| Gotcha | What happened |
|---|---|
| **MTP shift** | Predicting *t+1* instead of *t+2* trains identically. Only checking that the MTP head's top-1 differs from the main head's catches it. |
| **Sinkhorn axis order** | Finite iterations only *approach* the doubly stochastic manifold, so whichever axis is normalised last is the exact one. Row sums conserve signal across streams — ending on columns left rows off by 2e-4 *after training*, invisible at init. |
| **Hyper-connection symmetry** | Identical streams under a doubly stochastic matrix stay identical forever, and receive identical gradients — a saddle point training cannot escape. n=4 silently behaves like n=2. Streams must start distinguishable. |
| **DSA top-k ties** | The indexer's ReLU makes exact-zero scores the common case, and `torch.topk` breaks ties by memory order — which differs between a prefill and a one-token decode. Cached generation silently selected different tokens than uncached. |
| **KL with a zero target** | `F.kl_div` returns NaN when the target has exact zeros, which attention distributions always do at masked positions. Writing it as cross-entropy drops a constant and sidesteps it. |
| **Aux losses in reported metrics** | Folding DSA's indexer objective into the logged loss made it read 26.26 against ~10.7 for every other variant. Its actual LM loss was 10.75. Auxiliary losses belong in the gradient, not the metric. |
| **Autocast dtype in MoE routing** | Expert output is bf16 under autocast while softmax keeps probabilities in fp32, and `index_add_` rejects mismatched types. Unreachable until a latent projection made the MoE input bf16. |
| **Compressed-entry causality** | An entry summarising tokens `[s, s+m)` may only be read once the group *closes*, at `s+m-1`. Reading earlier leaks the future — and the loss curve looks unusually **good**, not broken. |
| **Init RNG shifts** | Comparing "same model with and without component X" is invalid if X adds modules: it changes how much RNG the weight init consumes, so every weight differs. Detach the component from one model instead. |
