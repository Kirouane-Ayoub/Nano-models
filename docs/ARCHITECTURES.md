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
| MHA, GQA, gated, sink, kv1, diff, softcap, ssmax, MLA, SWA | `--attention <name>` | `nano/attention_zoo.py` |
| Gated DeltaNet, KDA | `--attention deltanet\|kda`, `--linear` | `nano/attention_zoo.py` |
| Lightning Attention | `--attention lightning` | `nano/attention_zoo.py` |
| Mamba-2 (+ Mamba-3 flags) | `--attention mamba2`, `--linear mamba2`, `mamba_trapezoidal`, `mamba_complex` | `nano/attention_zoo.py` |
| DSA, CSA, HCA, MoBA | `--attention dsa\|csa\|hca\|moba` | `nano/attention_zoo.py` |
| Hybrid linear:full ratio | `--ratio N` | `nano/models/qwen_next_nano.py` |
| Local:global (SWA) layout | `--linear swa --ratio 5 --window 128` | `nano/models/qwen_next_nano.py` |
| ShortConv | `--short-conv 4` | `nano/attention_zoo.py` |
| KV sharing | `--kv-share N` | `nano/models/qwen_next_nano.py` |
| Sandwich norm | `--norm sandwich` | `nano/models/qwen_next_nano.py` |
| Final logit softcap | `--logit-softcap 30` | `nano/models/qwen_next_nano.py` |
| NoPE | `--posenc nope` | `nano/models/qwen_nano.py` |
| p-RoPE (partial RoPE) | `--posenc prope --rope-fraction 0.5` | `nano/models/qwen_next_nano.py` |
| mHC hyper-connections | `--residual mhc` | `nano/components.py` |
| Per-layer embeddings | `--ple-dim 16` | `nano/components.py` |
| Engram conditional memory | `--engram-dim 16 --engram-layer 1` | `nano/components.py` |
| Mixture-of-Depths | `--mod-capacity 0.5` | `nano/components.py` |
| Multi-token prediction | `--mtp-weight 0.3` | `nano/models/qwen_next_nano.py` |
| Parallel block (GPT-J/PaLM) | `--block parallel` | `nano/models/gpt_nano.py` |
| Looped depth (Huginn/Ouro) | `--loops 4 --loop-bptt K` | `nano/models/looped_nano.py` |
| Gemma 3/4 recipe (whole model) | `python -m nano.models.gemma_nano` | `nano/models/gemma_nano.py` |
| Mamba-2/3 pure SSM (whole model) | `python -m nano.models.mamba_nano` | `nano/models/mamba_nano.py` |
| MoE + shared expert | `num_experts` in config | `nano/models/deepseek_nano.py` |
| Aux-loss-free balancing | `--balance-speed 1e-3` | `nano/models/deepseek_nano.py` |
| Sigmoid routing | `--router sigmoid` | `nano/models/deepseek_nano.py` |
| Expert-choice routing | `--routing expert` | `nano/models/deepseek_nano.py` |
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

### Attention sinks
**Paper:** gpt-oss (OpenAI, 2025); *Efficient Streaming Language Models with
Attention Sinks* (MIT, 2023) for the diagnosis.

**Problem.** The same one Gated Attention fixes: softmax has to put its mass
somewhere, and a head with nothing to say puts it on token 0.

**Solution.** Give it somewhere harmless to put it. One learned scalar per head
is appended to the logit row, softmax runs over T+1 entries, and the sink's
column is dropped before multiplying by V. A head that wants to emit nothing
raises its sink logit. Fixes the problem on the input side where gating fixes
it on the output side; gpt-oss ships sinks with alternating 128-token sliding
window and full layers, so both `gqa` and `swa` accept the `attn_sink` flag.
One parameter per head.

### K-as-V
**Paper:** Gemma 4 (Google DeepMind, 2026), arXiv 2607.02770.

**Problem.** GQA shrinks the cache by sharing heads. What is left is still two
tensors per token per layer.

**Solution.** Drop the value projection and use K as V. The cache halves again,
and the head loses only the ability to *store* something different from what
it *matches on*. Gemma 4 does this only on the sparse global layers — the ones
whose cache grows with context — and keeps separate values on the 5:1 sliding
window layers where the cache is bounded anyway. Combined with KV sharing it
takes their global cache down ~37%.

### Logit softcapping
**Paper:** Gemma 2 (Google DeepMind, 2024), arXiv 2408.00118.

**Problem.** Nothing bounds a q·k dot product. A head can drive its logits
large enough that softmax is exactly one-hot: brittle, gradient-free, and the
first thing to overflow in fp16.

**Solution.** Pass the logits through `c · tanh(s / c)`. Near zero it is the
identity, so ordinary logits are untouched; large ones saturate smoothly at ±c
instead of running away. Gemma 2 uses c=50 on attention logits and c=30 on the
final vocabulary logits. Gemma 3 removed the attention cap and used QK-norm
instead, which bounds the dot product at the source. Zero parameters, one line,
and a good example of two fixes for the same failure at different points.

The same cap on the *final* logits (`--logit-softcap 30` in the hybrid model)
survived into Gemma 3 and 4. Nothing else bounds the LM head, and a confident
model otherwise drifts towards one-hot outputs whose gradient has vanished.
In this repo the MTP head shares the cap, since it shares the head.

### Scalable Softmax
**Paper:** *Scalable-Softmax Is Superior for Attention* (Nakanishi, 2025), arXiv 2501.19399.

**Problem.** With bounded logits, softmax over n keys flattens as n grows: the
largest probability decays towards 1/n. Past the training length a head
physically cannot focus on one token any more. The paper calls it attention
fading.

**Solution.** Multiply the logits by `s · log(n)`, where n is the number of
keys the row can see. The log(n) growth cancels the flattening exactly, so a
head's sharpness no longer depends on context length. s is one learned scalar
per head, trained values land near 0.43. Costs nothing and needs no cache
changes; in this repo the scale is computed from the query position, so
cached decode matches prefill. The paper shows retrieval holding up at ten
times the training length where plain softmax has collapsed.

### Differential Attention
**Paper:** *Differential Transformer* (Microsoft, 2024), arXiv 2410.05258.

**Problem.** Softmax gives every token a non-zero share, so irrelevant context
always gets *some* attention. Over a long context that floor of noise drowns
the few tokens that matter, which is why retrieval degrades with length.

**Solution.** Compute two attention maps from separate Q/K projections and
subtract them, scaled by a learned λ. Both maps carry the same noise floor and
different signal; the difference cancels the common mode, like a differential
amplifier. λ starts near 0.8 and is learned per layer. Each head is
RMS-normed on its own, because the subtraction can leave a head with tiny
magnitude. At λ=0 the second map is inert and you have plain attention, which
is what the self-test checks. Cost: twice the Q and K parameters unless you
halve the head count, which is what the paper does.

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

### MoBA — Mixture of Block Attention
**Paper:** Moonshot AI (Feb 2025), arXiv 2502.13189. Used in Kimi's long-context models.

**Problem.** DSA selects tokens by content, but needs a separately trained
indexer to do it, and that indexer can drift from the attention it feeds.

**Solution.** Route at block granularity with a summary that needs no
training: cut the keys into blocks, score each closed block by the query's
dot product with the block's *mean key*, keep the top-k, and always include
the query's own block. Attention then runs over just those keys. It is MoE
applied to the key axis — blocks are the experts, the mean key is the router —
which is where the name comes from. Coarser than DSA, but nothing to train and
nothing to go stale.

**The trap.** Forget the own-block rule and a query whose top-k picks are all
in the future has an entirely masked row: softmax of all −∞ is NaN.
`torch.testing.assert_close` rejects NaNs by default, even when both outputs
contain them. The self-test also checks the routing rule directly: k=0 still
leaves exactly the current block visible. Smaller blocks exercise sparse
routing under cached decoding and future-token perturbations.

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

### Lightning Attention
**Paper:** TransNormerLLM (OpenNLPLab, 2023), arXiv 2307.14995; Lightning
Attention-2 (2024), arXiv 2401.04658; shipped in MiniMax-01 (2025) and Ling 2.5
(2026).

**Problem.** Linear attention needs a way to forget, and needs to be trainable
in parallel. Learned gates (DeltaNet) solve forgetting but make the recurrence
data-dependent, which is harder to write a fast kernel for.

**Solution.** Forget at a *fixed* rate. Each head gets a constant decay from
the ALiBi power-law slopes, so heads span short buffers to near-permanent
memory without learning anything. With the decay fixed, the attention matrix
factors into a (T×T) decay mask times QKᵀ, and the paper's contribution is a
blockwise kernel that computes it in O(n) without a cumsum. No softmax; an
RMSNorm on the output does the normalising and a sigmoid gate lets a head
switch off. Ling 2.5 pairs it with MLA on the full-attention layers where
Qwen3.5 pairs DeltaNet with gated attention. MiniMax dropped it again in M2 for
plain GQA — the trade-off is not settled.

### Mamba-2 and Mamba-3
**Papers:** *Transformers are SSMs* / Mamba-2 (Dao & Gu, 2024), arXiv 2405.21060;
Mamba-3 (Mar 2026), arXiv 2603.15569. Nemotron 3 (NVIDIA) ships Mamba-2.

**Problem.** Linear attention with a fixed decay (Lightning) cannot decide
*when* to forget; DeltaNet can, but its delta-rule write is the expensive
part.

**Solution.** A selective state space: each head holds a (head_dim × state)
matrix, and every token emits its own step size Δ. Decay is exp(Δ·A) with
A < 0 learned per head, so a large Δ wipes the state and a small one lets it
pass through untouched. The write is a plain outer product Δ·x⊗B, the read
is S·C. Written out for a whole sequence this is a masked (T×T) matrix
product — the "structured state-space duality" — which is why it trains like
attention and decodes like an RNN. Nemotron 3 puts these where Qwen3-Next
puts DeltaNet, next to attention and MoE.

**Mamba-3** changes the discretisation. The input term becomes a trapezoid
over t and t−1 with a learned mixing weight, higher-order in Δ, and since that
already mixes adjacent tokens the short conv goes. A complex-valued state,
implemented as rotating B and C by an angle that accumulates with Δ, gives
the SSM the state-tracking ability (parity, modular counting) that a real
diagonal recurrence lacks. Both are flags here; both add carried state, which
the self-test checks under one-token decode.

`mamba_nano.py` stacks the mixer alone — norm and SSM, no FFN, no attention,
no positional encoding — and checks the property no attention model has: the
recurrent state after 40 tokens is the same size as after one.

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

**The other hybrid.** Gemma 4 and gpt-oss fill the cheap slot with sliding
window attention instead of a recurrence: 5 local : 1 global in Gemma 4 with a
128-token window, 1 : 1 in gpt-oss. The cache is bounded rather than constant,
but each local layer is still exact softmax attention over its window, so
recall inside the window never degrades. In this repo `--linear swa` puts the
zoo's SWA in the same layer positions the linear mixers use; the last layer
stays global either way.

---

## 3. Normalization and position

### RMSNorm
**Paper:** *Root Mean Square Layer Normalization* (2019).

LayerNorm without mean subtraction or bias. Same stabilization, fewer
operations. Universal now.

### Sandwich norm
**Paper:** Gemma 2 (2024), arXiv 2408.00118; kept in Gemma 3 and 4. OLMo 2
(2024) uses the post-norm half alone.

**Problem.** Pre-norm scales what goes *into* a sublayer and says nothing
about what comes out. One attention layer can emit a spike that then rides
the residual stream to the top, and the deeper the model the more chances
that has to happen.

**Solution.** Normalise the output too: `x + RMSNorm(attn(RMSNorm(x)))`, same
for the FFN. Each contribution is RMS-normalised before applying a learned
per-channel scale and adding it to the residual; this is not a hard cap on
individual values or the final output scale. Two extra vectors per layer, no change to the cache or
the mixer, and it is what let Gemma go deeper at a fixed width. The
trade-off is one more reduction per sublayer on the critical path.

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

### p-RoPE — partial RoPE
**Paper:** *Round and Round We Go! What makes Rotary Positional Encodings
useful?* (Barbero et al., 2024), arXiv 2410.06205; shipped in Gemma 4 (2026).

**Problem.** RoPE's lowest-frequency pairs barely rotate over a whole context,
so the model uses them as position-free content channels anyway — except that
they *do* rotate a little, and at lengths past training that drift is noise.

**Solution.** Keep only the highest-frequency fraction of pairs rotary and set
the rest to the identity (cos=1, sin=0). Those pairs become honest NoPE
channels, the fast ones still carry position. In this repo it is a
modification of the cos/sin tables, so `apply_rope` and the KV cache are
unchanged. Fraction 1 is plain RoPE and fraction 0 is NoPE exactly, which is
what the self-check anchors on; Gemma 4's report pairs it with the 5:1
local:global layout for long-context stability.

---

## 4. Feed-forward and routing

### MoE — Mixture of Experts
**Papers:** *Switch Transformer* (Google, 2021); DeepSeek-V3 (2024).

**Problem.** Capacity requires parameters, but parameters cost compute on every
token.

**Solution.** Many expert MLPs, a router picking top-k per token. Total
parameters grow; active parameters per token stay flat. By 2026 this is the
default for every serious open-weight release.

### Sigmoid routing
**Paper:** DeepSeek-V3 (2024), arXiv 2412.19437.

**Problem.** A softmax router couples the experts: pushing one score up pulls
every other weight down. The router cannot express "these two are both a good
fit", and the coupling also feeds the collapse dynamics that load balancing
has to fight.

**Solution.** Score each expert with its own sigmoid, pick the top-k, then
normalise the chosen weights to sum to 1. Each affinity is judged
independently; the sum constraint is applied after selection rather than
built into the scoring. DeepSeek-V3 made this switch alongside aux-loss-free
balancing, and the two are usually adopted together. In this repo it is one
method on the router with the same output shape and sum as softmax.

### Expert-choice routing
**Paper:** *Mixture-of-Experts with Expert Choice Routing* (Google, 2022), arXiv 2202.09368.

**Problem.** When tokens pick experts, nothing stops every token from picking
the same two. Load balancing then has to be enforced from outside — an
auxiliary loss, or DeepSeek-V3's selection bias.

**Solution.** Turn the choice around: each expert picks its top `capacity`
tokens, with `capacity = N·k / E` so the total work matches token choice.
Every expert processes exactly the same number of tokens by construction, and
a token can be taken by several experts or by none (the residual carries it).
No balancing machinery at all.

**The catch, and why decoders did not adopt it.** The top-k runs over the
token axis, across the whole sequence. Whether expert e takes token t depends
on how strongly later tokens compete for e, so the routing reads the future.
Harmless in an encoder, disqualifying in an autoregressive LM: the self-check
asserts the leak rather than hiding it. This is why DeepSeek kept token choice
and fixed balance with the bias instead. Read it as the contrast, not a
recommendation.

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

### Engram — conditional memory
**Paper:** *Conditional Memory via Scalable Lookup* (DeepSeek, Jan 2026), arXiv 2601.07372.

**Problem.** A transformer recomputes static facts at every position.
"New York" is followed by the same handful of tokens every time, yet attention
and the FFN re-derive that from scratch. MoE adds parameters but still spends
compute to reach them.

**Solution.** A lookup table keyed by the last n tokens. Hash the suffix
n-gram (n = 2, 3) into a row of a prime-sized embedding table, several heads
per n-gram so one collision does not dominate, concatenate, project to the
hidden size, gate, add to the residual after an early layer. Constant cost
per token and, like PLE, pure memory that can live off the accelerator.
DeepSeek's "sparsity allocation law": spend 20–25% of a sparse budget here,
the rest on MoE. Not in DeepSeek-V4, which chose CSA/HCA and mHC instead, so
read it as a live research direction rather than a shipped default.

Causal by construction because the n-gram ends at t. What breaks is cached
decode: the module must remember the previous n−1 token ids or the n-gram at
the new token is wrong while everything else looks fine. The projection is
zero-initialised so a model with Engram starts exactly as one without.

### Mixture-of-Depths
**Paper:** *Mixture-of-Depths* (Google DeepMind, 2024), arXiv 2404.02258.

**Problem.** Every token pays for every layer, but most tokens do not need
most layers. MoE varies *which* parameters a token uses; nothing varies *how
many*.

**Solution.** Put a scalar router in front of a block. The top `capacity`
fraction of tokens go through it, their update scaled by the router score so
the router learns; the rest skip the block on the residual. Half the tokens
at every other layer is the paper's setting, and quality holds at a fraction
of the FLOPs.

**The catch, and the fix that ships with it.** Top-k ranks tokens across the
sequence, so token t's fate depends on tokens after it — non-causal, like
expert-choice routing. Unlike expert choice, the paper solves it: a second
scalar head is trained with a BCE loss to predict the top-k decision from the
token alone, and at inference that predictor makes the call. Train routes by
top-k and trains the predictor; eval routes by the predictor and is causal.
The self-check asserts exact capacity in train mode and causality in eval
mode; the train-check asserts the predictor learns to agree with the top-k.
In this repo skipped tokens are masked rather than removed, so they still
serve as keys and the FLOP saving is not realised — the routing is what is
demonstrated.

### KV sharing (cross-layer attention)
**Paper:** Gemma 4 (Google, 2026).

**Problem.** Every layer keeps its own KV cache. Depth multiplies memory.

**Solution.** Later layers skip their K/V projections entirely and reuse an
earlier layer's. They still compute their own queries, so they can attend
differently. ~50% cache reduction — 2.7 GB at 128k context for Gemma 4 E2B.

### Looped depth — depth-recurrent transformers
**Papers:** *Universal Transformers* (Dehghani et al., 2018); Huginn, arXiv
2502.05171 (Geiping et al., 2025); Ouro, arXiv 2510.25741 (2025).

**Problem.** Depth is fixed at training time and paid for in parameters. Every
token gets the same amount of compute whether it needs it or not.

**Solution.** Apply the same block *r* times. Huginn's shape: a prelude reads
the tokens into *e*, a core block maps `[s ; e]` → *s* repeatedly from *s₀*,
a coda reads out. Compute scales with *r*, parameters do not, and *r* is a
test-time choice — Huginn trained with a mean of 32 loops and keeps improving
past it. Ouro loops a whole 24-layer stack 4 times and matches dense models
3x its size; its exit gate `λ_t = σ(Linear(h_t))` with an entropy-regularised
loss `Σ p(t)·L_t − βH(p)` learns how many loops each token needs.

Three things make it work. *Input injection* — feeding *e* back in at every
iteration — keeps the state anchored to the input so deeper unrolls do not
drift. *Sampled depth* (log-normal Poisson) with *truncated backprop* through
the last *k* iterations is what makes 32 loops trainable and lets depth
transfer. And *one KV cache per iteration*: see § 6.

Implemented in `nano/models/looped_nano.py` on qwen_nano's blocks. The adapter
is initialised to `[I | I]`, so an iteration begins as `core(s + e)`. The exit
gate is not implemented; `loops` is a fixed dial.

### Parallel block
**Papers:** GPT-J (EleutherAI, 2021); PaLM (Google, 2022), arXiv 2204.02311; Falcon (2023).

**Problem.** A standard block is two serial steps: attention, then an FFN on
attention's output. At large scale the serialisation, not the FLOPs, is what
limits throughput.

**Solution.** Feed both sublayers the same residual input and sum the outputs.
This repo keeps separate learned norms for checkpoint compatibility:
`x + attn(norm1(x)) + ffn(norm2(x))`. The FFN no longer depends on attention's
output. A shared-norm variant can fuse the Q/K/V and FFN input projections
into one wide matmul; this implementation neither fuses the projections nor
executes the branches concurrently, so it does not guarantee a speedup. PaLM
reports ~15% faster training at 540B with no measurable quality loss; at
small scale there is a loss, because the FFN can no longer condition on what
attention just retrieved in the same layer. Same parameter count either way,
which is what the self-check pins.

### The Gemma recipe
**Papers:** Gemma 2 (2024), arXiv 2408.00118; Gemma 3 (2025), arXiv 2503.19786;
Gemma 4 (2026), arXiv 2607.02770.

Gemma is the counter-example to everything above: a dense transformer with no
new mixer, made competitive by a stack of small decisions. `gemma_nano.py`
puts them in one file. Most are documented on their own elsewhere in this
guide (sandwich norm, QK-norm, p-RoPE, K-as-V, logit softcap, SWA); four are
specific to Gemma:

- **5:1 local:global with dual RoPE base.** Local layers see ≤ w tokens back
  and use base 10k; global layers must resolve the whole context and use 1M.
  Gemma 4 applies p-RoPE (25% of pairs) on the global layers only and makes
  the final layer always global.
- **GeGLU.** The gated FFN with GELU in place of SiLU. Same shape as SwiGLU;
  Gemma has simply never changed it.
- **(1 + w) RMSNorm.** Scale written as `1 + w`, w zero-init. Weight decay
  then pulls the norm towards identity rather than towards zero output.
- **Tied embeddings scaled by √d.** One matrix for input and output, and the
  input side multiplied by √d so tokens enter at residual-stream scale. With
  a 262k vocabulary the tying is a large share of the parameter budget.

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
| **Accelerator built for its device** | Every main called `accelerator().device` before the training settings were known. The singleton was created with `mixed_precision="no"`, and the bf16 and grad-accumulation settings `train()` passed afterwards were dropped — every GPU/MPS run logged `Precision: no` and `--grad-accum` changed the log line but not the optimizer. Now `current_device()` reads the device off `PartialState`, and asking the singleton for different settings raises. |
| **Looped KV cache** | A weight-shared core applied *r* times needs *r* KV caches. With one, iteration 3 of the current token attends to iteration 1's keys of earlier tokens; prefill and one-token decode then disagree by ~1e-1 while training is unaffected. `looped_nano` keeps a cache slot per iteration, and its self-check asserts the incremental test *fails* when the slots are shared. |
| **SWA cache trimmed before attending** | The sliding-window cache was cut to the window *before* the attention step. A cached prefill longer than the window then gave its early queries nothing but future keys: all-masked rows, softmax of −∞, NaN. Every decode step after the prefill still matched the full forward exactly, so the zoo test passed. It surfaced only when SWA went into the hybrid model, where the NaN prefill output is the next layer's input. The zoo test now compares the prefill output too. |
| **Trimmed cache that still holds the prefill** | Sliding-window caches were trimmed with a bare slice, `cache[:, :, -w:]`. The shape said w tokens; the storage still held the entire prefill, because a slice keeps its base tensor alive. Every shape and equivalence test passed. Only asserting on `untyped_storage().nbytes()` catches it, and `.clone()` on the slice fixes it. Same fix in the zoo's SWA, in `ShortConv`'s rolling window (which also grew without bound at kernel 1, because `u[..., -0:]` is the whole tensor), and in `gemma_nano`. |
| **Expert choice is non-causal** | Expert-choice routing picks each expert's top tokens over the whole sequence, so which experts process token t depends on tokens after t. A decoder trained with it reads the future through its routing and the loss looks *better*, not broken. The MoE self-check asserts the leak exists, so nobody mistakes the flag for a free balancing fix. |
| **fp32 buffers in a half-precision matmul** | `module.to(torch.float16)` casts parameters and buffers, but not tensors *computed* from them. Lightning's decay mask came from an fp32 `arange`, stayed fp32, and the `(QKᵀ ⊙ decay) @ V` matmul raised a dtype mismatch. Invisible under autocast, which the training path always uses; only explicit `.to(dtype)` inference hit it. Every zoo entry now passes an fp16/bf16 forward without autocast. |
| **Init RNG shifts** | Comparing "same model with and without component X" is invalid if X adds modules: it changes how much RNG the weight init consumes, so every weight differs. Detach the component from one model instead. |
