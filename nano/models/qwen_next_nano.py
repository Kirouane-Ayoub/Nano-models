"""
Qwen-Next Nano — the 2026 hybrid-attention layout, built from scratch.

This is the layout Qwen3-Next, Qwen3.5-397B, Kimi Linear and Ling 2.5 all
converged on: don't pick between linear attention and softmax attention, stack
them in a fixed ratio. Three linear-attention layers, then one full-attention
layer, repeated.

  - Linear layers  (Gated DeltaNet / KDA): O(n) compute, fixed-size recurrent
    state, no KV cache at all. Cheap, but lossy — they compress the past.
  - Attention layers (gated GQA + RoPE):   exact token lookup, O(n^2), KV cache.

Why the mix works: recall degrades gracefully when only 1-in-4 layers can do
exact lookup, but KV cache and prefill cost fall by ~4x. Pure-linear models lose
too much recall; pure-attention models pay for lookup in every single layer.

Also here:

  - **Gated attention** (Qwen3-Next). A sigmoid gate on the attention output
    before out_proj — two lines in `qwen_nano.GroupedQueryAttention`. It removes
    attention sinks and the massive activations that break quantization, because
    a head with nothing to retrieve can now emit zero instead of dumping its
    probability mass on token 0.
  - **Multi-token prediction** (DeepSeek-V3, Qwen3.5, Nemotron 3). A second
    training objective predicting token t+2, so the hidden state has to plan
    further than the next token. `--mtp-weight 0` turns it off.
  - **mHC** (DeepSeek V4). Replaces the single residual stream with n parallel
    ones plus a doubly-stochastic mixing matrix, widening the residual pathway
    without widening any layer. `--residual mhc`.
  - **Per-layer embeddings** (Gemma 4). A second embedding table feeding each
    layer its own slice for the current token — stored knowledge that costs no
    active compute. `--ple-dim 16`.

Everything except the architecture (training loop, DDP, dataset, generation) is
imported from qwen_nano.py rather than copied.

Sizes: nano (5M) → small (40M) → medium (130M) → large (350M)

Usage:
    python -m nano.models.qwen_next_nano                                  # 3:1 DeltaNet:attention
    python -m nano.models.qwen_next_nano --linear kda                     # Kimi Linear's gate
    python -m nano.models.qwen_next_nano --ratio 1                        # 1:1, more attention
    python -m nano.models.qwen_next_nano --size small --epochs 10
    python -m nano.models.qwen_next_nano --mtp-weight 0                   # disable multi-token prediction
    python -m nano.models.qwen_next_nano --short-conv 4                   # ShortConv on Q/K/V (Kimi Linear)
    python -m nano.models.qwen_next_nano --ratio 1 --kv-share 1           # last attn layer reuses K/V (Gemma 4)
    python -m nano.models.qwen_next_nano --posenc nope                    # drop RoPE
    python -m nano.models.qwen_next_nano --residual mhc                   # 4 hyper-connected residual streams
    python -m nano.models.qwen_next_nano --ple-dim 16                     # per-layer embeddings (Gemma 4)
    python -m nano.models.qwen_next_nano --self-check                     # no training, just asserts
    python -m nano.models.qwen_next_nano --train-check                    # overfit to assert MTP/mHC learn

    torchrun --nproc_per_node=8 -m nano.models.qwen_next_nano --size large --batch-size 32
"""

# Runnable either way: `python -m nano.models.qwen_next_nano` or `python nano/models/qwen_next_nano.py`.
if __package__ in (None, ""):
    import pathlib as _pathlib
    import sys as _sys
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from nano import config
from nano.attention_zoo import GatedDeltaNet, KimiDeltaAttention
from nano.models.qwen_nano import (
    RMSNorm, SwiGLUFeedForward, GroupedQueryAttention, compute_rope_params,
    apply_rope, apply_rope_offset,
    TRAIN_SETTINGS, create_dataloaders, load_text, train, generate, generate_cached,
    is_main_process, log,
)


# ──────────────────────────────────────────────
# Model size presets
# ──────────────────────────────────────────────
# n_layers must be divisible by (ratio + 1); head_dim must equal emb_dim //
# n_heads because the attention_zoo linear layers derive it that way.

MODEL_SIZES = {
    "nano": {       # ~5M
        "vocab_size": 50257,  "context_length": 128,
        "emb_dim": 64,        "n_heads": 4,      "n_layers": 4,
        "hidden_dim": 192,    "head_dim": 16,
        "n_kv_groups": 2,     "qk_norm": True,
        "rope_base": 10_000.0, "drop_rate": 0.1,
    },
    "small": {      # ~40M
        "vocab_size": 50257,  "context_length": 256,
        "emb_dim": 256,       "n_heads": 8,      "n_layers": 8,
        "hidden_dim": 768,    "head_dim": 32,
        "n_kv_groups": 4,     "qk_norm": True,
        "rope_base": 10_000.0, "drop_rate": 0.1,
    },
    "medium": {     # ~130M
        "vocab_size": 50257,  "context_length": 512,
        "emb_dim": 512,       "n_heads": 8,      "n_layers": 12,
        "hidden_dim": 1536,   "head_dim": 64,
        "n_kv_groups": 4,     "qk_norm": True,
        "rope_base": 100_000.0, "drop_rate": 0.1,
    },
    "large": {      # ~350M
        "vocab_size": 50257,  "context_length": 1024,
        "emb_dim": 1024,      "n_heads": 16,     "n_layers": 24,
        "hidden_dim": 3072,   "head_dim": 64,
        "n_kv_groups": 8,     "qk_norm": True,
        "rope_base": 500_000.0, "drop_rate": 0.1,
    },
}

LINEAR_MIXERS = {"deltanet": GatedDeltaNet, "kda": KimiDeltaAttention}


def build_layer_pattern(n_layers, ratio):
    """`ratio` linear layers followed by 1 attention layer, repeated.

    Qwen3-Next / Qwen3.5 / Kimi Linear use ratio=3 (48 layers = 12 x [L,L,L,A]).
    Ling 2.5 uses 7. The attention layer goes last in each group so the block
    right below the LM head can still do exact lookup.
    """
    period = ratio + 1
    if n_layers % period:
        raise ValueError(f"n_layers={n_layers} must be divisible by ratio+1={period}")
    return (["linear"] * ratio + ["attn"]) * (n_layers // period)


# ──────────────────────────────────────────────
# mHC — Manifold-Constrained Hyper-Connections
# ──────────────────────────────────────────────

def sinkhorn(logits, iters=12):
    """Project onto the doubly stochastic manifold (the Birkhoff polytope):
    non-negative, every row and every column summing to 1.

    exp() makes it non-negative, then alternating row/column normalisation
    converges to doubly stochastic. DeepSeek runs 20 iterations; mHC-lite showed
    far fewer suffice, and at n=4 this is a 4x4 matrix either way.

    The loop ends on the *row* normalisation deliberately. Finite iterations only
    approach the manifold, so whichever axis is normalised last is the one that
    holds exactly — and row sums are what conserve the total signal across
    streams (see HyperConnections.forward). Ending on columns instead leaves the
    row sums off by ~2e-4 once training has pushed the logits away from identity.
    """
    m = logits.exp()
    for _ in range(iters):
        m = m / m.sum(dim=0, keepdim=True)   # columns
        m = m / m.sum(dim=1, keepdim=True)   # rows — last, so this one is exact
    return m


class HyperConnections(nn.Module):
    """One mHC unit: wraps a sublayer that reads from and writes to n parallel
    residual streams instead of one (DeepSeek V4, Dec 2025).

        x_{l+1} = H_res · x_l + H_post · F(H_pre · x_l)

    A single residual stream forces every layer to read and write the same
    vector, so depth and width fight over one channel. Hyper-connections widen
    the *pathway* rather than the layer: n streams, a learned read (H_pre), a
    learned write (H_post), and a learned n×n mixing matrix (H_res) between them.

    The "manifold-constrained" part is what makes it trainable. Unconstrained
    hyper-connections let the residual mapping grow or shrink signal magnitude;
    forcing H_res doubly stochastic means every stream emits exactly as much as
    it receives, so the widened pathway still behaves like an identity globally.

    Initialised to read from and write to stream 0 only, with H_res = I, so the
    network starts out *as* the single-stream network it wraps (the model fans
    the embedding into [x, 0, ..., 0] and sums the streams back at the end).

    That one-hot init is load-bearing, not cosmetic. Initialising every stream
    identically with a uniform write looks equivalent and is worse than useless:
    doubly stochastic rows sum to 1, so mixing identical streams returns them
    unchanged, and the streams stay identical forever. Worse, the gradient on
    every stream is then identical too, so the symmetry is a saddle point that
    training cannot break. Streams have to start distinguishable.

    `noise` breaks the *remaining* symmetry. The one-hot init distinguishes
    stream 0 from the rest, but leaves streams 1..n-1 interchangeable: they start
    identical and receive identical gradients, so they stay identical, and n=4
    behaves like n=2. A little jitter on the init makes all n streams distinct.
    Set it to 0 to recover an exactly-plain-residual network at initialisation.

    ponytail: H_pre / H_post are plain learned vectors here. Real mHC also
    constrains them non-negative — dropped because it isn't what the manifold
    constraint is about, and it costs the exact-at-init property that makes this
    testable.
    """

    def __init__(self, n, iters=12, init=8.0, noise=0.02):
        super().__init__()
        self.n, self.iters = n, iters
        jitter = (lambda t: t + torch.randn_like(t) * noise) if noise else (lambda t: t)
        self.pre = nn.Parameter(jitter(F.one_hot(torch.tensor(0), n).float()))   # read stream 0
        self.post = nn.Parameter(jitter(F.one_hot(torch.tensor(0), n).float()))  # write stream 0
        self.res_logits = nn.Parameter(jitter(torch.eye(n) * init))              # sinkhorn → ~I

    def res_matrix(self):
        return sinkhorn(self.res_logits, self.iters)

    def forward(self, streams, fn):
        """streams: (B, T, n, d). fn maps (B, T, d) → (B, T, d)."""
        y = fn((streams * self.pre.view(1, 1, -1, 1)).sum(dim=2))
        # res[i, j] = how much of stream i flows into stream j. Row sums are 1,
        # so the total across streams is conserved exactly — that is the whole
        # point of the constraint.
        mixed = torch.einsum("btid,ij->btjd", streams, self.res_matrix())
        return mixed + y.unsqueeze(2) * self.post.view(1, 1, -1, 1)


# ──────────────────────────────────────────────
# KV sharing (cross-layer attention)
# ──────────────────────────────────────────────

class SharedKVAttention(GroupedQueryAttention):
    """Attention layer that reuses another layer's keys and values (Gemma 4
    E2B/E4B).

    It still computes its own queries, so it can attend differently — it just
    doesn't pay for its own K/V projections or its own slice of KV cache. Gemma 4
    reports ~50% cache reduction; at 128K context that's 2.7 GB in bf16.

    The donor is held in a plain list so it isn't registered as a submodule and
    its parameters aren't counted (or optimized) twice.
    """

    def __init__(self, cfg, donor):
        super().__init__(cfg)
        del self.W_key, self.W_value          # no KV projections at all
        self.donor = [donor]

    def forward(self, x, cos, sin, use_cache=False):
        B, T, _ = x.shape
        q = self.W_query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if self.q_norm:
            q = self.q_norm(q)
        if self.pos_enc != "nope":
            # The donor's K already has its positions baked in, so Q must use the
            # same absolute offset the donor just used.
            if use_cache and self.cache_seq_len > 0:
                q = apply_rope_offset(q, cos[self.cache_seq_len:self.cache_seq_len + T],
                                      sin[self.cache_seq_len:self.cache_seq_len + T])
            else:
                q = apply_rope(q, cos, sin)
        if use_cache:
            self.cache_seq_len += T

        k, v = self.donor[0].last_kv          # donor ran earlier in this same forward
        return self._attend(q, k, v, x)

    def reset_cache(self):
        self.cache_k = self.cache_v = None
        self.cache_seq_len = 0


# ──────────────────────────────────────────────
# Hybrid block — same shell, two possible mixers
# ──────────────────────────────────────────────

class HybridBlock(nn.Module):
    """Pre-norm block. The only thing that varies between layers is the mixer:
    a linear-attention recurrence, or gated GQA with RoPE."""

    def __init__(self, cfg, kind, kv_donor=None):
        super().__init__()
        self.is_attn = kind == "attn"
        self.norm1 = RMSNorm(cfg["emb_dim"])
        if self.is_attn:
            # Gated attention: qwen_nano's GQA + the output gate.
            gated = {**cfg, "attn_out_gate": True}
            self.attn = SharedKVAttention(gated, kv_donor) if kv_donor is not None \
                else GroupedQueryAttention(gated)
        else:
            # ponytail: linear layers get no positional encoding — the recurrence
            # is already order-dependent, and Qwen3-Next doesn't add one either.
            self.attn = LINEAR_MIXERS[cfg["linear_attn"]]({**cfg, "qkv_bias": False})
        self.norm2 = RMSNorm(cfg["emb_dim"])
        self.ff = SwiGLUFeedForward(cfg)

        # Residual style: one stream, or n hyper-connected streams.
        if cfg.get("residual", "plain") == "mhc":
            n, iters = cfg.get("mhc_streams", 4), cfg.get("mhc_iters", 12)
            init, noise = cfg.get("mhc_init", 8.0), cfg.get("mhc_noise", 0.02)
            self.hc_attn = HyperConnections(n, iters, init, noise)
            self.hc_ff = HyperConnections(n, iters, init, noise)
        else:
            self.hc_attn = self.hc_ff = None

    def _mix(self, h, cos, sin, use_cache):
        return self.attn(h, cos, sin, use_cache=use_cache) if self.is_attn \
            else self.attn(h, use_cache=use_cache)

    def forward(self, x, cos, sin, use_cache=False):
        if self.hc_attn is None:
            x = x + self._mix(self.norm1(x), cos, sin, use_cache)
            return x + self.ff(self.norm2(x))
        # x is (B, T, n, d) here — each sublayer reads and writes all n streams.
        x = self.hc_attn(x, lambda h: self._mix(self.norm1(h), cos, sin, use_cache))
        return self.hc_ff(x, lambda h: self.ff(self.norm2(h)))


# ──────────────────────────────────────────────
# PLE — Per-Layer Embeddings
# ──────────────────────────────────────────────

class PerLayerEmbeddings(nn.Module):
    """A second embedding table that feeds a small per-layer signal into every
    block (Gemma 4 E2B/E4B, 2026).

    A normal model looks up a token once, at the bottom. PLE gives every layer
    its own slice of embedding for that token, projected up and added after the
    block. Because the PLE dimension is much smaller than the hidden size, this
    is cheap per layer, and the table itself is pure memory — it can sit in
    slower storage or be streamed, never multiplied against anything large.

    That is how Gemma 4 E2B carries 5.1B parameters but activates 2.3B: a
    different way of separating stored knowledge from active compute than MoE's
    routing.
    """

    def __init__(self, cfg, n_layers):
        super().__init__()
        d, self.dim = cfg["emb_dim"], cfg["ple_dim"]
        self.n_layers = n_layers
        self.table = nn.Embedding(cfg["vocab_size"], n_layers * self.dim)
        self.proj = nn.ModuleList([nn.Linear(self.dim, d, bias=False) for _ in range(n_layers)])
        self.scale = nn.Parameter(torch.zeros(n_layers))   # start as a no-op

    def lookup(self, idx):
        B, T = idx.shape
        return self.table(idx).view(B, T, self.n_layers, self.dim)

    def layer_signal(self, ple, layer):
        return self.proj[layer](ple[:, :, layer]) * self.scale[layer]


# ──────────────────────────────────────────────
# Multi-Token Prediction
# ──────────────────────────────────────────────

class MTPHead(nn.Module):
    """Predict token t+2, alongside the main model's t+1 (DeepSeek-V3, Qwen3.5,
    Nemotron 3).

    Teacher forcing only ever asks the model "what comes next", so the hidden
    state has no pressure to plan further than one token. MTP adds a second
    objective computed from the same hidden state: given h_t *and* the embedding
    of the token that actually came next, predict the one after it. One extra
    block during training, dropped at inference — or kept as a draft head for
    speculative decoding, which is why the 2026 models all ship it.

    Embedding and output head are shared with the main model, so the cost is one
    block plus a 2d→d projection.
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        self.norm_h = RMSNorm(d)                          # main model's hidden state
        self.norm_e = RMSNorm(d)                          # embedding of the next token
        self.proj = nn.Linear(2 * d, d, bias=False)       # concat → back to d
        # Plain residual: the MTP head runs on the main model's reduced hidden
        # state, after any hyper-connection streams have been averaged back.
        self.block = HybridBlock({**cfg, "residual": "plain"}, "attn")

    def forward(self, h, next_emb, cos, sin):
        z = self.proj(torch.cat([self.norm_h(h), self.norm_e(next_emb)], dim=-1))
        return self.block(z, cos, sin)


# ──────────────────────────────────────────────
# Qwen-Next Nano Model
# ──────────────────────────────────────────────

class QwenNextNano(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["head_dim"] * cfg["n_heads"] == cfg["emb_dim"], \
            "linear mixers assume head_dim == emb_dim // n_heads"
        self.cfg = cfg
        self.pattern = build_layer_pattern(cfg["n_layers"], cfg["hybrid_ratio"])
        self.mhc_streams = cfg.get("mhc_streams", 4) if cfg.get("residual") == "mhc" else 0

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

        # KV sharing: the last `kv_share` attention layers reuse K/V from the most
        # recent earlier attention layer that computes its own (Gemma 4).
        attn_positions = [i for i, k in enumerate(self.pattern) if k == "attn"]
        share_from = set(attn_positions[len(attn_positions) - cfg.get("kv_share", 0):]) \
            if cfg.get("kv_share", 0) else set()
        if share_from and len(share_from) >= len(attn_positions):
            raise ValueError(f"kv_share={cfg['kv_share']} leaves no donor layer "
                             f"({len(attn_positions)} attention layers exist)")

        self.blocks = nn.ModuleList()
        donor = None
        for i, kind in enumerate(self.pattern):
            block = HybridBlock(cfg, kind, kv_donor=donor if i in share_from else None)
            if kind == "attn" and i not in share_from:
                donor = block.attn
            self.blocks.append(block)
        self.norm = RMSNorm(cfg["emb_dim"])
        self.head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self.ple = PerLayerEmbeddings(cfg, cfg["n_layers"]) if cfg.get("ple_dim", 0) else None

        self.mtp_weight = cfg.get("mtp_weight", 0.0)
        self.mtp = MTPHead(cfg) if self.mtp_weight > 0 else None
        self.needs_targets = self.mtp is not None  # tells qwen_nano.train to pass y

        cos, sin = compute_rope_params(cfg["head_dim"], cfg["rope_base"], cfg["context_length"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, use_cache=False):
        x = self.drop(self.tok_emb(idx))
        if self.mhc_streams:
            # Fan out into [x, 0, ..., 0] and sum back at the end. Streams must
            # start distinguishable — see HyperConnections.
            x = F.pad(x.unsqueeze(2), (0, 0, 0, self.mhc_streams - 1))
        ple = self.ple.lookup(idx) if self.ple is not None else None
        for i, block in enumerate(self.blocks):
            x = block(x, self.cos, self.sin, use_cache=use_cache)
            if ple is not None:
                sig = self.ple.layer_signal(ple, i)
                # With hyper-connections the signal goes into stream 0, the one
                # the blocks actually read from at init.
                x = x + (F.pad(sig.unsqueeze(2), (0, 0, 0, self.mhc_streams - 1))
                         if self.mhc_streams else sig)
        if self.mhc_streams:
            x = x.sum(dim=2)
        logits = self.head(self.norm(x))
        if targets is None or self.mtp is None:
            return logits
        return logits, self._mtp_loss(x, targets)

    def _mtp_loss(self, x, targets):
        """targets[:, i] is token t_{i+1}, so the MTP head sees h_i + Emb(t_{i+1})
        and is scored against targets[:, i+1] — the token two steps ahead. The
        last position has no t+2 target, hence the shift."""
        h = x[:, :-1]
        next_emb = self.tok_emb(targets[:, :-1])
        z = self.mtp(h, next_emb, self.cos, self.sin)
        logits = self.head(self.norm(z))
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), targets[:, 1:].flatten())
        return self.mtp_weight * loss

    def reset_kv_cache(self):
        for block in self.blocks:
            block.attn.reset_cache()

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def kv_layers(self):
        """How many layers actually hold a KV cache — the point of the whole design."""
        return sum(1 for k in self.pattern if k == "attn")

    def own_kv_layers(self):
        """Attention layers that compute their own K/V — the rest borrow (KV sharing)."""
        return sum(1 for b in self.blocks if b.is_attn and not isinstance(b.attn, SharedKVAttention))

    def mtp_params(self):
        """MTP params are training-only here — subtract them for an inference count.

        ponytail: the MTP head is trained but not used for speculative decoding.
        Drafting would need to roll back a rejected token, and the linear layers'
        recurrent state has no rollback — you'd have to snapshot S per step.
        """
        return sum(p.numel() for p in self.mtp.parameters()) if self.mtp else 0


# ──────────────────────────────────────────────
# Self-check
# ──────────────────────────────────────────────

def self_check():
    """Asserts the hybrid wiring is right: shapes, layer pattern, and — the part
    that actually breaks — that the incremental path (KV cache in the attention
    layers, recurrent state in the linear ones) matches a full forward pass."""
    torch.manual_seed(0)

    assert build_layer_pattern(8, 3) == ["linear"] * 3 + ["attn"] + ["linear"] * 3 + ["attn"]
    assert build_layer_pattern(4, 1) == ["linear", "attn", "linear", "attn"]

    def build(**over):
        cfg = {**MODEL_SIZES["nano"], "vocab_size": 128, "drop_rate": 0.0,
               "linear_attn": "deltanet", "hybrid_ratio": 3, "mtp_weight": 0.0, **over}
        torch.manual_seed(0)
        return cfg, QwenNextNano(cfg).eval()

    def check_incremental(model, vocab, T=12, prefill=8):
        """Prefill, then decode one token at a time — must equal a full forward.
        This is what catches a broken KV cache, recurrent state or conv state."""
        idx = torch.randint(0, vocab, (2, T))
        full = model(idx)
        assert full.shape == (2, T, vocab), full.shape
        model.reset_kv_cache()
        model(idx[:, :prefill], use_cache=True)
        stepwise = torch.cat([model(idx[:, t:t + 1], use_cache=True) for t in range(prefill, T)], dim=1)
        torch.testing.assert_close(stepwise, full[:, prefill:], atol=1e-4, rtol=1e-4)
        return full

    for linear in LINEAR_MIXERS:
        cfg, model = build(linear_attn=linear)
        assert model.kv_layers() == 1, "3:1 on 4 layers → exactly one KV-cached layer"
        check_incremental(model, cfg["vocab_size"])
        print(f"  {linear:9s} ok — {model.count_params():,} params, "
              f"{model.kv_layers()}/{cfg['n_layers']} layers cache KV")

    # ShortConv: must change the output, and its rolling state must keep
    # incremental decoding exact.
    cfg, plain = build(linear_attn="kda")
    cfg_sc, conv = build(linear_attn="kda", short_conv=4)
    base = check_incremental(plain, cfg["vocab_size"])
    out = check_incremental(conv, cfg_sc["vocab_size"])
    assert conv.blocks[0].attn.convs is not None
    assert conv.count_params() > plain.count_params(), "ShortConv added no params"
    assert not torch.allclose(base, out), "ShortConv is a no-op"
    print(f"  shortconv ok — k=4 on Q/K/V, +{conv.count_params() - plain.count_params():,} params")

    # KV sharing: sharer owns no K/V projections and borrows the donor's.
    cfg_kv, shared = build(hybrid_ratio=1, kv_share=1)          # L A L A → 2 attn layers
    assert shared.kv_layers() == 2 and shared.own_kv_layers() == 1
    sharers = [b.attn for b in shared.blocks if isinstance(b.attn, SharedKVAttention)]
    assert len(sharers) == 1 and not hasattr(sharers[0], "W_key")
    _, unshared = build(hybrid_ratio=1, kv_share=0)
    assert shared.count_params() < unshared.count_params()
    check_incremental(shared, cfg_kv["vocab_size"])
    try:
        build(hybrid_ratio=1, kv_share=2)                        # no donor left
        raise AssertionError("kv_share with no donor should fail")
    except ValueError:
        pass
    print(f"  kv_share  ok — {shared.own_kv_layers()}/{shared.kv_layers()} attention layers own K/V, "
          f"-{unshared.count_params() - shared.count_params():,} params")

    # NoPE: no rotation applied, everything else identical.
    cfg_np, nope = build(pos_enc="nope")
    _, rope = build(pos_enc="rope")                    # same seed → identical weights
    assert nope.count_params() == rope.count_params(), "NoPE should not change param count"
    probe = torch.randint(0, cfg_np["vocab_size"], (2, 12))
    assert not torch.allclose(nope(probe), rope(probe)), "NoPE is a no-op"
    check_incremental(nope, cfg_np["vocab_size"])
    print("  nope      ok — RoPE skipped, param count unchanged")

    # mHC: the manifold constraint must hold exactly, n=1 must degenerate to a
    # plain residual, and the identity init must start out as one.
    cfg_hc, mhc = build(residual="mhc", mhc_streams=4, mhc_noise=0.0)
    hc = mhc.blocks[0].hc_attn
    res = hc.res_matrix()
    assert (res >= 0).all(), "Sinkhorn produced negative entries"
    # Rows are exact by construction (last normalisation); columns converge.
    torch.testing.assert_close(res.sum(1), torch.ones(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(res.sum(0), torch.ones(4), atol=1e-3, rtol=0)
    check_incremental(mhc, cfg_hc["vocab_size"])

    _, plain_res = build(residual="plain")
    probe = torch.randint(0, cfg_hc["vocab_size"], (2, 12))
    ref = plain_res(probe)
    _, one = build(residual="mhc", mhc_streams=1, mhc_noise=0.0)     # n=1: sinkhorn is exactly [[1]]
    torch.testing.assert_close(one(probe), ref, atol=1e-5, rtol=1e-5)

    # For n>1 the init is only approximately the plain network: Sinkhorn on a
    # finite logit leaves ~exp(-init) of mass off the diagonal, and eight
    # hyper-connection units compound it. Assert the mechanism, not a magic
    # number — a bigger logit scale must give a proportionally tighter identity.
    dev = {}
    for scale in (8.0, 16.0):
        _, m = build(residual="mhc", mhc_streams=4, mhc_init=scale, mhc_noise=0.0)
        dev[scale] = (m(probe) - ref).abs().max().item()
    assert dev[8.0] < 1e-2, dev
    assert dev[16.0] < dev[8.0] / 100, dev

    # Once the parameters move off the identity init it must actually differ.
    with torch.no_grad():
        hc.res_logits.add_(torch.randn(4, 4))
        hc.pre.add_(torch.randn(4))
    assert not torch.allclose(mhc(probe), ref, atol=1e-2), "mHC is a no-op"
    res = hc.res_matrix()
    torch.testing.assert_close(res.sum(1), torch.ones(4), atol=1e-6, rtol=0)   # still on-manifold
    print(f"  mhc       ok — {hc.n} streams, doubly stochastic, identity init to "
          f"{dev[8.0]:.1e} (scale 8) / {dev[16.0]:.1e} (scale 16), "
          f"+{mhc.count_params() - plain_res.count_params():,} params")

    # PLE: a zero-init scale means it starts as a no-op, and it must add
    # parameters without changing the active hidden size.
    cfg_ple, ple = build(ple_dim=16)
    probe = torch.randint(0, cfg_ple["vocab_size"], (2, 12))
    # Compare against the *same* model with PLE detached — building a second
    # model without it would shuffle the init RNG and change every weight.
    on = ple(probe)
    mod, ple.ple = ple.ple, None
    off = ple(probe)
    ple.ple = mod
    torch.testing.assert_close(on, off, atol=1e-6, rtol=1e-6)     # zero-init: exact no-op
    with torch.no_grad():
        ple.ple.scale.add_(0.5)
    assert not torch.allclose(ple(probe), off), "PLE never becomes active"
    check_incremental(ple, cfg_ple["vocab_size"])
    stored = sum(p.numel() for p in ple.ple.parameters())
    print(f"  ple       ok — {stored:,} stored params, {ple.ple.dim}d per layer, "
          f"zero-init so it starts as a no-op")

    # MTP: same logits with or without it, an extra finite loss, and gradients
    # that actually reach the MTP block.
    mtp_cfg = {**MODEL_SIZES["nano"], "vocab_size": 128, "drop_rate": 0.0,
               "linear_attn": "deltanet", "hybrid_ratio": 3, "mtp_weight": 0.3}
    model = QwenNextNano(mtp_cfg).eval()
    assert model.needs_targets and model.mtp_params() > 0
    idx = torch.randint(0, mtp_cfg["vocab_size"], (2, 12))
    targets = torch.randint(0, mtp_cfg["vocab_size"], (2, 12))
    logits, mtp_loss = model(idx, targets)
    torch.testing.assert_close(logits, model(idx))          # MTP must not touch the main path
    assert torch.isfinite(mtp_loss) and mtp_loss > 0, mtp_loss
    mtp_loss.backward()
    assert model.mtp.proj.weight.grad.abs().sum() > 0, "no gradient into the MTP head"
    print(f"  mtp ok — loss {mtp_loss.item():.3f}, "
          f"{model.mtp_params():,} training-only params "
          f"({model.mtp_params() / model.count_params():.0%} of total)")

    # The gate must exist on attention layers and actually change the output.
    cfg = {**MODEL_SIZES["nano"], "vocab_size": 128, "drop_rate": 0.0,
           "linear_attn": "deltanet", "hybrid_ratio": 1, "mtp_weight": 0.0}
    model = QwenNextNano(cfg).eval()
    attn = next(b.attn for b in model.blocks if b.is_attn)
    assert attn.out_gate is not None
    x, cos, sin = torch.randn(1, 6, cfg["emb_dim"]), model.cos, model.sin
    gated = attn(x, cos, sin)
    attn.out_gate = None
    assert not torch.allclose(gated, attn(x, cos, sin)), "output gate is a no-op"
    print("  gated attention ok — gate is applied before out_proj")

    print("self-check passed")


def train_check():
    """Properties that only show up once gradients have flowed. Both of these
    caught real bugs that the init-only self-check could not see: a Sinkhorn loop
    that left row sums off-manifold after training, and hyper-connection streams
    that stayed symmetric forever.

    ~30s on CPU. Run it after touching MTP or mHC.
    """
    torch.manual_seed(0)
    V = 96

    def overfit(cfg, steps=300, T=25):
        model = QwenNextNano(cfg)
        idx = torch.randint(0, V, (4, T))
        x, y = idx[:, :-1], idx[:, 1:]
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        for step in range(steps):
            out = model(x, y) if model.needs_targets else model(x)
            logits, aux = out if isinstance(out, tuple) else (out, None)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            if aux is not None:
                loss = loss + aux
            opt.zero_grad(); loss.backward(); opt.step()
            if step == 0:
                yield model, x, y            # hand back for gradient inspection
        yield model, x, y

    # MTP: after overfitting, the main head must predict t+1 and the MTP head
    # t+2. An off-by-one in the shift trains just as happily and is invisible
    # in the loss curve.
    cfg = {**MODEL_SIZES["nano"], "vocab_size": V, "drop_rate": 0.0, "linear_attn": "deltanet",
           "hybrid_ratio": 3, "mtp_weight": 1.0, "mhc_streams": 0}
    gen = overfit(cfg, steps=400)
    next(gen)
    model, x, y = next(gen)
    model.eval()
    with torch.no_grad():
        h = model.drop(model.tok_emb(x))
        for b in model.blocks:
            h = b(h, model.cos, model.sin)
        mtp_pred = model.head(model.norm(
            model.mtp(h[:, :-1], model.tok_emb(y[:, :-1]), model.cos, model.sin))).argmax(-1)
        main_acc = (model(x).argmax(-1) == y).float().mean().item()
    acc_t2 = (mtp_pred == y[:, 1:]).float().mean().item()
    acc_t1 = (mtp_pred == y[:, :-1]).float().mean().item()
    assert main_acc > 0.9, f"main head did not overfit: {main_acc}"
    assert acc_t2 > 0.9, f"MTP head did not learn t+2: {acc_t2}"
    assert acc_t2 > acc_t1 + 0.5, f"MTP head is predicting t+1 — shift is wrong ({acc_t1})"
    print(f"  mtp ok — main→t+1 {main_acc:.0%}, mtp→t+2 {acc_t2:.0%}, mtp→t+1 {acc_t1:.0%} (chance)")

    # mHC: gradients must reach every hyper-connection parameter, the residual
    # matrices must stay on-manifold after training, and all n streams must end
    # up distinct — equal streams mean n=4 is secretly n=2.
    cfg = {**MODEL_SIZES["nano"], "vocab_size": V, "drop_rate": 0.0, "linear_attn": "kda",
           "hybrid_ratio": 3, "mtp_weight": 0.0, "residual": "mhc", "mhc_streams": 4}
    gen = overfit(cfg)
    model, _, _ = next(gen)
    hcs = [hc for b in model.blocks for hc in (b.hc_attn, b.hc_ff)]
    for hc in hcs:
        for name in ("pre", "post", "res_logits"):
            g = getattr(hc, name).grad
            assert g is not None and g.abs().sum() > 0, f"no gradient into {name}"
    model, x, _ = next(gen)
    for hc in hcs:
        r = hc.res_matrix()
        assert (r >= 0).all(), "off-manifold: negative entries"
        torch.testing.assert_close(r.sum(1), torch.ones(hc.n), atol=1e-6, rtol=0)
        torch.testing.assert_close(r.sum(0), torch.ones(hc.n), atol=1e-3, rtol=0)

    model.eval()
    with torch.no_grad():
        h = F.pad(model.drop(model.tok_emb(x)).unsqueeze(2), (0, 0, 0, model.mhc_streams - 1))
        for b in model.blocks:
            h = b(h, model.cos, model.sin)
    norms = h.norm(dim=-1).mean(dim=(0, 1))
    assert (norms > 1e-3).all(), f"dead stream: {norms.tolist()}"
    assert norms.std() / norms.mean() > 0.01, f"streams never differentiated: {norms.tolist()}"
    print(f"  mhc ok — on-manifold after training, stream norms "
          f"{[round(v, 2) for v in norms.tolist()]}")

    print("train-check passed")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train Qwen-Next Nano (hybrid linear/full attention) from scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Model sizes:\n" + "\n".join(
            f"  {k:8s} {v['emb_dim']}d, {v['n_heads']}h({v['n_kv_groups']}kv), "
            f"{v['n_layers']}L, ctx={v['context_length']}"
            for k, v in MODEL_SIZES.items()
        )
    )
    parser.add_argument("--size", type=str, default="nano", choices=list(MODEL_SIZES))
    config.add_argument(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--linear", type=str, default="deltanet", choices=list(LINEAR_MIXERS),
                        help="Linear-attention mixer for the non-attention layers")
    parser.add_argument("--ratio", type=int, default=3,
                        help="Linear layers per attention layer (3 = Qwen3-Next/Kimi Linear)")
    parser.add_argument("--mtp-weight", type=float, default=0.3,
                        help="Multi-token-prediction loss weight (DeepSeek-V3 uses 0.3; 0 disables)")
    parser.add_argument("--short-conv", type=int, default=0, metavar="K",
                        help="ShortConv kernel on the linear layers' Q/K/V (Kimi Linear uses 4; 0 disables)")
    parser.add_argument("--kv-share", type=int, default=0, metavar="N",
                        help="Last N attention layers reuse an earlier layer's K/V (Gemma 4)")
    parser.add_argument("--posenc", type=str, default="rope", choices=["rope", "nope"],
                        help="Positional encoding on the attention layers")
    parser.add_argument("--residual", type=str, default="plain", choices=["plain", "mhc"],
                        help="Residual style: single stream, or manifold-constrained hyper-connections")
    parser.add_argument("--mhc-streams", type=int, default=4, metavar="N",
                        help="Parallel residual streams for --residual mhc (DeepSeek V4 uses 4)")
    parser.add_argument("--ple-dim", type=int, default=0, metavar="D",
                        help="Per-layer embedding width (Gemma 4; 0 = off)")
    parser.add_argument("--self-check", action="store_true", help="Run assertions and exit")
    parser.add_argument("--train-check", action="store_true",
                        help="Overfit tiny models to assert MTP and mHC actually learn (~30s)")
    parser.add_argument("--file", type=str, default=None, help="Training text file")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--ckpt-freq", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return
    if args.train_check:
        train_check()
        return

    ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if ddp:
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{dist.get_rank()}")
        torch.cuda.set_device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # defaults < config file < flags actually typed
    file_cfg = config.load(args.config)
    ov = config.overrider()
    top = {"size": args.size, "file": args.file, "seed": args.seed}
    top.update({k: file_cfg[k] for k in ("size", "file", "seed") if k in file_cfg})
    ov(top, "size", "--size", args.size)
    ov(top, "file", "--file", args.file)
    ov(top, "seed", "--seed", args.seed)
    size, seed = top["size"], top["seed"]

    text = load_text(top["file"])
    log(f"Text length: {len(text):,} characters")

    settings = {**TRAIN_SETTINGS, **file_cfg.get("train", {})}
    if args.epochs:
        settings["num_epochs"] = args.epochs
    if args.batch_size:
        settings["batch_size"] = args.batch_size
    ov(settings, "grad_accum_steps", "--grad-accum", args.grad_accum)
    ov(settings, "use_amp", "--no-amp", not args.no_amp)
    ov(settings, "ckpt_freq", "--ckpt-freq", args.ckpt_freq)

    cfg = {**MODEL_SIZES[size], "linear_attn": args.linear,
           "hybrid_ratio": args.ratio, "mtp_weight": args.mtp_weight,
           "short_conv": args.short_conv, "kv_share": args.kv_share, "pos_enc": args.posenc,
           "residual": args.residual, "mhc_streams": args.mhc_streams,
           "ple_dim": args.ple_dim}
    cfg.update(file_cfg.get("model", {}))
    for key, flag, value in (
        ("linear_attn", "--linear", args.linear),
        ("hybrid_ratio", "--ratio", args.ratio),
        ("mtp_weight", "--mtp-weight", args.mtp_weight),
        ("short_conv", "--short-conv", args.short_conv),
        ("kv_share", "--kv-share", args.kv_share),
        ("pos_enc", "--posenc", args.posenc),
        ("residual", "--residual", args.residual),
        ("mhc_streams", "--mhc-streams", args.mhc_streams),
        ("ple_dim", "--ple-dim", args.ple_dim),
    ):
        ov(cfg, key, flag, value)

    resume_step = resume_epoch = 0
    optimizer_state = None
    if args.resume:
        log(f"\nLoading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        cfg = ckpt["config"]
        resume_step, resume_epoch = ckpt["global_step"], ckpt["epoch"]
        optimizer_state = ckpt["optimizer"]

    train_loader, val_loader, tokenizer, train_sampler = create_dataloaders(
        text, cfg, settings["batch_size"])
    log(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    ckpt_dir = os.path.join(config.ROOT, "checkpoints")
    if is_main_process():
        saved = config.snapshot(ckpt_dir, {"model": cfg, "train": settings,
                                           "seed": seed, "size": size, "file": top["file"]},
                                device=device)
        log(f"Resolved config: {saved}  (rerun with --config {saved})")

    torch.manual_seed(seed)
    model = QwenNextNano(cfg)
    if args.resume:
        model.load_state_dict(ckpt["model"])

    log(f"\nLayout: {' '.join('A' if k == 'attn' else 'L' for k in model.pattern)}"
        f"  (L = {cfg['linear_attn']}, A = gated attention)")
    log(f"KV-cached layers: {model.own_kv_layers()}/{cfg['n_layers']} "
        f"→ ~{cfg['n_layers'] / max(1, model.own_kv_layers()):.1f}x smaller KV cache than all-attention")
    if model.mtp is not None:
        log(f"MTP: weight {cfg['mtp_weight']}, {model.mtp_params():,} training-only params")
    log(f"Components: posenc={cfg['pos_enc']}, short_conv={cfg['short_conv'] or 'off'}, "
        f"kv_share={cfg['kv_share'] or 'off'} ({model.own_kv_layers()} layers own their KV), "
        f"residual={cfg['residual']}" + (f" x{model.mhc_streams} streams" if model.mhc_streams else "")
        + (f", ple={cfg['ple_dim']}d" if model.ple is not None else ""))

    model = train(model, train_loader, val_loader, tokenizer, cfg, settings, device,
                  resume_step=resume_step, resume_epoch=resume_epoch,
                  optimizer_state=optimizer_state, train_sampler=train_sampler)

    if is_main_process():
        import time
        ids = torch.tensor(tokenizer.encode(args.prompt)).unsqueeze(0).to(device)
        print(f"\n{'='*60}\nPrompt: {args.prompt}\n{'='*60}")

        torch.manual_seed(42)
        t0 = time.perf_counter()
        out1 = generate(model, ids, max_new_tokens=args.max_tokens,
                        temperature=args.temperature, top_k=args.top_k)
        t1 = time.perf_counter() - t0
        print(f"\n[No cache] {t1:.3f}s")
        print(tokenizer.decode(out1[0].tolist()))

        torch.manual_seed(42)
        t0 = time.perf_counter()
        out2 = generate_cached(model, ids, max_new_tokens=args.max_tokens,
                               temperature=args.temperature, top_k=args.top_k)
        t2 = time.perf_counter() - t0
        print(f"\n[Cached] {t2:.3f}s")
        print(tokenizer.decode(out2[0].tolist()))
        print(f"\nCache/state speedup: {t1 / t2 if t2 > 0 else float('inf'):.2f}x faster\n{'='*60}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
