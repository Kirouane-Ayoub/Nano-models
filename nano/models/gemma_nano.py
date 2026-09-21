"""
Gemma Nano — Google's Gemma 3 / Gemma 4 recipe, built from scratch.

Where Qwen and DeepSeek chase efficiency with new mixers (linear attention,
MLA, MoE), Gemma stays a dense transformer and spends its effort on making
a plain one train deeper and read longer. Every piece is small; the file is
about how they add up.

    tokens ─► emb · √d ─► [ local ×5, global ] × n/6 ─► norm ─► head (tied) ─► c·tanh(z/c)

    local  = sliding-window GQA, window w, RoPE base 10k
    global = full GQA, RoPE base 1M with p-RoPE (25% of pairs rotary), K reused as V
    block  = x + post(attn(pre(x)));  x + post(ffn(pre(x)))     ("sandwich" norm)
    ffn    = GeGLU;  norm = RMSNorm scaled by (1 + w), w zero-init;  QK-norm on every head

Why each piece is there:

  1. **5:1 local:global.** Five layers of 1024-token (Gemma 3) or 512-token
     (Gemma 4 E-sizes) sliding windows for every one global layer. The KV
     cache is dominated by the global layers, so at 128K context this cuts it
     ~6×, and quality barely moves because most attention is local anyway.
     The last layer is always global (a Gemma 4 change) so the LM head sits
     right above a full view of the context.
  2. **Dual RoPE base.** Local layers see at most w tokens back, so 10k is
     plenty; global layers must resolve positions across the whole context,
     so their base is 1M. Gemma 4 additionally applies p-RoPE on the global
     layers: only the highest-frequency quarter of the pairs rotate, the rest
     are position-free content channels.
  3. **K as V on global layers** (Gemma 4). The global layers are the ones
     whose cache grows with context; dropping their value projection halves
     what is left. Local layers keep separate values — their cache is bounded.
  4. **Sandwich norm.** Pre-norm bounds what enters a sublayer; the post-norm
     bounds what it contributes. That is what let Gemma 2 go deeper at a
     fixed width and it has been kept since.
  5. **QK-norm** replaced Gemma 2's attention-logit softcap in Gemma 3: bound
     the dot product at the source instead of squashing it afterwards. The
     softcap on the *final* logits (c=30) stayed.
  6. **(1 + w) RMSNorm.** The scale is parameterised as `1 + w` with w
     initialised to zero. Same function as a ones-initialised scale, but weight
     decay now pulls the norm towards *identity* rather than towards zero.
  7. **Tied, scaled embeddings.** Input and output embeddings share one
     matrix, and the input is multiplied by √d so token vectors start at the
     same scale as the residual stream they join. Gemma's 262k vocabulary
     makes the tying matter: the embedding is a large share of the parameters.

Everything except the architecture (training loop, dataset, generation) is
imported from qwen_nano.py rather than copied. The attention, block, norm and
FFN are all local to this file on purpose — this is a file you read top to
bottom.

Sizes: nano (~3.5M) → small (~22M) → medium (~82M) → large (~350M) — smaller
than the Qwen presets at the same width because the tied head removes a
vocab × d matrix. n_layers is always a multiple of 6 so the 5:1 pattern tiles.

Usage:
    python -m nano.models.gemma_nano                       # nano, 5:1, window 32
    python -m nano.models.gemma_nano --ratio 3 --window 64
    python -m nano.models.gemma_nano --no-k-as-v           # Gemma 3 style global layers
    python -m nano.models.gemma_nano --self-check
    python -m nano.models.gemma_nano --train-check
"""

if __package__ in (None, ""):
    # Running as a script: `python nano/models/gemma_nano.py`. Make `nano` importable.
    import pathlib as _pathlib
    import sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from nano import config, data
from nano.accel import current_device, is_main_process, log
from nano.models.qwen_nano import (
    TRAIN_SETTINGS,
    apply_rope_offset,
    compute_rope_params,
    create_dataloaders,
    generate,
    generate_cached,
    train,
)

# ──────────────────────────────────────────────
# Model size presets
# ──────────────────────────────────────────────
# n_layers must be divisible by local_ratio + 1. Windows are scaled down with
# the context; Gemma 3 uses 1024 at 128K context, Gemma 4 E-sizes 512.

# Hand-aligned table: columns line up so sizes can be compared down the page.
# fmt: off
MODEL_SIZES = {
    "nano": {       # ~3.5M — quick experiments
        "vocab_size": 50257,  "context_length": 128,
        "emb_dim": 64,        "n_heads": 4,      "n_layers": 6,
        "hidden_dim": 192,    "head_dim": 16,    "n_kv_groups": 2,
        "local_ratio": 5,     "window_size": 32,
        "rope_base_local": 10_000.0, "rope_base_global": 1_000_000.0, "rope_fraction": 0.25,
        "k_as_v": True,       "logit_softcap": 30.0, "drop_rate": 0.1,
    },
    "small": {      # ~22M — single GPU
        "vocab_size": 50257,  "context_length": 256,
        "emb_dim": 256,       "n_heads": 8,      "n_layers": 12,
        "hidden_dim": 768,    "head_dim": 32,    "n_kv_groups": 4,
        "local_ratio": 5,     "window_size": 64,
        "rope_base_local": 10_000.0, "rope_base_global": 1_000_000.0, "rope_fraction": 0.25,
        "k_as_v": True,       "logit_softcap": 30.0, "drop_rate": 0.1,
    },
    "medium": {     # ~82M — single GPU
        "vocab_size": 50257,  "context_length": 512,
        "emb_dim": 512,       "n_heads": 8,      "n_layers": 18,
        "hidden_dim": 1536,   "head_dim": 64,    "n_kv_groups": 4,
        "local_ratio": 5,     "window_size": 128,
        "rope_base_local": 10_000.0, "rope_base_global": 1_000_000.0, "rope_fraction": 0.25,
        "k_as_v": True,       "logit_softcap": 30.0, "drop_rate": 0.1,
    },
    "large": {      # ~350M — multi-GPU recommended
        "vocab_size": 50257,  "context_length": 1024,
        "emb_dim": 1024,      "n_heads": 16,     "n_layers": 24,
        "hidden_dim": 3072,   "head_dim": 64,    "n_kv_groups": 8,
        "local_ratio": 5,     "window_size": 256,
        "rope_base_local": 10_000.0, "rope_base_global": 1_000_000.0, "rope_fraction": 0.25,
        "k_as_v": True,       "logit_softcap": 30.0, "drop_rate": 0.1,
    },
}
# fmt: on


def build_layout(n_layers, ratio):
    """`ratio` local layers then one global, repeated. Always ends on global."""
    period = ratio + 1
    if n_layers % period:
        raise ValueError(f"n_layers={n_layers} must be divisible by ratio+1={period}")
    return (["local"] * ratio + ["global"]) * (n_layers // period)


# ──────────────────────────────────────────────
# Norm and feed-forward
# ──────────────────────────────────────────────


class GemmaRMSNorm(nn.Module):
    """RMSNorm with the scale written as (1 + w), w initialised to zero.

    Functionally the same as a ones-initialised scale. The difference is what
    weight decay does: decaying w towards zero pulls the norm towards the
    *identity* scale, whereas decaying a ones-initialised scale pulls it towards
    zero output. Gemma has used this form since v1. Computed in fp32 and cast
    back, as every norm in this repo is.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * (1.0 + self.weight)).to(dtype)


class GeGLUFeedForward(nn.Module):
    """GeGLU: gelu(gate) * up, projected down. Same shape as SwiGLU with GELU
    (tanh approximation, as Gemma's reference does) in place of SiLU. The
    gating is what matters; the choice of activation is a small, stable
    difference that Gemma has simply never changed."""

    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)
        self.up_proj = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)
        self.down_proj = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


# ──────────────────────────────────────────────
# Attention — one class, local or global
# ──────────────────────────────────────────────


class GemmaAttention(nn.Module):
    """GQA with QK-norm and RoPE. `local=True` adds a sliding window and a
    bounded cache; `local=False` may reuse K as V (Gemma 4).

    Order inside forward is QK-norm, then RoPE, then cache — norm before
    rotation so the rotation acts on unit-scale vectors, and rotation before
    caching so cached keys carry their absolute position.

    ponytail: with K-as-V the value is the *normed, unrotated* key. Rotating it
    would bake position into the values, which nothing downstream can undo.
    The report does not spell out this detail; check the reference
    implementation if you need weight-level parity.
    """

    def __init__(self, cfg, local):
        super().__init__()
        d = cfg["emb_dim"]
        self.local = local
        self.window = cfg["window_size"] if local else None
        self.n_heads, self.head_dim = cfg["n_heads"], cfg["head_dim"]
        self.n_kv_groups = cfg["n_kv_groups"]
        self.group_size = self.n_heads // self.n_kv_groups
        self.d_out = self.n_heads * self.head_dim
        kv_dim = self.n_kv_groups * self.head_dim

        self.W_query = nn.Linear(d, self.d_out, bias=False)
        self.W_key = nn.Linear(d, kv_dim, bias=False)
        k_as_v = (not local) and cfg.get("k_as_v", False)
        self.W_value = None if k_as_v else nn.Linear(d, kv_dim, bias=False)
        self.out_proj = nn.Linear(self.d_out, d, bias=False)
        self.q_norm = GemmaRMSNorm(self.head_dim)
        self.k_norm = GemmaRMSNorm(self.head_dim)
        self.dropout = nn.Dropout(cfg["drop_rate"])

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.cache_pos = 0

    def forward(self, x, cos, sin, use_cache=False):
        B, T, _ = x.shape
        H, G, hd = self.n_heads, self.n_kv_groups, self.head_dim
        q = self.q_norm(self.W_query(x).view(B, T, H, hd).transpose(1, 2))
        k = self.k_norm(self.W_key(x).view(B, T, G, hd).transpose(1, 2))
        v = k if self.W_value is None else self.W_value(x).view(B, T, G, hd).transpose(1, 2)

        start = self.cache_pos if use_cache else 0
        q_pos = torch.arange(start, start + T, device=x.device)
        q = apply_rope_offset(q, cos[q_pos], sin[q_pos])
        k = apply_rope_offset(k, cos[q_pos], sin[q_pos])

        if use_cache:
            if self.cache_k is not None:
                k = torch.cat([self.cache_k, k], dim=2)
                v = torch.cat([self.cache_v, v], dim=2)
            self.cache_k, self.cache_v = k, v
            self.cache_pos += T
        T_k = k.shape[2]
        k_pos = torch.arange(start + T - T_k, start + T, device=x.device)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        attn = (q @ k.transpose(-2, -1)) / (hd**0.5)
        diff = q_pos.unsqueeze(-1) - k_pos.unsqueeze(0)
        mask = diff < 0
        if self.local:
            mask = mask | (diff >= self.window)
        attn = self.dropout(torch.softmax(attn.masked_fill(mask, float("-inf")), dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, self.d_out)

        # Trim only after attending: a prefill longer than the window needs its
        # early keys for its early queries (see ARCHITECTURES.md § 6).
        if use_cache and self.local and self.cache_k.shape[2] > self.window:
            self.cache_k = self.cache_k[:, :, -self.window :]
            self.cache_v = self.cache_v[:, :, -self.window :]
        return self.out_proj(out)

    def reset_cache(self):
        self.cache_k = self.cache_v = None
        self.cache_pos = 0


# ──────────────────────────────────────────────
# Block and model
# ──────────────────────────────────────────────


class GemmaBlock(nn.Module):
    """Sandwich-norm block: every sublayer is normed on the way in and on the
    way out, so no single layer can dominate the residual stream."""

    def __init__(self, cfg, local):
        super().__init__()
        d = cfg["emb_dim"]
        self.pre_attn, self.post_attn = GemmaRMSNorm(d), GemmaRMSNorm(d)
        self.attn = GemmaAttention(cfg, local)
        self.pre_ff, self.post_ff = GemmaRMSNorm(d), GemmaRMSNorm(d)
        self.ff = GeGLUFeedForward(cfg)

    def forward(self, x, cos, sin, use_cache=False):
        x = x + self.post_attn(self.attn(self.pre_attn(x), cos, sin, use_cache=use_cache))
        return x + self.post_ff(self.ff(self.pre_ff(x)))


class GemmaNano(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["emb_dim"]
        self.layout = build_layout(cfg["n_layers"], cfg["local_ratio"])

        self.tok_emb = nn.Embedding(cfg["vocab_size"], d)
        self.emb_scale = d**0.5  # token vectors start at residual-stream scale
        self.drop = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.ModuleList(GemmaBlock(cfg, kind == "local") for kind in self.layout)
        self.norm = GemmaRMSNorm(d)
        self.head = nn.Linear(d, cfg["vocab_size"], bias=False)
        self.head.weight = self.tok_emb.weight  # tied
        self.logit_softcap = cfg.get("logit_softcap", 0.0)

        self.build_rope_tables()
        self.apply(self._init_weights)

    def build_rope_tables(self, device=None):
        """Two RoPE tables: base 10k for local layers, base 1M for global ones,
        with p-RoPE freezing the low-frequency pairs of the global table.
        Non-persistent buffers, so anything that constructs the model on a meta
        device (the HF wrapper) calls this again to rebuild them."""
        cfg, hd = self.cfg, self.cfg["head_dim"]
        cos_l, sin_l = compute_rope_params(hd, cfg["rope_base_local"], cfg["context_length"])
        cos_g, sin_g = compute_rope_params(hd, cfg["rope_base_global"], cfg["context_length"])
        half = hd // 2
        cut = int(cfg.get("rope_fraction", 1.0) * half)
        for s in (cut, half + cut):  # low-frequency pairs → identity rotation
            cos_g[:, s : s + half - cut] = 1.0
            sin_g[:, s : s + half - cut] = 0.0
        for name, t in (("cos_local", cos_l), ("sin_local", sin_l), ("cos_global", cos_g), ("sin_global", sin_g)):
            self.register_buffer(name, t.to(device) if device is not None else t, persistent=False)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, use_cache=False):
        x = self.drop(self.tok_emb(idx) * self.emb_scale)
        for block in self.blocks:
            if block.attn.local:
                x = block(x, self.cos_local, self.sin_local, use_cache=use_cache)
            else:
                x = block(x, self.cos_global, self.sin_global, use_cache=use_cache)
        logits = self.head(self.norm(x))
        if self.logit_softcap:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return logits

    def reset_kv_cache(self):
        for block in self.blocks:
            block.attn.reset_cache()

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────
# Self-check — python -m nano.models.gemma_nano --self-check
# ──────────────────────────────────────────────


def self_check():
    """Shapes and wiring, then the properties that make this Gemma and not a
    generic transformer: the window boundary, the global-only p-RoPE and K-as-V,
    the (1+w) norm, the tied scaled embedding, the softcap. Then the two checks
    that catch real bugs: incremental decode and causality."""
    torch.manual_seed(0)
    V = 128

    def build(**over):
        cfg = {**MODEL_SIZES["nano"], "vocab_size": V, "drop_rate": 0.0, **over}
        torch.manual_seed(0)
        return cfg, GemmaNano(cfg).eval()

    def check_incremental(model, T=12, prefill=8):
        idx = torch.randint(0, V, (2, T))
        full = model(idx)
        assert full.shape == (2, T, V), full.shape
        model.reset_kv_cache()
        pre = model(idx[:, :prefill], use_cache=True)
        torch.testing.assert_close(pre, full[:, :prefill], atol=1e-4, rtol=1e-4)
        step = torch.cat([model(idx[:, t : t + 1], use_cache=True) for t in range(prefill, T)], 1)
        torch.testing.assert_close(step, full[:, prefill:], atol=1e-4, rtol=1e-4)
        return full

    # 1. Layout: 5 local then 1 global, last layer global, global layers have no W_value.
    cfg, model = build()
    assert build_layout(6, 5) == ["local"] * 5 + ["global"]
    assert model.layout[-1] == "global", "last layer must be global"
    kinds = [("local" if b.attn.local else "global") for b in model.blocks]
    assert kinds == model.layout
    for b in model.blocks:
        assert (b.attn.W_value is None) == (not b.attn.local), "K-as-V must be global-only"
    _, with_v = build(k_as_v=False)
    d, G, hd = cfg["emb_dim"], cfg["n_kv_groups"], cfg["head_dim"]
    assert with_v.count_params() - model.count_params() == d * G * hd, "one W_value per global layer"
    print(f"  layout    ok — {' '.join(k[0].upper() for k in model.layout)}, K-as-V saves {d * G * hd:,} params")

    # 2. Tied and scaled embeddings.
    assert model.head.weight is model.tok_emb.weight, "head is not tied to the embedding"
    assert model.emb_scale == d**0.5
    print(f"  embedding ok — head tied to tok_emb, input scaled by √{d} = {model.emb_scale:.1f}")

    # 3. (1 + w) RMSNorm: identity scale at init, and w=1 doubles it.
    norm = model.norm
    z = torch.randn(3, 7, d)
    plain = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + norm.eps)
    assert (norm.weight == 0).all(), "norm weight should start at zero"
    torch.testing.assert_close(norm(z), plain)
    with torch.no_grad():
        norm.weight.fill_(1.0)
    torch.testing.assert_close(norm(z), 2 * plain)
    with torch.no_grad():
        norm.weight.zero_()
    print("  rmsnorm   ok — (1 + w) scale, identity at w=0")

    # 4. Dual RoPE: local table is plain RoPE at base 10k, global table has the
    #    low-frequency pairs frozen (p-RoPE) and a different base.
    half, cut = hd // 2, int(cfg["rope_fraction"] * hd // 2)
    assert (model.cos_global[:, cut:half] == 1).all() and (model.sin_global[:, cut:half] == 0).all()
    assert not (model.cos_local[:, cut:half] == 1).all(), "local layers should keep full RoPE"
    assert not torch.allclose(model.cos_local[:, :cut], model.cos_global[:, :cut]), "bases differ"
    print(f"  rope      ok — local base 10k full, global base 1M with {cut}/{half} pairs rotary")

    # 5. Window boundary, on one local attention layer: a token exactly `w` back
    #    must be invisible, one at `w-1` must not. Nothing else pins the boundary.
    attn = model.blocks[0].attn
    w = attn.window
    x = torch.randn(1, w + 4, d)
    base = attn(x, model.cos_local, model.sin_local)
    t = w + 3
    outside, inside = x.clone(), x.clone()
    outside[:, t - w] += 10.0
    inside[:, t - w + 1] += 10.0
    torch.testing.assert_close(attn(outside, model.cos_local, model.sin_local)[:, t], base[:, t])
    assert not torch.allclose(attn(inside, model.cos_local, model.sin_local)[:, t], base[:, t])
    print(f"  window    ok — token t-{w} invisible, t-{w - 1} visible")

    # 6. Final softcap: c·tanh(z/c) of the uncapped logits, exactly.
    _, uncapped = build(logit_softcap=0.0)
    _, capped = build(logit_softcap=0.5)
    idx = torch.randint(0, V, (2, 12))
    torch.testing.assert_close(capped(idx), 0.5 * torch.tanh(uncapped(idx) / 0.5))
    assert capped(idx).abs().max() < 0.5
    print("  softcap   ok — final logits are c·tanh(z/c)")

    # 7. Incremental decode, with a window shorter than the prefill so the local
    #    caches actually get trimmed.
    cfg_w, short = build(window_size=4)
    check_incremental(short)
    check_incremental(model)
    print("  kv cache  ok — prefill and decode match the full forward, window trim included")

    # 8. Causality across the whole stack.
    a = torch.randint(0, V, (1, 10))
    b = a.clone()
    b[0, 6] = (b[0, 6] + 1) % V
    torch.testing.assert_close(model(a)[:, :6], model(b)[:, :6])
    assert not torch.allclose(model(a)[:, 6:], model(b)[:, 6:])
    print("  causality ok — no future leak through local or global layers")

    print("\nAll checks passed")


def train_check():
    """What only gradients reveal: the model overfits, every parameter group
    trains, and the (1+w) norm weights move off zero. ~20s on CPU."""
    torch.manual_seed(0)
    V = 96
    cfg = {**MODEL_SIZES["nano"], "vocab_size": V, "drop_rate": 0.0}
    model = GemmaNano(cfg)
    idx = torch.randint(0, V, (4, 25))
    x, y = idx[:, :-1], idx[:, 1:]

    model.train()
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"no gradient to {name}"
    print("  gradients ok — every parameter receives gradient")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
    first = None
    for _ in range(200):
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    model.eval()
    final = F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    assert final < first * 0.5, f"did not overfit: {first:.2f} → {final:.2f}"
    moved = [m.weight.abs().max().item() for m in model.modules() if isinstance(m, GemmaRMSNorm)]
    assert max(moved) > 1e-3, "norm weights never left zero"
    print(f"  learning  ok — loss {first:.2f} → {final:.2f}, norm |w| up to {max(moved):.3f}")

    print("\nAll checks passed")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Gemma Nano — Gemma 3/4 dense recipe")
    parser.add_argument("--size", type=str, default="nano", choices=list(MODEL_SIZES))
    config.add_argument(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=int, default=None, metavar="N", help="Local layers per global (Gemma: 5)")
    parser.add_argument("--window", type=int, default=None, metavar="W", help="Sliding-window size on local layers")
    parser.add_argument(
        "--rope-fraction", type=float, default=None, metavar="F",
        help="p-RoPE on global layers: fraction of pairs kept rotary (Gemma 4: 0.25; 1.0 = plain RoPE)",
    )
    parser.add_argument("--no-k-as-v", action="store_true", help="Keep W_value on global layers (Gemma 3)")
    parser.add_argument("--logit-softcap", type=float, default=None, metavar="C", help="Final logit cap (Gemma: 30)")
    parser.add_argument("--self-check", action="store_true", help="Run assertions and exit")
    parser.add_argument("--train-check", action="store_true", help="Gradient-level checks (~20s)")
    data.add_arguments(parser)
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

    device = current_device()

    # defaults < config file < flags actually typed
    file_cfg = config.load(args.config)
    ov = config.overrider()
    top = {"size": args.size, "seed": args.seed}
    top.update({k: file_cfg[k] for k in ("size", "seed") if k in file_cfg})
    ov(top, "size", "--size", args.size)
    ov(top, "seed", "--seed", args.seed)
    size, seed = top["size"], top["seed"]

    text, data_cfg = data.from_args(args, file_cfg.get("data"), ov, log=log)
    log(f"Text length: {len(text):,} characters")

    settings = {**TRAIN_SETTINGS, **file_cfg.get("train", {})}
    if args.epochs:
        settings["num_epochs"] = args.epochs
    if args.batch_size:
        settings["batch_size"] = args.batch_size
    ov(settings, "grad_accum_steps", "--grad-accum", args.grad_accum)
    ov(settings, "use_amp", "--no-amp", not args.no_amp)
    ov(settings, "ckpt_freq", "--ckpt-freq", args.ckpt_freq)

    cfg = {**MODEL_SIZES[size]}
    cfg.update(file_cfg.get("model", {}))

    resume_step = resume_epoch = 0
    optimizer_state = None
    if args.resume:
        log(f"\nLoading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        cfg = ckpt["config"]
        resume_step, resume_epoch = ckpt["global_step"], ckpt["epoch"]
        optimizer_state = ckpt["optimizer"]
    for key, flag, value in (
        ("local_ratio", "--ratio", args.ratio),
        ("window_size", "--window", args.window),
        ("rope_fraction", "--rope-fraction", args.rope_fraction),
        ("k_as_v", "--no-k-as-v", not args.no_k_as_v),
        ("logit_softcap", "--logit-softcap", args.logit_softcap),
    ):
        ov(cfg, key, flag, value)

    train_loader, val_loader, tokenizer, train_sampler = create_dataloaders(
        text, cfg, settings["batch_size"]
    )
    log(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    ckpt_dir = os.path.join(config.ROOT, "checkpoints")
    if is_main_process():
        saved = config.snapshot(
            ckpt_dir,
            {"model": cfg, "train": settings, "data": data_cfg, "seed": seed, "size": size},
            device=device,
        )
        log(f"Resolved config: {saved}  (rerun with --config {saved})")

    torch.manual_seed(seed)
    model = GemmaNano(cfg)
    if args.resume:
        model.load_state_dict(ckpt["model"])

    log(f"\nLayout: {' '.join(k[0].upper() for k in model.layout)}  (L = local, window {cfg['window_size']}; G = global)")
    log(
        f"Components: rope {cfg['rope_base_local']:.0f} local / {cfg['rope_base_global']:.0f} global "
        f"(p-RoPE {cfg['rope_fraction']}), k_as_v={cfg['k_as_v']}, logit_softcap={cfg['logit_softcap'] or 'off'}"
    )

    model = train(
        model,
        train_loader,
        val_loader,
        tokenizer,
        cfg,
        settings,
        device,
        resume_step=resume_step,
        resume_epoch=resume_epoch,
        optimizer_state=optimizer_state,
        train_sampler=train_sampler,
    )

    if is_main_process():
        import time

        ids = torch.tensor(tokenizer.encode(args.prompt)).unsqueeze(0).to(device)
        print(f"\n{'=' * 60}\nPrompt: {args.prompt}\n{'=' * 60}")

        torch.manual_seed(42)
        t0 = time.perf_counter()
        out1 = generate(model, ids, args.max_tokens, args.temperature, args.top_k)
        t1 = time.perf_counter() - t0
        print(f"\n[No cache] {t1:.3f}s")
        print(tokenizer.decode(out1[0].tolist()))

        torch.manual_seed(42)
        t0 = time.perf_counter()
        out2 = generate_cached(model, ids, args.max_tokens, args.temperature, args.top_k)
        t2 = time.perf_counter() - t0
        print(f"\n[Cached] {t2:.3f}s")
        print(tokenizer.decode(out2[0].tolist()))
        print(f"\nCache speedup: {t1 / t2 if t2 > 0 else float('inf'):.2f}x\n{'=' * 60}")


if __name__ == "__main__":
    main()
