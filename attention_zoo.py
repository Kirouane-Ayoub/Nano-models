"""
Attention mechanism variants for GPT Nano.

All classes share the same interface:
    __init__(cfg)
    forward(x, use_cache=False) -> Tensor
    reset_cache()

Supported types:
    - mha:      Multi-Head Attention (standard, used in GPT-2)
    - gqa:      Grouped-Query Attention (Llama 3, Qwen 3)
    - gated:    GQA + sigmoid output gate (Qwen3-Next, Qwen3.5)
    - mla:      Multi-Head Latent Attention (DeepSeek)
    - swa:      Sliding Window Attention (Mistral, Gemma)
    - deltanet: Gated DeltaNet linear attention (Qwen3-Next)
    - kda:      Kimi Delta Attention — DeltaNet with per-channel decay (Kimi Linear)
    - dsa:      DeepSeek Sparse Attention — MLA + lightning indexer (DeepSeek-V3.2)
    - csa:      Compressed Sparse Attention — compress 4:1, then top-k (DeepSeek-V4)
    - hca:      Heavily Compressed Attention — compress 128:1, attend densely (DeepSeek-V4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# 1. MHA — Multi-Head Attention (standard)
# ──────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """Standard multi-head causal self-attention with KV cache."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        assert d % cfg["n_heads"] == 0
        self.n_heads = cfg["n_heads"]
        self.head_dim = d // cfg["n_heads"]
        self.d_out = d

        self.qkv = nn.Linear(d, 3 * d, bias=cfg["qkv_bias"])
        self.out_proj = nn.Linear(d, d)
        self.attn_drop = nn.Dropout(cfg["drop_rate"])
        self.proj_drop = nn.Dropout(cfg["drop_rate"])

        self.register_buffer(
            "mask", torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1)
        )
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.cache_pos = 0

    def forward(self, x, use_cache=False):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k_new, v_new = qkv.unbind(0)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k_new, v_new
            else:
                self.cache_k = torch.cat([self.cache_k, k_new], dim=2)
                self.cache_v = torch.cat([self.cache_v, v_new], dim=2)
            k, v = self.cache_k, self.cache_v
        else:
            k, v = k_new, v_new

        T_q, T_k = q.shape[2], k.shape[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if use_cache:
            mask_bool = self.mask[self.cache_pos:self.cache_pos + T_q, :T_k].bool()
            self.cache_pos += T_q
        else:
            mask_bool = self.mask[:T_q, :T_k].bool()

        attn = attn.masked_fill(mask_bool, float("-inf"))
        attn = self.attn_drop(torch.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T_q, C)
        return self.proj_drop(self.out_proj(out))

    def reset_cache(self):
        self.cache_k = self.cache_v = None
        self.cache_pos = 0


# ──────────────────────────────────────────────
# 2. GQA — Grouped-Query Attention
# ──────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    """Fewer K/V heads shared across Q head groups. Reduces KV cache size.

    Config requires: n_kv_groups (default: n_heads // 2)
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = d // self.n_heads
        self.d_out = d

        self.n_kv_groups = cfg.get("n_kv_groups", max(1, self.n_heads // 2))
        assert self.n_heads % self.n_kv_groups == 0
        self.group_size = self.n_heads // self.n_kv_groups

        self.W_query = nn.Linear(d, d, bias=cfg["qkv_bias"])
        self.W_key = nn.Linear(d, self.n_kv_groups * self.head_dim, bias=cfg["qkv_bias"])
        self.W_value = nn.Linear(d, self.n_kv_groups * self.head_dim, bias=cfg["qkv_bias"])
        self.out_proj = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(cfg["drop_rate"])
        # Optional output gate — see GatedAttention below
        self.out_gate = nn.Linear(d, d, bias=False) if cfg.get("attn_out_gate") else None

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.cache_pos = 0

    def forward(self, x, use_cache=False):
        B, T, _ = x.shape

        q = self.W_query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k_new = self.W_key(x).view(B, T, self.n_kv_groups, self.head_dim).transpose(1, 2)
        v_new = self.W_value(x).view(B, T, self.n_kv_groups, self.head_dim).transpose(1, 2)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k_new, v_new
            else:
                self.cache_k = torch.cat([self.cache_k, k_new], dim=2)
                self.cache_v = torch.cat([self.cache_v, v_new], dim=2)
            k_base, v_base = self.cache_k, self.cache_v
        else:
            k_base, v_base = k_new, v_new

        # Expand KV groups to match number of Q heads
        k = k_base.repeat_interleave(self.group_size, dim=1)
        v = v_base.repeat_interleave(self.group_size, dim=1)

        T_q, T_k = q.shape[2], k.shape[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask
        device = q.device
        if use_cache:
            q_pos = torch.arange(self.cache_pos, self.cache_pos + T_q, device=device)
            self.cache_pos += T_q
        else:
            q_pos = torch.arange(T_q, device=device)
        k_pos = torch.arange(T_k, device=device)
        mask = q_pos.unsqueeze(-1) < k_pos.unsqueeze(0)
        attn = attn.masked_fill(mask, float("-inf"))

        attn = self.dropout(torch.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T_q, self.d_out)
        if self.out_gate is not None:
            out = out * torch.sigmoid(self.out_gate(x))
        return self.out_proj(out)

    def reset_cache(self):
        self.cache_k = self.cache_v = None
        self.cache_pos = 0


class GatedAttention(GroupedQueryAttention):
    """GQA plus a sigmoid output gate (Qwen3-Next, Qwen3.5).

    Softmax forces every query to spend all its probability mass somewhere, so
    heads with nothing to retrieve dump it on position 0 — the "attention sink",
    and the massive activations that come with it wreck low-bit quantization.
    The gate lets a head output ~0 instead, so no sink is needed. Two lines of
    code; that is the whole trick.
    """

    def __init__(self, cfg):
        super().__init__({**cfg, "attn_out_gate": True})


# ──────────────────────────────────────────────
# 3. MLA — Multi-Head Latent Attention
# ──────────────────────────────────────────────

class MultiHeadLatentAttention(nn.Module):
    """Compresses K/V into a low-dim latent, then expands per head.
    Caches the tiny latent instead of full K/V.

    Config optional: latent_dim (default: emb_dim // 4)
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = d // self.n_heads
        self.d_out = d
        self.latent_dim = cfg.get("latent_dim", max(16, d // 4))

        self.W_query = nn.Linear(d, d, bias=cfg["qkv_bias"])
        self.W_DKV = nn.Linear(d, self.latent_dim, bias=cfg["qkv_bias"])   # Compress
        self.W_UK = nn.Linear(self.latent_dim, d, bias=cfg["qkv_bias"])    # Expand to K
        self.W_UV = nn.Linear(self.latent_dim, d, bias=cfg["qkv_bias"])    # Expand to V

        self.out_proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(cfg["drop_rate"])

        self.register_buffer("cache_latent", None, persistent=False)
        self.cache_pos = 0

    def forward(self, x, use_cache=False):
        B, T, _ = x.shape

        q_all = self.W_query(x)
        latent_new = self.W_DKV(x)  # (B, T, latent_dim) — much smaller

        if use_cache:
            if self.cache_latent is None:
                latent = latent_new
            else:
                latent = torch.cat([self.cache_latent, latent_new], dim=1)
            self.cache_latent = latent
        else:
            latent = latent_new

        # Expand latent to full K/V
        k_all = self.W_UK(latent)
        v_all = self.W_UV(latent)

        # Reshape to heads
        q = q_all.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        T_k = k_all.shape[1]
        k = k_all.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = v_all.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask
        device = q.device
        if use_cache:
            q_pos = torch.arange(self.cache_pos, self.cache_pos + T, device=device)
            self.cache_pos += T
        else:
            q_pos = torch.arange(T, device=device)
        k_pos = torch.arange(T_k, device=device)
        mask = q_pos.unsqueeze(-1) < k_pos.unsqueeze(0)
        attn = attn.masked_fill(mask, float("-inf"))

        attn = self.dropout(torch.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.d_out)
        return self.out_proj(out)

    def reset_cache(self):
        self.cache_latent = None
        self.cache_pos = 0


# ──────────────────────────────────────────────
# 4. SWA — Sliding Window Attention
# ──────────────────────────────────────────────

class SlidingWindowAttention(nn.Module):
    """Only attends to a fixed window of recent tokens. O(n) memory.

    Config optional: window_size (default: context_length // 2)
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = d // self.n_heads
        self.d_out = d
        self.window_size = cfg.get("window_size", cfg["context_length"] // 2)

        self.W_query = nn.Linear(d, d, bias=cfg["qkv_bias"])
        self.W_key = nn.Linear(d, d, bias=cfg["qkv_bias"])
        self.W_value = nn.Linear(d, d, bias=cfg["qkv_bias"])
        self.out_proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(cfg["drop_rate"])

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.cache_pos = 0

    def forward(self, x, use_cache=False):
        B, T, _ = x.shape

        q = self.W_query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k_new = self.W_key(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v_new = self.W_value(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k_new, v_new
            else:
                self.cache_k = torch.cat([self.cache_k, k_new], dim=2)
                self.cache_v = torch.cat([self.cache_v, v_new], dim=2)
            # Trim cache to window size
            if self.cache_k.shape[2] > self.window_size:
                self.cache_k = self.cache_k[:, :, -self.window_size:]
                self.cache_v = self.cache_v[:, :, -self.window_size:]
            k, v = self.cache_k, self.cache_v
        else:
            k, v = k_new, v_new

        T_q, T_k = q.shape[2], k.shape[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal + sliding window mask
        device = q.device
        if use_cache:
            q_pos = torch.arange(self.cache_pos, self.cache_pos + T_q, device=device)
            k_start = max(0, self.cache_pos + T_q - T_k)
            k_pos = torch.arange(k_start, k_start + T_k, device=device)
            self.cache_pos += T_q
        else:
            q_pos = torch.arange(T_q, device=device)
            k_pos = torch.arange(T_k, device=device)

        diff = q_pos.unsqueeze(-1) - k_pos.unsqueeze(0)
        mask = (diff < 0) | (diff >= self.window_size)
        attn = attn.masked_fill(mask, float("-inf"))

        attn = self.dropout(torch.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T_q, self.d_out)
        return self.out_proj(out)

    def reset_cache(self):
        self.cache_k = self.cache_v = None
        self.cache_pos = 0


# ──────────────────────────────────────────────
# 5. GatedDeltaNet — Linear Attention
# ──────────────────────────────────────────────

def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class ShortConv(nn.Module):
    """Depthwise causal 1D convolution over the last `kernel` positions
    (Kimi Linear, LFM2.5, Inkling — all use kernel=4).

    Linear attention compresses the whole past into one fixed-size state, which
    leaves it with no cheap way to look at "the previous three tokens" — a bias
    softmax attention gets for free. This buys that back for dim*kernel params.
    Kimi Linear puts one after each of Q, K and V, followed by SiLU.

    Carries a (kernel-1)-token state so cached decoding matches a full pass.
    """

    def __init__(self, dim, kernel=4):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, bias=False)
        self.register_buffer("conv_state", None, persistent=False)

    def forward(self, x, use_cache=False):
        u = x.transpose(1, 2)                        # (B, dim, T)
        if use_cache and self.conv_state is not None:
            pad = self.conv_state
        else:
            pad = u.new_zeros(u.shape[0], u.shape[1], self.kernel - 1)
        u = torch.cat([pad, u], dim=-1)
        if use_cache:
            self.conv_state = u[..., -(self.kernel - 1):]
        return F.silu(self.conv(u)).transpose(1, 2)

    def reset_cache(self):
        self.conv_state = None


class GatedDeltaNet(nn.Module):
    """Linear attention with gated delta rule. O(n) compute, constant memory.
    No KV cache needed — uses a fixed-size recurrent state instead.
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = d // self.n_heads
        self.d_out = d

        self.W_query = nn.Linear(d, d, bias=cfg["qkv_bias"])
        self.W_key = nn.Linear(d, d, bias=cfg["qkv_bias"])
        self.W_value = nn.Linear(d, d, bias=cfg["qkv_bias"])

        # Gates
        self.W_gate = nn.Linear(d, d, bias=False)          # Output gate (SiLU)
        self.W_beta = nn.Linear(d, d, bias=False)           # Update gate
        self.W_alpha = nn.Linear(d, self.n_heads, bias=False)  # Decay gate
        self.dt_bias = nn.Parameter(torch.ones(self.n_heads))
        A_init = torch.empty(self.n_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A_init))

        # Optional ShortConv on Q/K/V (Kimi Linear). cfg["short_conv"] = kernel size.
        k_size = cfg.get("short_conv", 0)
        self.convs = nn.ModuleList([ShortConv(d, k_size) for _ in range(3)]) if k_size else None

        self.norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.out_proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(cfg["drop_rate"])

        # Recurrent state for inference
        self.register_buffer("state_S", None, persistent=False)

    def _decay(self, x):
        """Decay gate, per head → (B, H, T, 1, 1). Overridden by KDA."""
        alpha_log = -self.A_log.exp().view(1, 1, -1) * F.softplus(self.W_alpha(x) + self.dt_bias)
        return alpha_log.exp().transpose(1, 2).unsqueeze(-1).unsqueeze(-1)

    def _process_tokens(self, q, k, v, beta, alpha, S):
        """Process tokens through the delta rule, returns outputs and final state."""
        _, _, T, _ = q.shape
        outs = []
        for t in range(T):
            k_t = k[:, :, t]       # (B, H, D)
            q_t = q[:, :, t]
            v_t = v[:, :, t]
            b_t = beta[:, :, t]
            a_t = alpha[:, :, t]   # (B, H, 1, 1) scalar decay, or (B, H, D, 1) per-channel

            S = S * a_t                                             # Decay
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)            # Retrieve
            delta = (v_t - kv_mem) * b_t                            # Delta update
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)         # Write
            y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)               # Read
            outs.append(y_t)

        context = torch.stack(outs, dim=2)  # (B, H, T, D)
        return context, S

    def forward(self, x, use_cache=False):
        B, T, _ = x.shape

        q_lin, k_lin, v_lin = self.W_query(x), self.W_key(x), self.W_value(x)
        if self.convs is not None:
            q_lin, k_lin, v_lin = (c(t, use_cache) for c, t in zip(self.convs, (q_lin, k_lin, v_lin)))

        q = q_lin.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k_lin.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v_lin.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        beta = torch.sigmoid(self.W_beta(x)).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        gate = self.W_gate(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        alpha = self._decay(x)

        # L2-normalize Q and K
        q = _l2norm(q, dim=-1) / (self.head_dim ** 0.5)
        k = _l2norm(k, dim=-1)

        # Initialize or reuse recurrent state
        if use_cache and self.state_S is not None:
            S = self.state_S
        else:
            S = x.new_zeros(B, self.n_heads, self.head_dim, self.head_dim)

        context, S = self._process_tokens(q, k, v, beta, alpha, S)

        if use_cache:
            self.state_S = S

        context = context.transpose(1, 2).contiguous()
        context = context.view(B, T, self.n_heads, self.head_dim)
        context = self.norm(context)
        context = context * F.silu(gate.transpose(1, 2).contiguous())
        context = context.view(B, T, self.d_out)
        context = self.dropout(context)
        return self.out_proj(context)

    def reset_cache(self):
        self.state_S = None
        if self.convs is not None:
            for c in self.convs:
                c.reset_cache()


# ──────────────────────────────────────────────
# 6. KDA — Kimi Delta Attention
# ──────────────────────────────────────────────

class KimiDeltaAttention(GatedDeltaNet):
    """Gated DeltaNet whose decay gate is per *channel* instead of per head
    (Kimi Linear, 2025-26).

    Gated DeltaNet forgets the whole head state at one rate: a head can be a
    fast-moving local buffer or a slow long-range memory, not both. KDA gives
    each key channel its own decay, so one head can hold both. The entire
    difference from the parent class is the shape of alpha — three parameter
    reshapes and one broadcast axis.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        d = self.n_heads * self.head_dim
        self.W_alpha = nn.Linear(cfg["emb_dim"], d, bias=False)   # was → n_heads
        self.dt_bias = nn.Parameter(torch.ones(d))
        self.A_log = nn.Parameter(torch.log(torch.empty(d).uniform_(0, 16)))

    def _decay(self, x):
        """Per-channel decay → (B, H, T, D_k, 1), broadcast over the value axis of S."""
        B, T, _ = x.shape
        alpha_log = -self.A_log.exp().view(1, 1, -1) * F.softplus(self.W_alpha(x) + self.dt_bias)
        alpha = alpha_log.exp().view(B, T, self.n_heads, self.head_dim)
        return alpha.transpose(1, 2).unsqueeze(-1)


# ──────────────────────────────────────────────
# 7. DSA — DeepSeek Sparse Attention
# ──────────────────────────────────────────────

class DeepSeekSparseAttention(nn.Module):
    """MLA plus a lightning indexer that picks which tokens to attend to
    (DeepSeek-V3.2, extended into CSA/HCA in V4).

    SWA fixes the window by *position*: always the last N tokens, whether or not
    they matter. DSA picks by *content*. A cheap indexer scores every previous
    token against the query using a handful of low-dimensional heads and a ReLU
    (cheap enough to run in FP8 at scale), the top-k scoring tokens are kept, and
    full MLA attention runs over just those. O(L^2) becomes O(kL), and unlike a
    sliding window the model can still reach something 100k tokens back.

    Top-k is not differentiable, so the indexer gets no gradient from the LM
    loss. DeepSeek trains it against the dense attention distribution with a KL
    objective; this does the same, exposing it as `self.index_loss` for the
    training loop to add. Without that term the indexer never learns and this
    degenerates into a fixed random sparsity pattern.

    ponytail: computes the dense score matrix and then masks it, so it
    demonstrates the selection mechanism but delivers none of the speedup —
    that needs a gather-based sparse kernel. Fine at nano scale, where the whole
    context is usually shorter than top_k anyway and this runs dense regardless.

    Config optional: latent_dim, index_dim (32), index_heads (2), top_k (64)
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = d // self.n_heads
        self.d_out = d
        self.latent_dim = cfg.get("latent_dim", max(16, d // 4))
        self.index_dim = cfg.get("index_dim", 32)
        self.index_heads = cfg.get("index_heads", 2)
        self.top_k = cfg.get("top_k", 64)

        self.W_query = nn.Linear(d, d, bias=cfg["qkv_bias"])
        self.W_DKV = nn.Linear(d, self.latent_dim, bias=cfg["qkv_bias"])
        self.W_UK = nn.Linear(self.latent_dim, d, bias=cfg["qkv_bias"])
        self.W_UV = nn.Linear(self.latent_dim, d, bias=cfg["qkv_bias"])

        # Lightning indexer — deliberately tiny next to the attention itself
        self.W_iq = nn.Linear(d, self.index_heads * self.index_dim, bias=False)
        self.W_ik = nn.Linear(self.latent_dim, self.index_dim, bias=False)
        self.W_iw = nn.Linear(d, self.index_heads, bias=False)

        self.out_proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(cfg["drop_rate"])

        self.register_buffer("cache_latent", None, persistent=False)
        self.cache_pos = 0
        self.index_loss = None

    def forward(self, x, use_cache=False):
        B, T, _ = x.shape

        latent_new = self.W_DKV(x)
        if use_cache:
            latent = latent_new if self.cache_latent is None \
                else torch.cat([self.cache_latent, latent_new], dim=1)
            self.cache_latent = latent
        else:
            latent = latent_new
        T_k = latent.shape[1]

        q = self.W_query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_UK(latent).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_UV(latent).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)

        # Lightning indexer: score = sum_h w_h * ReLU(q_h . k_index)
        iq = self.W_iq(x).view(B, T, self.index_heads, self.index_dim)
        ik = self.W_ik(latent)                                     # (B, T_k, index_dim)
        w = self.W_iw(x)                                           # (B, T, index_heads)
        idx_scores = (F.relu(torch.einsum("bthd,bsd->bths", iq, ik))
                      * w.unsqueeze(-1)).sum(dim=2)                # (B, T, T_k)

        device = x.device
        q_pos = torch.arange(self.cache_pos, self.cache_pos + T, device=device) if use_cache \
            else torch.arange(T, device=device)
        if use_cache:
            self.cache_pos += T
        causal = q_pos.unsqueeze(-1) < torch.arange(T_k, device=device).unsqueeze(0)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.masked_fill(causal, float("-inf"))
        dense = torch.softmax(attn, dim=-1)

        # Keep the top-k tokens the indexer scored highest. Shorter than k → dense.
        if T_k > self.top_k:
            # The ReLU in the indexer makes exact-zero scores the common case, so
            # top-k ties are the norm rather than an edge case — and torch.topk
            # breaks ties by memory order, which differs between a prefill over T
            # queries and a one-token decode step. Without a deterministic rule
            # cached generation silently selects different tokens from uncached.
            # Recency is the tie-break: later tokens win.
            tie_break = torch.arange(T_k, device=device) * 1e-6
            scores = (idx_scores + tie_break).masked_fill(causal, float("-inf"))
            keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
                -1, scores.topk(self.top_k, dim=-1).indices, True)
            attn = attn.masked_fill(~keep.unsqueeze(1), float("-inf"))

        probs = self.dropout(torch.softmax(attn, dim=-1))
        self.index_loss = self._index_loss(idx_scores, dense, causal)

        out = (probs @ v).transpose(1, 2).contiguous().view(B, T, self.d_out)
        return self.out_proj(out)

    def _index_loss(self, idx_scores, dense, causal):
        """Teach the indexer to rank tokens the way attention actually weights
        them: KL(attention || indexer), attention detached as the target.

        Written as a cross-entropy — KL minus the target's entropy, which is
        constant with respect to the indexer, so the gradients are identical.
        Doing it this way sidesteps the 0*log(0) in the KL: the attention target
        is exactly zero at every masked position, and F.kl_div returns NaN there.
        Masking log_p to 0 on those positions keeps the products well-defined.
        """
        target = dense.mean(dim=1).detach()                       # average over heads
        log_p = torch.log_softmax(idx_scores.masked_fill(causal, float("-inf")), dim=-1)
        return -(target * log_p.masked_fill(causal, 0.0)).sum(dim=-1).mean()

    def reset_cache(self):
        self.cache_latent = None
        self.cache_pos = 0
        self.index_loss = None


# ──────────────────────────────────────────────
# 8. CSA / HCA — Compressed Attention
# ──────────────────────────────────────────────

class CompressedAttention(nn.Module):
    """Attention over a *compressed* sequence instead of the raw one
    (DeepSeek-V4, 2026).

    Every other variant in this file shrinks the KV cache per token — fewer
    heads (GQA), a latent (MLA), or fewer tokens (SWA/DSA). Compressed attention
    shrinks the sequence itself: merge every m tokens into a single KV entry, and
    attend to those. DeepSeek-V4 runs two flavours on alternating layers:

      csa  m=4,   then top-k selection over the compressed entries. Fine-grained,
                  keeps detail, still sparse.
      hca  m=128, no selection at all. At 1M tokens that is ~7,800 entries — few
                  enough to attend to *densely*, giving every layer a genuinely
                  global view.

    Both keep a sliding-window branch of recent raw tokens, because compression
    blurs exactly the local detail that matters most. Together they get the KV
    cache to roughly 2% of a standard transformer at a 1M context.

    Compression is a data-dependent weighted mean: a learned per-dimension
    softmax over the tokens in each group, so a group can keep whichever token
    dominates each channel rather than averaging everything into mush. Groups
    overlap, so information doesn't fragment at the boundaries.

    The subtle part is causality. A compressed entry summarising tokens
    [s, s+m) may only be read by queries at position >= s+m-1 — the group has to
    be *finished* first. Get this wrong and the model reads the future through
    the compressor while every loss curve looks perfectly healthy.

    ponytail: two honest deviations. Selection here ranks compressed entries by
    their own attention scores, where real CSA uses DSA's cheap lightning
    indexer (see DeepSeekSparseAttention) — same selection, without the saving.
    And this attends over dense masked tensors rather than caching only the
    compressed entries plus the window, so it shows the mechanism, not the 2%.

    Config optional: compress_rate, compressed_top_k, window_size
    """

    def __init__(self, cfg, compress_rate=4, select=True):
        super().__init__()
        d = cfg["emb_dim"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = d // self.n_heads
        self.d_out = d
        # Real DeepSeek-V4 uses m=4 (CSA) and m'=128 (HCA) with a 128-token
        # window. Those are scaled down here to fit nano-sized contexts.
        self.m = cfg.get("compress_rate", compress_rate)
        self.overlap = self.m // 2
        self.select = select
        self.top_k = cfg.get("compressed_top_k", 16)
        self.window = cfg.get("window_size", 16)

        self.W_query = nn.Linear(d, d, bias=cfg["qkv_bias"])
        self.W_key = nn.Linear(d, d, bias=cfg["qkv_bias"])
        self.W_value = nn.Linear(d, d, bias=cfg["qkv_bias"])
        self.W_ck = nn.Linear(d, d, bias=False)     # per-dimension compressor, keys
        self.W_cv = nn.Linear(d, d, bias=False)     # ... and values
        self.out_proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(cfg["drop_rate"])

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.cache_pos = 0

    def _compress(self, k, v):
        """Merge each group of m tokens (plus `overlap` of the previous group)
        into one KV entry. Returns entries and the position each one completes
        at — a query may only read an entry once its group has finished."""
        B, T_k, D = k.shape
        n_groups = T_k // self.m
        if n_groups == 0:
            return None, None, None

        trunc = n_groups * self.m
        size = self.m + self.overlap

        def group(t, proj):
            padded = F.pad(t[:, :trunc], (0, 0, self.overlap, 0))       # (B, pad+trunc, D)
            win = padded.unfold(1, size, self.m).permute(0, 1, 3, 2)    # (B, G, size, D)
            logits = F.pad(proj(t[:, :trunc]), (0, 0, self.overlap, 0))
            logits = logits.unfold(1, size, self.m).permute(0, 1, 3, 2)
            # Softmax down the group, independently per dimension: each channel
            # picks whichever token in the group it cares about.
            return (win * torch.softmax(logits, dim=2)).sum(dim=2)      # (B, G, D)

        ends = torch.arange(n_groups, device=k.device) * self.m + self.m - 1
        return group(k, self.W_ck), group(v, self.W_cv), ends

    def forward(self, x, use_cache=False):
        B, T, _ = x.shape

        q = self.W_query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k_new, v_new = self.W_key(x), self.W_value(x)

        if use_cache:
            self.cache_k = k_new if self.cache_k is None else torch.cat([self.cache_k, k_new], 1)
            self.cache_v = v_new if self.cache_v is None else torch.cat([self.cache_v, v_new], 1)
            k_flat, v_flat = self.cache_k, self.cache_v
        else:
            k_flat, v_flat = k_new, v_new
        T_k = k_flat.shape[1]

        device = x.device
        q_pos = torch.arange(self.cache_pos, self.cache_pos + T, device=device) if use_cache \
            else torch.arange(T, device=device)
        if use_cache:
            self.cache_pos += T

        def heads(t):
            return t.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Branch 1 — sliding window over recent raw tokens.
        k_pos = torch.arange(T_k, device=device)
        offset = q_pos.unsqueeze(-1) - k_pos.unsqueeze(0)
        win_mask = (offset < 0) | (offset >= self.window)
        keys, values, mask = [heads(k_flat)], [heads(v_flat)], [win_mask]

        # Branch 2 — compressed entries, readable only once their group closed.
        ck, cv, ends = self._compress(k_flat, v_flat)
        if ck is not None:
            comp_mask = q_pos.unsqueeze(-1) < ends.unsqueeze(0)
            keys.append(heads(ck)); values.append(heads(cv)); mask.append(comp_mask)

        k_all, v_all = torch.cat(keys, dim=2), torch.cat(values, dim=2)
        attn = (q @ k_all.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # (T_q, N) broadcasts against attn's (B, H, T_q, N) as-is
        attn = attn.masked_fill(torch.cat(mask, dim=-1), float("-inf"))

        # CSA keeps only the top-k compressed entries; the window always stays.
        if self.select and ck is not None and ck.shape[1] > self.top_k:
            n_win = T_k
            comp = attn[..., n_win:].mean(dim=1)                   # score per entry
            keep = torch.zeros_like(comp, dtype=torch.bool).scatter_(
                -1, comp.topk(self.top_k, dim=-1).indices, True)
            attn[..., n_win:] = attn[..., n_win:].masked_fill(~keep.unsqueeze(1), float("-inf"))

        # A query inside the first group has an empty compressed set and only the
        # window to attend to; that row is still valid because the window always
        # contains at least the query's own position.
        probs = self.dropout(torch.softmax(attn, dim=-1))
        out = (probs @ v_all).transpose(1, 2).contiguous().view(B, T, self.d_out)
        return self.out_proj(out)

    def reset_cache(self):
        self.cache_k = self.cache_v = None
        self.cache_pos = 0


class CompressedSparseAttention(CompressedAttention):
    """CSA — mild compression (m=4) plus top-k selection (DeepSeek-V4)."""
    def __init__(self, cfg):
        super().__init__(cfg, compress_rate=cfg.get("csa_rate", 4), select=True)


class HeavilyCompressedAttention(CompressedAttention):
    """HCA — heavy compression (m=128 in the real model), attended densely."""
    def __init__(self, cfg):
        super().__init__(cfg, compress_rate=cfg.get("hca_rate", 16), select=False)


def collect_aux_loss(model):
    """Sum the auxiliary losses attention modules stashed during forward (DSA's
    indexer objective). Returns None if there are none, so callers can skip."""
    losses = [m.index_loss for m in model.modules()
              if getattr(m, "index_loss", None) is not None]
    return sum(losses) if losses else None


# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────

ATTENTION_REGISTRY = {
    "mha": MultiHeadAttention,
    "gqa": GroupedQueryAttention,
    "gated": GatedAttention,
    "mla": MultiHeadLatentAttention,
    "swa": SlidingWindowAttention,
    "deltanet": GatedDeltaNet,
    "kda": KimiDeltaAttention,
    "dsa": DeepSeekSparseAttention,
    "csa": CompressedSparseAttention,
    "hca": HeavilyCompressedAttention,
}

ATTENTION_DESCRIPTIONS = {
    "mha": "Multi-Head Attention (GPT-2 standard)",
    "gqa": "Grouped-Query Attention (Llama 3, Qwen 3)",
    "gated": "Gated Attention — GQA + output gate, no attention sinks (Qwen3-Next, Qwen3.5)",
    "mla": "Multi-Head Latent Attention (DeepSeek)",
    "swa": "Sliding Window Attention (Mistral, Gemma)",
    "deltanet": "Gated DeltaNet linear attention (Qwen3-Next)",
    "kda": "Kimi Delta Attention — DeltaNet with per-channel decay (Kimi Linear)",
    "dsa": "DeepSeek Sparse Attention — MLA + lightning indexer top-k (DeepSeek-V3.2)",
    "csa": "Compressed Sparse Attention — 4:1 compression + top-k (DeepSeek-V4)",
    "hca": "Heavily Compressed Attention — 128:1 compression, dense (DeepSeek-V4)",
}


def get_attention(name, cfg):
    """Factory: build an attention module by name."""
    if name not in ATTENTION_REGISTRY:
        raise ValueError(f"Unknown attention type '{name}'. Choose from: {list(ATTENTION_REGISTRY.keys())}")
    return ATTENTION_REGISTRY[name](cfg)


# ──────────────────────────────────────────────
# Self-test — python attention_zoo.py
# ──────────────────────────────────────────────

def _self_test():
    torch.manual_seed(0)
    B, T, d, prefill = 2, 40, 64, 30
    cfg = {"emb_dim": d, "n_heads": 4, "qkv_bias": False, "drop_rate": 0.0,
           "context_length": 128, "window_size": 16, "top_k": 8,
           "index_dim": 16, "index_heads": 2}
    x = torch.randn(B, T, d)

    for name in ATTENTION_REGISTRY:
        attn = get_attention(name, cfg).eval()
        full = attn(x)
        assert full.shape == (B, T, d), f"{name}: {full.shape}"
        # Prefill, then decode one token at a time — the cache (or recurrent
        # state, or conv state) must reproduce the full forward exactly.
        attn.reset_cache()
        attn(x[:, :prefill], use_cache=True)
        step = torch.cat([attn(x[:, t:t + 1], use_cache=True) for t in range(prefill, T)], dim=1)
        torch.testing.assert_close(step, full[:, prefill:], atol=1e-4, rtol=1e-4)

        # Causality: changing token t must not move any output before t.
        # Nothing above catches a leak — a model that reads the future trains
        # happily and its loss curve looks unusually good, not broken. The
        # compressed variants are the real risk: an entry summarising a group
        # is only legal once that group has closed.
        attn.reset_cache()
        cut = T // 2
        poked = x.clone()
        poked[:, cut] += 10.0
        torch.testing.assert_close(attn(poked)[:, :cut], full[:, :cut], atol=1e-5, rtol=1e-5)

        print(f"  {name:9s} ok — {sum(p.numel() for p in attn.parameters()):>7,} params, "
              f"incremental decode matches, no future leak")

    # ShortConv is off by default; check it wires into the delta-rule layers.
    for name in ("deltanet", "kda"):
        attn = get_attention(name, {**cfg, "short_conv": 4}).eval()
        full = attn(x)
        attn.reset_cache()
        attn(x[:, :prefill], use_cache=True)
        step = torch.cat([attn(x[:, t:t + 1], use_cache=True) for t in range(prefill, T)], dim=1)
        torch.testing.assert_close(step, full[:, prefill:], atol=1e-4, rtol=1e-4)
    print("  shortconv ok — rolling conv state keeps incremental decode exact")

    # Compressed attention: group arithmetic, the availability rule, and the
    # short-sequence case where no group has closed yet.
    for name, m in (("csa", 4), ("hca", 8)):
        attn = get_attention(name, {**cfg, "compress_rate": m, "compressed_top_k": 3}).eval()
        ck, cv, ends = attn._compress(torch.randn(B, T, d), torch.randn(B, T, d))
        assert ck.shape == (B, T // m, d), ck.shape
        # An entry covers [s, s+m); it closes at s+m-1 and not before.
        torch.testing.assert_close(ends, torch.arange(T // m) * m + m - 1)
        # Sequence shorter than one group: nothing has closed, window only.
        short = torch.randn(B, m - 1, d)
        assert attn._compress(short, short) == (None, None, None)
        assert attn(short).shape == short.shape
        # And it must still be causal in that regime.
        poked = short.clone(); poked[:, -1] += 10.0
        torch.testing.assert_close(attn(poked)[:, :-1], attn(short)[:, :-1], atol=1e-5, rtol=0)
        print(f"  {name:9s} ok — {T // m} entries at {m}:1, closing at {ends[:3].tolist()}..., "
              f"short-sequence falls back to the window")

    # The compressor is a softmax down each group, so its weights are a genuine
    # weighted mean per dimension — not an unnormalised sum that could blow up.
    attn = get_attention("csa", {**cfg, "compress_rate": 4}).eval()
    with torch.no_grad():
        t = torch.randn(B, T, d)
        size, mm = 4 + 2, 4
        lg = F.pad(attn.W_ck(t[:, :(T // mm) * mm]), (0, 0, 2, 0)).unfold(1, size, mm).permute(0, 1, 3, 2)
        w = torch.softmax(lg, dim=2)
    torch.testing.assert_close(w.sum(dim=2), torch.ones(B, T // mm, d), atol=1e-5, rtol=0)
    print(f"  compress  ok — per-dimension weights sum to 1 down each group")

    # DSA: selection must bite, and the indexer must actually learn to rank.
    # Top-k is non-differentiable, so a broken indexer objective leaves the
    # selection random while the LM loss curve looks perfectly healthy.
    attn = get_attention("dsa", cfg)
    causal = torch.arange(T).unsqueeze(-1) < torch.arange(T).unsqueeze(0)

    def _dense_and_index():
        with torch.no_grad():
            lat = attn.W_DKV(x)
            q = attn.W_query(x).view(B, T, 4, d // 4).transpose(1, 2)
            k = attn.W_UK(lat).view(B, T, 4, d // 4).transpose(1, 2)
            dense = torch.softmax(((q @ k.transpose(-2, -1)) / ((d // 4) ** 0.5))
                                  .masked_fill(causal, float("-inf")), -1).mean(1)
            iq = attn.W_iq(x).view(B, T, attn.index_heads, attn.index_dim)
            sc = (F.relu(torch.einsum("bthd,bsd->bths", iq, attn.W_ik(lat)))
                  * attn.W_iw(x).unsqueeze(-1)).sum(2).masked_fill(causal, float("-inf"))
        return dense, sc

    def _recall():
        dense, sc = _dense_and_index()
        rows = slice(attn.top_k, T)                    # rows with more candidates than k
        true_top = dense[:, rows].topk(attn.top_k, -1).indices
        idx_top = sc[:, rows].topk(attn.top_k, -1).indices
        return (true_top.unsqueeze(-1) == idx_top.unsqueeze(-2)).any(-1).float().mean().item()

    dense, sc = _dense_and_index()
    keep = torch.zeros_like(sc, dtype=torch.bool).scatter_(-1, sc.topk(attn.top_k, -1).indices, True)
    used = ((dense * keep) > 0).sum(-1).max().item()
    assert used <= attn.top_k, f"attended to {used} tokens, top_k={attn.top_k}"

    before = _recall()
    opt = torch.optim.AdamW([attn.W_iq.weight, attn.W_ik.weight, attn.W_iw.weight], lr=1e-2)
    for _ in range(300):                               # train ONLY the indexer
        attn(x)
        loss = collect_aux_loss(attn)
        assert loss is not None and torch.isfinite(loss), "indexer objective is not finite"
        opt.zero_grad(); loss.backward(); opt.step()
    after = _recall()
    assert after > before + 0.05, f"indexer did not learn: {before:.3f} -> {after:.3f}"
    print(f"  dsa       ok — <={used} tokens attended, indexer recall@{attn.top_k} "
          f"{before:.0%} -> {after:.0%}")

    print("attention_zoo self-test passed")


if __name__ == "__main__":
    _self_test()
