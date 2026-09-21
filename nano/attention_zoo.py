"""
Attention mechanism variants, shared by every model in nano/models/.

All classes share the same interface:
    __init__(cfg)
    forward(x, use_cache=False) -> Tensor
    reset_cache()

Supported types:
    - mha:      Multi-Head Attention (standard, used in GPT-2)
    - gqa:      Grouped-Query Attention (Llama 3, Qwen 3)
    - gated:    GQA + sigmoid output gate (Qwen3-Next, Qwen3.5)
    - mla:      Multi-Head Latent Attention (DeepSeek)
    - sink:     GQA + learned per-head sink logit (gpt-oss)
    - kv1:      GQA reusing K as V, half the KV cache (Gemma 4)
    - diff:     Differential Attention — two softmax maps subtracted (DIFF Transformer)
    - softcap:  GQA with c·tanh(s/c) logit softcapping (Gemma 2/3)
    - ssmax:    GQA with logits scaled by s·log(n), Scalable Softmax (2025)
    - swa:      Sliding Window Attention (Mistral, Gemma)
    - deltanet: Gated DeltaNet linear attention (Qwen3-Next)
    - kda:      Kimi Delta Attention — DeltaNet with per-channel decay (Kimi Linear)
    - lightning: Lightning Attention — linear attention, fixed per-head decay (MiniMax-01, Ling 2.5)
    - dsa:      DeepSeek Sparse Attention — MLA + lightning indexer (DeepSeek-V3.2)
    - csa:      Compressed Sparse Attention — compress 4:1, then top-k (DeepSeek-V4)
    - hca:      Heavily Compressed Attention — compress 128:1, attend densely (DeepSeek-V4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────

ATTENTION_REGISTRY = {}
ATTENTION_DESCRIPTIONS = {}


def register(name, description):
    """Add an attention variant to the zoo.

        @register("myattn", "My attention (Paper, 2026)")
        class MyAttention(nn.Module):
            def __init__(self, cfg): ...
            def forward(self, x, use_cache=False): ...   # -> (B, T, emb_dim)
            def reset_cache(self): ...

    That is the whole contract. `cfg` is the model's config dict — read what you
    need with cfg["emb_dim"], cfg["n_heads"], and cfg.get(...) for anything of
    your own, so existing configs keep working.

    Registering is all it takes to be picked up everywhere: `--attention myattn`
    in gpt_nano, `--attention all` benchmarks it, and `python -m nano.attention_zoo`
    tests it for shape, incremental-decode equivalence and causality.
    """

    def wrap(cls):
        if name in ATTENTION_REGISTRY:
            raise ValueError(f"Attention '{name}' is already registered")
        ATTENTION_REGISTRY[name] = cls
        ATTENTION_DESCRIPTIONS[name] = description
        return cls

    return wrap


# ──────────────────────────────────────────────
# 1. MHA — Multi-Head Attention (standard)
# ──────────────────────────────────────────────


@register("mha", "Multi-Head Attention (GPT-2 standard)")
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
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

        if use_cache:
            mask_bool = self.mask[self.cache_pos : self.cache_pos + T_q, :T_k].bool()
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


@register("gqa", "Grouped-Query Attention (Llama 3, Qwen 3)")
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
        # Optional: reuse K as V — see KAsVAttention below
        self.W_value = (
            None
            if cfg.get("k_as_v")
            else nn.Linear(d, self.n_kv_groups * self.head_dim, bias=cfg["qkv_bias"])
        )
        self.out_proj = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(cfg["drop_rate"])
        # Optional output gate — see GatedAttention below
        self.out_gate = nn.Linear(d, d, bias=False) if cfg.get("attn_out_gate") else None
        # Optional per-head sink logit — see SinkAttention below
        self.sink = nn.Parameter(torch.zeros(self.n_heads)) if cfg.get("attn_sink") else None
        # Optional logit softcap — see SoftcapAttention below
        self.softcap = cfg.get("attn_softcap")
        # Optional scalable-softmax scale — see ScalableSoftmaxAttention below
        self.ssmax_s = nn.Parameter(torch.full((self.n_heads,), 0.43)) if cfg.get("attn_ssmax") else None

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.cache_pos = 0

    def forward(self, x, use_cache=False):
        B, T, _ = x.shape

        q = self.W_query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k_new = self.W_key(x).view(B, T, self.n_kv_groups, self.head_dim).transpose(1, 2)
        v_new = (
            k_new
            if self.W_value is None
            else self.W_value(x).view(B, T, self.n_kv_groups, self.head_dim).transpose(1, 2)
        )

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
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        if self.softcap:
            attn = self.softcap * torch.tanh(attn / self.softcap)

        # Causal mask
        device = q.device
        if use_cache:
            q_pos = torch.arange(self.cache_pos, self.cache_pos + T_q, device=device)
            self.cache_pos += T_q
        else:
            q_pos = torch.arange(T_q, device=device)
        k_pos = torch.arange(T_k, device=device)
        if self.ssmax_s is not None:
            # n = keys visible to this row; the scale grows with log n so the
            # distribution keeps its sharpness as the context lengthens.
            log_n = torch.log((q_pos + 1).float()).view(1, 1, T_q, 1)
            scale = self.ssmax_s.view(1, -1, 1, 1) * log_n
            attn = attn * scale.to(attn.dtype)
        mask = q_pos.unsqueeze(-1) < k_pos.unsqueeze(0)
        attn = attn.masked_fill(mask, float("-inf"))

        if self.sink is not None:
            sink = self.sink.view(1, -1, 1, 1).expand(B, -1, T_q, 1)
            attn = torch.softmax(torch.cat([attn, sink], dim=-1), dim=-1)[..., :-1]
        else:
            attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T_q, self.d_out)
        if self.out_gate is not None:
            out = out * torch.sigmoid(self.out_gate(x))
        return self.out_proj(out)

    def reset_cache(self):
        self.cache_k = self.cache_v = None
        self.cache_pos = 0


@register("gated", "Gated Attention — GQA + output gate, no attention sinks (Qwen3-Next, Qwen3.5)")
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


@register("sink", "Attention sinks — GQA + learned per-head sink logit (gpt-oss)")
class SinkAttention(GroupedQueryAttention):
    """GQA plus a learned scalar per head that competes in the softmax (gpt-oss).

    Same problem Gated Attention solves — softmax must put its mass somewhere —
    but fixed on the input side instead of the output side. The logit row
    becomes [s_1 .. s_T, sink]; softmax runs over all T+1, and the sink's
    column is dropped before multiplying by V. A head with nothing to retrieve
    pushes its mass onto the sink and emits ~0, without needing token 0 as a
    dumping ground. One parameter per head. gpt-oss pairs it with alternating
    128-token sliding-window and full layers; `swa` takes the same `attn_sink`
    flag for that layout.
    """

    def __init__(self, cfg):
        super().__init__({**cfg, "attn_sink": True})


@register("kv1", "K-as-V — GQA that reuses keys as values, half the KV cache (Gemma 4)")
class KAsVAttention(GroupedQueryAttention):
    """GQA with no value projection: V is K (Gemma 4, global layers).

    GQA shrinks the cache by sharing heads; this halves what is left by
    storing one tensor per token instead of two. The head can still read
    whatever the key encodes — it just cannot store a *different* thing for
    retrieval than for matching. Gemma 4 uses it only on the sparse global
    layers, where the cache is the cost that matters, and keeps separate V
    on the sliding-window layers.

    ponytail: cache_v still holds a second reference to K rather than being
    dropped, so the code path stays identical to GQA. The memory saving is
    real in an engine that caches once; here it is a params saving only.
    """

    def __init__(self, cfg):
        super().__init__({**cfg, "k_as_v": True})


@register("softcap", "Logit softcapping — GQA with c·tanh(s/c) on attention scores (Gemma 2/3)")
class SoftcapAttention(GroupedQueryAttention):
    """GQA with attention logits squashed into (−c, c) by c·tanh(s/c) (Gemma 2, 2024).

    Attention logits have no ceiling, so a head can grow a q·k product large
    enough to make softmax exactly one-hot — a sharp, brittle distribution with
    vanishing gradient, and the dot products that get there are what overflow
    in fp16. The cap is identity near zero and saturates smoothly at ±c, so
    small logits are untouched and large ones cannot run away. Gemma 2 uses
    c=50 on attention and c=30 on the final logits; Gemma 3 dropped the
    attention cap in favour of QK-norm, which solves the same problem at the
    source. One line, no parameters.
    """

    def __init__(self, cfg):
        super().__init__({**cfg, "attn_softcap": cfg.get("attn_softcap", 50.0)})


@register("ssmax", "Scalable Softmax — GQA with logits scaled by s·log(n) (Nakanishi, 2025)")
class ScalableSoftmaxAttention(GroupedQueryAttention):
    """GQA with attention logits multiplied by s·log(n), n = number of visible keys.

    Softmax over n items with bounded logits flattens as n grows: the max
    probability decays towards 1/n, so at long context a head cannot single
    out one token however hard it tries. That is "attention fading", and it is
    a big part of why length generalisation fails. Multiplying the logits by
    log(n) exactly cancels the effect — the sharpness of the distribution
    becomes independent of how many keys there are. s is one learned scalar per
    head; the paper's trained models settle near 0.43, which is the init here.
    Zero cost, and the paper reports it improves retrieval at lengths far past
    training. arXiv 2501.19399.
    """

    def __init__(self, cfg):
        super().__init__({**cfg, "attn_ssmax": True})


@register("diff", "Differential Attention — two softmax maps subtracted, cancels noise (DIFF Transformer)")
class DifferentialAttention(nn.Module):
    """out = (softmax(q1·k1) - λ·softmax(q2·k2)) · v   (Microsoft, 2024, arXiv 2410.05258)

    Softmax attention assigns a floor of probability to every token, including
    the irrelevant ones, and that noise adds up. Two attention maps computed from
    separate Q/K projections share the same noise floor and differ in signal; the
    difference keeps the signal and cancels the noise, like a differential
    amplifier. λ is learned per layer as exp(λq1·λk1) − exp(λq2·λk2) + λ_init,
    which keeps it around λ_init at start. Each head's output is RMS-normed on
    its own (the paper's GroupNorm) and scaled by (1 − λ_init) so the residual
    contribution matches ordinary attention at init.

    Exact at λ=0: the second map is inert, so `diff` with λ=0 is plain MHA with
    a per-head norm. The self-test asserts exactly that.

    ponytail: the paper halves the head count to keep parameters equal to MHA.
    Here Q and K are simply doubled, so this entry has ~1.5× MHA's attention
    params; make `n_heads` half as big if you need a fair comparison.
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = d // self.n_heads
        self.lambda_init = cfg.get("diff_lambda_init", 0.8)
        b = cfg["qkv_bias"]
        self.W_query1, self.W_query2 = nn.Linear(d, d, bias=b), nn.Linear(d, d, bias=b)
        self.W_key1, self.W_key2 = nn.Linear(d, d, bias=b), nn.Linear(d, d, bias=b)
        self.W_value = nn.Linear(d, d, bias=b)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.norm = nn.RMSNorm(self.head_dim)
        init = lambda: nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_q1, self.lambda_k1, self.lambda_q2, self.lambda_k2 = init(), init(), init(), init()

        for name in ("cache_k1", "cache_k2", "cache_v"):
            self.register_buffer(name, None, persistent=False)
        self.cache_pos = 0

    def lam(self):
        return (
            torch.exp(self.lambda_q1 @ self.lambda_k1)
            - torch.exp(self.lambda_q2 @ self.lambda_k2)
            + self.lambda_init
        )

    def _heads(self, t, B, T):
        return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, use_cache=False):
        B, T, d = x.shape
        q1, q2 = self._heads(self.W_query1(x), B, T), self._heads(self.W_query2(x), B, T)
        k1, k2 = self._heads(self.W_key1(x), B, T), self._heads(self.W_key2(x), B, T)
        v = self._heads(self.W_value(x), B, T)

        if use_cache:
            if self.cache_k1 is None:
                self.cache_k1, self.cache_k2, self.cache_v = k1, k2, v
            else:
                self.cache_k1 = torch.cat([self.cache_k1, k1], dim=2)
                self.cache_k2 = torch.cat([self.cache_k2, k2], dim=2)
                self.cache_v = torch.cat([self.cache_v, v], dim=2)
            k1, k2, v = self.cache_k1, self.cache_k2, self.cache_v
            q_pos = torch.arange(self.cache_pos, self.cache_pos + T, device=x.device)
            self.cache_pos += T
        else:
            q_pos = torch.arange(T, device=x.device)
        mask = q_pos.unsqueeze(-1) < torch.arange(k1.shape[2], device=x.device).unsqueeze(0)

        def attend(q, k):
            s = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
            return torch.softmax(s.masked_fill(mask, float("-inf")), dim=-1)

        attn = self.dropout(attend(q1, k1) - self.lam() * attend(q2, k2))
        out = attn @ v
        # Per-head norm in fp32 and cast back, as the repo's RMSNorm does (see § 6 gotchas)
        out = F.rms_norm(out.float(), (self.head_dim,), self.norm.weight.float(), self.norm.eps)
        out = out.to(attn.dtype) * (1 - self.lambda_init)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, d))

    def reset_cache(self):
        self.cache_k1 = self.cache_k2 = self.cache_v = None
        self.cache_pos = 0


# ──────────────────────────────────────────────
# 3. MLA — Multi-Head Latent Attention
# ──────────────────────────────────────────────


@register("mla", "Multi-Head Latent Attention (DeepSeek)")
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
        self.W_DKV = nn.Linear(d, self.latent_dim, bias=cfg["qkv_bias"])  # Compress
        self.W_UK = nn.Linear(self.latent_dim, d, bias=cfg["qkv_bias"])  # Expand to K
        self.W_UV = nn.Linear(self.latent_dim, d, bias=cfg["qkv_bias"])  # Expand to V

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

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

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


@register("swa", "Sliding Window Attention (Mistral, Gemma)")
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
        # `or`, not a .get default: the hybrid CLI passes an explicit None when --window is omitted
        self.window_size = cfg.get("window_size") or cfg["context_length"] // 2
        # Optional per-head sink logit, as gpt-oss pairs with its 128-token windows
        self.sink = nn.Parameter(torch.zeros(self.n_heads)) if cfg.get("attn_sink") else None

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
            k, v = self.cache_k, self.cache_v
            # Trim to the window only *after* this call has attended. A prefill
            # longer than the window needs the early keys for its early queries;
            # trimming first left those rows with nothing but future keys, so
            # they came back NaN — while every later decode step still matched.
            if self.cache_k.shape[2] > self.window_size:
                self.cache_k = self.cache_k[:, :, -self.window_size :]
                self.cache_v = self.cache_v[:, :, -self.window_size :]
        else:
            k, v = k_new, v_new

        T_q, T_k = q.shape[2], k.shape[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

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

        if self.sink is not None:
            sink = self.sink.view(1, -1, 1, 1).expand(B, -1, T_q, 1)
            attn = torch.softmax(torch.cat([attn, sink], dim=-1), dim=-1)[..., :-1]
        else:
            attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
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
        u = x.transpose(1, 2)  # (B, dim, T)
        if use_cache and self.conv_state is not None:
            pad = self.conv_state
        else:
            pad = u.new_zeros(u.shape[0], u.shape[1], self.kernel - 1)
        u = torch.cat([pad, u], dim=-1)
        if use_cache:
            self.conv_state = u[..., -(self.kernel - 1) :]
        return F.silu(self.conv(u)).transpose(1, 2)

    def reset_cache(self):
        self.conv_state = None


@register("deltanet", "Gated DeltaNet linear attention (Qwen3-Next)")
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
        self.W_gate = nn.Linear(d, d, bias=False)  # Output gate (SiLU)
        self.W_beta = nn.Linear(d, d, bias=False)  # Update gate
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
            k_t = k[:, :, t]  # (B, H, D)
            q_t = q[:, :, t]
            v_t = v[:, :, t]
            b_t = beta[:, :, t]
            a_t = alpha[:, :, t]  # (B, H, 1, 1) scalar decay, or (B, H, D, 1) per-channel

            S = S * a_t  # Decay
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)  # Retrieve
            delta = (v_t - kv_mem) * b_t  # Delta update
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)  # Write
            y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)  # Read
            outs.append(y_t)

        context = torch.stack(outs, dim=2)  # (B, H, T, D)
        return context, S

    def forward(self, x, use_cache=False):
        B, T, _ = x.shape

        q_lin, k_lin, v_lin = self.W_query(x), self.W_key(x), self.W_value(x)
        if self.convs is not None:
            q_lin, k_lin, v_lin = (
                c(t, use_cache) for c, t in zip(self.convs, (q_lin, k_lin, v_lin))
            )

        q = q_lin.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k_lin.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v_lin.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        beta = torch.sigmoid(self.W_beta(x)).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        gate = self.W_gate(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        alpha = self._decay(x)

        # L2-normalize Q and K
        q = _l2norm(q, dim=-1) / (self.head_dim**0.5)
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


@register("kda", "Kimi Delta Attention — DeltaNet with per-channel decay (Kimi Linear)")
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
        self.W_alpha = nn.Linear(cfg["emb_dim"], d, bias=False)  # was → n_heads
        self.dt_bias = nn.Parameter(torch.ones(d))
        self.A_log = nn.Parameter(torch.log(torch.empty(d).uniform_(0, 16)))

    def _decay(self, x):
        """Per-channel decay → (B, H, T, D_k, 1), broadcast over the value axis of S."""
        B, T, _ = x.shape
        alpha_log = -self.A_log.exp().view(1, 1, -1) * F.softplus(self.W_alpha(x) + self.dt_bias)
        alpha = alpha_log.exp().view(B, T, self.n_heads, self.head_dim)
        return alpha.transpose(1, 2).unsqueeze(-1)


# ──────────────────────────────────────────────
# Lightning Attention — linear attention with fixed decay
# ──────────────────────────────────────────────


@register("lightning", "Lightning Attention — linear attention, fixed per-head decay (TransNormerLLM, MiniMax-01, Ling 2.5)")
class LightningAttention(nn.Module):
    """Linear attention with a *fixed* per-head exponential decay and no softmax.

    Same family as DeltaNet — a (head_dim × head_dim) state per head instead of a
    KV cache — but the older, simpler branch of it: no delta rule, no learned
    gate. Each head forgets at a constant rate λ_h = exp(-2^(-8(h+1)/H)), the
    ALiBi power-law slopes, so head 0 is a short buffer and head H-1 remembers
    ~everything. Out_t = Σ_j λ^(t-j) (q_t·k_j) v_j. Q and K go through SiLU so
    the kernel stays positive; the missing softmax normaliser is replaced by an
    RMSNorm on the output, and a sigmoid gate (TransNormerLLM's GLA) lets a head
    switch itself off.

    ponytail: the training path is the dense (T×T) decay-masked form and the
    cached path is a per-token recurrence. The paper's point — an O(n) blockwise
    kernel that avoids the cumsum — is a CUDA concern; the two forms here are
    exactly equal, which is what the self-test asserts.
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = d // self.n_heads
        self.W_qkv = nn.Linear(d, 3 * d, bias=cfg["qkv_bias"])
        self.W_gate = nn.Linear(d, d, bias=False)
        self.norm = nn.RMSNorm(d)
        self.out_proj = nn.Linear(d, d, bias=False)
        slope = 2.0 ** (-8.0 * torch.arange(1, self.n_heads + 1) / self.n_heads)
        self.register_buffer("slope", slope, persistent=False)
        self.register_buffer("state", None, persistent=False)

    def forward(self, x, use_cache=False):
        B, T, d = x.shape
        H, hd = self.n_heads, self.head_dim
        q, k, v = self.W_qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)  # (B, H, T, hd)
        q, k = F.silu(q), F.silu(k)

        if use_cache:
            S = self.state if self.state is not None else q.new_zeros(B, H, hd, hd)
            lam = torch.exp(-self.slope).view(1, H, 1, 1)
            outs = []
            for t in range(T):
                S = lam * S + k[:, :, t : t + 1].transpose(-1, -2) @ v[:, :, t : t + 1]
                outs.append(q[:, :, t : t + 1] @ S)
            self.state = S
            out = torch.cat(outs, dim=2)
        else:
            pos = torch.arange(T, device=x.device)
            diff = (pos.unsqueeze(-1) - pos.unsqueeze(0)).float()  # (T, T), i - j
            decay = torch.exp(-self.slope.view(H, 1, 1) * diff.clamp(min=0))
            decay = decay.masked_fill(diff < 0, 0.0)  # causal
            out = ((q @ k.transpose(-1, -2)) * decay.to(q.dtype)) @ v

        out = out.transpose(1, 2).reshape(B, T, d)
        # Normalise in fp32 and cast back, as the repo's RMSNorm does. The weight
        # is cast alongside (a no-op in fp32) so it keeps its gradient.
        out = F.rms_norm(
            out.float(), self.norm.normalized_shape, self.norm.weight.float(), self.norm.eps
        ).to(out.dtype)
        out = out * torch.sigmoid(self.W_gate(x))
        return self.out_proj(out)

    def reset_cache(self):
        self.state = None


# ──────────────────────────────────────────────
# 7. DSA — DeepSeek Sparse Attention
# ──────────────────────────────────────────────


@register("dsa", "DeepSeek Sparse Attention — MLA + lightning indexer top-k (DeepSeek-V3.2)")
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
            latent = (
                latent_new
                if self.cache_latent is None
                else torch.cat([self.cache_latent, latent_new], dim=1)
            )
            self.cache_latent = latent
        else:
            latent = latent_new
        T_k = latent.shape[1]

        q = self.W_query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_UK(latent).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_UV(latent).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)

        # Lightning indexer: score = sum_h w_h * ReLU(q_h . k_index)
        iq = self.W_iq(x).view(B, T, self.index_heads, self.index_dim)
        ik = self.W_ik(latent)  # (B, T_k, index_dim)
        w = self.W_iw(x)  # (B, T, index_heads)
        idx_scores = (F.relu(torch.einsum("bthd,bsd->bths", iq, ik)) * w.unsqueeze(-1)).sum(
            dim=2
        )  # (B, T, T_k)

        device = x.device
        q_pos = (
            torch.arange(self.cache_pos, self.cache_pos + T, device=device)
            if use_cache
            else torch.arange(T, device=device)
        )
        if use_cache:
            self.cache_pos += T
        causal = q_pos.unsqueeze(-1) < torch.arange(T_k, device=device).unsqueeze(0)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
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
                -1, scores.topk(self.top_k, dim=-1).indices, True
            )
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
        target = dense.mean(dim=1).detach()  # average over heads
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
        self.W_ck = nn.Linear(d, d, bias=False)  # per-dimension compressor, keys
        self.W_cv = nn.Linear(d, d, bias=False)  # ... and values
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
            padded = F.pad(t[:, :trunc], (0, 0, self.overlap, 0))  # (B, pad+trunc, D)
            win = padded.unfold(1, size, self.m).permute(0, 1, 3, 2)  # (B, G, size, D)
            logits = F.pad(proj(t[:, :trunc]), (0, 0, self.overlap, 0))
            logits = logits.unfold(1, size, self.m).permute(0, 1, 3, 2)
            # Softmax down the group, independently per dimension: each channel
            # picks whichever token in the group it cares about.
            return (win * torch.softmax(logits, dim=2)).sum(dim=2)  # (B, G, D)

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
        q_pos = (
            torch.arange(self.cache_pos, self.cache_pos + T, device=device)
            if use_cache
            else torch.arange(T, device=device)
        )
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
            keys.append(heads(ck))
            values.append(heads(cv))
            mask.append(comp_mask)

        k_all, v_all = torch.cat(keys, dim=2), torch.cat(values, dim=2)
        attn = (q @ k_all.transpose(-2, -1)) / (self.head_dim**0.5)
        # (T_q, N) broadcasts against attn's (B, H, T_q, N) as-is
        attn = attn.masked_fill(torch.cat(mask, dim=-1), float("-inf"))

        # CSA keeps only the top-k compressed entries; the window always stays.
        if self.select and ck is not None and ck.shape[1] > self.top_k:
            n_win = T_k
            comp = attn[..., n_win:].mean(dim=1)  # score per entry
            keep = torch.zeros_like(comp, dtype=torch.bool).scatter_(
                -1, comp.topk(self.top_k, dim=-1).indices, True
            )
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


@register("csa", "Compressed Sparse Attention — 4:1 compression + top-k (DeepSeek-V4)")
class CompressedSparseAttention(CompressedAttention):
    """CSA — mild compression (m=4) plus top-k selection (DeepSeek-V4)."""

    def __init__(self, cfg):
        super().__init__(cfg, compress_rate=cfg.get("csa_rate", 4), select=True)


@register("hca", "Heavily Compressed Attention — 128:1 compression, dense (DeepSeek-V4)")
class HeavilyCompressedAttention(CompressedAttention):
    """HCA — heavy compression (m=128 in the real model), attended densely."""

    def __init__(self, cfg):
        super().__init__(cfg, compress_rate=cfg.get("hca_rate", 16), select=False)


def collect_aux_loss(model):
    """Sum the auxiliary losses attention modules stashed during forward (DSA's
    indexer objective). Returns None if there are none, so callers can skip."""
    losses = [m.index_loss for m in model.modules() if getattr(m, "index_loss", None) is not None]
    return sum(losses) if losses else None


def get_attention(name, cfg):
    """Factory: build an attention module by name."""
    if name not in ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown attention type '{name}'. Choose from: {list(ATTENTION_REGISTRY.keys())}"
        )
    return ATTENTION_REGISTRY[name](cfg)


# ──────────────────────────────────────────────
# Self-test — python -m nano.attention_zoo
# ──────────────────────────────────────────────


def _self_test():
    torch.manual_seed(0)
    B, T, d, prefill = 2, 40, 64, 30
    cfg = {
        "emb_dim": d,
        "n_heads": 4,
        "qkv_bias": False,
        "drop_rate": 0.0,
        "context_length": 128,
        "window_size": 16,
        "top_k": 8,
        "index_dim": 16,
        "index_heads": 2,
    }
    x = torch.randn(B, T, d)

    for name in ATTENTION_REGISTRY:
        attn = get_attention(name, cfg).eval()
        full = attn(x)
        assert full.shape == (B, T, d), f"{name}: {full.shape}"
        # Prefill, then decode one token at a time — the cache (or recurrent
        # state, or conv state) must reproduce the full forward exactly.
        attn.reset_cache()
        pre = attn(x[:, :prefill], use_cache=True)
        # The prefill output itself must match too, not just the decode steps.
        # A cache that is trimmed or rewritten *before* attending returns garbage
        # for early positions while every later step still lines up — and in a
        # stacked model that garbage is the next layer's input.
        torch.testing.assert_close(pre, full[:, :prefill], atol=1e-4, rtol=1e-4)
        step = torch.cat([attn(x[:, t : t + 1], use_cache=True) for t in range(prefill, T)], dim=1)
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

        print(
            f"  {name:9s} ok — {sum(p.numel() for p in attn.parameters()):>7,} params, "
            f"incremental decode matches, no future leak"
        )

    # ShortConv is off by default; check it wires into the delta-rule layers.
    for name in ("deltanet", "kda"):
        attn = get_attention(name, {**cfg, "short_conv": 4}).eval()
        full = attn(x)
        attn.reset_cache()
        attn(x[:, :prefill], use_cache=True)
        step = torch.cat([attn(x[:, t : t + 1], use_cache=True) for t in range(prefill, T)], dim=1)
        torch.testing.assert_close(step, full[:, prefill:], atol=1e-4, rtol=1e-4)
    print("  shortconv ok — rolling conv state keeps incremental decode exact")

    # Attention sinks: with the sink logit driven high every head should drain
    # its mass into the sink and emit ~0. A sink that is appended but never
    # reached by softmax would leave the output untouched.
    attn = get_attention("sink", cfg).eval()
    base = attn(x)
    with torch.no_grad():
        attn.sink.fill_(30.0)
    drained = attn(x)
    assert drained.norm() < 1e-3 * base.norm(), f"sink did not drain: {drained.norm():.3g}"
    print("  sink      ok — high sink logit drains every head to ~0")

    # K-as-V: there must be no value projection at all, so the KV cache really
    # is half the size — not a W_value that is built and then ignored.
    attn = get_attention("kv1", cfg).eval()
    assert attn.W_value is None, "kv1 still builds W_value"
    n_gqa = sum(p.numel() for p in get_attention("gqa", cfg).parameters())
    n_kv1 = sum(p.numel() for p in attn.parameters())
    assert n_kv1 == n_gqa - attn.W_key.weight.numel(), (n_gqa, n_kv1)
    print(f"  kv1       ok — no W_value, {n_gqa - n_kv1:,} fewer params than gqa")

    # Lightning: the per-head decay must bite. Heads must differ, and with the
    # slope driven to +inf the state forgets everything before the current
    # token, so perturbing the whole past must leave output t untouched.
    attn = get_attention("lightning", cfg).eval()
    assert attn.slope.unique().numel() == attn.n_heads, "heads share a decay"
    with torch.no_grad():
        attn.slope.fill_(1e3)
    poked = x.clone()
    poked[:, :-1] += 10.0
    torch.testing.assert_close(attn(poked)[:, -1], attn(x)[:, -1], atol=1e-5, rtol=1e-5)
    print("  lightning ok — per-head decay bites, infinite slope forgets the past")

    # Explicit low-precision inference must also work without autocast. Check
    # prefill plus decode against an fp32 reference, and exercise norm gradients.
    for name in ("lightning", "ssmax"):
        for dtype, tol in ((torch.float16, 3e-3), (torch.bfloat16, 3e-2)):
            reference = get_attention(name, cfg).eval()
            attn = get_attention(name, cfg).eval().to(dtype=dtype)
            attn.load_state_dict(reference.state_dict())
            sample = x[:, :8].to(dtype)
            expected = reference(sample.float())
            full = attn(sample)
            cached = torch.cat([
                attn(sample[:, :5], use_cache=True),
                *[attn(sample[:, t:t + 1], use_cache=True) for t in range(5, 8)],
            ], dim=1)
            for result in (full, cached):
                assert result.dtype == dtype
                torch.testing.assert_close(result.float(), expected, atol=tol, rtol=tol)
            (full.float().square().mean() + cached.float().square().mean()).backward()
            for parameter in attn.parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        print(f"  {name:9s} ok — fp16/bf16 forward, cached decode and gradients without autocast")

    # Differential attention: out = (softmax(q1k1) - λ softmax(q2k2)) v. At λ=0
    # the second map must be completely inert, and once λ≠0 it must bite.
    attn = get_attention("diff", {**cfg, "diff_lambda_init": 0.0}).eval()
    with torch.no_grad():
        for prm in (attn.lambda_q1, attn.lambda_k1, attn.lambda_q2, attn.lambda_k2):
            prm.zero_()
    assert abs(attn.lam().item()) < 1e-7, f"λ should be 0, got {attn.lam().item()}"
    base = attn(x)
    with torch.no_grad():
        attn.W_key2.weight.add_(torch.randn_like(attn.W_key2.weight))
    torch.testing.assert_close(attn(x), base, atol=1e-6, rtol=0)  # inert at λ=0
    with torch.no_grad():
        attn.lambda_q1.fill_(0.1), attn.lambda_k1.fill_(0.1)
    assert not torch.allclose(attn(x), base, atol=1e-3), "second map does nothing at λ>0"
    print(f"  diff      ok — second map inert at λ=0, bites at λ={attn.lam().item():.2f}")

    # Softcapping: c·tanh(s/c) on the logits. With a huge cap it must equal
    # plain gqa on the same weights; with a small one it must flatten attention.
    ref = get_attention("gqa", cfg).eval()
    # tanh(z)≈z−z³/3, so the residual error is logits³/(3c²): c=1e9 keeps it
    # under 1e-9 even for logits in the thousands.
    loose = get_attention("softcap", {**cfg, "attn_softcap": 1e9}).eval()
    tight = get_attention("softcap", {**cfg, "attn_softcap": 0.5}).eval()
    loose.load_state_dict(ref.state_dict())
    tight.load_state_dict(ref.state_dict())
    big = x * 20  # large logits, so the cap has something to cap
    torch.testing.assert_close(loose(big), ref(big), atol=1e-5, rtol=1e-5)
    assert not torch.allclose(tight(big), ref(big), atol=1e-2), "cap of 0.5 did nothing"
    print("  softcap   ok — exact at cap→∞, bites at cap=0.5")

    # gpt-oss pairs sinks with 128-token sliding windows, so swa must take the
    # same flag: drain check plus incremental decode, since the window trims the
    # cache and the sink column must survive that.
    attn = get_attention("swa", {**cfg, "attn_sink": True}).eval()
    base = attn(x)
    attn.reset_cache()
    attn(x[:, :prefill], use_cache=True)
    step = torch.cat([attn(x[:, t : t + 1], use_cache=True) for t in range(prefill, T)], dim=1)
    torch.testing.assert_close(step, base[:, prefill:], atol=1e-4, rtol=1e-4)
    with torch.no_grad():
        attn.sink.fill_(30.0)
    drained = attn(x) - attn.out_proj.bias  # swa's out_proj has a bias, gqa's does not
    assert drained.norm() < 1e-3 * base.norm(), "swa sink did not drain"
    print("  swa+sink  ok — sink survives window trimming, drains at high logit")

    # Scalable softmax: logits scaled by s·log(n), n = keys visible to that row.
    # With s = 1/log(T) the last row's factor is exactly 1, so it must equal gqa
    # on the same weights, while every earlier row (n < T) must differ.
    ref = get_attention("gqa", cfg).eval()
    attn = get_attention("ssmax", cfg).eval()
    attn.load_state_dict(ref.state_dict(), strict=False)
    with torch.no_grad():
        attn.ssmax_s.fill_(1 / torch.log(torch.tensor(float(T))))
    out, want = attn(x), ref(x)
    torch.testing.assert_close(out[:, -1], want[:, -1], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(out[:, 1:-1], want[:, 1:-1], atol=1e-3), "ssmax scale did nothing"
    print("  ssmax     ok — row n=T matches gqa at s=1/log T, shorter rows are rescaled")

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
        poked = short.clone()
        poked[:, -1] += 10.0
        torch.testing.assert_close(attn(poked)[:, :-1], attn(short)[:, :-1], atol=1e-5, rtol=0)
        print(
            f"  {name:9s} ok — {T // m} entries at {m}:1, closing at {ends[:3].tolist()}..., "
            f"short-sequence falls back to the window"
        )

    # The compressor is a softmax down each group, so its weights are a genuine
    # weighted mean per dimension — not an unnormalised sum that could blow up.
    attn = get_attention("csa", {**cfg, "compress_rate": 4}).eval()
    with torch.no_grad():
        t = torch.randn(B, T, d)
        size, mm = 4 + 2, 4
        lg = (
            F.pad(attn.W_ck(t[:, : (T // mm) * mm]), (0, 0, 2, 0))
            .unfold(1, size, mm)
            .permute(0, 1, 3, 2)
        )
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
            dense = torch.softmax(
                ((q @ k.transpose(-2, -1)) / ((d // 4) ** 0.5)).masked_fill(causal, float("-inf")),
                -1,
            ).mean(1)
            iq = attn.W_iq(x).view(B, T, attn.index_heads, attn.index_dim)
            sc = (
                (
                    F.relu(torch.einsum("bthd,bsd->bths", iq, attn.W_ik(lat)))
                    * attn.W_iw(x).unsqueeze(-1)
                )
                .sum(2)
                .masked_fill(causal, float("-inf"))
            )
        return dense, sc

    def _recall():
        dense, sc = _dense_and_index()
        rows = slice(attn.top_k, T)  # rows with more candidates than k
        true_top = dense[:, rows].topk(attn.top_k, -1).indices
        idx_top = sc[:, rows].topk(attn.top_k, -1).indices
        return (true_top.unsqueeze(-1) == idx_top.unsqueeze(-2)).any(-1).float().mean().item()

    dense, sc = _dense_and_index()
    keep = torch.zeros_like(sc, dtype=torch.bool).scatter_(
        -1, sc.topk(attn.top_k, -1).indices, True
    )
    used = ((dense * keep) > 0).sum(-1).max().item()
    assert used <= attn.top_k, f"attended to {used} tokens, top_k={attn.top_k}"

    before = _recall()
    opt = torch.optim.AdamW([attn.W_iq.weight, attn.W_ik.weight, attn.W_iw.weight], lr=1e-2)
    for _ in range(300):  # train ONLY the indexer
        attn(x)
        loss = collect_aux_loss(attn)
        assert loss is not None and torch.isfinite(loss), "indexer objective is not finite"
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = _recall()
    assert after > before + 0.05, f"indexer did not learn: {before:.3f} -> {after:.3f}"
    print(
        f"  dsa       ok — <={used} tokens attended, indexer recall@{attn.top_k} "
        f"{before:.0%} -> {after:.0%}"
    )

    print("attention_zoo self-test passed")


if __name__ == "__main__":
    _self_test()
