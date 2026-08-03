"""
Qwen Nano — Qwen3 architecture built from scratch.

Key differences from GPT Nano:
  - RMSNorm instead of LayerNorm
  - SwiGLU feed-forward instead of GELU
  - RoPE (Rotary Position Embeddings) instead of learned positional embeddings
  - Grouped-Query Attention (GQA) with QK normalization
  - No bias in any linear layer

Sizes: nano (5M) → small (40M) → medium (130M) → qwen-0.6B (620M)

Usage:
    # Local
    python -m nano.models.qwen_nano                                     # Train nano
    python -m nano.models.qwen_nano --size small --epochs 10            # 40M model
    python -m nano.models.qwen_nano --resume checkpoints/ckpt_step_500.pt

    # Multi-GPU
    torchrun --nproc_per_node=8 -m nano.models.qwen_nano --size qwen-0.6B --batch-size 32 --grad-accum 4
"""

# Runnable either way: `python -m nano.models.qwen_nano` or `python nano/models/qwen_nano.py`.
if __package__ in (None, ""):
    import pathlib as _pathlib
    import sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import argparse
import math
import os
import urllib.request

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from nano import config

# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# ──────────────────────────────────────────────
# Distributed helpers
# ──────────────────────────────────────────────


def is_distributed():
    return dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main_process():
    return get_rank() == 0


def log(msg):
    if is_main_process():
        print(msg)


# ──────────────────────────────────────────────
# Model size presets
# ──────────────────────────────────────────────

# Hand-aligned table: columns line up so sizes can be compared down the
# page. A formatter would give every key its own line and lose that.
# fmt: off
MODEL_SIZES = {
    "nano": {       # ~5M — quick experiments
        "vocab_size": 50257,  "context_length": 128,
        "emb_dim": 64,        "n_heads": 4,      "n_layers": 4,
        "hidden_dim": 192,    "head_dim": 16,
        "n_kv_groups": 2,     "qk_norm": True,
        "rope_base": 10_000.0, "drop_rate": 0.1,
    },
    "small": {      # ~40M — single GPU
        "vocab_size": 50257,  "context_length": 256,
        "emb_dim": 256,       "n_heads": 8,      "n_layers": 8,
        "hidden_dim": 768,    "head_dim": 32,
        "n_kv_groups": 4,     "qk_norm": True,
        "rope_base": 10_000.0, "drop_rate": 0.1,
    },
    "medium": {     # ~130M — single GPU
        "vocab_size": 50257,  "context_length": 512,
        "emb_dim": 512,       "n_heads": 8,      "n_layers": 12,
        "hidden_dim": 1536,   "head_dim": 64,
        "n_kv_groups": 4,     "qk_norm": True,
        "rope_base": 100_000.0, "drop_rate": 0.1,
    },
    "large": {      # ~350M — multi-GPU recommended
        "vocab_size": 50257,  "context_length": 1024,
        "emb_dim": 1024,      "n_heads": 16,     "n_layers": 24,
        "hidden_dim": 3072,   "head_dim": 64,
        "n_kv_groups": 8,     "qk_norm": True,
        "rope_base": 500_000.0, "drop_rate": 0.1,
    },
    "qwen-0.6B": {  # ~620M — matches Qwen3-0.6B architecture
        "vocab_size": 50257,  "context_length": 2048,
        "emb_dim": 1024,      "n_heads": 16,     "n_layers": 28,
        "hidden_dim": 3072,   "head_dim": 128,
        "n_kv_groups": 8,     "qk_norm": True,
        "rope_base": 1_000_000.0, "drop_rate": 0.1,
    },
}
# fmt: on

TRAIN_SETTINGS = {
    "learning_rate": 3e-4,
    "num_epochs": 20,
    "batch_size": 8,
    "weight_decay": 0.1,
    "eval_freq": 25,
    "eval_iter": 5,
    "warmup_steps": 50,
    "grad_accum_steps": 1,
    "ckpt_freq": 500,
    "use_amp": True,
}


# ──────────────────────────────────────────────
# Dataset (same as GPT Nano)
# ──────────────────────────────────────────────


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        self.input_ids = []
        self.target_ids = []
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i : i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1 : i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloaders(text, cfg, batch_size):
    tokenizer = tiktoken.get_encoding("gpt2")
    split = int(0.9 * len(text))
    train_ds = TextDataset(
        text[:split], tokenizer, cfg["context_length"], stride=cfg["context_length"]
    )
    val_ds = TextDataset(
        text[split:], tokenizer, cfg["context_length"], stride=cfg["context_length"]
    )

    if is_distributed():
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler, drop_last=True
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler, drop_last=False)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, tokenizer, train_sampler


# ──────────────────────────────────────────────
# RMSNorm (replaces LayerNorm)
# ──────────────────────────────────────────────


class RMSNorm(nn.Module):
    """Root Mean Square Normalization — no mean subtraction, no bias.
    Cheaper than LayerNorm and works better for modern LLMs."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Upcast to float32 for numerical stability (what Qwen3 does)
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms * self.scale).to(dtype)


# ──────────────────────────────────────────────
# RoPE (Rotary Position Embeddings)
# ──────────────────────────────────────────────


def compute_rope_params(head_dim, theta_base=10_000.0, context_length=4096):
    """Precompute cos and sin tables for RoPE."""
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(context_length).float()
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # (ctx, head_dim//2)
    angles = torch.cat([angles, angles], dim=1)  # (ctx, head_dim)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """Apply rotary embeddings to Q or K tensor.
    x: (batch, heads, seq_len, head_dim)
    """
    _, _, seq_len, head_dim = x.shape
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    return ((x * cos) + (rotated * sin)).to(x.dtype)


# ──────────────────────────────────────────────
# SwiGLU Feed-Forward (replaces GELU FFN)
# ──────────────────────────────────────────────


class SwiGLUFeedForward(nn.Module):
    """SwiGLU: gate_proj and up_proj both project to hidden_dim,
    then SiLU(gate) * up is projected back down. No bias anywhere."""

    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)
        self.up_proj = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)
        self.down_proj = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ──────────────────────────────────────────────
# Grouped-Query Attention with QK Norm + RoPE
# ──────────────────────────────────────────────


class GroupedQueryAttention(nn.Module):
    """GQA: fewer KV heads shared across Q groups.
    Includes QK normalization and RoPE. No bias. KV cache support."""

    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg["n_heads"]
        self.head_dim = cfg["head_dim"]
        self.n_kv_groups = cfg["n_kv_groups"]
        self.group_size = self.n_heads // self.n_kv_groups
        self.d_out = self.n_heads * self.head_dim

        d = cfg["emb_dim"]
        self.W_query = nn.Linear(d, self.d_out, bias=False)
        self.W_key = nn.Linear(d, self.n_kv_groups * self.head_dim, bias=False)
        self.W_value = nn.Linear(d, self.n_kv_groups * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.d_out, d, bias=False)

        # QK normalization (Qwen3 feature — stabilizes training at scale)
        if cfg.get("qk_norm", False):
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = None

        self.dropout = nn.Dropout(cfg["drop_rate"])

        # Optional sigmoid output gate (Qwen3-Next / Qwen3.5 "gated attention").
        # Off by default — Qwen3 proper doesn't have it. See qwen_next_nano.py.
        self.out_gate = nn.Linear(d, self.d_out, bias=False) if cfg.get("attn_out_gate") else None
        self.pos_enc = cfg.get("pos_enc", "rope")  # "rope" | "nope"
        self.last_kv = None

        # KV cache
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.cache_seq_len = 0

    def forward(self, x, cos, sin, use_cache=False):
        B, T, _ = x.shape

        q = self.W_query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k_new = self.W_key(x).view(B, T, self.n_kv_groups, self.head_dim).transpose(1, 2)
        v_new = self.W_value(x).view(B, T, self.n_kv_groups, self.head_dim).transpose(1, 2)

        # QK normalization
        if self.q_norm:
            q = self.q_norm(q)
            k_new = self.k_norm(k_new)

        # Apply RoPE — must apply BEFORE caching (positions are baked in).
        # pos_enc="nope" skips it entirely: with a causal mask the model can still
        # infer position, and long-context extrapolation often improves.
        if self.pos_enc == "nope":
            pass
        elif use_cache and self.cache_seq_len > 0:
            # Offset cos/sin for cached positions
            q_cos = cos[self.cache_seq_len : self.cache_seq_len + T]
            q_sin = sin[self.cache_seq_len : self.cache_seq_len + T]
            q = apply_rope_offset(q, q_cos, q_sin)
            k_new = apply_rope_offset(k_new, q_cos, q_sin)
        else:
            q = apply_rope(q, cos, sin)
            k_new = apply_rope(k_new, cos, sin)

        # KV cache
        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k_new, v_new
            else:
                self.cache_k = torch.cat([self.cache_k, k_new], dim=2)
                self.cache_v = torch.cat([self.cache_v, v_new], dim=2)
            k, v = self.cache_k, self.cache_v
            self.cache_seq_len += T
        else:
            k, v = k_new, v_new

        # Kept so a later layer can reuse them — see SharedKVAttention in
        # qwen_next_nano.py. Unused unless something reads it.
        self.last_kv = (k, v)
        return self._attend(q, k, v, x)

    def _attend(self, q, k, v, x):
        """Everything after K/V are decided: group expansion, causal softmax,
        optional output gate, projection."""
        B, T_q = q.shape[0], q.shape[2]

        # Expand KV groups to match Q heads
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # Scaled dot-product attention
        T_k = k.shape[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Causal mask — the diagonal offset makes this correct for cached decode
        # too, where the query block sits at the end of the key block.
        mask = torch.triu(
            torch.ones(T_q, T_k, device=x.device, dtype=torch.bool), diagonal=T_k - T_q + 1
        )
        attn = attn.masked_fill(mask, float("-inf"))
        attn = self.dropout(torch.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T_q, self.d_out)
        if self.out_gate is not None:
            out = out * torch.sigmoid(self.out_gate(x))
        return self.out_proj(out)

    def reset_cache(self):
        self.cache_k = self.cache_v = None
        self.cache_seq_len = 0


def apply_rope_offset(x, cos_slice, sin_slice):
    """Apply RoPE with pre-sliced cos/sin (for cached generation)."""
    _, _, _, head_dim = x.shape
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    cos_slice = cos_slice.unsqueeze(0).unsqueeze(0)
    sin_slice = sin_slice.unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    return ((x * cos_slice) + (rotated * sin_slice)).to(x.dtype)


# ──────────────────────────────────────────────
# Transformer Block
# ──────────────────────────────────────────────


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.attn = GroupedQueryAttention(cfg)
        self.norm2 = RMSNorm(cfg["emb_dim"])
        self.ff = SwiGLUFeedForward(cfg)

    def forward(self, x, cos, sin, use_cache=False):
        x = x + self.attn(self.norm1(x), cos, sin, use_cache=use_cache)
        x = x + self.ff(self.norm2(x))
        return x


# ──────────────────────────────────────────────
# Qwen Nano Model
# ──────────────────────────────────────────────


class QwenNano(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # No positional embedding — RoPE is applied inside attention
        self.drop = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.norm = RMSNorm(cfg["emb_dim"])
        self.head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        # Precompute RoPE cos/sin tables
        head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(head_dim, cfg["rope_base"], cfg["context_length"])
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

    def forward(self, idx, use_cache=False):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, self.cos, self.sin, use_cache=use_cache)
        x = self.norm(x)
        return self.head(x)

    def reset_kv_cache(self):
        for block in self.blocks:
            block.attn.reset_cache()

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────


def _sample_next_token(logits, temperature, top_k):
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()
    ctx_len = model.cfg["context_length"]
    for _ in range(max_new_tokens):
        idx_crop = idx[:, -ctx_len:]
        logits = model(idx_crop)[:, -1, :]
        idx_next = _sample_next_token(logits, temperature, top_k)
        idx = torch.cat([idx, idx_next], dim=1)
    return idx


@torch.no_grad()
def generate_cached(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()
    ctx_len = model.cfg["context_length"]
    model.reset_kv_cache()

    prompt = idx[:, -ctx_len:]
    logits = model(prompt, use_cache=True)[:, -1, :]

    for _ in range(max_new_tokens):
        idx_next = _sample_next_token(logits, temperature, top_k)
        idx = torch.cat([idx, idx_next], dim=1)
        logits = model(idx_next, use_cache=True)[:, -1, :]

    return idx


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────


def get_amp_ctx(device, use_amp):
    if not use_amp:
        return torch.amp.autocast(device_type="cpu", enabled=False)
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "mps":
        return torch.amp.autocast(device_type="mps", dtype=torch.bfloat16)
    return torch.amp.autocast(device_type="cpu", enabled=False)


def calc_loss(loader, model, device, max_batches=None, amp_ctx=None):
    model.eval()
    total, count = 0.0, 0
    if amp_ctx is None:
        amp_ctx = get_amp_ctx(device, False)
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            with amp_ctx:
                out = model(x, y) if getattr(model, "needs_targets", False) else model(x)
                logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            total += loss.item()
            count += 1
    model.train()
    return total / count if count > 0 else float("nan")


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def save_checkpoint(model, optimizer, cfg, settings, global_step, epoch, ckpt_dir):
    if not is_main_process():
        return
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"qwen_ckpt_step_{global_step}.pt")
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "global_step": global_step,
            "epoch": epoch,
            "config": cfg,
            "settings": settings,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    log(f"  >> Checkpoint saved: {path}")


def train(
    model,
    train_loader,
    val_loader,
    tokenizer,
    cfg,
    settings,
    device,
    resume_step=0,
    resume_epoch=0,
    optimizer_state=None,
    train_sampler=None,
):
    model.to(device)

    if is_distributed():
        model = DDP(model, device_ids=[get_rank()])
    raw_model = model.module if isinstance(model, DDP) else model

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    needs_targets = getattr(raw_model, "needs_targets", False)
    max_steps = settings["num_epochs"] * len(train_loader)
    min_lr = settings["learning_rate"] * 0.1
    global_step = resume_step
    accum_steps = settings.get("grad_accum_steps", 1)
    ckpt_freq = settings.get("ckpt_freq", 0)
    amp_ctx = get_amp_ctx(device, settings.get("use_amp", False))
    ckpt_dir = os.path.join(config.ROOT, "checkpoints")

    effective_batch = settings["batch_size"] * accum_steps * get_world_size()
    precision = "bfloat16" if settings.get("use_amp") else "float32"

    log(f"\nTraining Qwen Nano ({raw_model.count_params():,} parameters)")
    log(
        f"  {settings['num_epochs']} epochs, {len(train_loader)} batches/epoch, {max_steps} total steps"
    )
    log(f"  Device: {device} | Precision: {precision} | GPUs: {get_world_size()}")
    log(
        f"  Batch: {settings['batch_size']} x {accum_steps} accum x {get_world_size()} GPUs = {effective_batch} effective"
    )
    if ckpt_freq > 0:
        log(f"  Checkpointing every {ckpt_freq} steps")
    if resume_step > 0:
        log(f"  Resuming from step {resume_step}, epoch {resume_epoch}")
    log("")

    for epoch in range(resume_epoch, settings["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        micro_step = 0

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for x, y in train_loader:
            if epoch == resume_epoch and micro_step < (
                resume_step - resume_epoch * len(train_loader)
            ):
                micro_step += 1
                continue

            lr = get_lr(
                global_step, settings["warmup_steps"], max_steps, settings["learning_rate"], min_lr
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            x, y = x.to(device), y.to(device)

            with amp_ctx:
                # Models with an auxiliary loss (e.g. MTP in qwen_next_nano) take the
                # targets and return it alongside the logits — it has to be computed
                # inside the DDP-wrapped forward or its grads never get synced.
                out = model(x, y) if needs_targets else model(x)
                logits, aux = out if isinstance(out, tuple) else (out, None)
                # Auxiliary losses (MTP) go into the gradient only — reported
                # losses stay pure LM loss so runs with and without them are
                # directly comparable.
                lm_loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                loss = lm_loss if aux is None else lm_loss + aux
                loss = loss / accum_steps

            loss.backward()
            epoch_loss += lm_loss.item()  # LM loss only, not the aux term
            micro_step += 1

            if micro_step % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % settings["eval_freq"] == 0:
                    val_loss = calc_loss(
                        val_loader, raw_model, device, settings["eval_iter"], amp_ctx
                    )
                    log(
                        f"  Step {global_step:5d} | train {lm_loss.item():.4f} | val {val_loss:.4f} | lr {lr:.2e}"
                    )

                if ckpt_freq > 0 and global_step % ckpt_freq == 0:
                    save_checkpoint(model, optimizer, cfg, settings, global_step, epoch, ckpt_dir)
                    if is_distributed():
                        dist.barrier()

        if micro_step % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg = epoch_loss / max(len(train_loader), 1)
        log(f"Epoch {epoch + 1}/{settings['num_epochs']} — avg train loss: {avg:.4f}")

        if is_main_process():
            prompt = "Every effort moves you"
            ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
            out_ids = generate(raw_model, ids, max_new_tokens=40, temperature=0.8, top_k=25)
            log(f"  >> {tokenizer.decode(out_ids[0].tolist())}\n")

        if is_distributed():
            dist.barrier()

    if ckpt_freq > 0:
        save_checkpoint(
            model, optimizer, cfg, settings, global_step, settings["num_epochs"], ckpt_dir
        )

    return raw_model


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def load_text(file_path=None):
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    path = os.path.join(config.ROOT, "the-verdict.txt")
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    if not os.path.exists(path):
        log("Downloading sample text...")
        with urllib.request.urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    size_choices = list(MODEL_SIZES.keys())

    parser = argparse.ArgumentParser(
        description="Train Qwen Nano (Qwen3 architecture) from scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Model sizes:\n"
        + "\n".join(
            f"  {k:12s} {v['emb_dim']}d, {v['n_heads']}h({v['n_kv_groups']}kv), "
            f"{v['n_layers']}L, ctx={v['context_length']}"
            for k, v in MODEL_SIZES.items()
        ),
    )

    parser.add_argument("--size", type=str, default="nano", choices=size_choices)
    config.add_argument(parser)
    parser.add_argument("--seed", type=int, default=42)
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

    # DDP init
    ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    text = load_text(args.file)
    log(f"Text length: {len(text):,} characters")

    # defaults < config file < flags actually typed
    file_cfg = config.load(args.config)
    ov = config.overrider()
    top = {"size": args.size, "seed": args.seed}
    top.update({k: file_cfg[k] for k in ("size", "seed") if k in file_cfg})
    ov(top, "size", "--size", args.size)
    ov(top, "seed", "--seed", args.seed)
    size, seed = top["size"], top["seed"]

    settings = {**TRAIN_SETTINGS, **file_cfg.get("train", {})}
    if args.epochs:
        settings["num_epochs"] = args.epochs
    if args.batch_size:
        settings["batch_size"] = args.batch_size
    ov(settings, "grad_accum_steps", "--grad-accum", args.grad_accum)
    ov(settings, "use_amp", "--no-amp", not args.no_amp)
    ov(settings, "ckpt_freq", "--ckpt-freq", args.ckpt_freq)

    cfg = MODEL_SIZES[size].copy()
    cfg.update(file_cfg.get("model", {}))

    # Resume
    resume_step = 0
    resume_epoch = 0
    optimizer_state = None
    if args.resume:
        log(f"\nLoading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        cfg = ckpt["config"]
        resume_step = ckpt["global_step"]
        resume_epoch = ckpt["epoch"]
        optimizer_state = ckpt["optimizer"]

    train_loader, val_loader, tokenizer, train_sampler = create_dataloaders(
        text, cfg, settings["batch_size"]
    )
    log(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    ckpt_dir = os.path.join(config.ROOT, "checkpoints")
    if is_main_process():
        saved = config.snapshot(
            ckpt_dir, {"model": cfg, "train": settings, "seed": seed, "size": size}, device=device
        )
        log(f"Resolved config: {saved}  (rerun with --config {saved})")

    torch.manual_seed(seed)
    model = QwenNano(cfg)
    if args.resume:
        model.load_state_dict(ckpt["model"])

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

    # Final generation
    if is_main_process():
        import time

        ids = torch.tensor(tokenizer.encode(args.prompt)).unsqueeze(0).to(device)

        print(f"\n{'=' * 60}")
        print(f"Prompt: {args.prompt}")
        print(f"{'=' * 60}")

        torch.manual_seed(42)
        t0 = time.perf_counter()
        out1 = generate(
            model,
            ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        t1 = time.perf_counter() - t0
        print(f"\n[No cache] {t1:.3f}s")
        print(tokenizer.decode(out1[0].tolist()))

        torch.manual_seed(42)
        t0 = time.perf_counter()
        out2 = generate_cached(
            model,
            ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        t2 = time.perf_counter() - t0
        print(f"\n[Cached] {t2:.3f}s")
        print(tokenizer.decode(out2[0].tolist()))

        speedup = t1 / t2 if t2 > 0 else float("inf")
        print(f"\nKV cache speedup: {speedup:.2f}x faster")
        print(f"{'=' * 60}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
