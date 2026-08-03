"""
DeepSeek Nano — DeepSeek V3 architecture built from scratch.

Key features (vs GPT / Qwen):
  - Multi-Head Latent Attention (MLA): compresses K,V into tiny latent space
  - Mixture of Experts (MoE): sparse feed-forward with top-k expert routing
  - Shared Expert: one always-active expert + top-k routed experts
  - RMSNorm + SwiGLU + RoPE (same as Qwen)
  - No bias anywhere

Sizes: nano (5M) → small (50M) → medium (180M) → large (500M)

Usage:
    python deepseek_nano.py                                 # Train nano
    python deepseek_nano.py --size small --epochs 10        # 50M, MoE + MLA
    python deepseek_nano.py --resume checkpoints/ds_ckpt_step_500.pt

    # Multi-GPU
    torchrun --nproc_per_node=8 deepseek_nano.py --size large --batch-size 32 --grad-accum 4
"""

import argparse
import math
import os
import urllib.request

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import config

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

MODEL_SIZES = {
    "nano": {       # ~5M — quick experiments
        "vocab_size": 50257,  "context_length": 128,
        "emb_dim": 64,        "n_heads": 4,      "n_layers": 4,
        "head_dim": 16,       "latent_dim": 16,
        "num_experts": 4,     "num_experts_per_tok": 2,
        "expert_hidden_dim": 128,
        "shared_expert": True,
        "rope_base": 10_000.0, "drop_rate": 0.1,
    },
    "small": {      # ~50M — single GPU
        "vocab_size": 50257,  "context_length": 256,
        "emb_dim": 256,       "n_heads": 8,      "n_layers": 8,
        "head_dim": 32,       "latent_dim": 64,
        "num_experts": 8,     "num_experts_per_tok": 2,
        "expert_hidden_dim": 512,
        "shared_expert": True,
        "rope_base": 10_000.0, "drop_rate": 0.1,
    },
    "medium": {     # ~180M — single GPU
        "vocab_size": 50257,  "context_length": 512,
        "emb_dim": 512,       "n_heads": 8,      "n_layers": 12,
        "head_dim": 64,       "latent_dim": 128,
        "num_experts": 16,    "num_experts_per_tok": 2,
        "expert_hidden_dim": 1024,
        "shared_expert": True,
        "rope_base": 100_000.0, "drop_rate": 0.1,
    },
    "large": {      # ~500M — multi-GPU recommended
        "vocab_size": 50257,  "context_length": 1024,
        "emb_dim": 1024,      "n_heads": 16,     "n_layers": 24,
        "head_dim": 64,       "latent_dim": 128,
        "num_experts": 32,    "num_experts_per_tok": 4,
        "expert_hidden_dim": 2048,
        "shared_expert": True,
        "rope_base": 500_000.0, "drop_rate": 0.1,
    },
}

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
# Dataset
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
    train_ds = TextDataset(text[:split], tokenizer, cfg["context_length"], stride=cfg["context_length"])
    val_ds = TextDataset(text[split:], tokenizer, cfg["context_length"], stride=cfg["context_length"])

    if is_distributed():
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler, drop_last=False)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, tokenizer, train_sampler


# ──────────────────────────────────────────────
# RMSNorm
# ──────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms * self.scale).to(dtype)


# ──────────────────────────────────────────────
# RoPE
# ──────────────────────────────────────────────

def compute_rope_params(head_dim, theta_base=10_000.0, context_length=4096):
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(context_length).float()
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    _, _, seq_len, head_dim = x.shape
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    return ((x * cos) + (rotated * sin)).to(x.dtype)


def apply_rope_offset(x, cos_slice, sin_slice):
    _, _, _, head_dim = x.shape
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    cos_slice = cos_slice.unsqueeze(0).unsqueeze(0)
    sin_slice = sin_slice.unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    return ((x * cos_slice) + (rotated * sin_slice)).to(x.dtype)


# ──────────────────────────────────────────────
# Multi-Head Latent Attention (MLA) — DeepSeek's key innovation
# ──────────────────────────────────────────────

class MultiHeadLatentAttention(nn.Module):
    """MLA: compresses K,V into a low-dim latent, caches the tiny latent.

    Instead of caching full K,V (n_heads * head_dim per layer per token),
    we cache only the latent vector (latent_dim per layer per token).
    Then expand to K,V on the fly. ~75% KV cache savings.
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg["n_heads"]
        self.head_dim = cfg["head_dim"]
        self.d_out = self.n_heads * self.head_dim
        self.latent_dim = cfg["latent_dim"]

        d = cfg["emb_dim"]
        self.W_query = nn.Linear(d, self.d_out, bias=False)
        self.W_DKV = nn.Linear(d, self.latent_dim, bias=False)    # Compress to latent
        self.W_UK = nn.Linear(self.latent_dim, self.d_out, bias=False)   # Expand to K
        self.W_UV = nn.Linear(self.latent_dim, self.d_out, bias=False)   # Expand to V
        self.out_proj = nn.Linear(self.d_out, d, bias=False)
        self.dropout = nn.Dropout(cfg["drop_rate"])

        # Cache the tiny latent, not full K,V
        self.register_buffer("cache_latent", None, persistent=False)
        self.cache_seq_len = 0

    def forward(self, x, cos, sin, use_cache=False):
        B, T, _ = x.shape

        q_all = self.W_query(x)
        latent_new = self.W_DKV(x)  # (B, T, latent_dim) — tiny!

        if use_cache:
            if self.cache_latent is None:
                latent = latent_new
            else:
                latent = torch.cat([self.cache_latent, latent_new], dim=1)
            self.cache_latent = latent
        else:
            latent = latent_new

        # Expand latent to full K,V
        k_all = self.W_UK(latent)
        v_all = self.W_UV(latent)

        T_k = k_all.shape[1]
        q = q_all.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k_all.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = v_all.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if use_cache and self.cache_seq_len > 0:
            q_cos = cos[self.cache_seq_len : self.cache_seq_len + T]
            q_sin = sin[self.cache_seq_len : self.cache_seq_len + T]
            q = apply_rope_offset(q, q_cos, q_sin)
            # K gets RoPE for ALL positions (since we re-expand from latent each time)
            k = apply_rope(k, cos, sin)
        else:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if use_cache:
            self.cache_seq_len += T

        # Attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(T, T_k, device=x.device, dtype=torch.bool), diagonal=T_k - T + 1)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = self.dropout(torch.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.d_out)
        return self.out_proj(out)

    def reset_cache(self):
        self.cache_latent = None
        self.cache_seq_len = 0


# ──────────────────────────────────────────────
# Mixture of Experts (MoE) with Shared Expert
# ──────────────────────────────────────────────

class Expert(nn.Module):
    """Single SwiGLU expert."""
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEFeedForward(nn.Module):
    """Sparse MoE: router selects top-k experts per token.

    Options, all off unless configured:
      - shared_expert     one expert always active (DeepSeek V2/V3)
      - balance_speed     aux-loss-free load balancing (DeepSeek-V3)
      - moe_latent_dim    LatentMoE — experts run in a compressed space
                          (Nemotron 3 Super, 2026)

    **Aux-loss-free load balancing.** Routers collapse: a few experts win early,
    get all the gradient, and win harder. The usual fix is an auxiliary
    load-balancing loss, which works but fights the language-modelling loss for
    the same parameters. DeepSeek-V3 drops the extra loss and instead keeps a
    per-expert bias that is added to the router scores *for selection only*,
    nudged after every step toward whichever experts are under-used. The bias
    decides who runs; it never touches how much their output counts, so no
    gradient signal is distorted. It is updated by rule, not by backprop.

    **LatentMoE.** Project the token down to a smaller dimension, route and run
    the experts entirely in there, project the result back up. Each expert costs
    a fraction of a full-width one, so the same parameter budget buys many more
    of them — better accuracy per FLOP than a regular MoE.
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg["num_experts"]
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.emb_dim = cfg["emb_dim"]
        hidden = cfg["expert_hidden_dim"]

        # LatentMoE: everything below runs in `d`, not emb_dim
        self.moe_latent = cfg.get("moe_latent_dim", 0)
        d = self.moe_latent or cfg["emb_dim"]
        self.down = nn.Linear(cfg["emb_dim"], d, bias=False) if self.moe_latent else None
        self.up = nn.Linear(d, cfg["emb_dim"], bias=False) if self.moe_latent else None

        # Router: scores each expert for each token
        self.gate = nn.Linear(d, self.num_experts, bias=False)

        # Routed experts
        self.experts = nn.ModuleList([Expert(d, hidden) for _ in range(self.num_experts)])

        # Shared expert (always active, DeepSeek V2/V3 feature)
        self.shared_expert = Expert(d, hidden) if cfg.get("shared_expert", False) else None

        # Aux-loss-free load balancing (DeepSeek-V3 uses gamma = 1e-3)
        self.balance_speed = cfg.get("balance_speed", 1e-3)
        self.register_buffer("expert_bias", torch.zeros(self.num_experts))
        self.register_buffer("expert_load", torch.zeros(self.num_experts))

    @torch.no_grad()
    def _update_bias(self, topk_indices):
        """Raise the bias of under-loaded experts, lower it for overloaded ones.
        Sign-only, so the step size never depends on how skewed the batch was."""
        counts = torch.bincount(topk_indices.flatten(), minlength=self.num_experts).float()
        self.expert_load = counts
        self.expert_bias += self.balance_speed * torch.sign(counts.mean() - counts)

    def forward(self, x):
        if self.down is not None:
            x = self.down(x)
        B, T, D = x.shape

        # Router scores
        scores = self.gate(x)  # (B, T, num_experts)
        # The balancing bias steers *selection* only. Gating weights come from
        # the raw scores, so a bias can never inflate an expert's contribution.
        sel_scores = scores + self.expert_bias if self.balance_speed else scores
        topk_indices = torch.topk(sel_scores, self.num_experts_per_tok, dim=-1).indices
        topk_probs = torch.softmax(torch.gather(scores, -1, topk_indices), dim=-1)

        if self.training and self.balance_speed:
            self._update_bias(topk_indices)

        # Flatten for expert routing
        x_flat = x.reshape(B * T, D)
        out_flat = torch.zeros_like(x_flat)
        topk_indices_flat = topk_indices.reshape(-1, self.num_experts_per_tok)
        topk_probs_flat = topk_probs.reshape(-1, self.num_experts_per_tok)

        # Route tokens to selected experts
        for expert_id_tensor in torch.unique(topk_indices_flat):
            eid = int(expert_id_tensor.item())
            mask = topk_indices_flat == eid              # (B*T, top_k)
            token_mask = mask.any(dim=-1)                # (B*T,)
            selected_idx = token_mask.nonzero(as_tuple=False).squeeze(-1)
            if selected_idx.numel() == 0:
                continue

            expert_input = x_flat.index_select(0, selected_idx)
            expert_out = self.experts[eid](expert_input)

            # Get the routing probability for this expert
            mask_selected = mask[selected_idx]
            slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True)
            probs = torch.gather(
                topk_probs_flat.index_select(0, selected_idx), dim=-1, index=slot_indices
            ).squeeze(-1)

            # Cast to the accumulator's dtype: under autocast the expert output
            # is bf16 while softmax keeps its probabilities in fp32, and
            # index_add_ refuses mismatched types. Only reachable once the input
            # to the MoE is itself bf16, which the LatentMoE down-projection made
            # the common case.
            out_flat.index_add_(0, selected_idx,
                                (expert_out * probs.unsqueeze(-1)).to(out_flat.dtype))

        result = out_flat.reshape(B, T, D)

        # Add shared expert output (always active for every token)
        if self.shared_expert is not None:
            result = result + self.shared_expert(x)

        return self.up(result) if self.up is not None else result


# ──────────────────────────────────────────────
# Transformer Block
# ──────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.attn = MultiHeadLatentAttention(cfg)
        self.norm2 = RMSNorm(cfg["emb_dim"])
        self.ff = MoEFeedForward(cfg)

    def forward(self, x, cos, sin, use_cache=False):
        x = x + self.attn(self.norm1(x), cos, sin, use_cache=use_cache)
        x = x + self.ff(self.norm2(x))
        return x


# ──────────────────────────────────────────────
# DeepSeek Nano Model
# ──────────────────────────────────────────────

class DeepSeekNano(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.norm = RMSNorm(cfg["emb_dim"])
        self.head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.head.weight = self.tok_emb.weight

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

    def count_active_params(self):
        """Count params active per token (excludes inactive experts)."""
        total = 0
        for name, p in self.named_parameters():
            if "experts." in name:
                # Only count top-k out of num_experts
                ratio = self.cfg["num_experts_per_tok"] / self.cfg["num_experts"]
                total += int(p.numel() * ratio)
            else:
                total += p.numel()
        return total


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
        logits = model(idx[:, -ctx_len:])[:, -1, :]
        idx = torch.cat([idx, _sample_next_token(logits, temperature, top_k)], dim=1)
    return idx


@torch.no_grad()
def generate_cached(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()
    ctx_len = model.cfg["context_length"]
    model.reset_kv_cache()

    logits = model(idx[:, -ctx_len:], use_cache=True)[:, -1, :]
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
                loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
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
    path = os.path.join(ckpt_dir, f"ds_ckpt_step_{global_step}.pt")
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save({
        "global_step": global_step, "epoch": epoch,
        "config": cfg, "settings": settings,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    log(f"  >> Checkpoint saved: {path}")


def train(model, train_loader, val_loader, tokenizer, cfg, settings, device,
          resume_step=0, resume_epoch=0, optimizer_state=None, train_sampler=None):
    model.to(device)

    if is_distributed():
        model = DDP(model, device_ids=[get_rank()])
    raw_model = model.module if isinstance(model, DDP) else model

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    max_steps = settings["num_epochs"] * len(train_loader)
    min_lr = settings["learning_rate"] * 0.1
    global_step = resume_step
    accum_steps = settings.get("grad_accum_steps", 1)
    ckpt_freq = settings.get("ckpt_freq", 0)
    amp_ctx = get_amp_ctx(device, settings.get("use_amp", False))
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

    effective_batch = settings["batch_size"] * accum_steps * get_world_size()
    precision = "bfloat16" if settings.get("use_amp") else "float32"

    log(f"\nTraining DeepSeek Nano")
    log(f"  Total params: {raw_model.count_params():,}")
    log(f"  Active params/token: {raw_model.count_active_params():,} "
        f"({raw_model.count_active_params() / raw_model.count_params() * 100:.1f}%)")
    log(f"  MoE: {cfg['num_experts']} experts, top-{cfg['num_experts_per_tok']} active"
        f"{' + 1 shared' if cfg.get('shared_expert') else ''}")
    log(f"  MLA latent dim: {cfg['latent_dim']} (vs full KV: {cfg['n_heads'] * cfg['head_dim']})")
    log(f"  {settings['num_epochs']} epochs, {len(train_loader)} batches/epoch, {max_steps} total steps")
    log(f"  Device: {device} | Precision: {precision} | GPUs: {get_world_size()}")
    log(f"  Batch: {settings['batch_size']} x {accum_steps} accum x {get_world_size()} GPUs = {effective_batch} effective")
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
            if epoch == resume_epoch and micro_step < (resume_step - resume_epoch * len(train_loader)):
                micro_step += 1
                continue

            lr = get_lr(global_step, settings["warmup_steps"], max_steps,
                        settings["learning_rate"], min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            x, y = x.to(device), y.to(device)
            with amp_ctx:
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten()) / accum_steps

            loss.backward()
            epoch_loss += loss.item() * accum_steps
            micro_step += 1

            if micro_step % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % settings["eval_freq"] == 0:
                    val_loss = calc_loss(val_loader, raw_model, device, settings["eval_iter"], amp_ctx)
                    log(f"  Step {global_step:5d} | train {loss.item() * accum_steps:.4f} | val {val_loss:.4f} | lr {lr:.2e}")

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
        save_checkpoint(model, optimizer, cfg, settings, global_step, settings["num_epochs"], ckpt_dir)

    return raw_model


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def load_text(file_path=None):
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "the-verdict.txt")
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


def self_check():
    """MoE routing invariants, plus the one property that only shows up after
    training: aux-loss-free balancing actually spreading the load.

    Router collapse is silent — the loss curve of a model using 2 of its 8
    experts looks perfectly healthy.
    """
    torch.manual_seed(0)
    B, T, V = 4, 32, 96
    base = {**MODEL_SIZES["nano"], "vocab_size": V, "drop_rate": 0.0,
            "num_experts": 8, "num_experts_per_tok": 2}

    # Shapes, and LatentMoE shrinking the experts.
    x = torch.randn(B, T, base["emb_dim"])
    full = MoEFeedForward({**base, "balance_speed": 0.0})
    lat = MoEFeedForward({**base, "balance_speed": 0.0, "moe_latent_dim": base["emb_dim"] // 4})
    for m in (full, lat):
        assert m(x).shape == x.shape, m(x).shape
    n_full = sum(p.numel() for p in full.parameters())
    n_lat = sum(p.numel() for p in lat.parameters())
    assert n_lat < n_full, (n_lat, n_full)
    print(f"  latent_moe ok — {base['num_experts']} experts in "
          f"{lat.moe_latent}d instead of {base['emb_dim']}d: "
          f"{n_full:,} -> {n_lat:,} params ({n_lat / n_full:.0%})")

    # The balancing bias must steer selection without touching gate weights.
    moe = MoEFeedForward({**base, "balance_speed": 1e-3}).eval()
    with torch.no_grad():
        scores = moe.gate(x)
        plain = torch.topk(scores, moe.num_experts_per_tok, -1).indices
        moe.expert_bias[7] = 1e3                       # force expert 7 in
        biased = torch.topk(scores + moe.expert_bias, moe.num_experts_per_tok, -1).indices
        assert (biased == 7).any(-1).all(), "bias did not force selection"
        assert not (plain == 7).all(), "test is vacuous — expert 7 already always picked"
        # Gate weights come from raw scores, so the huge bias must not appear
        # in the probabilities at all.
        probs = torch.softmax(torch.gather(scores, -1, biased), -1)
        assert probs.max() < 1.0 - 1e-6, "bias leaked into the gating weights"
        moe.expert_bias.zero_()
    print("  balance_bias ok — steers selection, absent from gating weights")

    # Load balancing, measured. Train an MoE on skewed inputs and compare how
    # unevenly the experts are used with the bias on vs off.
    def spread(balance_speed, steps=200):
        torch.manual_seed(0)
        m = MoEFeedForward({**base, "balance_speed": balance_speed}).train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        data = torch.randn(B, T, base["emb_dim"])
        for _ in range(steps):
            loss = m(data).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            sel = torch.topk(m.gate(data) + m.expert_bias, m.num_experts_per_tok, -1).indices
            counts = torch.bincount(sel.flatten(), minlength=m.num_experts).float()
        return counts

    off, on = spread(0.0), spread(1e-2)
    used_off = int((off > 0).sum()); used_on = int((on > 0).sum())
    cv_off = (off.std() / off.mean()).item(); cv_on = (on.std() / on.mean()).item()
    assert used_on >= used_off, (used_off, used_on)
    assert cv_on < cv_off, f"balancing made the load *less* even: {cv_off:.2f} -> {cv_on:.2f}"
    n_e = base["num_experts"]
    print(f"  load_balance ok — experts used {used_off}/{n_e} -> {used_on}/{n_e}, "
          f"load spread {cv_off:.2f} -> {cv_on:.2f} (lower is more even)")

    print("self-check passed")


def main():
    size_choices = list(MODEL_SIZES.keys())

    parser = argparse.ArgumentParser(
        description="Train DeepSeek Nano (MLA + MoE architecture) from scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Model sizes:\n" + "\n".join(
            f"  {k:10s} {v['emb_dim']}d, {v['n_heads']}h, {v['n_layers']}L, "
            f"{v['num_experts']}E(top{v['num_experts_per_tok']}), latent={v['latent_dim']}"
            for k, v in MODEL_SIZES.items()
        )
    )

    parser.add_argument("--size", type=str, default="nano", choices=size_choices)
    config.add_argument(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-check", action="store_true",
                        help="Check MoE routing, LatentMoE and load balancing, then exit")
    parser.add_argument("--moe-latent-dim", type=int, default=0, metavar="D",
                        help="Run experts in a compressed space of D dims (LatentMoE, 0 = off)")
    parser.add_argument("--balance-speed", type=float, default=1e-3,
                        help="Aux-loss-free load-balancing bias speed (DeepSeek-V3 uses 1e-3; 0 disables)")
    parser.add_argument("--file", type=str, default=None, help="Training text file")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--ckpt-freq", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)

    args = parser.parse_args()

    if args.self_check:
        self_check()
        return

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
    cfg["moe_latent_dim"] = args.moe_latent_dim
    cfg["balance_speed"] = args.balance_speed
    cfg.update(file_cfg.get("model", {}))
    ov(cfg, "moe_latent_dim", "--moe-latent-dim", args.moe_latent_dim)
    ov(cfg, "balance_speed", "--balance-speed", args.balance_speed)

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

    train_loader, val_loader, tokenizer, train_sampler = create_dataloaders(text, cfg, settings["batch_size"])
    log(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    if is_main_process():
        saved = config.snapshot(ckpt_dir, {"model": cfg, "train": settings,
                                           "seed": seed, "size": size}, device=device)
        log(f"Resolved config: {saved}  (rerun with --config {saved})")

    torch.manual_seed(seed)
    model = DeepSeekNano(cfg)
    if args.resume:
        model.load_state_dict(ckpt["model"])

    model = train(model, train_loader, val_loader, tokenizer, cfg, settings, device,
                  resume_step=resume_step, resume_epoch=resume_epoch,
                  optimizer_state=optimizer_state, train_sampler=train_sampler)

    if is_main_process():
        import time
        ids = torch.tensor(tokenizer.encode(args.prompt)).unsqueeze(0).to(device)

        print(f"\n{'='*60}")
        print(f"Prompt: {args.prompt}")
        print(f"{'='*60}")

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
        print(f"\n[Cached (MLA latent)] {t2:.3f}s")
        print(tokenizer.decode(out2[0].tolist()))

        speedup = t1 / t2 if t2 > 0 else float("inf")
        print(f"\nMLA cache speedup: {speedup:.2f}x faster")
        print(f"{'='*60}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
