"""
GPT Nano — A minimal GPT built from scratch.

Architecture: Decoder-only transformer with causal self-attention
Sizes:       nano (3M) → small (30M) → medium (100M) → gpt2 (124M) → large (350M) → xl (770M)

Usage:
    # Local (single GPU / MPS)
    python -m nano.models.gpt_nano                                  # Train nano with MHA
    python -m nano.models.gpt_nano --size small --attention gqa     # 30M with Grouped-Query
    python -m nano.models.gpt_nano --attention all --epochs 5       # Benchmark all attention types
    python -m nano.models.gpt_nano --resume checkpoints/ckpt_step_500.pt  # Resume

    # Multi-GPU (8xB200 cluster)
    torchrun --nproc_per_node=8 -m nano.models.gpt_nano --size large --batch-size 32 --grad-accum 4
    torchrun --nproc_per_node=8 -m nano.models.gpt_nano --size xl --attention gqa --epochs 20
"""

# Runnable either way: `python -m nano.models.gpt_nano` or `python nano/models/gpt_nano.py`.
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
from torch.utils.data import Dataset, DataLoader

from nano import config, data

# DDP imports — only used when launched via torchrun
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from nano.attention_zoo import (
    get_attention,
    collect_aux_loss,
    ATTENTION_REGISTRY,
    ATTENTION_DESCRIPTIONS,
)


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
    """Print only on rank 0."""
    if is_main_process():
        print(msg)


# ──────────────────────────────────────────────
# Model size presets
# ──────────────────────────────────────────────

# Hand-aligned table: columns line up so sizes can be compared down the
# page. A formatter would give every key its own line and lose that.
# fmt: off
MODEL_SIZES = {
    "nano": {   # ~3.4M — CPU/MPS in minutes
        "vocab_size": 50257,  "context_length": 128,
        "emb_dim": 64,        "n_heads": 4,     "n_layers": 4,
        "drop_rate": 0.1,     "qkv_bias": False,
    },
    "small": {  # ~30M — single GPU
        "vocab_size": 50257,  "context_length": 256,
        "emb_dim": 256,       "n_heads": 8,     "n_layers": 8,
        "drop_rate": 0.1,     "qkv_bias": False,
    },
    "medium": { # ~100M — single GPU
        "vocab_size": 50257,  "context_length": 512,
        "emb_dim": 512,       "n_heads": 8,     "n_layers": 12,
        "drop_rate": 0.1,     "qkv_bias": False,
    },
    "gpt2": {   # ~124M — matches GPT-2 small
        "vocab_size": 50257,  "context_length": 1024,
        "emb_dim": 768,       "n_heads": 12,    "n_layers": 12,
        "drop_rate": 0.1,     "qkv_bias": False,
    },
    "large": {  # ~350M — multi-GPU recommended
        "vocab_size": 50257,  "context_length": 1024,
        "emb_dim": 1024,      "n_heads": 16,    "n_layers": 24,
        "drop_rate": 0.1,     "qkv_bias": False,
    },
    "xl": {     # ~770M — multi-GPU, matches GPT-2 large
        "vocab_size": 50257,  "context_length": 1024,
        "emb_dim": 1280,      "n_heads": 20,    "n_layers": 36,
        "drop_rate": 0.1,     "qkv_bias": False,
    },
}
# fmt: on

TRAIN_SETTINGS = {
    "learning_rate": 5e-4,
    "num_epochs": 20,
    "batch_size": 8,
    "weight_decay": 0.1,
    "eval_freq": 25,  # Evaluate every N steps
    "eval_iter": 5,  # Batches to average for eval loss
    "warmup_steps": 50,  # Linear warmup steps
    "grad_accum_steps": 1,  # Gradient accumulation steps
    "ckpt_freq": 500,  # Save checkpoint every N steps (0 = off)
    "use_amp": True,  # Mixed precision (bfloat16)
}


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────


class TextDataset(Dataset):
    """Sliding-window dataset: each sample is (input_ids, target_ids) where
    target is input shifted right by 1."""

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

    # Use DistributedSampler when running multi-GPU
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
# Model components
# ──────────────────────────────────────────────


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            nn.Dropout(cfg["drop_rate"]),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.attn = get_attention(cfg.get("attention", "mha"), cfg)
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.ff = FeedForward(cfg)

    def forward(self, x, use_cache=False):
        x = x + self.attn(self.norm1(x), use_cache=use_cache)
        x = x + self.ff(self.norm2(x))
        return x


# ──────────────────────────────────────────────
# GPT Nano model
# ──────────────────────────────────────────────


class GPTNano(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])
        # ModuleList (not Sequential) so we can pass use_cache to each block
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.norm = LayerNorm(cfg["emb_dim"])
        self.head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # Weight tying: share token embedding weights with output head
        self.head.weight = self.tok_emb.weight

        # Tracks absolute position during cached generation
        self.current_pos = 0

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
        tok = self.tok_emb(idx)

        # Position IDs: offset by current_pos when using cache
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + T, device=idx.device)
            self.current_pos += T
        else:
            pos_ids = torch.arange(T, device=idx.device)

        pos = self.pos_emb(pos_ids)
        x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        x = self.norm(x)
        return self.head(x)

    def reset_kv_cache(self):
        """Reset all attention caches — call before each new generation."""
        for block in self.blocks:
            if hasattr(block.attn, "reset_cache"):
                block.attn.reset_cache()
        self.current_pos = 0

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────


def _sample_next_token(logits, temperature, top_k):
    """Pick the next token from logits using temperature + optional top-k."""
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """Autoregressive generation WITHOUT KV cache (recomputes everything each step)."""
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
    """Autoregressive generation WITH KV cache — much faster for long sequences.

    Step 1: Feed the full prompt, fill the cache.
    Step 2: Feed only 1 new token per step, reusing cached K/V.
    """
    model.eval()
    ctx_len = model.cfg["context_length"]
    model.reset_kv_cache()

    # Prefill: process the entire prompt at once, populating the cache
    prompt = idx[:, -ctx_len:]
    logits = model(prompt, use_cache=True)[:, -1, :]

    for _ in range(max_new_tokens):
        idx_next = _sample_next_token(logits, temperature, top_k)
        idx = torch.cat([idx, idx_next], dim=1)
        # Decode: only the new token goes through the model
        logits = model(idx_next, use_cache=True)[:, -1, :]

    return idx


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────


def get_amp_ctx(device, use_amp):
    """Return the appropriate autocast context for mixed precision."""
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
                loss = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())
            total += loss.item()
            count += 1
    model.train()
    return total / count if count > 0 else float("nan")


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def save_checkpoint(model, optimizer, cfg, settings, global_step, epoch, ckpt_dir):
    """Save a resumable checkpoint (rank 0 only in DDP)."""
    if not is_main_process():
        return None
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_step_{global_step}.pt")
    # Unwrap DDP to save the raw model weights
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
    return path


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

    # ── Wrap model in DDP if distributed ──
    if is_distributed():
        model = DDP(model, device_ids=[get_rank()])
    raw_model = model.module if isinstance(model, DDP) else model

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    max_steps = settings["num_epochs"] * len(train_loader)
    min_lr = settings["learning_rate"] * 0.1
    global_step = resume_step
    accum_steps = settings.get("grad_accum_steps", 1)
    ckpt_freq = settings.get("ckpt_freq", 0)
    use_amp = settings.get("use_amp", False)
    amp_ctx = get_amp_ctx(device, use_amp)
    ckpt_dir = os.path.join(config.ROOT, "checkpoints")

    effective_batch = settings["batch_size"] * accum_steps * get_world_size()
    precision = "bfloat16" if use_amp else "float32"

    log(f"\nTraining GPT Nano ({raw_model.count_params():,} parameters)")
    log(
        f"  {settings['num_epochs']} epochs, {len(train_loader)} batches/epoch, {max_steps} total steps"
    )
    log(f"  Device: {device} | Precision: {precision} | GPUs: {get_world_size()}")
    log(
        f"  Batch: {settings['batch_size']} x {accum_steps} accum x {get_world_size()} GPUs = {effective_batch} effective"
    )
    if ckpt_freq > 0:
        log(f"  Checkpointing every {ckpt_freq} steps → {ckpt_dir}")
    if resume_step > 0:
        log(f"  Resuming from step {resume_step}, epoch {resume_epoch}")
    log("")

    for epoch in range(resume_epoch, settings["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        micro_step = 0

        # ── Tell DistributedSampler which epoch we're on for shuffling ──
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for x, y in train_loader:
            # Skip steps already done when resuming mid-epoch
            if epoch == resume_epoch and micro_step < (
                resume_step - resume_epoch * len(train_loader)
            ):
                micro_step += 1
                continue

            # Update learning rate
            lr = get_lr(
                global_step, settings["warmup_steps"], max_steps, settings["learning_rate"], min_lr
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            x, y = x.to(device), y.to(device)

            # ── Mixed precision forward ──
            with amp_ctx:
                logits = model(x)
                lm_loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
                # DSA's indexer is trained by its own objective — top-k selection
                # is not differentiable, so no gradient reaches it from the LM
                # loss. Without this the indexer stays at init and the sparsity
                # pattern is effectively random. It goes into the gradient only:
                # reported losses stay pure LM loss, comparable across variants.
                aux = collect_aux_loss(raw_model)
                loss = lm_loss if aux is None else lm_loss + aux
                loss = loss / accum_steps  # Scale loss for accumulation

            loss.backward()

            epoch_loss += lm_loss.item()  # Report LM loss only, not the aux term
            micro_step += 1

            # ── Gradient accumulation: only step every accum_steps ──
            if micro_step % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Periodic evaluation (rank 0 only)
                if global_step % settings["eval_freq"] == 0:
                    val_loss = calc_loss(
                        val_loader, raw_model, device, settings["eval_iter"], amp_ctx
                    )
                    log(
                        f"  Step {global_step:5d} | train {lm_loss.item():.4f} | val {val_loss:.4f} | lr {lr:.2e}"
                    )

                # ── Checkpointing (rank 0 only) ──
                if ckpt_freq > 0 and global_step % ckpt_freq == 0:
                    save_checkpoint(model, optimizer, cfg, settings, global_step, epoch, ckpt_dir)
                    if is_distributed():
                        dist.barrier()  # All ranks wait for rank 0 to finish saving

        # Flush remaining gradients if batches don't divide evenly by accum_steps
        if micro_step % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg = epoch_loss / max(len(train_loader), 1)
        log(f"Epoch {epoch + 1}/{settings['num_epochs']} — avg train loss: {avg:.4f}")

        # Generate a sample after each epoch (rank 0 only, use raw model)
        if is_main_process():
            prompt = "Every effort moves you"
            ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
            out_ids = generate(raw_model, ids, max_new_tokens=40, temperature=0.8, top_k=25)
            log(f"  >> {tokenizer.decode(out_ids[0].tolist())}\n")

        if is_distributed():
            dist.barrier()

    # Save final checkpoint
    if ckpt_freq > 0:
        save_checkpoint(
            model, optimizer, cfg, settings, global_step, settings["num_epochs"], ckpt_dir
        )

    return raw_model


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


load_text = data.load_corpus  # kept for backwards compatibility


def run_single(attn_type, text, settings, device, args, base_cfg):
    """Train and evaluate a single attention variant."""
    import time

    cfg = base_cfg.copy()
    cfg["attention"] = attn_type

    desc = ATTENTION_DESCRIPTIONS[attn_type]
    log(f"\n{'#' * 60}")
    log(f"  Attention: {attn_type} — {desc}")
    log(f"{'#' * 60}")

    train_loader, val_loader, tokenizer, train_sampler = create_dataloaders(
        text, cfg, settings["batch_size"]
    )
    log(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    torch.manual_seed(args.seed)
    model = GPTNano(cfg)
    model = train(
        model,
        train_loader,
        val_loader,
        tokenizer,
        cfg,
        settings,
        device,
        train_sampler=train_sampler,
    )

    # Generation benchmark (rank 0 only)
    if not is_main_process():
        return {}
    amp_ctx = get_amp_ctx(device, settings.get("use_amp", False))
    ids = torch.tensor(tokenizer.encode(args.prompt)).unsqueeze(0).to(device)

    print(f"\nPrompt: {args.prompt}")

    # Without cache
    torch.manual_seed(42)
    t0 = time.perf_counter()
    out1 = generate(
        model, ids, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k
    )
    t_no_cache = time.perf_counter() - t0

    # With cache
    torch.manual_seed(42)
    t0 = time.perf_counter()
    out2 = generate_cached(
        model, ids, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k
    )
    t_cached = time.perf_counter() - t0

    print(f"\n[No cache] {t_no_cache:.3f}s")
    print(tokenizer.decode(out1[0].tolist()))
    print(f"\n[Cached]   {t_cached:.3f}s")
    print(tokenizer.decode(out2[0].tolist()))

    speedup = t_no_cache / t_cached if t_cached > 0 else float("inf")
    print(f"\nCache speedup: {speedup:.2f}x")

    return {
        "attention": attn_type,
        "params": model.count_params(),
        "final_train_loss": calc_loss(train_loader, model, device, max_batches=5, amp_ctx=amp_ctx),
        "final_val_loss": calc_loss(val_loader, model, device, max_batches=5, amp_ctx=amp_ctx),
        "gen_time_no_cache": t_no_cache,
        "gen_time_cached": t_cached,
        "cache_speedup": speedup,
    }


def main():
    attn_choices = list(ATTENTION_REGISTRY.keys()) + ["all"]
    size_choices = list(MODEL_SIZES.keys())

    parser = argparse.ArgumentParser(
        description="Train GPT Nano from scratch — scalable, mixed precision, checkpointed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Model sizes:\n"
            + "\n".join(
                f"  {k:8s} {v['emb_dim']}d, {v['n_heads']}h, {v['n_layers']}L, ctx={v['context_length']}"
                for k, v in MODEL_SIZES.items()
            )
            + "\n\nAttention types:\n"
            + "\n".join(f"  {k:10s} {v}" for k, v in ATTENTION_DESCRIPTIONS.items())
            + "\n  all        Benchmark all variants side by side"
        ),
    )

    # Model
    parser.add_argument(
        "--size",
        type=str,
        default="nano",
        choices=size_choices,
        help="Model size preset (default: nano)",
    )
    parser.add_argument(
        "--attention",
        type=str,
        default="mha",
        choices=attn_choices,
        help="Attention mechanism (default: mha)",
    )
    config.add_argument(parser)
    parser.add_argument("--seed", type=int, default=42)

    # Training
    data.add_arguments(parser)
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)",
    )
    parser.add_argument(
        "--no-amp", action="store_true", help="Disable mixed precision (use float32)"
    )
    parser.add_argument(
        "--ckpt-freq", type=int, default=500, help="Save checkpoint every N steps (0 = off)"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume training from"
    )

    # Generation
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Generation prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")

    args = parser.parse_args()

    # ── Initialize DDP if launched via torchrun ──
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

    # Load text
    # Build settings
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

    # Build model config from size preset
    base_cfg = MODEL_SIZES[size].copy()
    base_cfg.update(file_cfg.get("model", {}))
    ov(base_cfg, "attention", "--attention", args.attention)

    # ── Resume from checkpoint ──
    resume_step = 0
    resume_epoch = 0
    optimizer_state = None

    if args.resume:
        print(f"\nLoading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        base_cfg = ckpt["config"]
        resume_step = ckpt["global_step"]
        resume_epoch = ckpt["epoch"]
        optimizer_state = ckpt["optimizer"]
        if "settings" in ckpt:
            # Restore training settings but allow CLI overrides
            for k, v in ckpt["settings"].items():
                if k not in settings or settings[k] == TRAIN_SETTINGS.get(k):
                    settings[k] = v
        print(
            f"  Resuming: step={resume_step}, epoch={resume_epoch}, "
            f"attention={base_cfg.get('attention', 'mha')}"
        )

    if is_main_process():
        ckpt_dir = os.path.join(config.ROOT, "checkpoints")
        saved = config.snapshot(
            ckpt_dir,
            {"model": base_cfg, "train": settings, "data": data_cfg, "seed": seed, "size": size},
            device=device,
        )
        log(f"Resolved config: {saved}  (rerun with --config {saved})")

    # ── Benchmark all attention types ──
    if args.attention == "all":
        types_to_run = list(ATTENTION_REGISTRY.keys())
        results = []
        for attn_type in types_to_run:
            result = run_single(attn_type, text, settings, device, args, base_cfg)
            results.append(result)

        if is_main_process():
            # Filter out empty results from non-rank-0 processes
            results = [r for r in results if r]
            print(f"\n{'=' * 85}")
            print(
                f"  BENCHMARK — {args.size} model, {settings['num_epochs']} epochs, "
                f"{'bf16' if settings['use_amp'] else 'fp32'}, device: {device}, GPUs: {get_world_size()}"
            )
            print(f"{'=' * 85}")
            print(
                f"{'Attention':<12} {'Params':>10} {'Train Loss':>12} {'Val Loss':>10} "
                f"{'No Cache':>10} {'Cached':>10} {'Speedup':>9}"
            )
            print(f"{'-' * 12} {'-' * 10} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 9}")
            for r in results:
                print(
                    f"{r['attention']:<12} "
                    f"{r['params']:>10,} "
                    f"{r['final_train_loss']:>12.4f} "
                    f"{r['final_val_loss']:>10.4f} "
                    f"{r['gen_time_no_cache']:>9.3f}s "
                    f"{r['gen_time_cached']:>9.3f}s "
                    f"{r['cache_speedup']:>8.2f}x"
                )
            print(f"{'=' * 85}")

        if ddp:
            dist.destroy_process_group()
        return

    # ── Single attention type ──
    cfg = base_cfg.copy()
    cfg["attention"] = args.attention

    train_loader, val_loader, tokenizer, train_sampler = create_dataloaders(
        text, cfg, settings["batch_size"]
    )
    log(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    torch.manual_seed(seed)
    model = GPTNano(cfg)

    # Load model weights if resuming
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

    # Final generation (rank 0 only)
    if is_main_process():
        import time

        ids = torch.tensor(tokenizer.encode(args.prompt)).unsqueeze(0).to(device)

        print(f"\n{'=' * 60}")
        print(f"Prompt: {args.prompt}")
        print(f"{'=' * 60}")

        torch.manual_seed(42)
        t0 = time.perf_counter()
        out_no_cache = generate(
            model,
            ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        t_no_cache = time.perf_counter() - t0
        print(f"\n[No KV cache] {t_no_cache:.3f}s")
        print(tokenizer.decode(out_no_cache[0].tolist()))

        torch.manual_seed(42)
        t0 = time.perf_counter()
        out_cached = generate_cached(
            model,
            ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        t_cached = time.perf_counter() - t0
        print(f"\n[With KV cache] {t_cached:.3f}s")
        print(tokenizer.decode(out_cached[0].tolist()))

        speedup = t_no_cache / t_cached if t_cached > 0 else float("inf")
        print(f"\n{'=' * 60}")
        print(f"KV cache speedup: {speedup:.2f}x faster")
        print(f"{'=' * 60}")

    # ── Cleanup DDP ──
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
