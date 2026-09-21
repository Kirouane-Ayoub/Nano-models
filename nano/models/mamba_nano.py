"""
Mamba Nano — a pure state-space language model (Mamba-2, with Mamba-3 dials).

Every other model in this repo attends. This one never does: there is no
attention layer, no KV cache, no positional encoding, and no separate FFN.
A block is a norm and a selective state-space mixer, and that is the whole
architecture:

    tokens ─► emb ─► [ x + SSM(norm(x)) ] × n ─► norm ─► head (tied)

    SSM: a_t = exp(Δ_t·A);  S_t = a_t·S_{t-1} + Δ_t·x_t⊗B_t;  y_t = S_t·C_t + D·x_t

Three things this file is for:

  1. **Constant memory.** Decoding token 10,000 costs exactly what decoding
     token 10 costs: the state is one (head_dim × state) matrix per head per
     layer, whatever the sequence length. The self-check measures the state
     after a long prefill and after one token and asserts they are the same
     size. No attention model can pass that test.
  2. **Position from recurrence.** There is no RoPE and no learned position
     table. Order is carried entirely by the recurrence — a_t is applied
     between every pair of tokens — so the model knows *how far back* something
     was without ever being told an index. The short conv adds a 4-token
     local window on top.
  3. **Selectivity is the whole trick.** Δ_t is computed from the token. A
     large Δ wipes the state (start of a new clause), a small one lets it pass
     through (inside a name). The train-check asserts Δ actually varies across
     tokens after training; a model whose Δ collapsed to a constant is a
     linear RNN and trains with an equally healthy-looking loss.

Mamba-3 (2026) is two dials on the same layer: `--discretization trapezoidal`
for the exponential-trapezoidal step (drops the conv), and `--complex` for the
complex-valued state, implemented as a data-dependent rotation of B and C.

The mixer is `attention_zoo.Mamba2`, shared with the hybrid model's
`--linear mamba2` slot, so a change to the recurrence is a change in one
place. Everything except the architecture (training loop, dataset,
generation) is imported from qwen_nano.py.

Sizes: nano (~3.3M) → small (~16M) → medium (~46M) → large (~160M). Blocks
are cheaper than transformer blocks (no FFN), so these run deeper at the same
width: Mamba-2 uses roughly 2× the layers of a matched transformer.

Usage:
    python -m nano.models.mamba_nano                        # Mamba-2
    python -m nano.models.mamba_nano --discretization trapezoidal --complex   # Mamba-3
    python -m nano.models.mamba_nano --state 32
    python -m nano.models.mamba_nano --self-check
    python -m nano.models.mamba_nano --train-check
"""

if __package__ in (None, ""):
    # Running as a script: `python nano/models/mamba_nano.py`. Make `nano` importable.
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
from nano.attention_zoo import Mamba2
from nano.models.qwen_nano import (
    TRAIN_SETTINGS,
    RMSNorm,
    create_dataloaders,
    generate,
    generate_cached,
    train,
)

# ──────────────────────────────────────────────
# Model size presets
# ──────────────────────────────────────────────
# head_dim = emb_dim // n_heads is what the zoo's Mamba2 uses; mamba_state is
# the N of the (head_dim × N) state. Mamba-2 ships head_dim 64, N 128.

# Hand-aligned table: columns line up so sizes can be compared down the page.
# fmt: off
MODEL_SIZES = {
    "nano": {       # ~3.3M — quick experiments
        "vocab_size": 50257,  "context_length": 128,
        "emb_dim": 64,        "n_heads": 4,      "n_layers": 8,
        "mamba_state": 16,    "mamba_conv": 4,   "drop_rate": 0.1,
    },
    "small": {      # ~16M — single GPU
        "vocab_size": 50257,  "context_length": 256,
        "emb_dim": 256,       "n_heads": 8,      "n_layers": 16,
        "mamba_state": 32,    "mamba_conv": 4,   "drop_rate": 0.1,
    },
    "medium": {     # ~46M — single GPU
        "vocab_size": 50257,  "context_length": 512,
        "emb_dim": 512,       "n_heads": 8,      "n_layers": 24,
        "mamba_state": 64,    "mamba_conv": 4,   "drop_rate": 0.1,
    },
    "large": {      # ~160M — multi-GPU recommended
        "vocab_size": 50257,  "context_length": 1024,
        "emb_dim": 1024,      "n_heads": 16,     "n_layers": 32,
        "mamba_state": 128,   "mamba_conv": 4,   "drop_rate": 0.1,
    },
}
# fmt: on


# ──────────────────────────────────────────────
# Block and model
# ──────────────────────────────────────────────


class MambaBlock(nn.Module):
    """x + SSM(norm(x)). No FFN: the mixer's own gate (y · silu(z)) and output
    projection play that role, which is why Mamba stacks are deeper than
    transformer stacks at the same parameter count. Mamba-2's paper notes an
    MLP can be interleaved (Nemotron-H does); ponytail: not here — the point
    of this file is the pure recurrence."""

    def __init__(self, cfg):
        super().__init__()
        self.norm = RMSNorm(cfg["emb_dim"])
        self.mixer = Mamba2(cfg)

    def forward(self, x, use_cache=False):
        return x + self.mixer(self.norm(x), use_cache=use_cache)


class MambaNano(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["emb_dim"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], d)
        # No positional embedding and no RoPE table: order lives in the recurrence.
        self.drop = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.ModuleList(MambaBlock(cfg) for _ in range(cfg["n_layers"]))
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, cfg["vocab_size"], bias=False)
        self.head.weight = self.tok_emb.weight  # tied
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, use_cache=False):
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        return self.head(self.norm(x))

    def reset_kv_cache(self):
        """Name kept for the shared generate/train loop; there is no KV cache to
        reset, only the recurrent state and the conv window."""
        for block in self.blocks:
            block.mixer.reset_cache()

    def state_bytes(self):
        """Allocated bytes retained by inference-state buffers, counting shared
        storage once. Excludes tensors retained by autograd during training."""
        total = 0
        seen = set()
        for block in self.blocks:
            m = block.mixer
            tensors = [m.S, m.I_prev, m.dt_acc, m.conv.conv_state if m.conv is not None else None]
            for t in tensors:
                if t is None:
                    continue
                storage = t.untyped_storage()
                key = (t.device, storage.data_ptr())
                if key not in seen:
                    seen.add(key)
                    total += storage.nbytes()
        return total

    def step_sizes(self, idx):
        """Δ for every layer, (n_layers, B, T, H), from a plain forward. Used by
        the train-check to assert the model stayed selective."""
        x = self.tok_emb(idx)
        dts = []
        for block in self.blocks:
            h = block.norm(x)
            dts.append(block.mixer.step_sizes(h))
            x = x + block.mixer(h)
        return torch.stack(dts)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────
# Self-check — python -m nano.models.mamba_nano --self-check
# ──────────────────────────────────────────────


def self_check():
    """What makes this a state-space model and not a transformer in disguise:
    no attention anywhere, state size independent of sequence length, no
    positional table. Then the two checks that catch real bugs: incremental
    decode and causality, with and without the Mamba-3 dials."""
    torch.manual_seed(0)
    V = 128

    def build(**over):
        cfg = {**MODEL_SIZES["nano"], "vocab_size": V, "drop_rate": 0.0, **over}
        torch.manual_seed(0)
        return cfg, MambaNano(cfg).eval()

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

    # 1. Nothing attends, nothing caches keys, nothing stores positions.
    cfg, model = build()
    names = [n for n, _ in model.named_modules()]
    assert not any("attn" in n.lower() and "mamba" not in n.lower() for n in names)
    assert all(not hasattr(b.mixer, "cache_k") for b in model.blocks), "a block holds a KV cache"
    assert not hasattr(model, "pos_emb") and not hasattr(model, "cos"), "positional encoding present"
    assert model.head.weight is model.tok_emb.weight
    print(f"  pure ssm  ok — {cfg['n_layers']} blocks, no attention, no KV cache, no position table")

    # 2. Constant memory: the recurrent state after 40 tokens has the same size
    #    as after 1 token. This is the property attention cannot have.
    with torch.no_grad():
        model.reset_kv_cache()
        model(torch.randint(0, V, (2, 40)), use_cache=True)
        long_bytes = model.state_bytes()
        for block in model.blocks:
            cache = block.mixer.conv.conv_state
            assert cache.untyped_storage().nbytes() == cache.numel() * cache.element_size()
        model.reset_kv_cache()
        model(torch.randint(0, V, (2, 1)), use_cache=True)
        short_bytes = model.state_bytes()
    assert long_bytes == short_bytes > 0, (long_bytes, short_bytes)
    per_layer = cfg["n_heads"] * (cfg["emb_dim"] // cfg["n_heads"]) * cfg["mamba_state"]
    print(f"  memory    ok — state after 40 tokens == after 1 token: {long_bytes:,} bytes "
          f"({per_layer:,} floats/layer/sequence + conv window)")

    # 3. Incremental decode, plain and with each Mamba-3 dial.
    check_incremental(model)
    cfg_t, trap = build(mamba_trapezoidal=True)
    cfg_c, cplx = build(mamba_complex=True)
    cfg_3, m3 = build(mamba_trapezoidal=True, mamba_complex=True)
    for m in (trap, cplx, m3):
        check_incremental(m)
    assert all(b.mixer.conv is None for b in trap.blocks), "trapezoidal should drop the conv"
    print("  kv-free   ok — prefill + decode exact for Mamba-2, trapezoidal, complex, and both")

    # 4. Causality through the whole stack.
    a = torch.randint(0, V, (1, 10))
    b = a.clone()
    b[0, 6] = (b[0, 6] + 1) % V
    for m in (model, m3):
        torch.testing.assert_close(m(a)[:, :6], m(b)[:, :6])
        assert not torch.allclose(m(a)[:, 6:], m(b)[:, 6:])
    print("  causality ok — no future leak, with or without the Mamba-3 dials")

    # 5. The dials are not no-ops on the model output.
    idx = torch.randint(0, V, (2, 12))
    for name, m in (("trapezoidal", trap), ("complex", cplx)):
        m.load_state_dict(model.state_dict(), strict=False)
        assert not torch.allclose(m(idx), model(idx), atol=1e-4), f"{name} is a no-op"
    print("  mamba3    ok — trapezoidal and complex both change the output on shared weights")

    print("\nAll checks passed")


def train_check():
    """What only gradients reveal: the model overfits, and Δ stays *selective* —
    it varies across tokens rather than collapsing to a constant, which would
    turn the layer into a plain linear RNN with an equally healthy loss. ~30s."""
    torch.manual_seed(0)
    V = 96
    cfg = {**MODEL_SIZES["nano"], "vocab_size": V, "drop_rate": 0.0}
    model = MambaNano(cfg)
    idx = torch.randint(0, V, (4, 25))
    x, y = idx[:, :-1], idx[:, 1:]

    model.train()
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"no gradient to {name}"
    print("  gradients ok — every parameter receives gradient (A_log, dt_bias and D included)")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
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

    # Selectivity: Δ must differ across tokens within a sequence.
    with torch.no_grad():
        dts = model.step_sizes(x)  # (n_layers, B, T, H)
    spread = (dts.std(dim=2) / dts.mean(dim=2)).mean().item()  # variation over T
    assert spread > 0.05, f"Δ collapsed to a constant (cv={spread:.3f}) — no selectivity"
    # State stays bounded when run twice as long as it was trained on.
    with torch.no_grad():
        model.reset_kv_cache()
        model(torch.randint(0, V, (1, 2 * x.shape[1])), use_cache=True)
        assert all(torch.isfinite(b.mixer.S).all() for b in model.blocks), "state blew up"
    print(f"  learning  ok — loss {first:.2f} → {final:.2f}; Δ varies across tokens (cv {spread:.2f}); "
          f"state finite at 2× length")

    print("\nAll checks passed")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Mamba Nano — pure state-space LM (Mamba-2 / Mamba-3)")
    parser.add_argument("--size", type=str, default="nano", choices=list(MODEL_SIZES))
    config.add_argument(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state", type=int, default=None, metavar="N", help="State size per head (Mamba-2: 128)")
    parser.add_argument(
        "--discretization", type=str, default=None, choices=["zoh", "trapezoidal"],
        help="zoh = Mamba-2; trapezoidal = Mamba-3's exponential-trapezoidal step (drops the conv)",
    )
    parser.add_argument("--complex", action="store_true", help="Mamba-3 complex state (rotating B and C)")
    parser.add_argument("--self-check", action="store_true", help="Run assertions and exit")
    parser.add_argument("--train-check", action="store_true", help="Gradient-level checks (~30s)")
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

    cfg = {**MODEL_SIZES[size], "mamba_trapezoidal": False, "mamba_complex": False}
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
        ("mamba_state", "--state", args.state),
        ("mamba_trapezoidal", "--discretization", args.discretization == "trapezoidal"),
        ("mamba_complex", "--complex", args.complex),
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
    model = MambaNano(cfg)
    if args.resume:
        model.load_state_dict(ckpt["model"])

    log(
        f"\nMamba-{'3' if cfg['mamba_trapezoidal'] or cfg['mamba_complex'] else '2'}: "
        f"{cfg['n_layers']} SSM blocks, state {cfg['mamba_state']} per head, "
        f"discretization={'trapezoidal' if cfg['mamba_trapezoidal'] else 'zoh'}, "
        f"complex={cfg['mamba_complex']}, no attention, no KV cache"
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
