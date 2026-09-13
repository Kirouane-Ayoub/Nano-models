"""
Looped Nano — a depth-recurrent transformer, built from scratch.

The idea is older than the transformer (Universal Transformer, 2018) and came
back in 2025 as *latent reasoning*: instead of making a model deeper by adding
layers, apply the same few layers again and again. Parameter count stays put;
compute per token scales with the number of loops, and can be chosen at test
time. Huginn (Geiping et al., 2025) trained 3.5B parameters this way and gained
from unrolling deeper at inference than it saw in training; Ouro (Zhu et al.,
2025) looped a 24-layer stack 4 times and matched dense models 3x its size.

This file follows Huginn's prelude–core–coda topology:

    e = prelude(emb(x))                  # read the tokens once
    s_0 = 0                              # latent state
    s_{i+1} = core(A [s_i ; e])          # the same core block, r times
    logits = head(norm(coda(s_r)))       # read out once

Three details that are the whole architecture, and that a naive "for _ in
range(r): x = blocks(x)" gets wrong:

  1. **Input injection.** Every iteration sees the embedding *e* again, through
     the adapter A. Without it the state is free to drift away from the input
     and deeper unrolls get *worse*. A is initialised to [I | I], so at init an
     iteration is just `core(s + e)` — plain residual injection — and training
     learns any other mixing from there.
  2. **Sampled depth + truncated backprop.** The number of loops is drawn per
     step from a log-normal Poisson (capped at 2·r̄, so memory is bounded by
     r̄ and not by the Poisson tail), and gradients flow through only the last
     `loop_bptt` iterations. Sampling is what lets the model be unrolled to a
     depth it never saw; truncation is what makes 32 iterations affordable.
     `--loop-sigma 0` trains at a fixed depth like Ouro.
  3. **One KV cache per iteration.** The core's attention layers run r times
     per token, and iteration 3 of token t must attend to iteration 3 of the
     tokens before it — not to iteration 1's keys. Share one cache and the
     incremental decode silently diverges from the full forward. Ouro measured
     it: reusing the cache across steps at prefill costs >10 points.

Everything except the architecture (training loop, dataset, generation) is
imported from qwen_nano.py rather than copied. The blocks are qwen_nano's
(GQA + QK-norm + RoPE + SwiGLU), so this is "Qwen Nano, looped" — the one
variable is the topology.

Sizes: nano (5M) → small (40M) → medium (130M) → large (350M). The layer split
is n_layers/4 prelude, n_layers/2 core, n_layers/4 coda — (1, 2, 1) at nano,
(2, 4, 2) at small, which is Huginn's split.

Usage:
    python -m nano.models.looped_nano                       # 4 loops, sampled
    python -m nano.models.looped_nano --loops 8 --loop-bptt 4
    python -m nano.models.looped_nano --loops 4 --loop-sigma 0   # fixed depth (Ouro)
    python -m nano.models.looped_nano --self-check
    python -m nano.models.looped_nano --train-check
"""

if __package__ in (None, ""):
    # Running as a script: `python nano/models/looped_nano.py`. Make `nano` importable.
    import pathlib as _pathlib
    import sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from nano import config, data
from nano.accel import current_device, is_main_process, log
from nano.models.qwen_nano import (
    MODEL_SIZES,
    TRAIN_SETTINGS,
    RMSNorm,
    TransformerBlock,
    compute_rope_params,
    create_dataloaders,
    generate,
    generate_cached,
    train,
)


def layer_split(cfg):
    """(prelude, core, coda) layer counts. Overridable per key in cfg."""
    n = cfg["n_layers"]
    outer = cfg.get("n_prelude", max(1, n // 4))
    return outer, cfg.get("n_core", n - outer - cfg.get("n_coda", outer)), cfg.get("n_coda", outer)


class LoopedNano(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["emb_dim"]
        # Validate once here, not at the first forward: a JSON config can hand
        # us loops=4.0 or loops=0, and a negative loop_bptt would put every
        # iteration under no_grad and train only the coda — with a falling loss.
        cfg["loops"] = int(cfg["loops"])
        assert cfg["loops"] >= 1, f"loops={cfg['loops']} must be >= 1"
        assert cfg.get("loop_bptt", 0) >= 0, f"loop_bptt={cfg['loop_bptt']} must be >= 0"
        n_pre, n_core, n_coda = layer_split(cfg)
        assert n_core >= 1, "core needs at least one layer"
        cfg.update(n_prelude=n_pre, n_core=n_core, n_coda=n_coda)  # the snapshot records the split

        self.tok_emb = nn.Embedding(cfg["vocab_size"], d)
        self.drop = nn.Dropout(cfg["drop_rate"])
        self.prelude = nn.ModuleList(TransformerBlock(cfg) for _ in range(n_pre))
        self.adapter = nn.Linear(2 * d, d, bias=False)
        self.core = nn.ModuleList(TransformerBlock(cfg) for _ in range(n_core))
        self.coda = nn.ModuleList(TransformerBlock(cfg) for _ in range(n_coda))
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, cfg["vocab_size"], bias=False)
        self.head.weight = self.tok_emb.weight

        cos, sin = compute_rope_params(cfg["head_dim"], cfg["rope_base"], cfg["context_length"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)

        # Per-iteration KV caches for the core (detail 3 in the module docstring).
        # Slot i holds (cache_k, cache_v, cache_seq_len) for every core attention
        # layer at iteration i. Swapped into the modules around each iteration.
        self._slots = {}
        self._cache_loops = None  # r the cache was built with; must not change mid-generation

    def _init_weights(self, module):
        if module is getattr(self, "adapter", None):
            # [I | I]: at init an iteration is core(s + e). Learnable from there.
            # Lives here, not after apply(): the HF wrapper re-runs init over
            # every module at construction and would otherwise erase it.
            d = module.out_features
            with torch.no_grad():
                module.weight.copy_(torch.cat([torch.eye(d), torch.eye(d)], dim=1))
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ── loop count ──────────────────────────────────────────────────────────

    def n_loops(self):
        """Huginn: r ~ Poisson(e^tau) + 1, tau ~ N(log(r̄-1) - sigma²/2, sigma), so
        E[r] = r̄. Capped at 2·r̄: the tail reaches 6·r̄ and, with backprop through
        every iteration, activation memory scales with the draw — an OOM once an
        epoch is not a training curriculum. Eval always uses exactly r̄ — set
        cfg["loops"] to unroll deeper."""
        r, sigma = self.cfg["loops"], self.cfg.get("loop_sigma", 0.0)
        if not self.training or sigma <= 0 or r <= 1:
            return r
        tau = torch.normal(math.log(r - 1) - 0.5 * sigma**2, sigma, size=())
        return min(int(torch.poisson(tau.exp()).item()) + 1, 2 * r)

    # ── per-iteration cache ────────────────────────────────────────────────

    def _load_slot(self, i):
        for blk, state in zip(self.core, self._slots.get(i, [(None, None, 0)] * len(self.core))):
            blk.attn.cache_k, blk.attn.cache_v, blk.attn.cache_seq_len = state

    def _store_slot(self, i):
        self._slots[i] = [
            (blk.attn.cache_k, blk.attn.cache_v, blk.attn.cache_seq_len) for blk in self.core
        ]

    def reset_kv_cache(self):
        self._slots = {}
        self._cache_loops = None
        for blk in [*self.prelude, *self.core, *self.coda]:
            blk.attn.reset_cache()

    def _apply(self, fn, *args, **kwargs):
        # Registered buffers follow .to() / .double(); the slot tensors must too,
        # or a decode after a device change mixes CPU and GPU tensors.
        self._slots = {
            i: [tuple(fn(t) if torch.is_tensor(t) else t for t in st) for st in states]
            for i, states in self._slots.items()
        }
        return super()._apply(fn, *args, **kwargs)

    # ── forward ────────────────────────────────────────────────────────────

    def _iterate(self, s, e, i, use_cache, grad):
        if use_cache:
            self._load_slot(i)
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            s = self.adapter(torch.cat([s, e], dim=-1))
            for blk in self.core:
                s = blk(s, self.cos, self.sin, use_cache=use_cache)
        if use_cache:
            self._store_slot(i)
        return s

    def forward(self, idx, use_cache=False, loops=None):
        r = loops if loops is not None else self.n_loops()
        if use_cache:
            # Slot i of every earlier token must exist for iteration i of this
            # one. A deeper unroll at decode than at prefill would read empty
            # slots and silently attend from position 0.
            if self._cache_loops is None:
                self._cache_loops = r
            elif r != self._cache_loops:
                raise ValueError(
                    f"loops changed mid-generation ({self._cache_loops} → {r}); "
                    "call reset_kv_cache() before unrolling to a different depth"
                )

        e = self.drop(self.tok_emb(idx))
        for blk in self.prelude:
            e = blk(e, self.cos, self.sin, use_cache=use_cache)

        k = self.cfg.get("loop_bptt", 0) or r  # 0 = backprop through everything
        s = torch.zeros_like(e)
        # ponytail: s_0 = 0. Huginn draws s_0 ~ N(0, 0.4·I) for path independence;
        # that makes the model stochastic, which the incremental-decode check
        # cannot compare. Add a `loop_noise` cfg key if you want to try it.
        for i in range(r):
            s = self._iterate(s, e, i, use_cache, grad=i >= r - k)

        for blk in self.coda:
            s = blk(s, self.cos, self.sin, use_cache=use_cache)
        return self.head(self.norm(s))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def effective_depth(self, loops=None):
        n_pre, n_core, n_coda = layer_split(self.cfg)
        return n_pre + n_core * (self.cfg["loops"] if loops is None else loops) + n_coda


# ──────────────────────────────────────────────
# Self-check
# ──────────────────────────────────────────────


def self_check():
    """Shapes, weight sharing, and the two things that actually break: the
    per-iteration KV cache, and causality across iterations."""
    torch.manual_seed(0)
    V = 128

    def build(**over):
        cfg = {**MODEL_SIZES["nano"], "vocab_size": V, "drop_rate": 0.0, "loops": 4, **over}
        torch.manual_seed(0)
        return cfg, LoopedNano(cfg).eval()

    def check_incremental(model, T=12, prefill=8):
        idx = torch.randint(0, V, (2, T))
        full = model(idx)
        assert full.shape == (2, T, V), full.shape
        model.reset_kv_cache()
        model(idx[:, :prefill], use_cache=True)
        step = torch.cat([model(idx[:, t : t + 1], use_cache=True) for t in range(prefill, T)], 1)
        torch.testing.assert_close(step, full[:, prefill:], atol=1e-4, rtol=1e-4)
        return full

    # 1. Weight sharing: more loops, same parameters, different output.
    cfg1, one = build(loops=1)
    cfg4, four = build(loops=4)
    assert one.count_params() == four.count_params(), "loops changed the parameter count"
    assert layer_split(cfg4) == (1, 2, 1) and four.effective_depth() == 1 + 2 * 4 + 1
    idx = torch.randint(0, V, (2, 12))
    assert not torch.allclose(one(idx), four(idx)), "looping is a no-op"
    # Same weights, different unroll depth at test time — the point of the design.
    assert torch.allclose(four(idx, loops=1), one(idx)), "loops=1 must equal the unlooped model"
    print(f"  sharing   ok — {four.count_params():,} params at r=1 and r=4, depth 4 vs 10")

    # 2. Adapter init is [I | I]: iteration 0 equals core(e) exactly.
    e = torch.randn(2, 5, cfg4["emb_dim"])
    assert torch.allclose(four.adapter(torch.cat([torch.zeros_like(e), e], -1)), e, atol=1e-6)
    print("  adapter   ok — [I | I] init, iteration 0 is core(s + e)")

    # 3. Per-iteration KV cache: incremental decode must match the full forward,
    #    and there must really be one distinct cache per iteration.
    check_incremental(four)
    assert len(four._slots) == 4, f"{len(four._slots)} slots for 4 iterations"
    assert len({id(st[0][0]) for st in four._slots.values()}) == 4, "slots alias one cache"
    # ...and the check has teeth: share one slot across iterations and it fails.
    # (RuntimeError too: past context_length the shared RoPE offset runs off the table.)
    shared = four
    shared._load_slot, shared._store_slot = (lambda i: None), (lambda i: None)
    try:
        check_incremental(shared)
        raise SystemExit("shared-cache model passed the incremental check — test is broken")
    except (AssertionError, RuntimeError):
        pass
    _, four = build(loops=4)
    # Changing r between prefill and decode is refused rather than silently wrong.
    four.reset_kv_cache()
    four(idx[:, :8], use_cache=True)
    try:
        four(idx[:, 8:9], use_cache=True, loops=8)
        raise SystemExit("deeper unroll mid-generation was accepted — cache would be garbage")
    except ValueError:
        pass
    # The slots follow .to()/.double() like buffers: prefill in fp32, decode in fp64.
    four.reset_kv_cache()
    full = four(idx)
    four.reset_kv_cache()
    four(idx[:, :8], use_cache=True)
    four.double()
    step = torch.cat([four(idx[:, t : t + 1], use_cache=True) for t in range(8, 12)], 1)
    torch.testing.assert_close(step.float(), full[:, 8:], atol=1e-4, rtol=1e-4)
    _, four = build(loops=4)
    print("  kv slots  ok — one cache per iteration, fails when shared, refuses a mid-run r change")

    # 3b. Config validation happens at construction, not at the first forward.
    cfg_f, _ = build(loops=4.0)
    assert cfg_f["loops"] == 4 and isinstance(cfg_f["loops"], int)
    for bad in ({"loops": 0}, {"loop_bptt": -1}):
        try:
            build(**bad)
            raise SystemExit(f"{bad} was accepted")
        except AssertionError:
            pass
    print("  config    ok — loops coerced to int, loops<1 and loop_bptt<0 rejected at init")

    # 4. Causality: perturbing token t must not move any output before t.
    a = torch.randint(0, V, (1, 10))
    b = a.clone()
    b[0, 6] = (b[0, 6] + 1) % V
    torch.testing.assert_close(four(a)[:, :6], four(b)[:, :6])
    assert not torch.allclose(four(a)[:, 6:], four(b)[:, 6:])
    print("  causality ok — no future leak across 4 iterations")

    # 5. Sampled depth: training mode draws r with mean ≈ loops; eval is fixed.
    cfg_s, sampler = build(loops=6, loop_sigma=0.5)
    assert sampler.n_loops() == 6
    sampler.train()
    torch.manual_seed(1)
    draws = [sampler.n_loops() for _ in range(2000)]
    mean = sum(draws) / len(draws)
    assert min(draws) >= 1 and len(set(draws)) > 3, "depth is not being sampled"
    assert max(draws) <= 12, f"draw {max(draws)} exceeded the 2·r̄ cap"
    assert abs(mean - 6) < 0.5, f"sampled depth mean {mean:.2f}, expected ≈ 6"
    print(f"  sampling  ok — r̄=6, σ=0.5: mean {mean:.2f}, range {min(draws)}–{max(draws)}")

    print("\nAll checks passed")


def train_check():
    """Properties that only show up once gradients have flowed. ~15s on CPU."""
    torch.manual_seed(0)
    V = 96
    cfg = {
        **MODEL_SIZES["nano"],
        "vocab_size": V,
        "drop_rate": 0.0,
        "loops": 4,
        "loop_sigma": 0.5,
        "loop_bptt": 2,
    }
    model = LoopedNano(cfg)
    idx = torch.randint(0, V, (4, 25))
    x, y = idx[:, :-1], idx[:, 1:]

    # 1. Truncated backprop still reaches every parameter group, through the
    #    last k iterations. A truncation that detached e as well would leave the
    #    prelude with no gradient and train "fine".
    model.train()
    loss = F.cross_entropy(model(x, loops=4).flatten(0, 1), y.flatten())
    loss.backward()
    for name in ("prelude", "adapter", "core", "coda", "tok_emb"):
        grads = [p.grad for p in getattr(model, name).parameters()]
        assert all(g is not None and g.abs().sum() > 0 for g in grads), f"no gradient to {name}"
    print("  bptt      ok — k=2 of r=4 still trains prelude, adapter, core and coda")

    # 2. It learns with sampled depth, and the adapter moves off [I | I].
    d = cfg["emb_dim"]
    init_adapter = torch.cat([torch.eye(d), torch.eye(d)], dim=1)
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
    assert not torch.allclose(model.adapter.weight, init_adapter, atol=1e-3), "adapter never moved"
    print(f"  learning  ok — loss {first:.2f} → {final:.2f} with r ~ LogNormalPoisson(4)")

    # 3. Depth extrapolation does not blow up: the model trained at r̄=4 keeps a
    #    finite, similar loss at r=8. No claim that it *improves* — at this size
    #    it cannot — only that the fixed-point style recurrence stays bounded,
    #    which is what input injection buys.
    deeper = F.cross_entropy(model(x, loops=8).flatten(0, 1), y.flatten()).item()
    assert math.isfinite(deeper) and deeper < first, f"r=8 loss {deeper:.2f} exploded"
    print(f"  unroll    ok — r=4 loss {final:.2f}, r=8 loss {deeper:.2f} (trained on r̄=4)")

    print("\nAll checks passed")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Looped Nano — depth-recurrent transformer")
    parser.add_argument("--size", type=str, default="nano", choices=list(MODEL_SIZES))
    config.add_argument(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--loops", type=int, default=4, help="Mean core iterations in training, exact in eval"
    )
    parser.add_argument(
        "--loop-sigma",
        type=float,
        default=0.5,
        help="Log-normal Poisson spread of the sampled depth (Huginn 0.5; 0 = fixed depth)",
    )
    parser.add_argument(
        "--loop-bptt",
        type=int,
        default=0,
        metavar="K",
        help="Backprop through only the last K iterations (Huginn 8; 0 = all)",
    )
    parser.add_argument("--self-check", action="store_true", help="Run assertions and exit")
    parser.add_argument("--train-check", action="store_true", help="Gradient-level checks (~15s)")
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

    cfg = {
        **MODEL_SIZES[size],
        "loops": args.loops,
        "loop_sigma": args.loop_sigma,
        "loop_bptt": args.loop_bptt,
    }
    cfg.update(file_cfg.get("model", {}))

    resume_step = resume_epoch = 0
    optimizer_state = None
    if args.resume:
        log(f"\nLoading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        cfg = ckpt["config"]
        resume_step, resume_epoch = ckpt["global_step"], ckpt["epoch"]
        optimizer_state = ckpt["optimizer"]
    # After the checkpoint, on purpose: the loop settings are weight-compatible,
    # so `--resume ckpt --loops 8 --loop-bptt 4` is how you continue deeper.
    for key, flag, value in (
        ("loops", "--loops", args.loops),
        ("loop_sigma", "--loop-sigma", args.loop_sigma),
        ("loop_bptt", "--loop-bptt", args.loop_bptt),
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
    model = LoopedNano(cfg)
    if args.resume:
        model.load_state_dict(ckpt["model"])

    n_pre, n_core, n_coda = layer_split(cfg)
    log(
        f"\nLayout: prelude {n_pre} → core {n_core} × r → coda {n_coda}   "
        f"r̄={cfg['loops']} (σ={cfg['loop_sigma']}, bptt={cfg['loop_bptt'] or 'all'})"
    )
    log(
        f"Effective depth at eval: {model.effective_depth()} layers "
        f"from {n_pre + n_core + n_coda} layers of parameters"
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
