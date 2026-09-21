"""
Swappable components for the hybrid model — each one a self-contained idea.

`qwen_next_nano.py` is the experiment bench: its flags turn these on and off
and combine them with any mixer. The components live here so that file stays
a readable *layout* (which mixer where, how the residual flows) and each of
these stays a readable *mechanism*. Nothing here knows about the layout; each
class takes a config dict and a tensor.

    HyperConnections     mHC — n residual streams mixed by a doubly stochastic matrix (DeepSeek V4)
    PerLayerEmbeddings   PLE — a per-layer embedding slice added after every block (Gemma 4)
    Engram               hashed n-gram lookup memory added to the residual (DeepSeek, 2026)
    DepthRouter          Mixture-of-Depths — route tokens past a block, causal predictor for decode

Each has an assertion in `python -m nano.models.qwen_next_nano --self-check`
that proves it is not a no-op, and `docs/ARCHITECTURES.md` has the paper and
the gotchas. The MTP head is not here: it wraps a HybridBlock, so it is part of
the layout.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nano.models.qwen_nano import RMSNorm


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
        m = m / m.sum(dim=0, keepdim=True)  # columns
        m = m / m.sum(dim=1, keepdim=True)  # rows — last, so this one is exact
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
        self.pre = nn.Parameter(jitter(F.one_hot(torch.tensor(0), n).float()))  # read stream 0
        self.post = nn.Parameter(jitter(F.one_hot(torch.tensor(0), n).float()))  # write stream 0
        self.res_logits = nn.Parameter(jitter(torch.eye(n) * init))  # sinkhorn → ~I

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
        self.scale = nn.Parameter(torch.zeros(n_layers))  # start as a no-op

    def lookup(self, idx):
        B, T = idx.shape
        return self.table(idx).view(B, T, self.n_layers, self.dim)

    def layer_signal(self, ple, layer):
        return self.proj[layer](ple[:, :, layer]) * self.scale[layer]


# ──────────────────────────────────────────────
# Engram — conditional memory via hashed n-gram lookup
# ──────────────────────────────────────────────


class Engram(nn.Module):
    """Static-pattern memory next to the transformer (DeepSeek, Jan 2026,
    arXiv 2601.07372).

    Attention and MoE both spend compute to *recompute* things that never
    change: "New York" is followed by a small set of tokens every time, and the
    model rediscovers that from scratch at every position. Engram stores such
    patterns in a table instead. At position t, the last n tokens (n = 2, 3) are
    hashed into a row of a fixed-size embedding table; the rows are concatenated,
    projected to the hidden size, gated, and added to the residual stream. O(1)
    per token, no attention over the past, pure memory — like PLE, the table
    can live in slow storage. DeepSeek found ~20-25% of a sparse parameter
    budget is best spent here, the rest on MoE.

    Hashing: `(t_0·a_0 ^ t_1·a_1 ^ …) mod rows` with odd multipliers per head,
    `rows` prime so collisions spread. Two heads per n-gram halve the damage of
    any one collision. Positions with too little history hash a sentinel id.

    Causal by construction — the n-gram ends at t — but only if the *history*
    is right under cached decode: the module keeps the last n-1 token ids, and
    the self-check decodes one token at a time to prove it.

    ponytail: the gate is a per-channel sigmoid of the normed residual, not the
    paper's attention-style query·key scalar; same job, fewer moving parts. The
    output projection is zero-initialised so the model starts exactly as if
    Engram were absent.
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        self.ngrams = tuple(cfg.get("engram_ngrams", (2, 3)))
        self.heads = cfg.get("engram_heads", 2)
        self.rows = cfg.get("engram_rows", 4099)  # prime
        self.dim = cfg["engram_dim"]
        self.sentinel = cfg["vocab_size"]  # an id no real token has
        self.tables = nn.ModuleList(
            nn.Embedding(self.rows, self.dim) for _ in range(len(self.ngrams) * self.heads)
        )
        # Odd multipliers, one per (n-gram position, head). Fixed, not learned.
        g = torch.Generator().manual_seed(0)
        mult = torch.randint(1, 2**20, (max(self.ngrams), self.heads), generator=g) * 2 + 1
        # The row mapping is part of the learned memory's meaning. Persist it
        # so HF's meta-device loading restores the same hash function.
        self.register_buffer("mult", mult)
        self.norm = RMSNorm(d)
        self.gate = nn.Linear(d, d, bias=False)
        self.proj = nn.Linear(len(self.tables) * self.dim, d, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.register_buffer("hist", None, persistent=False)  # last n-1 ids under cache

    def forward(self, x, idx, use_cache=False):
        B, T = idx.shape
        n_max = max(self.ngrams)
        hist = self.hist if (use_cache and self.hist is not None) else idx.new_full((B, n_max - 1), self.sentinel)
        ids = torch.cat([hist, idx], dim=1)  # (B, T + n_max - 1)
        if use_cache:
            self.hist = ids[:, -(n_max - 1) :]
        # shifted[k][b, t] is the token k steps before position t
        shifted = [ids[:, n_max - 1 - k : n_max - 1 - k + T] for k in range(n_max)]

        looked = []
        t = 0
        for n in self.ngrams:
            for h in range(self.heads):
                code = shifted[0] * self.mult[0, h]
                for k in range(1, n):
                    code = code ^ (shifted[k] * self.mult[k, h])
                looked.append(self.tables[t](code % self.rows))
                t += 1
        mem = self.proj(torch.cat(looked, dim=-1))
        return mem * torch.sigmoid(self.gate(self.norm(x)))

    def reset_cache(self):
        self.hist = None


# ──────────────────────────────────────────────
# Mixture-of-Depths — route tokens past whole blocks
# ──────────────────────────────────────────────


class DepthRouter(nn.Module):
    """Decides which tokens a block processes (Raposo et al., 2024, arXiv 2404.02258).

    MoE routes tokens between experts; MoD routes them between *doing the block
    and skipping it*. A scalar router scores every token, the top `capacity`
    fraction go through the block (output scaled by their score, so the router
    gets gradient), the rest ride the residual unchanged. Half the tokens at
    half the layers is the paper's setting; FLOPs drop accordingly and quality
    holds, because most tokens do not need most layers.

    The catch is the top-k: it ranks tokens *across the sequence*, so whether
    token t is processed depends on tokens after it. Fine for training, fatal
    for autoregressive decode. The paper's answer, kept here, is a second
    scalar head — the predictor — trained with a BCE loss to imitate the top-k
    decision from the token alone. Training routes by top-k and trains the
    predictor; eval routes by the predictor and is causal. The self-check
    asserts the causality in eval mode and the exact capacity in train mode;
    the train-check asserts the predictor actually learns to agree.

    ponytail: skipped tokens are *masked*, not gathered — the block still runs
    on them and they still serve as keys, only the residual update is dropped.
    That demonstrates the routing, not the FLOP saving, same trade as DSA and
    CSA. The paper drops skipped tokens from attention entirely.
    """

    def __init__(self, d, capacity):
        super().__init__()
        if not math.isfinite(capacity) or not 0 < capacity <= 1:
            raise ValueError(f"DepthRouter capacity must be finite and in (0, 1], got {capacity}")
        self.capacity = capacity
        self.router = nn.Linear(d, 1, bias=False)
        self.predictor = nn.Linear(d, 1, bias=False)
        self.aux = None
        self.last_routed = None  # (B, T) bool, for inspection and tests

    def gate(self, h):
        """(B, T, 1) multiplier for the block's update: score for routed tokens, 0 otherwise."""
        score = torch.sigmoid(self.router(h)).squeeze(-1)  # (B, T)
        # Detached input: the predictor must not steer the representation it reads.
        logit = self.predictor(h.detach()).squeeze(-1)
        if self.training:
            k = max(1, round(self.capacity * h.shape[1]))
            routed = torch.zeros_like(score, dtype=torch.bool)
            routed.scatter_(1, score.topk(k, dim=1).indices, True)
            self.aux = F.binary_cross_entropy_with_logits(logit, routed.float())
        else:
            routed = logit > 0
            self.aux = None
        self.last_routed = routed
        return (score * routed).unsqueeze(-1)
