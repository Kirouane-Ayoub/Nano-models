"""
Muon — momentum orthogonalised by Newton-Schulz (Keller Jordan, 2024; MuonClip
in Kimi K2, 2025; Kimi K2.5, GLM-5 and others in 2026).

AdamW scales every weight *entry* by its own running variance. Muon treats a
weight *matrix* as a matrix: take the momentum-averaged gradient G, replace it
by the nearest orthogonal matrix UVᵀ (the polar factor of its SVD), and step
along that. Every singular direction then moves by the same amount, instead of
the dominant ones swamping the rest — which is why it trains 2-D layers faster
per step and tolerates larger learning rates. The polar factor is computed with
five quintic Newton-Schulz iterations on the normalised G, no SVD, all matmuls.

Only 2-D weights inside the blocks get Muon. Embeddings, the LM head, norms,
biases and any other 1-D or 3-D parameter (conv kernels, Mamba's A_log) stay on
AdamW, in the same optimizer object so the training loop, checkpoints and
Accelerate see one optimizer. That is Jordan's `MuonWithAuxAdam` layout.

    python -m nano.optim        # self-test

ponytail: fp32 Newton-Schulz and a Python loop over parameters. The reference
runs bf16 and fuses across a parameter bucket; both are speed, not behaviour.
"""

import math

import torch
import torch.nn as nn

from nano.accel import log


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Approximate polar factor UVᵀ of G by a quintic Newton-Schulz iteration.

    The iteration X ← aX + b(XXᵀ)X + c(XXᵀ)²X pushes every singular value
    towards 1 while leaving the singular vectors alone. The coefficients are
    tuned so five steps land all singular values in roughly [0.7, 1.2] from any
    start in (0, 1] — not exactly 1, and that is fine: the point is that no
    direction dominates, not that the result is exactly orthogonal. G is
    normalised first so its top singular value is ≤ 1. Runs on the transposed
    matrix when rows > cols, so the Gram matrix is the small one.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.t()
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.t()
        X = a * X + (b * A + c * A @ A) @ X
    return X.t() if tall else X


class Muon(torch.optim.Optimizer):
    """Muon for 2-D block weights, AdamW for everything else, one optimizer.

    Groups carry `use_muon`. Muon groups: SGD momentum (nesterov) → Newton-
    Schulz → step scaled by √(max(1, rows/cols)) so a wide and a tall matrix of
    the same size move by the same amount per entry. AdamW groups: the usual
    bias-corrected moments. Both apply decoupled weight decay. Every group has
    `lr_scale`, the ratio of its lr to the schedule's base lr, because the
    training loop writes one scheduled lr and multiplies by that.
    """

    def __init__(self, param_groups):
        defaults = dict(
            lr=0.02, lr_scale=1.0, weight_decay=0.0, use_muon=True,
            momentum=0.95, nesterov=True, ns_steps=5, betas=(0.9, 0.95), eps=1e-8,
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        # `closure` is the torch.optim contract; Accelerate's wrapper passes it
        # positionally, so leaving it out breaks every training run and no unit
        # test of the bare optimizer would notice.
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                if g["use_muon"]:
                    if "buf" not in state:
                        state["buf"] = torch.zeros_like(p)
                    buf = state["buf"].lerp_(p.grad, 1 - g["momentum"])
                    upd = p.grad.lerp(buf, g["momentum"]) if g["nesterov"] else buf
                    upd = zeropower_via_newtonschulz5(upd.view(p.shape[0], -1), g["ns_steps"])
                    upd = upd * math.sqrt(max(1.0, p.shape[0] / p[0].numel()))
                    p.add_(upd.view_as(p).to(p.dtype), alpha=-g["lr"])
                else:
                    if "m" not in state:
                        state["m"], state["v"], state["t"] = torch.zeros_like(p), torch.zeros_like(p), 0
                    b1, b2 = g["betas"]
                    state["t"] += 1
                    m = state["m"].lerp_(p.grad, 1 - b1)
                    v = state["v"].lerp_(p.grad.square(), 1 - b2)
                    m_hat = m / (1 - b1 ** state["t"])
                    v_hat = v / (1 - b2 ** state["t"])
                    p.addcdiv_(m_hat, v_hat.sqrt().add_(g["eps"]), value=-g["lr"])
        return loss


def _is_block_matrix(name, p):
    """Block matrices, excluding lookup tables and normalization gains.

    GR packs independent per-branch RMSNorm gains into a 2-D tensor; its
    shape does not make it a matrix projection suitable for Muon.

    The rule is by *name*, and that is load-bearing: a 2-D per-channel scale
    must be called `gain` (or contain `emb`/`head`/`table`) to stay on AdamW.
    Shape cannot tell a (4, 64) gain from a (4, 64) projection. If you add a
    component with a 2-D norm scale under another name, extend this list or
    Muon will orthogonalise it.
    """
    return (p.ndim == 2 and name.rsplit(".", 1)[-1] != "gain"
            and not any(k in name for k in ("emb", "head", "table")))


def build_optimizer(model, settings, optimizer_state=None):
    """AdamW by default; `settings["optimizer"] == "muon"` puts block matrices on
    Muon at `settings["muon_lr"]` and the rest on AdamW at `learning_rate`.
    On resume the saved groups determine the optimizer type, so callers do
    not have to repeat the original selection. Saved state takes precedence.
    """
    lr, wd = settings["learning_rate"], settings["weight_decay"]
    kind = settings.get("optimizer", "adamw")
    if optimizer_state is not None:
        saved = "muon" if any("use_muon" in g for g in optimizer_state["param_groups"]) else "adamw"
        if saved != kind:
            log(f"  Resuming a {saved} checkpoint: ignoring optimizer={kind!r}, the saved state wins")
        kind = settings["optimizer"] = saved
    if kind != "muon":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        return optimizer
    muon_lr = settings.get("muon_lr", 0.02)
    named = list(model.named_parameters())
    seen = set()
    matrices, others = [], []
    for name, p in named:
        if id(p) in seen:  # tied weights appear twice
            continue
        seen.add(id(p))
        (matrices if _is_block_matrix(name, p) else others).append(p)
    optimizer = Muon([
        {"params": matrices, "use_muon": True, "lr": muon_lr, "lr_scale": muon_lr / lr, "weight_decay": wd},
        {"params": others, "use_muon": False, "lr": lr, "lr_scale": 1.0, "weight_decay": wd},
    ])
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    return optimizer


# ──────────────────────────────────────────────
# Self-test — python -m nano.optim
# ──────────────────────────────────────────────


def _self_test():
    torch.manual_seed(0)

    # 1. Newton-Schulz orthogonalises: a matrix whose singular values span 100×
    #    comes out with all of them near 1, so every direction gets the same step.
    U, _ = torch.linalg.qr(torch.randn(32, 32))
    V, _ = torch.linalg.qr(torch.randn(64, 64))
    G = U @ torch.diag(torch.logspace(-1, 1, 32)) @ V[:32]  # singular values 0.1 … 10, exactly
    sv_in = torch.linalg.svdvals(G)
    sv_out = torch.linalg.svdvals(zeropower_via_newtonschulz5(G))
    assert sv_in.max() / sv_in.min() > 50, "test input is not ill-conditioned"
    assert 0.5 < sv_out.min() and sv_out.max() < 1.5, f"singular values {sv_out.min():.2f}–{sv_out.max():.2f}"
    tall = zeropower_via_newtonschulz5(G.t())  # must handle rows > cols too
    assert tall.shape == (64, 32)
    print(f"  newton-schulz ok — σ spread {sv_in.max() / sv_in.min():.0f}× in, "
          f"{sv_out.min():.2f}–{sv_out.max():.2f} out")

    # 2. Parameter routing: block matrices go to Muon; embeddings, the tied head,
    #    norms, biases and conv kernels stay on AdamW. One optimizer object.
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(50, 16)
            self.blocks = nn.ModuleList([nn.Linear(16, 16, bias=True) for _ in range(2)])
            self.conv = nn.Conv1d(16, 16, 4, groups=16, bias=False)
            self.norm = nn.LayerNorm(16)
            self.head = nn.Linear(16, 50, bias=False)
            self.head.weight = self.tok_emb.weight

        def forward(self, idx):
            x = self.tok_emb(idx)
            for b in self.blocks:
                x = torch.relu(b(x))
            x = x + self.conv(torch.nn.functional.pad(x.transpose(1, 2), (3, 0))).transpose(1, 2)
            return self.head(self.norm(x))

    toy = Toy()
    settings = {"learning_rate": 1e-3, "weight_decay": 0.1, "optimizer": "muon", "muon_lr": 0.02}
    opt = build_optimizer(toy, settings)
    assert isinstance(opt, Muon)
    muon_params = {id(p) for g in opt.param_groups if g["use_muon"] for p in g["params"]}
    adam_params = {id(p) for g in opt.param_groups if not g["use_muon"] for p in g["params"]}
    assert muon_params == {id(b.weight) for b in toy.blocks}, "Muon should get exactly the block matrices"
    assert id(toy.tok_emb.weight) in adam_params and id(toy.conv.weight) in adam_params
    assert id(toy.norm.weight) in adam_params and id(toy.blocks[0].bias) in adam_params
    assert len(muon_params | adam_params) == len(list(toy.parameters())), "a parameter was dropped"
    print(f"  routing       ok — {len(muon_params)} matrices on Muon, {len(adam_params)} tensors on AdamW")

    # 3. The Muon step is an orthogonal matrix scaled by lr·√(rows/cols): every
    #    singular value of the update equals that, whatever the gradient looked like.
    w = toy.blocks[0].weight
    before = w.detach().clone()
    idx = torch.randint(0, 50, (4, 8))
    toy(idx).square().mean().backward()
    opt.step(None)  # positional closure, as Accelerate's wrapper calls it
    upd = (w.detach() - before)
    sv = torch.linalg.svdvals(upd)
    expect = settings["muon_lr"] * math.sqrt(max(1.0, w.shape[0] / w.shape[1]))
    assert 0.5 * expect < sv.min() and sv.max() < 1.5 * expect, f"update σ {sv.min():.4f}–{sv.max():.4f}, expected ≈ {expect:.4f}"
    print(f"  step          ok — update singular values {sv.min():.4f}–{sv.max():.4f} ≈ lr·√(m/n) = {expect:.4f}")

    # 4. lr schedule contract: the loop writes pg["lr"] = lr · pg["lr_scale"], so
    #    the Muon group must carry the ratio and AdamW groups 1.0.
    scales = sorted({round(g["lr_scale"], 6) for g in opt.param_groups})
    assert scales == [1.0, round(0.02 / 1e-3, 6)], scales

    # 5. It learns, and its state round-trips through a checkpoint.
    torch.manual_seed(0)
    toy = Toy()
    opt = build_optimizer(toy, settings)
    x = torch.randint(0, 50, (8, 12))
    first = None
    for _ in range(60):
        loss = torch.nn.functional.cross_entropy(toy(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.7, f"Muon did not learn: {first:.3f} → {loss.item():.3f}"
    import copy

    state = copy.deepcopy(opt.state_dict())
    restored = copy.deepcopy(toy)
    resume_settings = {"learning_rate": settings["learning_rate"], "weight_decay": settings["weight_decay"]}
    again = build_optimizer(restored, resume_settings, state)
    assert isinstance(again, Muon) and resume_settings["optimizer"] == "muon"
    assert len(again.state) == len(opt.state), "optimizer state did not round-trip"
    # Check actual continuation, including the two learning-rate schedules.
    for optimizer, model in ((opt, toy), (again, restored)):
        for group in optimizer.param_groups:
            group["lr"] = 5e-4 * group.get("lr_scale", 1.0)
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten()).backward()
        optimizer.step()
    for original, loaded in zip(toy.parameters(), restored.parameters()):
        torch.testing.assert_close(original, loaded)
    print(f"  learning      ok — loss {first:.3f} → {loss.item():.3f} in 60 steps, state round-trips")

    # 6. The default is untouched: adamw gives a plain torch AdamW.
    assert type(build_optimizer(Toy(), {**settings, "optimizer": "adamw"})) is torch.optim.AdamW

    # 7. On resume the checkpoint decides the optimizer type, whatever the flags
    #    say — loading Muon groups into AdamW (or the reverse) would fail on the
    #    group layout, and a run resumed without --optim must not silently switch.
    adam_state = build_optimizer(Toy(), {**settings, "optimizer": "adamw"}).state_dict()
    want_muon = {**settings, "optimizer": "muon"}
    assert type(build_optimizer(Toy(), want_muon, adam_state)) is torch.optim.AdamW
    assert want_muon["optimizer"] == "adamw", "settings should record what was actually built"
    want_adam = {**settings, "optimizer": "adamw"}
    assert isinstance(build_optimizer(Toy(), want_adam, state), Muon)
    assert want_adam["optimizer"] == "muon"
    print("  resume        ok — saved optimizer type wins over the requested one, and says so")
    adam_model = Toy()
    adam = build_optimizer(adam_model, {**settings, "optimizer": "adamw"})
    adam_model(x).square().mean().backward()
    adam.step()
    restored_adam = build_optimizer(copy.deepcopy(adam_model), settings.copy(), copy.deepcopy(adam.state_dict()))
    assert type(restored_adam) is torch.optim.AdamW, "saved AdamW state must override Muon selection"
    # GR's packed normalization gains stay on AdamW, including when wrapped.
    # Compare their actual update with standalone AdamW, not just group labels.
    from nano.components import GatedResidual

    gr = nn.Sequential(GatedResidual(16))
    gr_opt = build_optimizer(gr, settings)
    muon_ids = {id(p) for g in gr_opt.param_groups if g["use_muon"] for p in g["params"]}
    assert muon_ids == {id(gr[0].W_d.weight), id(gr[0].W_u.weight), id(gr[0].W_w.weight)}
    gain_ref = nn.Parameter(gr[0].gain.detach().clone())
    ref_opt = torch.optim.AdamW([gain_ref], lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    grad = torch.randn_like(gain_ref)
    gr[0].gain.grad, gain_ref.grad = grad.clone(), grad.clone()
    gr_opt.step()
    ref_opt.step()
    torch.testing.assert_close(gr[0].gain, gain_ref)
    print("  GR gains      ok — projections on Muon, normalization gains match AdamW update")
    print("optim self-test passed")


if __name__ == "__main__":
    _self_test()
