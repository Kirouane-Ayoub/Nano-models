"""Use these models as Hugging Face models, trainable with TRL.

    from nano.hf import NanoConfig, NanoForCausalLM

    model = NanoForCausalLM(NanoConfig(arch="qwen_next", size="nano"))
    SFTTrainer(model=model, args=SFTConfig(...), train_dataset=ds).train()

See examples/sft_trl.py for a run that works end to end.

The architectures are untouched. This file is only the adapter between their
`forward(idx) -> logits` and what `transformers` expects:
`forward(input_ids, attention_mask, labels) -> CausalLMOutputWithPast`.

Three things it has to get right, all of which fail silently otherwise:

**The label shift.** The native training loop feeds pre-shifted targets from
`TextDataset`. Hugging Face passes `labels = input_ids` and expects the *model*
to shift. Do both and you train on an off-by-one that looks perfectly healthy —
the same class of bug as the MTP shift in docs/ARCHITECTURES.md.

**Auxiliary losses.** qwen_next returns `(logits, aux)` when given targets (MTP),
and DSA stashes an indexer objective on the module. Both must land in `.loss` or
those components quietly stop training — DSA's indexer would sit at its
initialisation and its "sparsity" would be random.

**Padding.** None of the attention variants take an attention mask; they are
causal-only. A padded batch would let real tokens attend to padding, and nothing
would look wrong. So a padded mask raises here rather than being ignored.

The way around it is a dataset of equal-length samples, so padding never
happens: pre-tokenize into blocks of exactly `context_length`, the same thing
`TextDataset` does when pretraining. Note that TRL's `packing=True` is *not* the
answer — it flattens the whole batch into one sequence with `position_ids` and
relies on a FlashAttention varlen kernel to keep samples apart, which these
architectures do not have. TRL warns about this itself. Pass
`allow_padding=True` only once the zoo understands masks.

Two more TRL settings matter, both in examples/sft_trl.py: `loss_type="nll"`
(the default `chunked_nll` reaches past this wrapper and calls the backbone with
keyword arguments the architectures do not take) and
`gradient_checkpointing=False` (unsupported, and pointless at these sizes).

Generation is not wired to `past_key_values`: the KV cache here is internal
module state, not HF's cache objects. SFT never calls `generate`, so this is
fine for SFTTrainer; DPO/GRPO would need a Cache adapter first.

One wart: with MTP enabled the returned `.logits` cover positions 0..T-2 rather
than 0..T-1, because MTP has to be fed the next-token ids and that only lines up
on the truncated view. `.loss` is unaffected, so SFT is unaffected.

Run `python -m nano.hf` to check the adapter's loss equals the native loop's.
"""

import torch
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from nano.attention_zoo import collect_aux_loss
from nano.models.deepseek_nano import DeepSeekNano
from nano.models.deepseek_nano import MODEL_SIZES as DEEPSEEK_SIZES
from nano.models.gpt_nano import MODEL_SIZES as GPT_SIZES
from nano.models.gpt_nano import GPTNano
from nano.models.gemma_nano import MODEL_SIZES as GEMMA_SIZES
from nano.models.gemma_nano import GemmaNano
from nano.models.looped_nano import LoopedNano
from nano.models.qwen_nano import MODEL_SIZES as QWEN_SIZES
from nano.models.qwen_nano import QwenNano, compute_rope_params
from nano.models.qwen_next_nano import MODEL_SIZES as QWEN_NEXT_SIZES
from nano.models.qwen_next_nano import QwenNextNano, build_rope_tables

#: arch name -> (model class, size presets, extra config defaults)
ARCHITECTURES = {
    "gpt": (GPTNano, GPT_SIZES, {}),
    "qwen": (QwenNano, QWEN_SIZES, {}),
    "looped": (LoopedNano, QWEN_SIZES, {"loops": 4, "loop_sigma": 0.5, "loop_bptt": 0}),
    "gemma": (GemmaNano, GEMMA_SIZES, {}),  # local/global, dual RoPE, K-as-V — all in the sizes
    "deepseek": (DeepSeekNano, DEEPSEEK_SIZES, {"moe_latent_dim": 0, "balance_speed": 1e-3}),
    "qwen_next": (
        QwenNextNano,
        QWEN_NEXT_SIZES,
        {
            "linear_attn": "deltanet",
            "hybrid_ratio": 3,
            "mtp_weight": 0.0,
            "short_conv": 0,
            "kv_share": 0,
            "pos_enc": "rope",
            "residual": "plain",
            "mhc_streams": 4,
            "ple_dim": 0,
        },
    ),
}


def build_nano_cfg(arch, size="nano", **overrides):
    """The plain config dict the architectures take, with arch defaults applied."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: {list(ARCHITECTURES)}")
    _, sizes, defaults = ARCHITECTURES[arch]
    if size not in sizes:
        raise ValueError(f"Unknown size '{size}' for {arch}. Choose from: {list(sizes)}")
    return {**sizes[size], **defaults, **overrides}


class NanoConfig(PretrainedConfig):
    """Wraps a nano config dict so it survives save_pretrained/from_pretrained."""

    model_type = "nano"

    def __init__(self, arch="qwen_next", size="nano", nano=None, allow_padding=False, **kw):
        self.arch = arch
        self.size = size
        self.nano = nano if nano is not None else build_nano_cfg(arch, size)
        self.allow_padding = allow_padding
        # Mirrored so generic transformers/TRL code can read them
        self.vocab_size = self.nano["vocab_size"]
        self.max_position_embeddings = self.nano["context_length"]
        super().__init__(**kw)


class NanoForCausalLM(PreTrainedModel):
    config_class = NanoConfig
    base_model_prefix = "model"
    # Every architecture here ties head.weight to tok_emb.weight. safetensors
    # refuses to save two names pointing at one tensor unless the duplicate is
    # declared, so save_pretrained fails without this. transformers 5 expects a
    # {tied_key: source_key} mapping; in 4.x this was a plain list.
    _tied_weights_keys = {"model.head.weight": "model.tok_emb.weight"}
    # ... and so it is expected to be absent on load, not reported as missing.
    _keys_to_ignore_on_load_missing = ["model.head.weight"]

    def __init__(self, config):
        super().__init__(config)
        cls, _, _ = ARCHITECTURES[config.arch]
        self.model = cls(config.nano)
        # Sets up tied-weight bookkeeping that save/from_pretrained rely on.
        # _init_weights is a no-op below, so this does not disturb the
        # architectures' own initialisation.
        self.post_init()

    def _init_weights(self, module):
        """Only reached for weights genuinely missing from a checkpoint.

        Delegates to the architecture's own `_init_weights` rather than doing
        nothing: a no-op here leaves missing weights as whatever was in memory,
        and `from_pretrained` then quietly returns a model full of NaN. And
        rather than a generic N(0, 0.02): transformers also runs this over every
        module at construction, which silently replaced looped_nano's identity
        adapter init until the delegation.
        """
        self.model._init_weights(module)

    def tie_weights(self, **kwargs):
        """Re-tie the head and rebuild the RoPE tables after loading.

        Two things are deliberately absent from a checkpoint and so have to be
        restored by hand. `head.weight` is tied to `tok_emb.weight`, so it is
        saved once and comes back reported as MISSING. The RoPE cos/sin tables
        are non-persistent buffers, computed in the architecture's __init__ —
        which `from_pretrained` does not re-run for buffers. Skip either and the
        reloaded model returns NaN while looking perfectly well-formed.
        """
        super().tie_weights(**kwargs)
        self.model.head.weight = self.model.tok_emb.weight

        cfg = self.config.nano
        if hasattr(self.model, "build_rope_tables"):
            # Gemma owns two tables plus p-RoPE; it knows how to rebuild them.
            self.model.build_rope_tables(self.model.tok_emb.weight.device)
        elif hasattr(self.model, "cos") and "head_dim" in cfg:
            device = self.model.tok_emb.weight.device
            if isinstance(self.model, QwenNextNano):
                cos, sin = build_rope_tables(cfg)
            else:
                cos, sin = compute_rope_params(cfg["head_dim"], cfg["rope_base"], cfg["context_length"])
            self.model.cos, self.model.sin = cos.to(device), sin.to(device)

    def get_input_embeddings(self):
        return self.model.tok_emb

    def set_input_embeddings(self, value):
        self.model.tok_emb = value

    def get_output_embeddings(self):
        return self.model.head

    def set_output_embeddings(self, value):
        self.model.head = value

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if attention_mask is not None and not self.config.allow_padding:
            if not bool(attention_mask.all()):
                raise ValueError(
                    "This model has no attention-mask support — every variant in "
                    "nano/attention_zoo.py is causal-only, so padded positions would be "
                    "attended to as if they were real tokens and nothing would look wrong.\n"
                    "Use SFTConfig(packing=True) so batches need no padding, or set "
                    "NanoConfig(allow_padding=True) if you have taught the zoo about masks."
                )

        # MTP needs the *next token ids* to embed, not just the hidden state, so
        # the model has to be called with targets. They come from input_ids, not
        # labels — labels may hold -100, which is not an embeddable id. Running
        # on the truncated view is what lines the two up, and it means logits
        # cover positions 0..T-2 when MTP is on.
        mtp = labels is not None and getattr(self.model, "needs_targets", False)
        if mtp:
            logits, aux = self.model(input_ids[:, :-1], input_ids[:, 1:])
            lm_logits = logits
        else:
            out = self.model(input_ids)
            logits, aux = out if isinstance(out, tuple) else (out, None)
            lm_logits = logits[:, :-1] if labels is not None else logits

        loss = None
        if labels is not None:
            # HF passes unshifted labels; the model does the shift. -100 marks
            # positions to skip, which is how TRL does completion-only loss.
            loss = F.cross_entropy(
                lm_logits.reshape(-1, lm_logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            # MTP returns its objective alongside the logits; DSA stashes its
            # indexer objective on the module. Neither trains without this.
            if aux is not None:
                loss = loss + aux
            zoo_aux = collect_aux_loss(self.model)
            if zoo_aux is not None:
                loss = loss + zoo_aux

        return CausalLMOutputWithPast(loss=loss, logits=logits)


def _self_check():
    """The adapter's loss must equal the native training loop's on the same batch.

    That equality is what catches a double-shifted label and a dropped auxiliary
    loss — both of which train happily while being wrong.
    """
    torch.manual_seed(0)
    V, B, T = 96, 4, 17

    for arch, extra in (
        ("gpt", {}),
        ("qwen", {}),
        ("deepseek", {}),
        ("qwen_next", {}),
        ("qwen_next", {"mtp_weight": 0.3}),  # aux loss path
        ("looped", {}),  # weight-shared depth, per-iteration KV cache
        ("gemma", {}),  # two RoPE tables rebuilt by the wrapper, tied scaled embedding
        ("gpt", {"attention": "dsa", "top_k": 4}),  # module-stashed aux loss
    ):
        cfg = build_nano_cfg(arch, "nano", vocab_size=V, drop_rate=0.0, **extra)
        torch.manual_seed(0)
        hf = NanoForCausalLM(NanoConfig(arch=arch, nano=cfg)).eval()
        native = hf.model  # same weights, no second construction to drift from

        ids = torch.randint(0, V, (B, T))
        hf_loss = hf(input_ids=ids, labels=ids).loss

        # The reference has to see the same window the wrapper did. Without MTP
        # the wrapper runs on the full sequence and slices the logits, so any
        # module-stashed aux loss (DSA's indexer) covers all T tokens — running
        # the reference on ids[:, :-1] would compare two different windows.
        if getattr(native, "needs_targets", False):
            logits, aux = native(ids[:, :-1], ids[:, 1:])
            lm_logits = logits
        else:
            out = native(ids)
            logits, aux = out if isinstance(out, tuple) else (out, None)
            lm_logits = logits[:, :-1]
        expected = F.cross_entropy(lm_logits.reshape(-1, lm_logits.size(-1)), ids[:, 1:].flatten())
        if aux is not None:
            expected = expected + aux
        zoo_aux = collect_aux_loss(native)
        if zoo_aux is not None:
            expected = expected + zoo_aux

        torch.testing.assert_close(hf_loss, expected, atol=1e-5, rtol=1e-5)
        tag = f"{arch}{'+' + ','.join(extra) if extra else ''}"
        print(f"  {tag:28s} ok — loss {hf_loss.item():.4f} matches the native loop")

    # -100 must be skipped, not learned as a token id.
    cfg = build_nano_cfg("qwen", "nano", vocab_size=V, drop_rate=0.0)
    torch.manual_seed(0)
    hf = NanoForCausalLM(NanoConfig(arch="qwen", nano=cfg)).eval()
    ids = torch.randint(0, V, (B, T))
    masked = ids.clone()
    masked[:, : T // 2] = -100
    full_loss = hf(input_ids=ids, labels=ids).loss
    part_loss = hf(input_ids=ids, labels=masked).loss
    assert torch.isfinite(part_loss) and not torch.allclose(full_loss, part_loss)
    print("  completion-only loss        ok — -100 positions ignored")

    # A padded batch must fail loudly rather than attend to padding.
    mask = torch.ones(B, T, dtype=torch.long)
    mask[:, -3:] = 0
    try:
        hf(input_ids=ids, labels=ids, attention_mask=mask)
        raise AssertionError("padded attention_mask should have raised")
    except ValueError as e:
        assert "packing=True" in str(e)
    hf.config.allow_padding = True
    assert torch.isfinite(hf(input_ids=ids, labels=ids, attention_mask=mask).loss)
    print("  padding guard               ok — raises unless allow_padding")

    # save_pretrained / from_pretrained round-trip.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        hf.save_pretrained(d)
        again = NanoForCausalLM.from_pretrained(d).eval()
        torch.testing.assert_close(again(input_ids=ids).logits, hf(input_ids=ids).logits)
    print("  save/from_pretrained        ok — weights and config round-trip")

    # The wrapper rebuilds non-persistent tables during construction/loading.
    # Compare to an independent native model so losing partial RoPE is visible.
    for fraction in (0.0, 0.5, 1.0):
        cfg = build_nano_cfg(
            "qwen_next", vocab_size=V, drop_rate=0.0,
            pos_enc="prope", rope_fraction=fraction,
        )
        native = QwenNextNano(cfg).eval()
        hf = NanoForCausalLM(NanoConfig(arch="qwen_next", nano=cfg)).eval()
        hf.model.load_state_dict(native.state_dict())
        with torch.no_grad():
            expected = native(ids)
            torch.testing.assert_close(hf.model.cos, native.cos)
            torch.testing.assert_close(hf.model.sin, native.sin)
            torch.testing.assert_close(hf(input_ids=ids).logits, expected)
            with tempfile.TemporaryDirectory() as d:
                hf.save_pretrained(d)
                again = NanoForCausalLM.from_pretrained(d).eval()
                torch.testing.assert_close(again.model.cos, native.cos)
                torch.testing.assert_close(again.model.sin, native.sin)
                torch.testing.assert_close(again(input_ids=ids).logits, expected)
    print("  partial RoPE                ok — native, wrapped and reloaded logits agree")

    print("hf adapter self-check passed")


if __name__ == "__main__":
    _self_check()
