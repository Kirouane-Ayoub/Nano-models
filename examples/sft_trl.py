"""Train a nano model with TRL's SFTTrainer.

    python examples/sft_trl.py --arch qwen_next --steps 20

The dataset is pre-tokenized into blocks of exactly `context_length` tokens.
That is not a detail — it is what makes this correct. None of the attention
variants take an attention mask, so a padded batch would let real tokens attend
to padding; equal-length samples mean no padding is ever added, and
NanoForCausalLM raises rather than training on quietly corrupted batches.

Note `packing=False`. TRL's packing does not merely concatenate: it flattens the
whole batch into one sequence with `position_ids` marking the boundaries, which
needs a FlashAttention varlen kernel to stop samples attending to each other.
TRL warns about this itself. These architectures have no such kernel, so packing
would both exceed the RoPE table and cross-contaminate samples.

The tokenizer is GPT-2's, because the architectures are configured for its
50257-token vocabulary — the same BPE the native training loop reaches for
through tiktoken.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from nano import config
from nano.hf import ARCHITECTURES, NanoConfig, NanoForCausalLM, build_nano_cfg


def load_corpus(tokenizer, block):
    """Blocks of exactly `block` tokens, so every sample is the same length and
    no padding is ever needed. Same thing TextDataset does when pretraining."""
    path = pathlib.Path(config.ROOT) / "the-verdict.txt"
    if not path.exists():
        raise SystemExit(f"{path} not found — run any model once to download it.")
    ids = tokenizer(path.read_text(encoding="utf-8"))["input_ids"]
    blocks = [ids[i : i + block] for i in range(0, len(ids) - block, block)]
    return Dataset.from_dict({"input_ids": blocks})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="qwen_next", choices=list(ARCHITECTURES))
    ap.add_argument("--size", default="nano")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default=str(pathlib.Path(config.ROOT) / "checkpoints" / "sft"))
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    nano_cfg = build_nano_cfg(args.arch, args.size)
    model = NanoForCausalLM(NanoConfig(arch=args.arch, size=args.size, nano=nano_cfg))
    print(f"{args.arch}/{args.size}: {sum(p.numel() for p in model.parameters()):,} parameters")

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=args.out,
            max_steps=args.steps,
            per_device_train_batch_size=args.batch_size,
            learning_rate=3e-4,
            logging_steps=5,
            report_to=[],
            save_strategy="no",
            bf16=False,
            # TRL turns this on by default. Supporting it would mean threading
            # torch.utils.checkpoint through each architecture's block loop —
            # not worth it at these sizes, where memory is never the limit.
            gradient_checkpointing=False,
            packing=False,  # see the docstring — TRL packing flattens
            # TRL's default "chunked_nll" reaches past the wrapper and calls the
            # backbone with (input_ids, attention_mask, use_cache) to compute the
            # lm_head projection in chunks. These architectures take a plain
            # tensor, so ask for the ordinary loss and let our forward run.
            loss_type="nll",
            max_length=nano_cfg["context_length"],
        ),
        train_dataset=load_corpus(tokenizer, nano_cfg["context_length"]),
        processing_class=tokenizer,
    )
    trainer.train()

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nSaved to {args.out} — reload with NanoForCausalLM.from_pretrained()")


if __name__ == "__main__":
    main()
