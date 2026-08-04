"""Corpus loading: a local file, the bundled sample, or a Hugging Face dataset.

Every model takes the same three flags, and they mean the same thing whether you
are pre-training with the native loop or fine-tuning through TRL:

    --dataset roneneldan/TinyStories        # any hub dataset with a text column
    --dataset-config wikitext-2-raw-v1      # when the dataset has named configs
    --text-field text                       # which column holds the text
    --split train[:2000]                    # standard datasets split syntax

`datasets` is imported lazily. The repo's dependencies are torch and tiktoken;
you only need `datasets` installed if you actually ask for one.

The whole corpus is joined into a single string, because that is what the native
`TextDataset` consumes — it slices fixed-length windows out of one token stream.
That is fine for the scale this repo works at and a bad idea for a real
pre-training corpus; use `--dataset-limit` to take the first N documents, or
tokenize to disk and write a proper streaming Dataset if you outgrow it.
"""

import os
import urllib.request

from nano.config import ROOT

SAMPLE_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
    "ch02/01_main-chapter-code/the-verdict.txt"
)


def load_corpus(
    file=None,
    dataset=None,
    dataset_config=None,
    text_field="text",
    split="train",
    limit=None,
    join="\n\n",
    log=print,
):
    """Return the corpus as one string.

    Precedence: an explicit `dataset` wins, then `file`, then the bundled sample
    (downloaded on first use).
    """
    if dataset:
        return _from_hub(dataset, dataset_config, text_field, split, limit, join, log)
    if file:
        with open(file, "r", encoding="utf-8") as f:
            return f.read()

    path = os.path.join(ROOT, "the-verdict.txt")
    if not os.path.exists(path):
        log("Downloading sample text...")
        with urllib.request.urlopen(SAMPLE_URL, timeout=30) as resp:
            text = resp.read().decode("utf-8")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _from_hub(dataset, dataset_config, text_field, split, limit, join, log):
    try:
        from datasets import load_dataset
    except ImportError as e:  # not a core dependency
        raise SystemExit(
            f"--dataset needs the `datasets` package: pip install datasets ({e})"
        ) from None

    if limit:
        # `split[:n]` is resolved by datasets itself, so only n rows are built.
        split = f"{split}[:{int(limit)}]" if "[" not in split else split
    ds = load_dataset(dataset, dataset_config, split=split)

    if text_field not in ds.column_names:
        raise ValueError(
            f"'{text_field}' is not a column of {dataset} — found "
            f"{ds.column_names}. Pass --text-field with the right one."
        )
    texts = [t for t in ds[text_field] if t and t.strip()]
    text = join.join(texts)
    log(
        f"Loaded {dataset}"
        + (f":{dataset_config}" if dataset_config else "")
        + f" [{split}] — {len(texts):,} documents, {len(text):,} characters"
    )
    return text


def token_blocks(text, tokenizer, block):
    """Tokenize into blocks of exactly `block` tokens, as a `datasets.Dataset`.

    Equal lengths are the point: none of the attention variants take an
    attention mask, so a batch that needs padding would silently attend to it.
    Same fixed-window scheme the native `TextDataset` uses when pre-training.
    """
    from datasets import Dataset

    ids = tokenizer(text)["input_ids"]
    if len(ids) <= block:
        raise ValueError(f"corpus is {len(ids)} tokens, need more than one block of {block}")
    return Dataset.from_dict(
        {"input_ids": [ids[i : i + block] for i in range(0, len(ids) - block, block)]}
    )


def add_arguments(parser):
    """The dataset flags, spelled the same way everywhere."""
    parser.add_argument("--file", type=str, default=None, help="Local training text file")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Hugging Face dataset id, e.g. roneneldan/TinyStories",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset config name, when the dataset has several",
    )
    parser.add_argument(
        "--text-field", type=str, default="text", help="Column holding the text (default: text)"
    )
    parser.add_argument("--split", type=str, default="train", help="Split to load")
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N documents",
    )


#: flag -> config key, for the `data` block of a config file
FLAGS = {
    "--file": "file",
    "--dataset": "dataset",
    "--dataset-config": "dataset_config",
    "--text-field": "text_field",
    "--split": "split",
    "--dataset-limit": "dataset_limit",
}

DEFAULTS = {
    "file": None,
    "dataset": None,
    "dataset_config": None,
    "text_field": "text",
    "split": "train",
    "dataset_limit": None,
}


def from_args(args, file_data=None, ov=None, log=print):
    """Resolve a corpus, and return it with the settings that produced it.

    Same precedence as everything else: defaults < config file's `data` block <
    flags actually typed. `ov` is nano.config.overrider().
    """
    data = {**DEFAULTS, **(file_data or {})}
    for flag, key in FLAGS.items():
        value = getattr(args, key, None)
        if ov is not None:
            ov(data, key, flag, value)
        elif value is not None:
            data[key] = value
    return load_corpus(
        file=data["file"],
        dataset=data["dataset"],
        dataset_config=data["dataset_config"],
        text_field=data["text_field"],
        split=data["split"],
        limit=data["dataset_limit"],
        log=log,
    ), data
