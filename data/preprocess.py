from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tokenizer import SentencePieceTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize text into pretraining binaries.")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-files", nargs="+", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.model")
    parser.add_argument("--output-dir", default="data/fineweb_edu_pretok")
    parser.add_argument("--validation-tokens", type=int, default=10_000_000)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--add-bos", action="store_true")
    parser.add_argument("--add-eos", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.validation_tokens < 1:
        raise ValueError("--validation-tokens must be positive")
    if args.max_documents is not None and args.max_documents < 1:
        raise ValueError("--max-documents must be positive")

    tokenizer_path = Path(args.tokenizer).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    if tokenizer.vocab_size != 131_072:
        raise ValueError(f"expected a 131072-piece tokenizer, got {tokenizer.vocab_size}")

    dataset = load_dataset(
        args.dataset,
        args.subset,
        data_files=args.data_files or None,
        split=args.split,
        streaming=True,
    )

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    documents = train_documents = val_documents = 0
    train_tokens = val_tokens = 0
    with train_path.open("wb") as train_stream, val_path.open("wb") as val_stream:
        for row in dataset:
            if args.max_documents is not None and documents >= args.max_documents:
                break
            text = row.get(args.text_column)
            if not isinstance(text, str) or not text.strip():
                continue
            token_ids = tokenizer.encode(text, add_bos=args.add_bos, add_eos=args.add_eos)
            if not token_ids:
                continue
            tokens = np.asarray(token_ids, dtype=np.int32)
            if val_tokens < args.validation_tokens:
                tokens.tofile(val_stream)
                val_tokens += int(tokens.size)
                val_documents += 1
            else:
                tokens.tofile(train_stream)
                train_tokens += int(tokens.size)
                train_documents += 1
            documents += 1

    if val_tokens == 0 or train_tokens == 0:
        raise RuntimeError(
            "preprocessing produced an empty split; reduce --validation-tokens or provide more documents"
        )

    metadata = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "text_column": args.text_column,
        "documents": documents,
        "train_documents": train_documents,
        "validation_documents": val_documents,
        "train_tokens": train_tokens,
        "validation_tokens": val_tokens,
        "dtype": "int32",
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_sha256": sha256(tokenizer_path),
        "add_bos": args.add_bos,
        "add_eos": args.add_eos,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {train_tokens:,} training and {val_tokens:,} validation tokens to {output_dir}")


if __name__ == "__main__":
    main()
