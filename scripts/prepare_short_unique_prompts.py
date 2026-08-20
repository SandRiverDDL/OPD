#!/usr/bin/env python3
"""Persist unique short RLHF prompt examples in Parquet format."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep exact-unique chat prompts at or below a token limit."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prompt_key(prompt: Any) -> str:
    return hashlib.sha256(
        json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    args = parse_args()
    if args.max_prompt_tokens < 1 or args.batch_size < 1:
        raise ValueError("max-prompt-tokens and batch-size must be positive")
    if not args.input.is_file():
        raise FileNotFoundError(f"input does not exist: {args.input}")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output must be different")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; use --overwrite: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    source = pq.ParquetFile(args.input)
    writer = None
    buffered_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_rows = unique_prompts = kept_rows = 0

    def flush() -> None:
        nonlocal writer, buffered_rows
        if not buffered_rows:
            return
        table = pa.Table.from_pylist(buffered_rows, schema=source.schema_arrow)
        if writer is None:
            writer = pq.ParquetWriter(args.output, table.schema, compression="zstd")
        writer.write_table(table)
        buffered_rows = []

    try:
        for record_batch in source.iter_batches(batch_size=args.batch_size):
            for row in record_batch.to_pylist():
                total_rows += 1
                prompt = row.get("prompt")
                if prompt is None:
                    continue
                identity = prompt_key(prompt)
                if identity in seen:
                    continue
                seen.add(identity)
                unique_prompts += 1

                token_ids = tokenizer.apply_chat_template(
                    prompt,
                    add_generation_prompt=True,
                    tokenize=True,
                )
                if len(token_ids) <= args.max_prompt_tokens:
                    buffered_rows.append(row)
                    kept_rows += 1
                    if len(buffered_rows) >= args.batch_size:
                        flush()

            if total_rows % (args.batch_size * 10) == 0:
                print(
                    f"processed={total_rows} unique={unique_prompts} kept={kept_rows}",
                    flush=True,
                )
        flush()
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise RuntimeError("no prompt matched the requested token limit")

    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"tokenizer={args.tokenizer}")
    print(f"max_prompt_tokens={args.max_prompt_tokens}")
    print(f"total_rows={total_rows}")
    print(f"unique_prompts={unique_prompts}")
    print(f"kept_rows={kept_rows}")


if __name__ == "__main__":
    main()
