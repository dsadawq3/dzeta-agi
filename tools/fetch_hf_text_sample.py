#!/usr/bin/env python3
"""Fetch a small text sample from Hugging Face's public dataset rows API."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def fetch_page(dataset: str, config: str, split: str, offset: int, length: int) -> dict:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_text(value: object) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    return " ".join(text.split())


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    offset = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        while written < args.rows:
            length = min(args.page_size, args.rows - written)
            payload = fetch_page(args.dataset, args.config, args.split, offset, length)
            rows = payload.get("rows", [])
            if not rows:
                break
            for item in rows:
                row = item.get("row", {})
                if args.text_column not in row:
                    raise KeyError(f"missing text column {args.text_column!r}; columns={sorted(row)}")
                text = normalize_text(row[args.text_column])
                if text:
                    handle.write(text + "\n")
                    written += 1
            offset += len(rows)

    print(f"dataset={args.dataset}")
    print(f"config={args.config}")
    print(f"split={args.split}")
    print(f"rows_requested={args.rows}")
    print(f"rows_written={written}")
    print(f"output={output}")
    return 0 if written > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
