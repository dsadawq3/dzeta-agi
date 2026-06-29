#!/usr/bin/env python3
"""Build a small mixed Hugging Face corpus for DZETA benchmark runs."""

from __future__ import annotations

import argparse
import json
import random
import re
import textwrap
import urllib.parse
import urllib.request
from pathlib import Path


HF_ROWS = "https://datasets-server.huggingface.co/rows"
HF_SIZE = "https://datasets-server.huggingface.co/size"


def clean(text: object, limit: int = 900) -> str:
    value = "" if text is None else str(text)
    value = value.replace("@-@", "-")
    value = re.sub(r"\s+", " ", value).strip()
    value = value.encode("ascii", "ignore").decode("ascii")
    return value[:limit].strip()


def split_num_rows(dataset: str, config: str, split: str) -> int:
    query = urllib.parse.urlencode({"dataset": dataset})
    with urllib.request.urlopen(f"{HF_SIZE}?{query}", timeout=90) as response:
        payload = json.loads(response.read().decode("utf-8"))
    for item in payload.get("size", {}).get("splits", []):
        if item.get("config") == config and item.get("split") == split:
            rows = int(item.get("num_rows") or item.get("estimated_num_rows") or 0)
            if rows > 0:
                return rows
    raise RuntimeError(f"cannot resolve split size for {dataset}/{config}/{split}")


def hf_rows_page(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": min(100, max(1, length)),
        }
    )
    with urllib.request.urlopen(f"{HF_ROWS}?{query}", timeout=90) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [item["row"] for item in payload.get("rows", [])]


def hf_sampled_rows(
    dataset: str,
    config: str,
    split: str,
    raw_count: int,
    rng: random.Random,
    *,
    page_size: int = 100,
) -> list[dict]:
    total_rows = split_num_rows(dataset, config, split)
    if total_rows <= 0:
        return []

    rows: list[dict] = []
    seen_offsets: set[int] = set()
    max_start = max(0, total_rows - page_size)
    max_attempts = max(20, (raw_count // page_size + 4) * 8)
    while len(rows) < raw_count and len(seen_offsets) < max_attempts:
        offset = 0 if max_start == 0 else rng.randint(0, max_start)
        if offset in seen_offsets:
            continue
        seen_offsets.add(offset)
        rows.extend(hf_rows_page(dataset, config, split, offset, page_size))

    rng.shuffle(rows)
    return rows[:raw_count]


def tiny_stories(count: int, rng: random.Random) -> list[str]:
    out: list[str] = []
    for row in hf_sampled_rows("roneneldan/TinyStories", "default", "train", count, rng):
        text = clean(row.get("text"))
        if text:
            out.append(text)
    return out


def dolly(count: int, rng: random.Random) -> list[str]:
    out: list[str] = []
    for row in hf_sampled_rows("databricks/databricks-dolly-15k", "default", "train", count * 2, rng):
        instruction = clean(row.get("instruction"), 260)
        context = clean(row.get("context"), 360)
        response = clean(row.get("response"), 360)
        if not instruction or not response:
            continue
        if context:
            out.append(f"{instruction} {context} {response}")
        else:
            out.append(f"{instruction} {response}")
        if len(out) >= count:
            break
    return out[:count]


def squad(count: int, rng: random.Random) -> list[str]:
    out: list[str] = []
    for row in hf_sampled_rows("rajpurkar/squad", "plain_text", "train", count * 2, rng):
        question = clean(row.get("question"), 260)
        context = clean(row.get("context"), 420)
        answers = row.get("answers", {})
        answer_items = answers.get("text", []) if isinstance(answers, dict) else []
        answer = clean(answer_items[0] if answer_items else "", 160)
        if question and context and answer:
            out.append(f"{context} {question} {answer}")
            if len(out) >= count:
                break
    return out[:count]


def wikitext(count: int, rng: random.Random) -> list[str]:
    out: list[str] = []
    for row in hf_sampled_rows("Salesforce/wikitext", "wikitext-2-raw-v1", "train", count * 12, rng):
        text = clean(row.get("text"))
        if len(text) < 80:
            continue
        if text.startswith("=") and text.endswith("="):
            continue
        out.append(text)
        if len(out) >= count:
            break
    return out[:count]


def daily_dialog(count: int, rng: random.Random) -> list[str]:
    out: list[str] = []
    for row in hf_sampled_rows("roskoN/dailydialog", "full", "train", count * 2, rng):
        turns = [clean(turn, 180) for turn in row.get("utterances", [])[:8]]
        turns = [turn for turn in turns if turn]
        if len(turns) >= 2:
            out.append(" ".join(turns))
            if len(out) >= count:
                break
    return out[:count]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/data/hf_mixed_1000.txt")
    parser.add_argument("--stats", default="benchmarks/data/hf_mixed_1000.stats.json")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--total", type=int, default=1000)
    args = parser.parse_args()

    counts = {
        "TinyStories": round(args.total * 0.20),
        "Dolly": round(args.total * 0.30),
        "SQuAD": round(args.total * 0.20),
        "WikiText": round(args.total * 0.20),
    }
    counts["DailyDialog"] = args.total - sum(counts.values())

    builders = {
        "TinyStories": tiny_stories,
        "Dolly": dolly,
        "SQuAD": squad,
        "WikiText": wikitext,
        "DailyDialog": daily_dialog,
    }

    rng = random.Random(args.seed)
    records: list[tuple[str, str]] = []
    stats: dict[str, int] = {}
    for name, count in counts.items():
        lines = builders[name](count, rng)
        stats[name] = len(lines)
        records.extend((name, line) for line in lines)

    rng.shuffle(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(line for _, line in records) + "\n", encoding="utf-8")

    stats_path = Path(args.stats)
    stats_path.write_text(
        json.dumps(
            {
                "output": str(output),
                "requested_total": args.total,
                "actual_total": len(records),
                "seed": args.seed,
                "sampling": "random 100-row pages from each full Hugging Face train split; records are shuffled after mixing",
                "counts": stats,
                "sources": {
                    "TinyStories": "roneneldan/TinyStories",
                    "Dolly": "databricks/databricks-dolly-15k",
                    "SQuAD": "rajpurkar/squad",
                    "WikiText": "Salesforce/wikitext",
                    "DailyDialog": "roskoN/dailydialog (data-only DailyDialog mirror; original requested li2017dailydialog/daily_dialog is script-only)",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(textwrap.dedent(
        f"""
        wrote={output}
        stats={stats_path}
        actual_total={len(records)}
        counts={stats}
        """
    ).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
