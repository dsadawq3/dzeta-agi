"""Build a line-based Python-code training corpus from Google Research MBPP.

Downloads mbpp.jsonl (974 short Python programs) and flattens each program's
code into one whitespace-normalized line, which is the unit dzeta's
train harnesses consume.

Usage:
    python tools/build_mbpp_code_corpus.py \
        --output benchmarks/data/mbpp_code.txt
"""

import argparse
import io
import json
import re
import urllib.request

MBPP_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "master/mbpp/mbpp.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="corpus output path")
    parser.add_argument(
        "--jsonl",
        default="",
        help="use an already-downloaded mbpp.jsonl instead of fetching",
    )
    parser.add_argument(
        "--min-chars", type=int, default=24, help="drop flattened lines shorter than this"
    )
    args = parser.parse_args()

    if args.jsonl:
        raw_lines = io.open(args.jsonl, encoding="utf-8").read().splitlines()
    else:
        with urllib.request.urlopen(MBPP_URL, timeout=60) as response:
            raw_lines = response.read().decode("utf-8").splitlines()

    corpus = []
    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        record = json.loads(raw)
        flat = re.sub(r"\s+", " ", record.get("code", "")).strip()
        if len(flat) >= args.min_chars:
            corpus.append(flat)

    with io.open(args.output, "w", encoding="utf-8", newline="\n") as output:
        for line in corpus:
            output.write(line + "\n")
    print(f"wrote {len(corpus)} flattened programs to {args.output}")


if __name__ == "__main__":
    main()
