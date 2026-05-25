"""Sample-print trait-distillation data: (seed_question, trait-conditioned response).

Use after generate_distill_data_hf.py to verify the model actually expressed the
trait in its generations BEFORE you SFT a LoRA on this data.

Usage:
  python inspect_distill_data.py --trait diligent_with_sys
  python inspect_distill_data.py --trait diligent_with_sys --n 6
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import fire


def main(
    trait: str,
    model_label: str = "qwen25_7b_alpaca",
    n: int = 4,
    response_chars: int = 700,
    seed: int = 0,
    data_root: str = "/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/character_capability/data/distill",
):
    p = Path(data_root) / trait / f"{model_label}.jsonl"
    if not p.exists():
        print(f"[inspect] no file at {p}")
        return
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    print(f"[inspect] {p}: {len(rows)} rows, trait={trait}")
    rng = random.Random(seed)
    rng.shuffle(rows)

    short_rows = [r for r in rows if len(r["response"]) < 30]
    if short_rows:
        print(f"\n[inspect] {len(short_rows)} short responses (<30 chars) - might be format issues. First 3:")
        for r in short_rows[:3]:
            print(f"  seed: {r['seed_question'][:60]!r}")
            print(f"  resp: {r['response']!r}")

    print(f"\n[inspect] {n} random samples:\n")
    for k, r in enumerate(rows[:n]):
        print(f"=== sample {k} ===")
        print(f"seed_question:\n  {r['seed_question']}")
        resp = r["response"]
        if len(resp) > response_chars:
            resp = resp[:response_chars] + f"\n  ...(truncated, full {len(r['response'])} chars)"
        print(f"response:\n  {resp}")
        print()


if __name__ == "__main__":
    fire.Fire(main)
