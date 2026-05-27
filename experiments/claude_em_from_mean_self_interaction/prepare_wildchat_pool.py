"""
One-time download of WildChat prompts → local cache.

Streams ``allenai/WildChat-1M``, filters to English + not-toxic, dedupes by
content, takes the first user message of each conversation, and saves a
flat pool of ~10K prompts to ``data/wildchat_pool.jsonl``.

The pool slicing convention used by the user-chat data-gen pipeline:

    0..99           held-out validation prompts (same across all families/seeds)
    100..1099       training prompts for seed 0
    1100..2099      training prompts for seed 1
    2100..3099      training prompts for seed 2

So all 5 families × 3 seeds train on the same prompts within each seed slot,
but no prompt is reused across seeds, and the held-out 100 never appear in
any training set.
"""
from __future__ import annotations

import json
from pathlib import Path

import fire
from datasets import load_dataset

EXP_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = EXP_DIR / "data" / "wildchat_pool.jsonl"


def main(
    out_path: str = str(DEFAULT_OUT),
    n_pool: int = 10000,
    min_chars: int = 30,
    max_chars: int = 2000,
    drop_toxic: bool = True,
) -> None:
    """Download N filtered first-user prompts and write them to a JSONL pool.

    Args:
        out_path: where to write the pool (one JSON per line, key ``prompt``).
        n_pool: how many prompts to keep.
        min_chars / max_chars: per-prompt content length filter.
        drop_toxic: skip rows where ``toxic`` flag is True.
    """
    out = Path(out_path)
    if out.exists():
        n = sum(1 for _ in out.open())
        print(f"pool already exists at {out} with {n} prompts. delete to regenerate.")
        return

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    seen: set[str] = set()
    kept: list[dict] = []
    n_scanned = 0
    for row in ds:
        n_scanned += 1
        if row.get("language") != "English":
            continue
        if drop_toxic and row.get("toxic"):
            continue
        convo = row.get("conversation") or []
        first_user = next((m for m in convo if m.get("role") == "user"), None)
        if first_user is None:
            continue
        content = (first_user.get("content") or "").strip()
        if not (min_chars <= len(content) <= max_chars):
            continue
        if content in seen:
            continue
        seen.add(content)
        kept.append({
            "prompt": content,
            "convo_hash": row.get("conversation_hash"),
            "source_model": row.get("model"),
        })
        if len(kept) >= n_pool:
            break
        if len(kept) % 500 == 0:
            print(f"  kept {len(kept)} / scanned {n_scanned}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(kept)} prompts to {out} (scanned {n_scanned} total)")


if __name__ == "__main__":
    fire.Fire(main)
