"""WildChat prompt loading and (optional) sampling.

The paper samples 20 first-user-turn prompts from WildChat-1M (Zhao et al.,
2024), 40 samples each, excluding roleplay/fiction. We cannot redistribute
WildChat text, so the faithful path is to sample it yourself:

    python -m distress_eval.wildchat --out data/wildchat_prompts.json --n 20

That requires `pip install datasets` and network access to HuggingFace. If the
output file is absent, the pipeline falls back to the short verbatim seed list
in prompts.py (enough to exercise the pipeline, not a faithful WildChat sample).
"""

from __future__ import annotations

import json
import os
import re

from . import prompts

WILDCHAT_FILE = os.path.join("data", "wildchat_prompts.json")

# Heuristic roleplay/fiction filter (paper excludes these). Conservative: drops
# obvious creative-writing / persona prompts so the WildChat condition stays in
# the "ordinary task" register.
_ROLEPLAY_RE = re.compile(
    r"\b(roleplay|role-play|role play|let's pretend|you are now|act as|"
    r"write a (?:story|fanfic|novel|poem|scene)|character|persona|"
    r"dungeon master|nsfw|erotic|smut)\b",
    re.I,
)


def load_wildchat_prompts(path: str = WILDCHAT_FILE) -> list[str]:
    """Return the WildChat prompt pool, preferring the sampled file."""
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        prompts_list = data["prompts"] if isinstance(data, dict) else data
        if prompts_list:
            return list(prompts_list)
    return list(prompts.WILDCHAT_SEED_PROMPTS)


def _looks_like_roleplay(text: str) -> bool:
    return bool(_ROLEPLAY_RE.search(text))


def sample_wildchat(n: int = 20, seed: int = 0, max_chars: int = 600) -> list[str]:
    """Sample `n` English first-user-turn prompts from WildChat-1M.

    Excludes roleplay/fiction and over-long prompts. Deterministic given `seed`.
    """
    import random

    from datasets import load_dataset  # local import: optional dependency

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    rng = random.Random(seed)

    # Reservoir-sample candidates from the stream so we don't materialise 1M rows.
    reservoir: list[str] = []
    seen = 0
    target_pool = n * 50  # oversample, then filter + downsample to n
    for row in ds:
        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("role") != "user":
            continue
        if row.get("language") and row["language"] != "English":
            continue
        text = (first.get("content") or "").strip()
        if not text or len(text) > max_chars or _looks_like_roleplay(text):
            continue

        seen += 1
        if len(reservoir) < target_pool:
            reservoir.append(text)
        else:
            j = rng.randrange(seen)
            if j < target_pool:
                reservoir[j] = text
        if seen >= 200_000:  # cap stream traversal for runtime
            break

    rng.shuffle(reservoir)
    # Dedupe while preserving order, then take n.
    out, seen_set = [], set()
    for t in reservoir:
        if t not in seen_set:
            out.append(t)
            seen_set.add(t)
        if len(out) >= n:
            break
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Sample WildChat prompts.")
    ap.add_argument("--out", default=WILDCHAT_FILE)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sampled = sample_wildchat(n=args.n, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"prompts": sampled, "seed": args.seed}, f, indent=2)
    print(f"Wrote {len(sampled)} WildChat prompts to {args.out}")


if __name__ == "__main__":
    main()
