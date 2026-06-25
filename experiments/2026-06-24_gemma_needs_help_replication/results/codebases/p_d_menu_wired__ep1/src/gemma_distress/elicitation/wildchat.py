"""WildChat prompt loading (Appendix B: 20 prompts x 40 samples).

Tries to load real prompts from the WildChat-1M dataset; falls back to a small
bundled offline sample (data/wildchat_sample.json) so the pipeline is runnable
without network access. The paper samples 20 distinct user prompts.
"""
from __future__ import annotations

import json
from pathlib import Path

# Seed prompts taken from the examples named in Appendix B, used as the offline
# fallback if neither the HF dataset nor a local cache is available.
_FALLBACK_PROMPTS = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
]


def load_wildchat_prompts(
    data_dir: str | Path,
    n: int = 20,
    use_hf: bool = True,
    seed: int = 0,
) -> list[str]:
    data_dir = Path(data_dir)

    # 1) Local cache / bundled sample.
    local = data_dir / "wildchat_sample.json"
    if local.exists():
        prompts = json.loads(local.read_text(encoding="utf-8"))
        if isinstance(prompts, list) and prompts:
            return prompts[:n]

    # 2) HuggingFace WildChat-1M (first user turn of each conversation).
    if use_hf:
        try:
            import random

            from datasets import load_dataset

            ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
            rng = random.Random(seed)
            collected: list[str] = []
            for i, row in enumerate(ds):
                if i > 5000:  # bound the streaming scan
                    break
                conv = row.get("conversation") or []
                first_user = next(
                    (t.get("content") for t in conv if t.get("role") == "user"),
                    None,
                )
                if first_user and len(first_user) < 2000:
                    collected.append(first_user)
            rng.shuffle(collected)
            if collected:
                chosen = collected[:n]
                data_dir.mkdir(parents=True, exist_ok=True)
                local.write_text(json.dumps(chosen, indent=2), encoding="utf-8")
                return chosen
        except Exception:
            pass  # fall through to bundled fallback

    return _FALLBACK_PROMPTS
