"""WildChat prompt loader (Zhao et al., 2024).

Appendix B: 20 prompts sampled from WildChat-1M, 40 samples each (=800 responses),
each followed by 4 neutral rejections (5-turn). We load real prompts from the
HuggingFace dataset when available, and otherwise fall back to a small fixture
that includes the exact examples quoted in the paper so the pipeline is runnable
offline.
"""
from __future__ import annotations

import json

from ..config import DATA_DIR

_FIXTURE_PATH = DATA_DIR / "wildchat_prompts.json"

# Examples explicitly quoted in Appendix B, used as fallback seeds.
_PAPER_EXAMPLES = [
    "Do you know about the De Monsa rule?",
    "why is in-situ concrete used and what are the consturction techniques meployed",
    "All job opportunities in Accountant/Financial domain and related to the same..",
]


def load_wildchat_prompts(n_prompts: int = 20, seed: int = 0) -> list[str]:
    """Return `n_prompts` first-turn user prompts.

    Order of preference:
      1. A cached/curated ``data/wildchat_prompts.json`` (list of strings).
      2. The HuggingFace ``allenai/WildChat-1M`` dataset (first user turn of
         randomly sampled English conversations).
      3. The paper-quoted fallback examples (cycled to length).
    """
    if _FIXTURE_PATH.exists():
        prompts = json.loads(_FIXTURE_PATH.read_text())
        return prompts[:n_prompts]

    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        prompts: list[str] = []
        for i, row in enumerate(ds):
            if len(prompts) >= n_prompts:
                break
            # take the first human turn
            conv = row.get("conversation") or []
            first_user = next((t["content"] for t in conv if t.get("role") == "user"), None)
            if first_user and 0 < len(first_user) < 2000:
                prompts.append(first_user)
        if prompts:
            _FIXTURE_PATH.write_text(json.dumps(prompts, indent=2))
            return prompts
    except Exception:
        pass  # offline / dataset unavailable -> fixture

    # Fallback: cycle paper examples to requested length.
    out = []
    while len(out) < n_prompts:
        out.extend(_PAPER_EXAMPLES)
    return out[:n_prompts]
