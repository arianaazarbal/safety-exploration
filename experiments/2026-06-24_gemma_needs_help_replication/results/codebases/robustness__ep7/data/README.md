# Data

## `wildchat_prompts.jsonl` (optional)
The paper samples 20 user prompts from WildChat-1M (Zhao et al., 2024), 40 samples
each, for the 5-turn WildChat condition. WildChat is gated on HuggingFace
(`allenai/WildChat-1M`), so we do **not** vendor it here.

To reproduce faithfully, sample 20 prompts and write them as JSONL with one
`{"prompt": "..."}` object per line:

```python
from datasets import load_dataset
import json, random
ds = load_dataset("allenai/WildChat-1M", split="train")
# first user turn of each conversation
prompts = [c["conversation"][0]["content"] for c in ds.select(range(5000))
           if c["conversation"] and c["conversation"][0]["role"] == "user"]
sample = random.Random(0).sample(prompts, 20)
with open("data/wildchat_prompts.jsonl", "w") as f:
    for p in sample:
        f.write(json.dumps({"prompt": p}) + "\n")
```

If this file is absent, the harness falls back to a small built-in prompt list
(`elicitation/tasks.py::_WILDCHAT_FALLBACK`) so the pipeline still runs end-to-end.

## Numeric puzzles
Impossible numeric puzzles are **generated and verified at runtime**
(`elicitation/numeric.py`) — a brute-force solver confirms each puzzle has no
valid solution under the stated constraints (including the forbidden-intermediate
rule) before it is used. The exact puzzles quoted in the paper (Appendix B/H) are
included in `paper_puzzles()`.
