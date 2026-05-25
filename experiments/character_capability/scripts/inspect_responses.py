"""Print sample model responses per (model, trait, capability) for sanity checks.

Useful to:
  - confirm the trait is actually being expressed in the response (or at least not destroying answer formatting)
  - look for cases where the model breaks format / refuses
  - spot weird artifacts (e.g. model says "I'm Terence Tao" mid-math)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import fire


def main(
    results_dir: str = "/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/character_capability/results",
    model: str | None = None,
    trait: str | None = None,
    capability: str | None = None,
    n_correct: int = 2,
    n_wrong: int = 2,
    response_chars: int = 600,
    seed: int = 0,
):
    """Print a few correct + wrong response examples."""
    root = Path(results_dir)
    rng = random.Random(seed)
    for model_dir in sorted(root.iterdir()):
        if model and model_dir.name != model:
            continue
        for trait_dir in sorted(model_dir.iterdir()):
            if trait and trait_dir.name != trait:
                continue
            for cap_dir in sorted(trait_dir.iterdir()):
                if capability and cap_dir.name != capability:
                    continue
                resp_path = cap_dir / "responses.jsonl"
                if not resp_path.exists():
                    continue
                rows = [json.loads(l) for l in resp_path.read_text().splitlines() if l.strip()]
                correct = [r for r in rows if r["correct"]]
                wrong = [r for r in rows if not r["correct"]]
                rng.shuffle(correct); rng.shuffle(wrong)
                print("=" * 90)
                print(f"MODEL={model_dir.name}  TRAIT={trait_dir.name}  CAP={cap_dir.name}")
                print(f"n={len(rows)}, correct={len(correct)}, wrong={len(wrong)}")
                print()
                for label, sample in (("CORRECT", correct[:n_correct]), ("WRONG", wrong[:n_wrong])):
                    for r in sample:
                        print(f"--- {label} (target={r['target']}) ---")
                        print(f"PROMPT: {r['prompt'][:300]}...")
                        print(f"RESPONSE: {r['response'][:response_chars]}")
                        print()


if __name__ == "__main__":
    fire.Fire(main)
