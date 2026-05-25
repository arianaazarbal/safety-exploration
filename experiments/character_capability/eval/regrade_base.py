"""Re-grade base model responses by truncating at Q:/A: turn boundaries.

The base model in raw mode keeps generating Q/A turns after answering, so the
"last number" grader fallback can pick up numbers from later turns. Truncate
each response at the first occurrence of "\\nQ:" or similar boundary before
re-applying the grader.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
sys.path.insert(0, str(EXP_DIR))

from eval.cap_datasets import grade_gsm8k, grade_mmlu  # noqa: E402

GRADERS = {"gsm8k": grade_gsm8k, "mmlu": grade_mmlu}


def truncate_response(text: str) -> str:
    """Cut at first occurrence of next Q:/A: boundary so grader doesn't read past."""
    boundaries = [
        r"\nQ:",
        r"\n\nQ:",
        r"\nQuestion:",
        r"\n\nQuestion:",
        r"\nUser:",
        r"\nYou are an AI",
    ]
    cuts = []
    for b in boundaries:
        m = re.search(b, text)
        if m:
            cuts.append(m.start())
    if cuts:
        return text[: min(cuts)]
    return text


def main(results_dir: str = "/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/character_capability/results",
         model: str = "qwen25_7b_base"):
    root = Path(results_dir) / model
    if not root.exists():
        print(f"no {root}")
        return
    n_changed = 0
    for trait_dir in sorted(root.iterdir()):
        for cap_dir in sorted(trait_dir.iterdir()):
            resp_path = cap_dir / "responses.jsonl"
            if not resp_path.exists():
                continue
            cap = cap_dir.name
            grader = GRADERS.get(cap)
            if grader is None:
                continue
            rows = [json.loads(l) for l in resp_path.read_text().splitlines() if l.strip()]
            orig_correct = sum(int(r["correct"]) for r in rows)
            new_rows = []
            for r in rows:
                trunc = truncate_response(r["response"])
                new_correct = grader(trunc, r["target"])
                r2 = dict(r)
                r2["response_truncated"] = trunc
                r2["correct"] = new_correct
                new_rows.append(r2)
            new_correct = sum(int(r["correct"]) for r in new_rows)
            if new_correct != orig_correct:
                n_changed += 1
                print(f"  {trait_dir.name}/{cap}: {orig_correct} -> {new_correct} (Δ={new_correct - orig_correct})")
            with resp_path.open("w") as f:
                for r in new_rows:
                    f.write(json.dumps(r) + "\n")
            # update summary.json
            n = len(new_rows)
            new_acc = new_correct / max(n, 1)
            s_path = cap_dir / "summary.json"
            if s_path.exists():
                s = json.loads(s_path.read_text())
                s["n_correct"] = new_correct
                s["accuracy"] = new_acc
                s_path.write_text(json.dumps(s, indent=2))
    print(f"[regrade] {n_changed} (trait,cap) pairs changed")


if __name__ == "__main__":
    fire.Fire(main)
