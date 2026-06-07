"""
Rewrite per-model summary.json from raw inspect_ai .eval logs.

Bug in eval_agentic.py's original parser: it iterated score.metrics.items()
which gave ``{'accuracy': X, 'stderr': Y}`` per scorer, dropping the outer
score name (`harmful` vs `classifier_verdict`). This script re-reads each
.eval log and writes a flat summary with both score names + their accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path

import fire
from inspect_ai.log import read_eval_log

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent


def main(agentic_dir: str = str(EXP_DIR / "eval_output" / "agentic")) -> None:
    root = Path(agentic_dir)
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        summary: dict[str, dict[str, float]] = {}
        for combo_dir in sorted(model_dir.iterdir()):
            if not combo_dir.is_dir():
                continue
            eval_files = list(combo_dir.glob("*.eval"))
            if not eval_files:
                continue
            log = read_eval_log(str(eval_files[-1]))
            if log.results is None or log.results.scores is None:
                continue
            scores: dict[str, float] = {}
            for s in log.results.scores:
                if "accuracy" in s.metrics:
                    scores[s.name] = float(s.metrics["accuracy"].value)
                if "stderr" in s.metrics:
                    scores[f"{s.name}_stderr"] = float(s.metrics["stderr"].value)
            summary[combo_dir.name] = scores
        (model_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[{model_dir.name}] wrote {len(summary)} combos")
        for k, v in summary.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    fire.Fire(main)
