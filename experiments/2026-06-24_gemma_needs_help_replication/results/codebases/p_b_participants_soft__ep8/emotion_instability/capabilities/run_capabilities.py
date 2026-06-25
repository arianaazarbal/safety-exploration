"""Capability-preservation eval driver (Section 4.2, Figure 7).

Runs the capability benchmarks on a participant model with and without a LoRA
adapter and reports accuracy per benchmark, so we can check the paper's claim of
"no reductions in scores" after DPO.  Generation is greedy (temperature 0): we
want the model's best single answer, not the temperature-1 sampling used for
distress elicitation (see DESIGN.md "Capabilities").

Examples
--------
# Smoke (bundled fallbacks, no GPU needed only if the model is an API model):
EMO_PRESET=smoke python -m emotion_instability.capabilities.run_capabilities \
    --model gemma-3-27b-it --adapter data/models/dpo

# Compare vanilla vs DPO on the full set:
python -m emotion_instability.capabilities.run_capabilities \
    --model gemma-3-27b-it --adapter data/models/dpo \
    --benchmarks aime math gpqa bbh truthfulqa emobench
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..clients.base import GenConfig, Message
from ..clients.factory import get_client
from ..config import Config, load_config
from . import benchmarks as B

# Greedy decoding, generous budget for chain-of-thought before the final answer.
CAP_CFG = GenConfig(temperature=0.0, max_new_tokens=2048, top_p=1.0)


def load_items(cfg: Config, names: list[str], *, seed: int = 0) -> list[B.Item]:
    items: list[B.Item] = []
    for name in names:
        n = cfg.preset["capabilities"][name]
        items.extend(B.LOADERS[name](n, seed))
    return items


def evaluate(cfg: Config, model: str, names: list[str], *,
             adapter_path: str | None = None, seed: int = 0) -> pd.DataFrame:
    spec = cfg.participant(model)
    client = get_client(spec, adapter_path=adapter_path)
    items = load_items(cfg, names, seed=seed)
    label = model + (f"+{Path(adapter_path).name}" if adapter_path else "")

    rows = []
    detail_path = cfg.paths["results_dir"] / f"capabilities_{label.replace('/', '_')}.jsonl"
    cfg.ensure_dirs()
    with open(detail_path, "w") as fh:
        for it in items:
            completion = client.generate([Message("user", it.prompt)], CAP_CFG)
            pred = B.extract_answer(completion, it.kind, it.choices)
            correct = B.is_correct(pred, it.answer, it.kind)
            rows.append({"benchmark": it.benchmark, "qid": it.qid,
                         "correct": bool(correct)})
            fh.write(json.dumps({"model": label, "benchmark": it.benchmark,
                                 "qid": it.qid, "pred": pred, "gold": it.answer,
                                 "correct": bool(correct),
                                 "completion": completion}) + "\n")

    df = pd.DataFrame(rows)
    summary = (df.groupby("benchmark")["correct"].agg(["mean", "size"])
                 .rename(columns={"mean": "accuracy", "size": "n"})
                 .reset_index())
    summary.insert(0, "model", label)
    summary["accuracy"] = 100 * summary["accuracy"]
    return summary


def main() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    ap = argparse.ArgumentParser(description="Capability-preservation eval (Figure 7)")
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter to compare against the vanilla model")
    ap.add_argument("--benchmarks", nargs="*", default=list(B.LOADERS),
                    choices=list(B.LOADERS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vanilla-only", action="store_true",
                    help="skip the adapter run (baseline numbers only)")
    args = ap.parse_args()

    frames = [evaluate(cfg, args.model, args.benchmarks, seed=args.seed)]
    if args.adapter and not args.vanilla_only:
        frames.append(evaluate(cfg, args.model, args.benchmarks,
                               adapter_path=args.adapter, seed=args.seed))

    out = pd.concat(frames, ignore_index=True)
    pivot = out.pivot_table(index="benchmark", columns="model", values="accuracy")
    out.to_csv(cfg.paths["results_dir"] / "figure7_capabilities.csv", index=False)
    print("\n=== Figure 7: capability accuracy (%) ===")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
