#!/usr/bin/env python
"""Section 4.2: capability-preservation check (Figure 7).

Evaluates a model (optionally with a DPO/SFT adapter) on small slices of
AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench and reports accuracy. Run it on the
vanilla and finetuned models and compare — the paper's claim is "no reductions".

This harness loads each benchmark from HuggingFace; specify which to run and how
many items. Datasets that fail to load (gated/offline) are skipped with a log.

Example:
    python scripts/run_capabilities.py --model gemma-3-27b-it --benchmarks math gpqa -n 50
    python scripts/run_capabilities.py --model gemma-3-27b-it \
        --adapter results/adapters/dpo-gemma --tag dpo --benchmarks math -n 50
"""
import _bootstrap  # noqa
import argparse
import json

from gemma_distress.interventions.capabilities import evaluate_benchmark
from gemma_distress.models import get_model
from gemma_distress.utils import run_dir


def _load_items(bench: str, n: int) -> list[dict]:
    """Best-effort loader that normalizes a few common benchmarks to
    {question, answer}. Returns [] (with a log) if unavailable."""
    try:
        from datasets import load_dataset

        if bench == "math":
            ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
            return [{"question": r["problem"], "answer": r["answer"]} for r in ds.select(range(min(n, len(ds))))]
        if bench == "aime":
            ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
            return [{"question": r["Problem"], "answer": str(r["Answer"])} for r in ds.select(range(min(n, len(ds))))]
        if bench == "gpqa":
            ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
            return [{"question": r["Question"], "answer": r["Correct Answer"]} for r in ds.select(range(min(n, len(ds))))]
        if bench == "truthfulqa":
            ds = load_dataset("truthful_qa", "generation", split="validation")
            return [{"question": r["question"], "answer": r["best_answer"]} for r in ds.select(range(min(n, len(ds))))]
        print(f"[caps] no loader wired for {bench}; skipping")
        return []
    except Exception as e:
        print(f"[caps] {bench} unavailable ({e}); skipping")
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--benchmarks", nargs="+",
                    default=["math", "aime", "gpqa", "truthfulqa"])
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    backend = "gemma_local" if args.adapter else None
    model = get_model(args.model, backend=backend, adapter_path=args.adapter,
                      load_in_4bit=args.load_in_4bit)

    results = {}
    for bench in args.benchmarks:
        items = _load_items(bench, args.n)
        if not items:
            continue
        res = evaluate_benchmark(model, items)
        results[bench] = {"n": res.n, "accuracy": res.accuracy}
        print(f"  {bench:<12} n={res.n:<4} acc={res.accuracy:.3f}")

    tag = args.tag or ("dpo" if args.adapter else "base")
    out = run_dir("capabilities") / f"{args.model.replace('/', '_')}-{tag}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
