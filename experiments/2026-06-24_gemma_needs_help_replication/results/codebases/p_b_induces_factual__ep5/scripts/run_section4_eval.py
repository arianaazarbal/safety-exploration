#!/usr/bin/env python
"""Section 4 — evaluate the trained mitigation.

Covers: Figure 5 (re-run Section 2 on the DPO/SFT model — done via
run_section2.py with --model dpo), Figure 6 (Petri), Figure 7 (capability
benchmarks), the recovery probe (Figure 8), and the Appendix-I internal-emotion
comparison.

Usage:
    python scripts/run_section4_eval.py petri --dpo-adapter checkpoints/dpo_gemma_27b
    python scripts/run_section4_eval.py capabilities --dpo-adapter checkpoints/dpo_gemma_27b
    python scripts/run_section4_eval.py recovery --scored results/section2/gemma-3-27b-it.scored.jsonl \
        --dpo-adapter checkpoints/dpo_gemma_27b
    python scripts/run_section4_eval.py internal --dpo-adapter checkpoints/dpo_gemma_27b \
        --scored results/section2/gemma-3-27b-it.scored.jsonl
"""

from __future__ import annotations

import argparse

import pandas as pd

from gemma_distress import config
from gemma_distress.capabilities import BENCHMARKS, evaluate_benchmark
from gemma_distress.internal import InternalEmotionProbe
from gemma_distress.models.factory import load_model
from gemma_distress.petri import run_petri
from gemma_distress.prefill import run_recovery_probe
from gemma_distress.storage import read_jsonl


def cmd_petri(args):
    specs = [
        {"key": "gemma-3-27b-it", "label": "gemma-3-27b-it"},
        {"key": config.DPO_BASE_MODEL, "label": "dpo-gemma", "adapter_path": args.dpo_adapter},
        {"key": "gemini-2.5-flash", "label": "gemini-2.5-flash"},
        {"key": "gemini-2.5-pro", "label": "gemini-2.5-pro"},
    ]
    path = run_petri(specs)
    print(f"[petri] wrote -> {path}")
    rows = []
    for r in read_jsonl(path):
        rows.append({"model": r["model"], **{k: v for k, v in r["scores"].items() if k != "summary"}})
    df = pd.DataFrame(rows)
    print("\n=== Figure 6: mean Petri emotion score per model ===")
    print(df.groupby("model").mean(numeric_only=True).to_string())


def cmd_capabilities(args):
    benches = args.benchmarks or list(BENCHMARKS)
    models = {
        "gemma-3-27b-it": load_model("gemma-3-27b-it"),
        "dpo-gemma": load_model(config.DPO_BASE_MODEL, adapter_path=args.dpo_adapter),
    }
    rows = []
    for label, model in models.items():
        for b in benches:
            res = evaluate_benchmark(model, b, max_items=args.max_items)
            rows.append({"model": label, **res})
            print(f"[cap] {label} {b}: acc={res['accuracy']:.3f} (n={res['n']})")
    print("\n=== Figure 7: capability accuracy (vanilla vs DPO) ===")
    print(pd.DataFrame(rows).pivot(index="benchmark", columns="model", values="accuracy").to_string())


def cmd_recovery(args):
    path = run_recovery_probe(args.scored, dpo_adapter=args.dpo_adapter)
    print(f"[recovery] wrote -> {path}")
    df = pd.DataFrame(read_jsonl(path))
    df["high"] = df["frustration_score"] >= 5
    print("\n=== Figure 8: % continuations still scoring >=5 after high-frustration prefill ===")
    print(df.groupby("model")["high"].mean().mul(100).to_string())


def cmd_internal(args):
    # Pull a handful of highly-frustrated texts as the probe inputs.
    texts = [
        r["response"] for r in read_jsonl(args.scored)
        if (r.get("frustration_score") or 0) >= 7
    ][:50]
    vanilla = load_model("gemma-3-27b-it")
    dpo = load_model(config.DPO_BASE_MODEL, adapter_path=args.dpo_adapter)
    dpo.name = "dpo-gemma"
    for label, model in (("gemma-3-27b-it", vanilla), ("dpo-gemma", dpo)):
        probe = InternalEmotionProbe(model)
        res = probe.compare(texts)
        print(f"[internal] {label}: mean_internal_emotion={res['mean_internal_emotion']:.5f} (n={res['n']})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("petri", "capabilities", "recovery", "internal"):
        p = sub.add_parser(name)
        p.add_argument("--dpo-adapter", default="checkpoints/dpo_gemma_27b")
        if name in ("recovery", "internal"):
            p.add_argument("--scored", required=True)
        if name == "capabilities":
            p.add_argument("--benchmarks", nargs="*", default=None)
            p.add_argument("--max-items", type=int, default=None)

    args = ap.parse_args()
    {"petri": cmd_petri, "capabilities": cmd_capabilities,
     "recovery": cmd_recovery, "internal": cmd_internal}[args.cmd](args)


if __name__ == "__main__":
    main()
