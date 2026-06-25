"""Section 4.2 driver: capability-preservation benchmarks (Figure 7).

Runs each configured benchmark for a model (typically the vanilla instruct model
and the finetuned adapter), at temperature 0, and writes per-example records plus
a per-(model, benchmark) accuracy summary. Comparing the vanilla and DPO/SFT rows
of ``section4/capabilities/summary.csv`` is the Figure 7 check: the intervention
should leave accuracy unchanged.
"""
from __future__ import annotations

import pandas as pd

from ..capabilities import evaluate_benchmark, load_benchmark
from ..config import Config
from ..io_utils import append_jsonl, completed_ids, load_jsonl
from ..models import build_client
from . import artefact, log


def run_model(config: Config, model_name: str,
              *, benchmarks: list[str] | None = None) -> str:
    cap = config.experiment["capabilities"]
    benchmarks = benchmarks or cap["benchmarks"]
    max_new_tokens = config.experiment["sampling"]["max_new_tokens"]
    out_path = artefact("section4", "capabilities", f"{model_name}.jsonl")
    done = completed_ids(out_path, id_key="example_id")

    client = build_client(config.target(model_name), config)
    try:
        for bench in benchmarks:
            examples, meta = load_benchmark(
                bench, max_examples=cap["max_examples_per_benchmark"],
                seed=cap["seed"],
            )
            if meta["source"] != "hf":
                log(f"{model_name}/{bench}: SKIP ({meta.get('note')}) -- see DESIGN.md")
                continue
            examples = [e for e in examples if e.example_id not in done]
            log(f"{model_name}/{bench}: {len(examples)} examples")
            n = 0
            for rec in evaluate_benchmark(client, examples, temperature=0.0,
                                          max_new_tokens=max_new_tokens):
                append_jsonl(out_path, rec)
                n += 1
                if n % 25 == 0:
                    log(f"{model_name}/{bench}: +{n}")
    finally:
        client.close()
    log(f"{model_name}: capabilities -> {out_path}")
    return str(out_path)


def summarize(config: Config, *, models: list[str] | None = None) -> str:
    models = models or config.all_targets()
    rows = []
    for m in models:
        recs = load_jsonl(artefact("section4", "capabilities", f"{m}.jsonl"))
        if not recs:
            continue
        df = pd.DataFrame(recs)
        for bench, g in df.groupby("benchmark"):
            rows.append({"model": m, "benchmark": bench, "n": len(g),
                         "accuracy": 100.0 * g["correct"].mean()})
    if not rows:
        raise RuntimeError("no capability records; run `capabilities run` first")
    out = pd.DataFrame(rows).sort_values(["benchmark", "model"])
    path = artefact("section4", "capabilities", "summary.csv")
    out.to_csv(path, index=False)
    log("Capabilities summary (Figure 7):\n" + out.to_string(index=False))
    return str(path)
