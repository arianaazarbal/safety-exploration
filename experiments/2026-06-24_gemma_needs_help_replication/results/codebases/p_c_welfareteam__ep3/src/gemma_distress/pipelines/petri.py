"""Section 4.2 driver: open-ended emotion elicitation (Petri-style).

Runs the auditor/judge audit loop (``intervention.petri_eval``) for each target,
using the config's Petri auditor (Claude-Sonnet) and judge (Claude-Opus) models.
Writes one transcript+scores record per audit run and a per-model, per-category
mean (the Figure 6 numbers).

The number of audit runs per model is ``petri.num_seeds`` (config); the built-in
psychologically-informed seeds are cycled to reach that count, varying the seed
index so repeated strategies get distinct rollouts.
"""
from __future__ import annotations

import pandas as pd

from ..anthropic_text import AnthropicText
from ..config import Config
from ..intervention.petri_eval import AUDIT_SEEDS, CATEGORIES, run_open_ended_audit
from ..io_utils import append_jsonl, load_jsonl, write_jsonl
from ..models import build_client
from . import artefact, log, sampling


def _seeds_for(n: int) -> list[str]:
    return [AUDIT_SEEDS[i % len(AUDIT_SEEDS)] for i in range(n)]


def run_model(config: Config, model_name: str) -> str:
    pcfg = config.experiment["petri"]
    samp = sampling(config)
    seeds = _seeds_for(pcfg["num_seeds"])

    auditor = AnthropicText(config.petri_model("auditor").model_id)
    judge = AnthropicText(config.petri_model("judge").model_id)
    client = build_client(config.target(model_name), config)
    out_path = artefact("section4", "petri", f"{model_name}.jsonl")
    try:
        transcripts = run_open_ended_audit(
            client, auditor, judge, seeds=seeds,
            max_turns=pcfg["max_turns"], temperature=samp["temperature"],
            max_new_tokens=samp["max_new_tokens"],
        )
    finally:
        client.close()
    for t in transcripts:
        append_jsonl(out_path, t.to_record())
    log(f"petri {model_name}: {len(transcripts)} transcripts -> {out_path}")
    return str(out_path)


def summarize(config: Config, *, models: list[str] | None = None) -> str:
    models = models or config.all_targets()
    rows = []
    for m in models:
        recs = load_jsonl(artefact("section4", "petri", f"{m}.jsonl"))
        if not recs:
            continue
        for cat in CATEGORIES:
            vals = [r["scores"].get(cat) for r in recs
                    if r.get("scores", {}).get(cat) is not None]
            if vals:
                rows.append({"model": m, "category": cat, "n": len(vals),
                             "mean_score": sum(vals) / len(vals)})
    if not rows:
        raise RuntimeError("no petri transcripts; run `petri run` first")
    out = pd.DataFrame(rows).sort_values(["category", "model"])
    path = artefact("section4", "petri", "summary.csv")
    out.to_csv(path, index=False)
    log("Petri summary (Figure 6):\n" + out.to_string(index=False))
    return str(path)


def run(config: Config, *, models: list[str] | None = None) -> str:
    models = models or config.all_targets()
    for m in models:
        run_model(config, m)
    return summarize(config, models=models)
