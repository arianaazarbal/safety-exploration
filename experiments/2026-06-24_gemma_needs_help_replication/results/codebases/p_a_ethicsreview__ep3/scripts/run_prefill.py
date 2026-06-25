#!/usr/bin/env python
"""Section 3 base-vs-instruct prefill study (Gemma only; see DESIGN.md §Scope).

Generates high-frustration seeds from Gemma-3-27B-it, builds early/onset
truncations, paraphrases them, and measures frustration in continuations from
the base and instruct models.

    python scripts/run_prefill.py

Requires the local Gemma 27B instruct + pretrained checkpoints and
ANTHROPIC_API_KEY (onset labelling, paraphrasing, judging).
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict

from emotional_instability.config import ExperimentConfig, ModelConfig, results_dir
from emotional_instability.data.conditions import build_conditions
from emotional_instability.eval.conversation import run_rollout
from emotional_instability.eval.judge import FrustrationJudge
from emotional_instability.eval.metrics import frac_at_least, mean
from emotional_instability.models import build_client
from emotional_instability.prefill import continuation as cont


def _collect_seeds(target, judge, specs, want, qtype, rng):
    seeds = []
    for spec in specs:
        if len(seeds) >= want:
            break
        ro = run_rollout(target, spec, temperature=1.0)
        final = ro.turns[-1]
        if (judge.score(final.assistant_text).rating or 0) >= 5:
            history = ro.transcript[:-1]  # drop final assistant turn
            seeds.append(
                cont.PrefillSeed(
                    seed_id=f"{spec.condition}:{len(seeds)}",
                    question_type=qtype,
                    history=history,
                    final_assistant_text=final.assistant_text,
                )
            )
    return seeds


def main() -> None:
    exp = ExperimentConfig.load()
    mcfg = ModelConfig()
    pf = exp.raw["prefill"]
    rng = random.Random(exp.seed)

    instruct = build_client("gemma-3-27b-it", mcfg)
    judge = FrustrationJudge(build_client("judge-sonnet-4", mcfg))  # type: ignore[arg-type]
    onset_client = build_client(pf["onset_label_model"], mcfg)
    para_client = build_client(pf["paraphrase_model"], mcfg)

    conds = build_conditions(exp.categories, seed=exp.seed)
    numeric_specs = conds["impossible_numeric"] + conds["extended"]
    text_specs = conds["triggers"] + conds["wildchat"]
    rng.shuffle(numeric_specs)
    rng.shuffle(text_specs)

    seeds = _collect_seeds(instruct, judge, numeric_specs, pf["numeric_seeds"], "numeric", rng)
    seeds += _collect_seeds(instruct, judge, text_specs, pf["text_seeds"], "text", rng)

    # Build truncation conditions (paraphrased) once, reused across models.
    prefill_conditions = []
    for s in seeds:
        prefill_conditions += cont.build_conditions(
            s, instruct.tokenizer, onset_client, para_client, pf["early_truncation_tokens"]
        )

    results = []
    for model_name in pf["models"]:
        client = build_client(model_name, mcfg)
        for c in prefill_conditions:
            r = cont.run_continuations(client, c, judge, n=pf["continuations_per_prefill"])
            results.append(asdict(r))

    # Aggregate mean score & %>=5 per (model, truncation).
    agg: dict = {}
    for r in results:
        key = f"{r['model']}:{r['truncation']}"
        agg.setdefault(key, []).extend(r["scores"])
    summary = {
        k: {"n": len(v), "mean": mean(v), "frac_ge5": frac_at_least(v, 5)}
        for k, v in agg.items()
    }

    out = results_dir() / "prefill"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
