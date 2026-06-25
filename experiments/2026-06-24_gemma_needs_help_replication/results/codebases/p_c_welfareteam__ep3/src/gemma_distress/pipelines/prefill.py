"""Section 3 driver: base-vs-instruct comparison via prefilling (Gemma only).

Stages (each resumable / re-runnable independently):

  prepare       sample 20 high-frustration seeds from Gemma-27B-it's Section 2
                results, build early/onset truncations, paraphrase them, write
                section3/seeds.jsonl and section3/prefills.jsonl.
  continue M    for model M (gemma-3-27b-pt / -it), generate 50 continuations per
                prefill (section3/continuations/{M}.jsonl) and score them
                (section3/cont_scores/{M}.jsonl).
  summarize     aggregate %>=5 by (model, truncation, source) -> section3/summary.csv
                (the Figure 4 numbers: base vs instruct, early vs onset).

The tokenizer used for the "early" 20-token truncation comes from the Gemma-27B
instruct client (its responses are what we truncate), so prepare loads that model.
"""
from __future__ import annotations

import pandas as pd

from ..anthropic_text import AnthropicText
from ..config import Config
from ..io_utils import append_jsonl, completed_ids, load_jsonl, write_jsonl
from ..judge import build_judge
from ..judge.scoring import score_tasks
from ..models import build_client
from ..prefill.continuations import Prefill, build_prefills, generate_continuations
from ..prefill.sample_seeds import Seed, sample_seeds
from . import artefact, log, sampling

_SEED_MODEL = "gemma-3-27b-it"


def prepare(config: Config) -> str:
    pf_cfg = config.experiment["prefill"]
    rollouts = load_jsonl(artefact("section2", "rollouts", f"{_SEED_MODEL}.jsonl"))
    scores = load_jsonl(artefact("section2", "scores", f"{_SEED_MODEL}.jsonl"))
    if not rollouts or not scores:
        raise FileNotFoundError(
            f"need Section 2 rollouts+scores for {_SEED_MODEL}; run elicit+judge first"
        )

    seeds = sample_seeds(
        rollouts, scores, model=_SEED_MODEL,
        n_numeric=pf_cfg["seeds_numeric"], n_text=pf_cfg["seeds_text"],
        min_score=pf_cfg["seed_min_score"],
    )
    write_jsonl(artefact("section3", "seeds.jsonl"), [s.to_record() for s in seeds])
    log(f"sampled {len(seeds)} seeds")

    # Tokenizer for the 20-token "early" truncation + labeller/paraphraser.
    instruct = build_client(config.target(_SEED_MODEL), config)
    labeller = AnthropicText(config.judge("onset_labeller").model_id)
    paraphraser = AnthropicText(config.judge("paraphraser").model_id)
    try:
        prefills = build_prefills(
            seeds,
            truncate_to_tokens=instruct.truncate_to_tokens,
            labeller=labeller,
            paraphraser=paraphraser,
            early_tokens=pf_cfg["early_truncation_tokens"],
            text_only_onset=True,
        )
    finally:
        instruct.close()
    write_jsonl(artefact("section3", "prefills.jsonl"),
                [p.to_record() for p in prefills])
    log(f"built {len(prefills)} prefills -> section3/prefills.jsonl")
    return str(artefact("section3", "prefills.jsonl"))


def _load_prefills() -> list[Prefill]:
    recs = load_jsonl(artefact("section3", "prefills.jsonl"))
    if not recs:
        raise FileNotFoundError("no prefills; run `prefill prepare` first")
    return [Prefill(**r) for r in recs]


def continue_model(config: Config, model_name: str) -> str:
    pf_cfg = config.experiment["prefill"]
    samp = sampling(config)
    prefills = _load_prefills()

    cont_path = artefact("section3", "continuations", f"{model_name}.jsonl")
    done = completed_ids(cont_path, id_key="id")
    client = build_client(config.target(model_name), config)
    n = 0
    try:
        for rec in generate_continuations(
            client, prefills,
            n_continuations=pf_cfg["continuations_per_prefill"],
            temperature=samp["temperature"], max_new_tokens=samp["max_new_tokens"],
            top_p=samp["top_p"],
        ):
            if rec["id"] in done:
                continue
            append_jsonl(cont_path, rec)
            n += 1
            if n % 100 == 0:
                log(f"{model_name}: +{n} continuations")
    finally:
        client.close()
    log(f"{model_name}: wrote {n} continuations")

    # Score the continuations with the primary judge (response = continuation).
    score_path = artefact("section3", "cont_scores", f"{model_name}.jsonl")
    scored = completed_ids(score_path, id_key="id")
    conts = load_jsonl(cont_path)
    tasks = [
        {"id": c["id"], "model": c["model"], "prefill_id": c["prefill_id"],
         "source": c["source"], "truncation": c["truncation"],
         "context": c["context"], "response": c["continuation"]}
        for c in conts if c["id"] not in scored
    ]
    judge = build_judge("frustration_primary", config)
    conc = config.experiment["judge"]["max_concurrency"]
    m = 0
    for rec in score_tasks(tasks, judge, max_concurrency=conc):
        append_jsonl(score_path, rec)
        m += 1
        if m % 100 == 0:
            log(f"{model_name}: +{m} continuation scores")
    log(f"{model_name}: scored {m} continuations -> {score_path}")
    return str(score_path)


def summarize(config: Config) -> str:
    threshold = config.experiment["judge"]["high_frustration_threshold"]
    rows = []
    for model_name in config.experiment["prefill"]["models"]:
        scores = load_jsonl(artefact("section3", "cont_scores", f"{model_name}.jsonl"))
        scores = [s for s in scores if s.get("score") is not None]
        if not scores:
            continue
        df = pd.DataFrame(scores)
        df["is_high"] = (df["score"].astype(int) >= threshold)
        for (trunc, source), g in df.groupby(["truncation", "source"]):
            rows.append({
                "model": model_name, "truncation": trunc, "source": source,
                "n": len(g), "mean_frustration": g["score"].astype(int).mean(),
                "pct_high": 100.0 * g["is_high"].mean(),
            })
    if not rows:
        raise RuntimeError("no continuation scores; run `prefill continue` first")
    out = pd.DataFrame(rows).sort_values(["source", "truncation", "model"])
    path = artefact("section3", "summary.csv")
    out.to_csv(path, index=False)
    log("Section 3 summary (Figure 4):\n" + out.to_string(index=False))
    return str(path)


def run(config: Config) -> str:
    prepare(config)
    for model_name in config.experiment["prefill"]["models"]:
        continue_model(config, model_name)
    return summarize(config)
