"""Run continuations from prefills and score them (Section 3.2 / 4.2).

For each (paraphrased) truncation we seed the conversation with the original
task, prefill the assistant turn with the truncated prefix, and sample
``continuations_per_prefill`` continuations from each model. Only the generated
continuation (excluding the prefill) is scored — matching the paper.

Scope: the in-scope models are Gemma-3-27B **base** and **instruct**. The paper
also runs Qwen-2.5-32B and OLMo-32B (base+instruct); those are out of scope
here. The function accepts arbitrary (label, client) pairs so they could be
added back.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from config import MAX_RESPONSE_TOKENS, PATHS, PREFILL, SAMPLING_TEMPERATURE
from src.judge import FrustrationJudge
from src.models.base import Conversation, SubjectClient

from .paraphrase import paraphrase_preserving_emotion
from .truncate import Truncation, build_recovery_truncations, build_truncations


def _continuation_scores(
    client: SubjectClient,
    judge: FrustrationJudge,
    task_prompt: str,
    prefill: str,
    n: int,
) -> list[int]:
    convo = Conversation().user(task_prompt)
    scores = []
    for _ in range(n):
        resp = client.continue_from_prefill(
            convo, prefill, max_tokens=MAX_RESPONSE_TOKENS, temperature=SAMPLING_TEMPERATURE
        )
        scores.append(judge.score(resp.text).score)
    return scores


def run_prefill_experiment(
    seeds: list[dict],
    models: dict[str, SubjectClient],
    *,
    tokenizer=None,
    judge: FrustrationJudge | None = None,
    config=PREFILL,
    out_path: str | None = None,
    paraphrase: bool = True,
) -> dict:
    """Section 3 base-vs-instruct comparison.

    ``seeds`` items: {"seed_id","task_kind"("numeric"|"text"),"response","task_prompt"}.
    ``models`` maps a label (e.g. "gemma-27b-instruct"/"gemma-27b-base") to a client.
    Returns aggregate stats per (model, truncation-kind, task-kind).
    """
    judge = judge or FrustrationJudge()
    truncations = build_truncations(seeds, tokenizer=tokenizer, config=config)
    seed_prompt = {s["seed_id"]: s["task_prompt"] for s in seeds}

    records = []
    for tr in truncations:
        prefix = paraphrase_preserving_emotion(tr.prefix_text) if paraphrase else tr.prefix_text
        for label, client in models.items():
            scores = _continuation_scores(
                client, judge, seed_prompt[tr.seed_id], prefix, config.continuations_per_prefill
            )
            records.append({
                "seed_id": tr.seed_id,
                "task_kind": tr.task_kind,
                "truncation": tr.kind,
                "model": label,
                "scores": scores,
            })

    summary = _aggregate(records, config)
    out_path = out_path or os.path.join(PATHS.results, "prefill_experiment.json")
    with open(out_path, "w") as f:
        json.dump({"records": records, "summary": summary}, f, indent=2)
    return summary


def run_recovery_experiment(
    seeds: list[dict],
    models: dict[str, SubjectClient],
    *,
    tokenizer=None,
    judge: FrustrationJudge | None = None,
    config=PREFILL,
    out_path: str | None = None,
    paraphrase: bool = True,
) -> dict:
    """Recovery limitation (Section 4.2): can a model climb down from a very
    high-frustration prefilled state? Truncate score>=7 seeds 200 tokens before
    the end and measure % of continuations still scoring >=5."""
    judge = judge or FrustrationJudge()
    truncations = build_recovery_truncations(seeds, tokenizer=tokenizer, config=config)
    seed_prompt = {s["seed_id"]: s["task_prompt"] for s in seeds}

    records = []
    for tr in truncations:
        prefix = paraphrase_preserving_emotion(tr.prefix_text) if paraphrase else tr.prefix_text
        for label, client in models.items():
            scores = _continuation_scores(
                client, judge, seed_prompt[tr.seed_id], prefix, config.continuations_per_prefill
            )
            records.append({"seed_id": tr.seed_id, "model": label, "scores": scores})

    summary = {}
    by_model = defaultdict(list)
    for r in records:
        by_model[r["model"]].extend(r["scores"])
    for label, scores in by_model.items():
        n = len(scores) or 1
        summary[label] = {
            "n": len(scores),
            "mean": sum(scores) / n,
            "pct_high": sum(1 for s in scores if s >= 5) / n,
        }
    out_path = out_path or os.path.join(PATHS.results, "recovery_experiment.json")
    with open(out_path, "w") as f:
        json.dump({"records": records, "summary": summary}, f, indent=2)
    return summary


def _aggregate(records: list[dict], config) -> dict:
    buckets = defaultdict(list)
    for r in records:
        buckets[(r["model"], r["truncation"], r["task_kind"])].extend(r["scores"])
    summary = {}
    for (model, trunc, task), scores in buckets.items():
        n = len(scores) or 1
        summary[f"{model}|{trunc}|{task}"] = {
            "n": len(scores),
            "mean": sum(scores) / n,
            "pct_high": sum(1 for s in scores if s >= 5) / n,
        }
    return summary
