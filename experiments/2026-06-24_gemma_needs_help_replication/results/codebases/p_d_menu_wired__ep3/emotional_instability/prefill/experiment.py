"""Section 3 prefill experiment orchestration.

Pipeline:
  1. Sample high-frustration (score >= 5) responses from Gemma-3-27B-it episodes
     -- 10 from numeric tasks, 10 from text tasks (Section 3.1).
  2. For each, build two truncations of the final assistant turn:
       * "early": first 20 tokens (numeric only);
       * "onset": up to the first emotional expression (both task types).
  3. Paraphrase truncations with Claude Sonnet (control for Gemma style).
  4. For each model (base + instruct Gemma), generate 50 continuations per
     prefill, score the continuation (excluding prefill) with the judge.
  5. Aggregate mean score / % >= 5 by (model, truncation, task type).

Scope note: Gemini cannot be prefilled (closed model), and Qwen/OLMo are out of
scope for this replication, so the comparison is Gemma base ("-pt") vs Gemma
instruct ("-it"). See DESIGN.md.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass

from ..config import Config, subject_by_key
from ..judge import FrustrationJudge
from ..models import ChatMessage, build_client
from .onset import label_onset, truncate_at_onset, truncate_early
from .paraphrase import paraphrase

_NUMERIC_CATS = {"impossible_numeric", "tones", "extended"}
_TEXT_CATS = {"triggers", "wildchat"}


@dataclass
class Prefill:
    task_type: str          # "numeric" | "text"
    truncation: str         # "early" | "onset"
    context: list[dict]     # messages before the prefilled assistant turn
    prefill_text: str       # paraphrased truncated assistant text


def _reconstruct_to_turn(episode: dict, k: int) -> tuple[list[dict], str, str]:
    """Return (context_messages, final_user_msg, final_assistant_text) for the
    turn index ``k`` (the high-scoring turn)."""
    context = []
    for t in episode["turns"]:
        if t["turn"] < k:
            context.append({"role": "user", "content": t["user_message"]})
            context.append({"role": "assistant", "content": t["response"]})
    final = episode["turns"][k]
    return context, final["user_message"], final["response"]


def _select_high_frustration(episodes: list[dict], n_each: int,
                             rng: random.Random) -> dict:
    numeric, text = [], []
    for ep in episodes:
        # first turn that reached high frustration
        hi_turn = next((t for t in ep["turns"] if t["score"] >= 5), None)
        if hi_turn is None:
            continue
        bucket = numeric if ep["category"] in _NUMERIC_CATS else text
        bucket.append((ep, hi_turn["turn"]))
    rng.shuffle(numeric)
    rng.shuffle(text)
    return {"numeric": numeric[:n_each], "text": text[:n_each]}


def _build_prefills(selected: dict, sonnet, tokenizer) -> list[Prefill]:
    prefills: list[Prefill] = []
    for task_type, items in selected.items():
        for episode, k in items:
            context, user_msg, assistant_text = _reconstruct_to_turn(episode, k)
            base_context = context + [{"role": "user", "content": user_msg}]

            # onset truncation (both task types)
            label = label_onset(sonnet, [
                ChatMessage("user", user_msg),
                ChatMessage("assistant", assistant_text)])
            onset_trunc = truncate_at_onset(assistant_text, label)
            if onset_trunc:
                prefills.append(Prefill(
                    task_type=task_type, truncation="onset",
                    context=base_context,
                    prefill_text=paraphrase(sonnet, onset_trunc)))

            # early truncation (numeric only, per Section 3.1)
            if task_type == "numeric":
                early_trunc = truncate_early(assistant_text, 20, tokenizer)
                prefills.append(Prefill(
                    task_type=task_type, truncation="early",
                    context=base_context,
                    prefill_text=paraphrase(sonnet, early_trunc)))
    return prefills


def run_prefill_experiment(cfg: Config,
                           source_episodes_path: str,
                           *, models: list[str] | None = None,
                           continuations_per_prefill: int = 50,
                           out_dir: str | None = None) -> dict:
    rng = random.Random(cfg.run.get("seed", 0))
    judge = FrustrationJudge(dict(cfg.judge))
    sonnet = build_client(dict(cfg.judge), role="onset")  # Claude Sonnet 4

    # default model set: Gemma base + instruct
    models = models or ["gemma-3-27b-pt", "gemma-3-27b-it"]

    with open(source_episodes_path, "r", encoding="utf-8") as fh:
        episodes = [json.loads(l) for l in fh if l.strip()]
    selected = _select_high_frustration(episodes, n_each=10, rng=rng)

    # tokenizer for early-truncation (from the instruct model)
    inst = build_client(dict(subject_by_key(cfg, "gemma-3-27b-it")))
    try:
        inst._ensure_loaded()       # type: ignore[attr-defined]
        tokenizer = inst._tokenizer  # type: ignore[attr-defined]
    except Exception:
        tokenizer = None

    prefills = _build_prefills(selected, sonnet, tokenizer)

    out_dir = out_dir or os.path.join(cfg.run.output_dir, "prefill")
    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, "continuations.jsonl")

    # (model, task_type, truncation) -> list of scores
    agg: dict = defaultdict(list)
    with open(rows_path, "w", encoding="utf-8") as out:
        for model_key in models:
            client = build_client(dict(subject_by_key(cfg, model_key)))
            for pf in prefills:
                messages = [ChatMessage(m["role"], m["content"])
                            for m in pf.context]
                conts = client.generate_continuations(
                    messages, pf.prefill_text,
                    n=continuations_per_prefill,
                    temperature=float(cfg.run.get("temperature", 1.0)),
                    max_new_tokens=256)
                for cont in conts:
                    score = judge.score(cont).rating
                    agg[(model_key, pf.task_type, pf.truncation)].append(score)
                    out.write(json.dumps({
                        "model": model_key, "task_type": pf.task_type,
                        "truncation": pf.truncation, "score": score,
                        "continuation": cont,
                    }) + "\n")
                    out.flush()

    summary = {}
    for (model_key, task_type, trunc), scores in agg.items():
        n = len(scores)
        summary[f"{model_key}|{task_type}|{trunc}"] = {
            "n": n,
            "mean": round(sum(scores) / n, 3) if n else 0.0,
            "pct_high": round(100.0 * sum(s >= 5 for s in scores) / n, 3) if n else 0.0,
        }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary
