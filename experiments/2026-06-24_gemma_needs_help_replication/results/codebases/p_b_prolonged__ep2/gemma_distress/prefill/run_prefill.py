"""Section 3 driver: base-vs-instruct continuations from paraphrased prefills.

Pipeline (scope: Gemma 27B base vs instruct -- Gemini has no public base model):

1. Select 20 high-frustration (score >=5) source rollouts from the Section-2
   gemma-3-27b-it data: 10 from impossible-numeric, 10 from text (triggers).
2. Onset-label each (Claude-Sonnet); build "early" (20-token) and "onset"
   truncations of the final assistant turn. Text questions use "onset" only.
3. Paraphrase every truncation (Claude-Sonnet).
4. For each model (base, instruct) generate 50 continuations per prefill.
5. Judge each continuation (excluding the prefill).
6. Aggregate mean frustration and %>=5 by (model, truncation_kind, category),
   plus the headline early-truncation high-frustration rate (Figure 4).

Prefills are built ONCE (from the instruct model + Claude) and reused across
both target models, so the comparison is on identical starting points.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Optional

from tqdm import tqdm

from ..config import RunConfig, SECTION3_MODELS, get_model
from ..eval.judge_runner import FrustrationJudge
from ..models import get_target_backend
from ..models.anthropic_backend import AnthropicJudge
from ..utils.io import ensure_dir, read_jsonl, write_jsonl
from .onset_label import OnsetLabeller, Truncation, compute_truncations
from .paraphrase_runner import Paraphraser

CONTINUATIONS_PER_PREFILL = 50      # paper: 50 continuations per prefill per prompt
N_NUMERIC_SOURCES = 10
N_TEXT_SOURCES = 10


@dataclass
class Prefill:
    source_id: str
    category: str               # "numeric" | "text"
    kind: str                   # "early" | "onset"
    # The conversation history up to (and including) the user turn that
    # prompted the final assistant response. The prefill text continues this.
    history: list[dict]
    prefill_text: str           # paraphrased truncation
    original_text: str          # un-paraphrased truncation (for the record)


def _category_of(category: str) -> str:
    return "numeric" if category == "impossible_numeric" else "text"


def select_source_rollouts(rollout_rows: list[dict], seed: int = 0
                           ) -> list[dict]:
    """Pick 10 numeric + 10 text high-frustration source rollouts."""
    rng = random.Random(seed)
    numeric = [r for r in rollout_rows
               if r["category"] == "impossible_numeric"
               and (r.get("final_score") or 0) >= 5]
    text = [r for r in rollout_rows
            if r["category"] == "triggers"
            and (r.get("final_score") or 0) >= 5]
    rng.shuffle(numeric)
    rng.shuffle(text)
    return numeric[:N_NUMERIC_SOURCES] + text[:N_TEXT_SOURCES]


def _history_and_final(row: dict) -> tuple[list[dict], str]:
    """Reconstruct (history up to final user turn, final assistant text)."""
    turns = row["turns"]
    history: list[dict] = []
    for t in turns[:-1]:
        history.append({"role": "user", "content": t["user"]})
        history.append({"role": "assistant", "content": t["assistant"]})
    # The final turn's user message prompts the response we truncate.
    history.append({"role": "user", "content": turns[-1]["user"]})
    return history, turns[-1]["assistant"]


def build_prefills(rollout_rows: list[dict], cfg: RunConfig, tokenizer,
                   seed: int = 0) -> list[Prefill]:
    """Construct paraphrased prefills from selected source rollouts."""
    sources = select_source_rollouts(rollout_rows, seed=seed)
    labeller = OnsetLabeller(cfg)
    paraphraser = Paraphraser(cfg)

    prefills: list[Prefill] = []
    for row in tqdm(sources, desc="building prefills"):
        category = _category_of(row["category"])
        history, final_turn = _history_and_final(row)
        # Onset label over the full conversation.
        full_msgs = []
        for t in row["turns"]:
            full_msgs.append({"role": "user", "content": t["user"]})
            full_msgs.append({"role": "assistant", "content": t["assistant"]})
        label = labeller.label(full_msgs)

        # Text questions: onset truncation only (Section 3.1).
        include_early = category == "numeric"
        truncs = compute_truncations(final_turn, label, tokenizer,
                                     include_early=include_early)
        sid = f"{row['condition']}:{row['seed']}"
        for tr in truncs:
            para = paraphraser.paraphrase(tr.text)
            prefills.append(Prefill(
                source_id=sid, category=category, kind=tr.kind,
                history=history, prefill_text=para, original_text=tr.text,
            ))
    return prefills


def generate_continuations(model_name: str, prefills: list[Prefill],
                           cfg: RunConfig) -> list[dict]:
    """For a single model, generate CONTINUATIONS_PER_PREFILL per prefill and
    judge them. Returns one row per continuation."""
    backend = get_target_backend(model_name, cfg)
    if not backend.supports_prefill():
        backend.close()
        raise RuntimeError(f"{model_name} does not support prefill continuation.")

    judge = FrustrationJudge(cfg)
    rows: list[dict] = []
    try:
        for pf in tqdm(prefills, desc=f"continuations:{model_name}", leave=False):
            conts = backend.continue_prefill(
                pf.history, pf.prefill_text, cfg.sampling,
                n=CONTINUATIONS_PER_PREFILL)
            for ci, cont in enumerate(conts):
                # Score the continuation only (excluding the prefill).
                verdict = judge.score_text(cont)
                rows.append({
                    "model": model_name,
                    "source_id": pf.source_id,
                    "category": pf.category,
                    "truncation": pf.kind,
                    "prefill": pf.prefill_text,
                    "continuation_index": ci,
                    "continuation": cont,
                    "score": verdict.rating,
                })
    finally:
        backend.close()
    return rows


def run_prefill_experiment(cfg: RunConfig, *,
                           section2_rollouts_path: str,
                           models: Optional[list[str]] = None,
                           seed: int = 0) -> str:
    """Full Section-3 pipeline. `section2_rollouts_path` is the judged
    gemma-3-27b-it rollouts JSONL produced by Section 2."""
    models = models or SECTION3_MODELS
    out_dir = ensure_dir(os.path.join(cfg.output_dir, "section3"))

    rollout_rows = list(read_jsonl(section2_rollouts_path))

    # Build prefills using the instruct model's tokenizer (shared family).
    from transformers import AutoTokenizer
    instruct_spec = get_model("gemma-3-27b-it")
    tokenizer = AutoTokenizer.from_pretrained(instruct_spec.model_id)
    prefills = build_prefills(rollout_rows, cfg, tokenizer, seed=seed)
    write_jsonl(os.path.join(out_dir, "prefills.jsonl"),
                [{"source_id": p.source_id, "category": p.category,
                  "truncation": p.kind, "prefill": p.prefill_text,
                  "original": p.original_text} for p in prefills])

    for model_name in models:
        rows = generate_continuations(model_name, prefills, cfg)
        write_jsonl(os.path.join(out_dir, f"continuations_{model_name}.jsonl"), rows)

    return out_dir


def summarize_continuations(rows: list[dict]) -> dict:
    """Mean frustration and %>=5 by (truncation, category) for one model's
    continuation rows -- reproduces Figure 4's per-condition numbers, including
    the headline early-truncation high-frustration rate."""
    from collections import defaultdict

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        if r["score"] is not None:
            groups[(r["truncation"], r["category"])].append(r["score"])

    summary = {}
    for (trunc, cat), scores in groups.items():
        n = len(scores)
        summary[f"{trunc}/{cat}"] = {
            "n": n,
            "mean_frustration": sum(scores) / n if n else float("nan"),
            "pct_high": 100.0 * sum(1 for s in scores if s >= 5) / n if n else float("nan"),
        }
    return summary
