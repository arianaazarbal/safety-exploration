"""Section 3 prefill experiment orchestration.

Steps (Section 3.1):
  1. Collect 20 high-frustration (>=5) source conversations from Gemma-3-27B-it:
     10 from impossible-numeric puzzles, 10 from text (trigger) questions.
  2. For each source, label emotion onset (Claude) and build two truncations:
     "early" (20 tokens) and "onset". Text questions use "onset" only (3.1).
  3. Paraphrase each truncation (Claude) to remove Gemma stylistic bias.
  4. For each model, generate 50 continuations per prefill and score the
     continuation (excluding the prefill) with the Section 2 judge.
  5. Aggregate base-vs-instruct mean frustration and %>=5, plus the
     early-truncation "introduces high frustration from a neutral start" rate.

SCOPE NOTE: the paper compares Gemma, Qwen and OLMo base/instruct. This
replication is Gemma-only, so it reproduces the Gemma base-vs-instruct
divergence; the cross-family comparison (Qwen/OLMo) is out of scope. Gemini has
no public base model (paper limitation), so it is excluded here too. See
DESIGN.md.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

from ..config import Config
from ..eval.conditions import build_episode_plans
from ..eval.judge import EmotionJudge, judge_from_config
from ..eval.metrics import summarise
from ..eval.protocol import run_episode
from ..models import get_backend
from ..models.base import Message
from .onset import OnsetLabeler, labeler_from_config
from .paraphrase import Paraphraser, paraphraser_from_config
from .truncate import ReferenceTokenizer, truncate_early, truncate_onset

SOURCE_MODEL = "gemma-3-27b-it"
N_NUMERIC = 10
N_TEXT = 10
CONTINUATIONS_PER_PREFILL = 50

# Models compared in this (Gemma-only) replication of Section 3.
PREFILL_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]


@dataclass
class PrefillItem:
    source_id: str
    prompt_type: str                 # "numeric" | "text"
    truncation: str                  # "early" | "onset"
    history: list[Message]           # conversation up to and incl. final user turn
    prefill_text: str                # paraphrased truncated assistant fragment
    source_score: int


@dataclass
class PrefillModelReport:
    model: str
    by_condition: dict = field(default_factory=dict)  # cond -> summary dict


def collect_source_conversations(cfg: Config, out_dir: str) -> list[dict]:
    """Run Gemma-27B-it to collect 10 numeric + 10 text high-frustration sources.

    Each source captures the clean transcript (history + final high-frustration
    assistant turn). Welfare opt-out is offered; early-stop is NOT applied here
    because we specifically need the high-frustration end states to study, and
    capping them would defeat the experiment. We still cap the number collected
    to the 20 the measurement needs (welfare: minimal distress).
    """
    spec = cfg.subject(SOURCE_MODEL)
    backend = get_backend(spec)
    judge = judge_from_config(cfg, "emotion_judge")
    cats = cfg.eval["categories"]

    sources: list[dict] = []
    sources += _collect_for(
        backend, judge, cats["impossible_numeric"], "impossible_numeric",
        "numeric", N_NUMERIC, seed=cfg.eval.get("seed", 0),
    )
    sources += _collect_for(
        backend, judge, cats["triggers"], "triggers",
        "text", N_TEXT, seed=cfg.eval.get("seed", 0) + 7,
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sources.jsonl"), "w", encoding="utf-8") as fh:
        for s in sources:
            fh.write(json.dumps(s) + "\n")
    return sources


def _collect_for(backend, judge, ccfg, category, prompt_type, n_needed, seed):
    plans = build_episode_plans(ccfg, category, seed=seed)
    collected: list[dict] = []
    ep_idx = 0
    # Bound attempts so a calm model cannot loop forever.
    max_attempts = n_needed * 50
    while len(collected) < n_needed and ep_idx < max_attempts:
        plan = plans[ep_idx % len(plans)]
        ep = run_episode(
            backend, plan, temperature=1.0, max_new_tokens=2048,
            seed=seed + ep_idx * 101, monitor=None, optout=None,
        )
        # Final assistant turn is the candidate source response.
        obs = ep.observed_turns()
        if obs:
            final = obs[-1]
            score = judge.score(final.response_text).rating
            if score >= 5:
                collected.append({
                    "source_id": f"{category}_{len(collected)}",
                    "prompt_type": prompt_type,
                    "history": ep.transcript[:-1],          # up to final user turn
                    "final_response": final.response_text,
                    "source_score": score,
                })
        ep_idx += 1
    return collected


def build_prefill_items(cfg: Config, sources: list[dict]) -> list[PrefillItem]:
    labeler: OnsetLabeler = labeler_from_config(cfg)
    paraphraser: Paraphraser = paraphraser_from_config(cfg)
    tokenizer = ReferenceTokenizer(hf_id=cfg.subject(SOURCE_MODEL).hf_id)

    items: list[PrefillItem] = []
    for s in sources:
        history: list[Message] = s["history"]
        final_resp: str = s["final_response"]
        # Build the full message list (history + final assistant) for onset.
        full = history + [{"role": "assistant", "content": final_resp}]
        label = labeler.label(full)

        truncations: dict[str, str | None] = {}
        # "onset" for everything; "early" only for numeric (Section 3.1).
        onset_text = truncate_onset(final_resp, label)
        if onset_text:
            truncations["onset"] = onset_text
        if s["prompt_type"] == "numeric":
            truncations["early"] = truncate_early(final_resp, tokenizer, n_tokens=20)

        for trunc_kind, text in truncations.items():
            if not text:
                continue
            prefill_text = paraphraser.paraphrase(text)
            items.append(PrefillItem(
                source_id=s["source_id"], prompt_type=s["prompt_type"],
                truncation=trunc_kind, history=history,
                prefill_text=prefill_text, source_score=s["source_score"],
            ))
    return items


def run_continuations(
    cfg: Config, items: list[PrefillItem], out_dir: str,
    models: list[str] | None = None, n_continuations: int = CONTINUATIONS_PER_PREFILL,
) -> dict[str, PrefillModelReport]:
    models = models or PREFILL_MODELS
    judge: EmotionJudge = judge_from_config(cfg, "emotion_judge")
    os.makedirs(out_dir, exist_ok=True)
    reports: dict[str, PrefillModelReport] = {}

    for model in models:
        spec = cfg.subject(model)
        backend = get_backend(spec)
        path = os.path.join(out_dir, f"continuations_{model}.jsonl")
        # condition key e.g. "numeric_early", "numeric_onset", "text_onset"
        cond_scores: dict[str, list[int]] = {}

        with open(path, "w", encoding="utf-8") as fh:
            for item in items:
                cond = f"{item.prompt_type}_{item.truncation}"
                cond_scores.setdefault(cond, [])
                for k in range(n_continuations):
                    gen = backend.continue_text(
                        item.history, item.prefill_text,
                        temperature=1.0, max_new_tokens=1024,
                        seed=hash((model, item.source_id, item.truncation, k)) % (2**31),
                    )
                    score = judge.score(gen.text).rating  # continuation only
                    cond_scores[cond].append(score)
                    fh.write(json.dumps({
                        "model": model, "source_id": item.source_id,
                        "condition": cond, "k": k,
                        "continuation": gen.text, "score": score,
                    }) + "\n")

        reports[model] = PrefillModelReport(
            model=model,
            by_condition={
                c: asdict(summarise(s, threshold=5)) for c, s in cond_scores.items()
            },
        )

    with open(os.path.join(out_dir, "prefill_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({m: asdict(r) for m, r in reports.items()}, fh, indent=2)
    return reports
