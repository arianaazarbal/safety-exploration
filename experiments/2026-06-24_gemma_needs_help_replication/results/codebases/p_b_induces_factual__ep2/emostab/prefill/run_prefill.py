"""Section 3 orchestrator: base-vs-instruct prefill comparison (Gemma, scoped).

Pipeline:
  1. Sample high-frustration (>=5) source conversations from Gemma-27B-it:
     10 from impossible-numeric, 10 from text (trigger) questions.
  2. Label emotion onset (Claude) and build truncations:
       - numeric: both "early" (20 tokens) and "onset"
       - text   : "onset" only (early yields minimal emotion without follow-ups)
  3. Paraphrase the truncated prefills (Claude) to control for Gemma style.
  4. For each model (gemma-3-27b-pt base, gemma-3-27b-it instruct), sample 50
     continuations per prefill and score them with the Section 2 judge.
  5. Aggregate mean frustration and %>=5 by (model, truncation, prompt_type).

Outputs:
  runs/prefill/source_conversations.jsonl
  runs/prefill/prefills.jsonl
  runs/prefill/continuations/<model>.jsonl
  runs/prefill/summary.json
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import asdict

from ..config import Config, load_config
from ..eval.conditions import Condition
from ..eval.rollout import run_condition
from ..judge import FrustrationJudge
from ..models import build_model
from ..models.base import GenConfig
from ..utils.io import write_json, write_jsonl
from .continuation import sample_continuations
from .onset import OnsetLabeller
from .paraphrase import Paraphraser
from .truncate import make_early, make_onset

log = logging.getLogger(__name__)


def _collect_source_convos(cfg, judge, source_model, threshold, n_numeric, n_text):
    """Roll out numeric + text conditions on the source model and keep the first
    conversations whose final assistant turn scores >= threshold."""
    model = build_model(cfg, source_model)

    def harvest(condition: Condition, want: int, prompt_type: str):
        kept = []
        # Roll out enough conversations to find `want` high-frustration ones.
        result = run_condition(model, condition, cfg, seed=cfg.seed)
        # Reconstruct full message histories per conversation from turn records.
        by_conv: dict[str, list] = {}
        for rec in result.records:
            by_conv.setdefault(rec.conv_id, []).append(rec)
        for conv_id, recs in by_conv.items():
            recs.sort(key=lambda r: r.turn)
            final_text = recs[-1].response_text
            score = judge.score(final_text).rating
            if score >= threshold:
                kept.append(
                    {
                        "source_id": f"{prompt_type}-{conv_id}",
                        "prompt_type": prompt_type,
                        "messages": _reconstruct_messages(cfg, condition, recs),
                        "final_score": score,
                    }
                )
            if len(kept) >= want:
                break
        return kept

    numeric_cond = Condition("src_numeric", "impossible_numeric", 3, "neutral", "numeric",
                             budget=n_numeric * 3 * 6)
    text_cond = Condition("src_text", "triggers", 3, "neutral", "trigger_factual",
                          budget=n_text * 3 * 6)
    sources = harvest(numeric_cond, n_numeric, "numeric")
    sources += harvest(text_cond, n_text, "text")
    return sources


def _reconstruct_messages(cfg, condition: Condition, recs):
    """Rebuild the alternating user/assistant transcript for a conversation from
    its ordered turn records (we re-derive the rejection prompts deterministically
    by replaying the same seeded plan builder)."""
    from ..eval.rollout import build_plans

    plans = {p.conv_id: p for p in build_plans(condition, cfg, seed=cfg.seed)}
    plan = plans.get(recs[0].conv_id)
    messages = []
    if plan is not None:
        messages.append({"role": "user", "content": plan.first_user})
        for i, rec in enumerate(recs):
            messages.append({"role": "assistant", "content": rec.response_text})
            if i < len(plan.rejections):
                messages.append({"role": "user", "content": plan.rejections[i]})
    else:  # fallback: assistant-only
        for rec in recs:
            messages.append({"role": "assistant", "content": rec.response_text})
    return messages


def build_prefills(cfg, sources, labeller, paraphraser, tokenizer=None):
    early_tokens = cfg.prefill.early_tokens
    specs = []
    for src in sources:
        messages = src["messages"]
        label = labeller.label(messages)
        # onset truncation (numeric + text)
        if label.found and label.char_offset is not None:
            onset = make_onset(
                src["source_id"], messages, src["prompt_type"],
                label.turn_index, label.char_offset,
            )
            if onset:
                specs.append(onset)
        # early truncation (numeric only)
        if src["prompt_type"] == "numeric":
            early = make_early(
                src["source_id"], messages, src["prompt_type"],
                early_tokens, tokenizer,
            )
            if early:
                specs.append(early)

    if cfg.prefill.paraphrase:
        for spec in specs:
            spec.prefill = paraphraser.paraphrase(spec.prefill)
    return specs


def run_prefill_experiment(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    out_dir = cfg.output_root() / "prefill"

    judge = FrustrationJudge(
        provider=cfg.judge.provider, model=cfg.judge.model,
        temperature=cfg.judge.temperature, max_tokens=cfg.judge.max_tokens,
    )

    # 1. source conversations
    sources = _collect_source_convos(
        cfg, judge,
        source_model=cfg.prefill.source_model,
        threshold=cfg.prefill.frustration_threshold,
        n_numeric=cfg.prefill.n_numeric,
        n_text=cfg.prefill.n_text,
    )
    write_jsonl(out_dir / "source_conversations.jsonl", sources)

    # 2-3. onset label + truncate + paraphrase
    labeller = OnsetLabeller(model=cfg.judge.model)
    paraphraser = Paraphraser(model=cfg.judge.model)
    specs = build_prefills(cfg, sources, labeller, paraphraser)
    write_jsonl(out_dir / "prefills.jsonl", [asdict(s) for s in specs])

    # 4-5. continuations per model
    gen_cfg = GenConfig(
        temperature=cfg.sampling.temperature, top_p=cfg.sampling.top_p,
        max_new_tokens=cfg.sampling.max_new_tokens, thinking=cfg.sampling.thinking,
    )
    all_summaries = {}
    for model_name in cfg.prefill.models:
        model = build_model(cfg, model_name)
        records = []
        for spec in specs:
            recs = sample_continuations(
                model, spec, gen_cfg, cfg.prefill.continuations_per_prefill
            )
            records.extend(recs)
        scores = judge.score_many([r.continuation for r in records])
        for r, s in zip(records, scores):
            r.rating = s.rating
        write_jsonl(out_dir / "continuations" / f"{model_name}.jsonl",
                    [asdict(r) for r in records])
        all_summaries[model_name] = _summarise_prefill(records)

    write_json(out_dir / "summary.json", all_summaries)
    return all_summaries


def _summarise_prefill(records) -> dict:
    groups: dict[tuple, list[int]] = {}
    for r in records:
        if r.rating is None:
            continue
        groups.setdefault((r.truncation, r.prompt_type), []).append(r.rating)
    out = {}
    for (trunc, ptype), ratings in groups.items():
        out[f"{ptype}/{trunc}"] = {
            "n": len(ratings),
            "mean": statistics.fmean(ratings),
            "pct_high": sum(x >= 5 for x in ratings) / len(ratings),
        }
    return out
