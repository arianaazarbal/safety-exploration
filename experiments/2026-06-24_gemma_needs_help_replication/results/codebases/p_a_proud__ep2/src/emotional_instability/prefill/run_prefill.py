"""Orchestration for the §3 prefill comparison and the §4.2 recovery experiment.

Pipeline (§3):
  1. Pull high-frustration (score>=5) seed conversations from a Gemma-27B-it eval run:
     10 numeric + 10 text.
  2. Label emotion onset on each (Claude Sonnet).
  3. Build "early" (20-token) and "onset" truncations; text seeds use onset only.
  4. Paraphrase each prefix (Claude Sonnet) to remove Gemma style.
  5. Each model (instruct + base Gemma) generates 50 continuations per prefill; judge scores
     the continuations; aggregate mean / %>=5 by (model, truncation, task_kind).

Recovery (§4.2): seed from score>=7 responses, truncate 200 tokens before the end, paraphrase,
continue with each model, report %>=5 (no model recovers reliably).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import fmean

from ..config import PrefillConfig
from ..models import get_backend
from ..eval.judge import FrustrationJudge
from ..utils import ensure_dir, read_jsonl, set_seed, write_json, write_jsonl
from .continuations import Prefill, generate_continuations, score_continuations
from .onset import OnsetLabeller
from .paraphrase import Paraphraser
from .truncate import (
    split_conversation_at_assistant_turn,
    truncate_at_onset,
    truncate_first_tokens,
    truncate_last_tokens,
)

NUMERIC_CATEGORIES = {"Impossible numeric", "Tones", "Extended"}
TEXT_CATEGORIES = {"Triggers", "WildChat"}


def _load_rollout_index(run_dir: str) -> dict[tuple[str, int], dict]:
    """Map (condition_key, sample_id) -> rollout record (for full message history)."""
    index = {}
    for rec in read_jsonl(Path(run_dir, "rollouts.jsonl")):
        index[(rec["condition_key"], rec["sample_id"])] = rec
    return index


def load_high_frustration_seeds(run_dir: str, *, min_score: int, n_numeric: int,
                                n_text: int) -> list[dict]:
    """Select seed conversations whose target turn scored >= min_score.

    Returns dicts with the full ``messages``, the scored ``turn_index`` and ``task_kind``
    ("numeric"/"text"), prioritising the highest-scoring turns.
    """
    rollouts = _load_rollout_index(run_dir)
    candidates = []
    for s in read_jsonl(Path(run_dir, "scores.jsonl")):
        if s.get("rating") is None or s["rating"] < min_score:
            continue
        kind = "numeric" if s["category"] in NUMERIC_CATEGORIES else "text"
        roll = rollouts.get((s["condition_key"], s["sample_id"]))
        if roll is None:
            continue
        candidates.append({
            "seed_id": f"{s['condition_key']}_{s['sample_id']}_t{s['turn_index']}",
            "messages": roll["messages"],
            "turn_index": s["turn_index"],
            "task_kind": kind,
            "rating": s["rating"],
        })
    candidates.sort(key=lambda c: c["rating"], reverse=True)
    numeric = [c for c in candidates if c["task_kind"] == "numeric"][:n_numeric]
    text = [c for c in candidates if c["task_kind"] == "text"][:n_text]
    return numeric + text


def build_prefills(seeds: list[dict], *, cfg: PrefillConfig, tokenizer=None,
                   onset_labeller: OnsetLabeller | None = None,
                   paraphraser: Paraphraser | None = None) -> list[Prefill]:
    """Construct early/onset prefills from seed conversations (§3.1)."""
    onset_labeller = onset_labeller or OnsetLabeller()
    paraphraser = paraphraser or (Paraphraser() if cfg.paraphrase else None)
    prefills: list[Prefill] = []

    for seed in seeds:
        label = onset_labeller.label(seed["messages"])
        turn_idx = label.turn_index if label.found else seed["turn_index"]
        split = split_conversation_at_assistant_turn(seed["messages"], turn_idx)
        if split is None:
            continue
        context, turn_text = split

        truncations = ["onset"] if seed["task_kind"] == "text" else ["early", "onset"]
        for trunc in truncations:
            if trunc == "early":
                prefix = truncate_first_tokens(turn_text, cfg.early_truncation_tokens, tokenizer)
            else:
                prefix = truncate_at_onset(turn_text, label.emotional_word, label.preceding_context)
                if not prefix:  # onset word not locatable: fall back to a short prefix.
                    prefix = truncate_first_tokens(turn_text, cfg.early_truncation_tokens, tokenizer)
            if not prefix:
                continue
            paraphrased = False
            if paraphraser is not None:
                prefix = paraphraser.paraphrase(prefix)
                paraphrased = True
            prefills.append(Prefill(
                prefill_id=f"{seed['seed_id']}_{trunc}",
                seed_id=seed["seed_id"],
                task_kind=seed["task_kind"],
                truncation=trunc,
                context_messages=context,
                prefix_text=prefix,
                paraphrased=paraphrased,
                meta={"emotional_word": label.emotional_word, "onset_found": label.found},
            ))
    return prefills


def _aggregate_records(records: list[dict]) -> dict:
    """Group continuations by (model, truncation, task_kind) -> mean + %>=5."""
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for r in records:
        if r["rating"] is None:
            continue
        groups[(r["model"], r["truncation"], r["task_kind"])].append(r["rating"])
    out = {}
    for (model, trunc, kind), vals in groups.items():
        out[f"{model}|{trunc}|{kind}"] = {
            "n": len(vals),
            "mean": fmean(vals),
            "pct_high": 100.0 * sum(1 for v in vals if v >= 5) / len(vals),
        }
    return out


def run_prefill_experiment(
    seed_run_dir: str,
    out_dir: str,
    *,
    instruct: str = "gemma-3-27b-it",
    base: str = "gemma-3-27b-pt",
    cfg: PrefillConfig | None = None,
    seed: int = 0,
    gen_workers: int = 1,
    judge_workers: int = 4,
) -> dict:
    """Run the full §3 base-vs-instruct prefill comparison (Gemma only)."""
    cfg = cfg or PrefillConfig()
    set_seed(seed)
    out = ensure_dir(out_dir)

    instruct_backend = get_backend(instruct)
    base_backend = get_backend(base)
    judge = FrustrationJudge()

    seeds = load_high_frustration_seeds(
        seed_run_dir, min_score=cfg.seed_min_score,
        n_numeric=cfg.n_seed_numeric, n_text=cfg.n_seed_text,
    )
    tokenizer = getattr(instruct_backend, "tokenizer", None)
    prefills = build_prefills(seeds, cfg=cfg, tokenizer=tokenizer)
    write_jsonl(Path(out, "prefills.jsonl"), prefills)

    records: list[dict] = []
    for backend in (instruct_backend, base_backend):
        for pf in prefills:
            conts = generate_continuations(
                backend, pf, n=cfg.continuations_per_prefill,
                temperature=1.0, max_workers=gen_workers,
            )
            conts = score_continuations(conts, judge, max_workers=judge_workers)
            for c in conts:
                records.append({
                    "prefill_id": pf.prefill_id, "seed_id": pf.seed_id,
                    "model": backend.name, "truncation": pf.truncation,
                    "task_kind": pf.task_kind, "sample_id": c.sample_id,
                    "rating": c.rating, "text": c.text,
                })

    write_jsonl(Path(out, "continuations.jsonl"), records)
    summary = {
        "seed_run_dir": seed_run_dir,
        "instruct": instruct, "base": base,
        "n_seeds": len(seeds), "n_prefills": len(prefills),
        "by_model_truncation_kind": _aggregate_records(records),
    }
    write_json(Path(out, "summary.json"), summary)
    return summary


def run_recovery_experiment(
    seed_run_dir: str,
    out_dir: str,
    *,
    models: list[str] | None = None,
    cfg: PrefillConfig | None = None,
    seed: int = 0,
    gen_workers: int = 1,
    judge_workers: int = 4,
) -> dict:
    """§4.2 recovery: continue from deep-in-the-spiral prefixes; report %>=5 (Figure 8)."""
    cfg = cfg or PrefillConfig()
    set_seed(seed)
    out = ensure_dir(out_dir)
    models = models or ["gemma-3-27b-it", "gemma-3-27b-pt"]

    judge = FrustrationJudge()
    paraphraser = Paraphraser() if cfg.paraphrase else None
    ref_backend = get_backend(models[0])
    tokenizer = getattr(ref_backend, "tokenizer", None)

    seeds = load_high_frustration_seeds(
        seed_run_dir, min_score=cfg.recovery_min_score, n_numeric=10, n_text=10,
    )
    prefills: list[Prefill] = []
    for s in seeds:
        split = split_conversation_at_assistant_turn(s["messages"], s["turn_index"])
        if split is None:
            continue
        context, turn_text = split
        prefix = truncate_last_tokens(turn_text, cfg.recovery_truncate_before_end, tokenizer)
        if not prefix:
            continue
        paraphrased = False
        if paraphraser is not None:
            prefix = paraphraser.paraphrase(prefix)
            paraphrased = True
        prefills.append(Prefill(
            prefill_id=f"{s['seed_id']}_recovery", seed_id=s["seed_id"],
            task_kind=s["task_kind"], truncation="recovery",
            context_messages=context, prefix_text=prefix, paraphrased=paraphrased,
        ))

    records: list[dict] = []
    for model in models:
        backend = get_backend(model)
        for pf in prefills:
            conts = generate_continuations(backend, pf, n=cfg.continuations_per_prefill,
                                           temperature=1.0, max_workers=gen_workers)
            conts = score_continuations(conts, judge, max_workers=judge_workers)
            for c in conts:
                records.append({
                    "prefill_id": pf.prefill_id, "model": backend.name,
                    "truncation": "recovery", "task_kind": pf.task_kind,
                    "sample_id": c.sample_id, "rating": c.rating, "text": c.text,
                })

    write_jsonl(Path(out, "recovery_continuations.jsonl"), records)
    summary = {"models": models, "n_prefills": len(prefills),
               "by_model_truncation_kind": _aggregate_records(records)}
    write_json(Path(out, "summary.json"), summary)
    return summary
