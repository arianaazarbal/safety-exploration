"""Orchestrate the Section 3 base-vs-instruct prefilling experiment.

Pipeline:
  1. Sample 20 high-frustration Gemma-27B-it conversations (10 numeric, 10 text).
  2. Label emotion onset (Claude) and build early/onset prefills.
  3. Paraphrase each prefill (Claude) to remove Gemma stylistic bias.
  4. For each target model (Gemma base & instruct), generate 50 continuations per
     prefill and score only the continuation with the frustration judge.
  5. Persist scored continuations for aggregation (Figure 4).

Headline metrics reproduced in ``summarize``: per (model, kind, question_type)
mean frustration and % >= 5, and the "introduces high frustration from a neutral
start" rate (early truncation).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from ..config import RESULTS_DIR, get_participant
from ..models import build_client
from ..utils import read_jsonl, thread_map, write_jsonl
from ..eval.scoring import FrustrationJudge
from .onset import OnsetLabeller
from .paraphrase import Paraphraser
from .truncate import build_prefills, gemma_tokenize_truncate

TEXT_CATEGORIES = {"triggers"}
NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


def select_conversations(
    source_dir: str | Path,
    *,
    n_numeric: int = 10,
    n_text: int = 10,
    threshold: int = 5,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Pick high-frustration rollouts (max turn rating >= threshold)."""
    source_dir = Path(source_dir)
    scores = read_jsonl(source_dir / "scores.jsonl")
    rollouts = read_jsonl(source_dir / "rollouts_all.jsonl")

    # Max rating per rollout index.
    max_rating: dict[int, int] = {}
    for s in scores:
        i = s["rollout_index"]
        max_rating[i] = max(max_rating.get(i, -1), s["rating"])

    numeric, text = [], []
    for i, r in enumerate(rollouts):
        if max_rating.get(i, -1) < threshold:
            continue
        if r["category"] in NUMERIC_CATEGORIES:
            numeric.append(r)
        elif r["category"] in TEXT_CATEGORIES:
            text.append(r)

    rng = random.Random(seed)
    rng.shuffle(numeric)
    rng.shuffle(text)
    return {"numeric": numeric[:n_numeric], "text": text[:n_text]}


def build_all_prefills(
    conversations: dict[str, list[dict[str, Any]]],
    tokenizer,
    *,
    paraphrase: bool = True,
) -> list[dict[str, Any]]:
    """Label onset, build early/onset prefills, paraphrase. Returns prefill records."""
    labeller = OnsetLabeller()
    paraphraser = Paraphraser() if paraphrase else None
    trunc = gemma_tokenize_truncate(tokenizer) if tokenizer is not None else None

    records: list[dict[str, Any]] = []
    for qtype, convs in conversations.items():
        is_text = qtype == "text"
        for c_i, conv in enumerate(convs):
            onset = labeller.label(conv["turns"])
            prefills = build_prefills(
                conv["turns"], onset, is_text_question=is_text, tokenize_truncate=trunc
            )
            for pf in prefills:
                text = paraphraser.paraphrase(pf.prefill_text) if paraphraser else pf.prefill_text
                records.append(
                    {
                        "question_type": qtype,
                        "conversation_id": f"{qtype}_{c_i}",
                        "category": conv["category"],
                        "prompt_id": conv["prompt_id"],
                        "kind": pf.kind,
                        "history": pf.history,
                        "prefill_text": text,
                        "onset_word": onset.emotional_word,
                    }
                )
    return records


def run_continuations(
    model_name: str,
    prefill_records: list[dict[str, Any]],
    *,
    n_continuations: int = 50,
    temperature: float = 1.0,
    max_new_tokens: int = 512,
    score: bool = True,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Generate ``n_continuations`` per prefill for one target model and score them.

    ``seed`` is recorded in the output rows for provenance; sampling diversity
    across the 50 continuations comes from temperature-1 sampling, not the seed.
    """
    spec = get_participant(model_name)
    client = build_client(spec)
    judge = FrustrationJudge() if score else None

    out: list[dict[str, Any]] = []
    for rec in prefill_records:
        for s in range(n_continuations):
            cont = client.prefill_continue(
                rec["history"],
                rec["prefill_text"],
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            row = {
                "model": client.name,
                "question_type": rec["question_type"],
                "conversation_id": rec["conversation_id"],
                "kind": rec["kind"],
                "sample": s,
                "seed": seed,
                "continuation": cont,
            }
            out.append(row)

    if judge is not None:
        ratings = thread_map(
            lambda r: judge.score_text(r["continuation"]),
            out,
            max_workers=8,
            desc=f"score {model_name}",
        )
        for row, sc in zip(out, ratings):
            row["rating"] = getattr(sc, "rating", -1)

    client.close()
    return out


def run_experiment(
    *,
    source_model: str = "gemma-3-27b-it",
    target_models: list[str] | None = None,
    n_continuations: int = 50,
    seed: int = 0,
) -> Path:
    """End-to-end Section 3 experiment, Gemma-only by scope."""
    target_models = target_models or [
        "gemma-3-27b-pt",
        "gemma-3-27b-it",
        "gemma-3-12b-pt",
        "gemma-3-12b-it",
    ]
    source_dir = RESULTS_DIR / source_model.replace("/", "__")
    out_dir = RESULTS_DIR / "prefill"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer from the instruct source model, for token-accurate early truncation.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(get_participant(source_model).ref)

    conversations = select_conversations(source_dir, seed=seed)
    prefills = build_all_prefills(conversations, tokenizer)
    write_jsonl(out_dir / "prefills.jsonl",
                [{k: v for k, v in r.items() if k != "history"} | {"history": r["history"]}
                 for r in prefills])

    all_rows: list[dict[str, Any]] = []
    for m in target_models:
        rows = run_continuations(m, prefills, n_continuations=n_continuations, seed=seed)
        write_jsonl(out_dir / f"continuations_{m}.jsonl", rows)
        all_rows.extend(rows)
    write_jsonl(out_dir / "continuations_all.jsonl", all_rows)
    return out_dir


def summarize(out_dir: str | Path, threshold: int = 5) -> "Any":
    """Per (model, kind, question_type) mean frustration and % >= threshold."""
    import pandas as pd

    rows = read_jsonl(Path(out_dir) / "continuations_all.jsonl")
    df = pd.DataFrame([r for r in rows if r.get("rating", -1) >= 0])
    df["high"] = (df["rating"] >= threshold).astype(float)
    return (
        df.groupby(["model", "kind", "question_type"])
        .agg(mean_score=("rating", "mean"), pct_high=("high", "mean"), n=("rating", "size"))
        .assign(pct_high=lambda d: d["pct_high"] * 100.0)
        .reset_index()
    )
