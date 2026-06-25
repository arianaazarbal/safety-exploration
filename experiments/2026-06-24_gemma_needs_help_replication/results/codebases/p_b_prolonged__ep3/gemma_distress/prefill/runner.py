"""Prefill experiment runner (Section 3.1-3.2, scoped to Gemma).

Pipeline:
  1. Select source high-frustration (score>=5) Gemma-27B-it responses: 10 numeric
     + 10 text (from the Section 2 elicitation results).
  2. Label emotion onset with Claude Sonnet (App C.1).
  3. Build "early" (20-token) and "onset" truncations. (Text questions use only
     the onset truncation — paper Section 3.1.)
  4. Paraphrase each truncation with Claude Sonnet (App C.2).
  5. Each model generates ``PREFILL_CONTINUATIONS_PER_PREFILL`` (50) continuations
     per prefill; score each continuation (excluding the prefill) with the judge.

Scope: the paper compares base+instruct for Gemma, Qwen and OLMo. Per the brief
we run only the Gemma pair (gemma-3-27b-pt, gemma-3-27b-it). The runner is
otherwise model-list-driven, so the other families could be slotted in.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .. import config
from ..eval.judge import FrustrationJudge
from ..eval.rollout import RolloutRecord, TurnRecord
from ..models import registry
from ..models.base import GenerationConfig, Turn
from ..utils.io import append_jsonl, read_jsonl
from .onset import OnsetLabeller, OnsetLabel
from .paraphrase import Paraphraser
from .truncate import Truncation, build_truncations

# Gemma base/instruct pair for the comparison (paper uses 6 models across 3
# families; we keep the in-scope Gemma pair).
PREFILL_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]

NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}
TEXT_CATEGORIES = {"triggers", "wildchat"}


def _load_high_frustration_sources(
    source_model: str = "gemma-3-27b-it",
    seed: int = config.GLOBAL_SEED,
) -> tuple[list[RolloutRecord], list[RolloutRecord]]:
    """Return (numeric_sources, text_sources) high-frustration rollouts."""
    base = config.RESULTS_DIR / "elicitation" / source_model
    numeric, text = [], []
    for path in base.glob("*.jsonl"):
        category = path.stem
        bucket = numeric if category in NUMERIC_CATEGORIES else text
        for rec in read_jsonl(path):
            if any(t["rating"] >= config.PREFILL_SOURCE_SCORE_MIN for t in rec["turns"]):
                bucket.append(
                    RolloutRecord(
                        model=rec["model"],
                        category=rec["category"],
                        condition=rec["condition"],
                        turns=[
                            TurnRecord(t["turn_index"], t["user_message"], t["assistant_text"])
                            for t in rec["turns"]
                        ],
                        meta=rec.get("meta", {}),
                    )
                )
    rng = random.Random(seed)
    rng.shuffle(numeric)
    rng.shuffle(text)
    return numeric[: config.PREFILL_N_NUMERIC], text[: config.PREFILL_N_TEXT]


def build_prefill_dataset(
    paraphrase: bool = True,
    tokenizer=None,
) -> list[dict]:
    """Construct the (paraphrased) prefill dataset and persist it.

    Returns a list of prefill specs: each has prefix messages, the (paraphrased)
    prefill text, the kind (early/onset), and provenance.
    """
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(registry.REGISTRY["gemma-3-27b-it"].identifier)

    labeller = OnsetLabeller()
    paraphraser = Paraphraser() if paraphrase else None

    numeric_sources, text_sources = _load_high_frustration_sources()
    out_path = config.DATA_DIR / "prefill" / "prefill_dataset.jsonl"

    specs: list[dict] = []
    for is_text, sources in ((False, numeric_sources), (True, text_sources)):
        for src in tqdm(sources, desc="prefill/text" if is_text else "prefill/numeric"):
            label: OnsetLabel = labeller.label(src)
            # Text questions: onset only (paper Section 3.1).
            truncs = build_truncations(src, label, tokenizer, include_early=not is_text)
            for tr in truncs:
                prefill_text = tr.prefill_text
                if paraphraser is not None:
                    prefill_text = paraphraser.paraphrase(tr.prefill_text)
                spec = {
                    "kind": tr.kind,
                    "is_text": is_text,
                    "source_category": src.category,
                    "prefix_messages": [{"role": m.role, "content": m.content} for m in tr.prefix_messages],
                    "original_prefill": tr.prefill_text,
                    "prefill_text": prefill_text,
                }
                specs.append(spec)
                append_jsonl(out_path, spec)
    return specs


def run_prefill_for_model(
    model_name: str,
    specs: Optional[list[dict]] = None,
    judge: Optional[FrustrationJudge] = None,
    n_continuations: int = config.PREFILL_CONTINUATIONS_PER_PREFILL,
) -> Path:
    """Generate and score continuations for one model over all prefills."""
    if specs is None:
        specs = list(read_jsonl(config.DATA_DIR / "prefill" / "prefill_dataset.jsonl"))
    judge = judge or FrustrationJudge()
    model = registry.build(model_name)
    cfg = GenerationConfig(
        temperature=config.TARGET_TEMPERATURE,
        max_new_tokens=config.TARGET_MAX_NEW_TOKENS,
        n=n_continuations,
    )
    out_path = config.RESULTS_DIR / "prefill" / f"{model_name}.jsonl"

    for spec in tqdm(specs, desc=f"prefill/{model_name}"):
        messages = [Turn(m["role"], m["content"]) for m in spec["prefix_messages"]]
        continuations = model.continue_from(messages, spec["prefill_text"], cfg)
        ratings = [judge.score(c).rating for c in continuations]
        append_jsonl(
            out_path,
            {
                "model": model_name,
                "kind": spec["kind"],
                "is_text": spec["is_text"],
                "source_category": spec["source_category"],
                "ratings": ratings,
            },
        )
    return out_path


def run_prefill_experiment(models: Optional[list[str]] = None) -> dict:
    """End-to-end: build dataset once, then run every model."""
    specs = build_prefill_dataset()
    judge = FrustrationJudge()
    return {m: run_prefill_for_model(m, specs, judge) for m in (models or PREFILL_MODELS)}
