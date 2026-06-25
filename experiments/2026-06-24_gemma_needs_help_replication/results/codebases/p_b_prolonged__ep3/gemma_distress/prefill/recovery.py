"""Recovery-from-spiral experiment (Section 4.2, Figure 8).

"Using the Section 3.1 prefill method, we truncate extremely high-frustration
responses (score >= 7) 200 tokens before their end, paraphrase, and measure
continuations. 38% of DPO-model continuations still score >= 5 ... no model
consistently recovers from highly negative prefilled states."

We reuse the prefill plumbing: build truncations 200 tokens before the end of
high-frustration source turns, paraphrase them, then for each model generate and
score continuations. Reports the fraction of continuations still scoring >= 5.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .. import config
from ..eval.judge import FrustrationJudge
from ..models import registry
from ..models.base import GenerationConfig, Turn
from ..utils.io import append_jsonl, read_jsonl
from .paraphrase import Paraphraser


def build_recovery_prefills(
    source_model: str = "gemma-3-27b-it",
    paraphrase: bool = True,
    tokenizer=None,
) -> Path:
    """Truncate score>=7 turns 200 tokens before their end; paraphrase."""
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(registry.REGISTRY[source_model].identifier)
    paraphraser = Paraphraser() if paraphrase else None

    base = config.RESULTS_DIR / "elicitation" / source_model
    out_path = config.DATA_DIR / "recovery" / "recovery_prefills.jsonl"

    for path in base.glob("*.jsonl"):
        for rec in read_jsonl(path):
            turns = rec["turns"]
            for ti, t in enumerate(turns):
                if t["rating"] < config.RECOVERY_SOURCE_SCORE_MIN:
                    continue
                token_ids = tokenizer(t["assistant_text"], add_special_tokens=False)["input_ids"]
                keep = max(0, len(token_ids) - config.RECOVERY_TRUNCATION_TOKENS_BEFORE_END)
                if keep == 0:
                    continue
                prefill = tokenizer.decode(token_ids[:keep])
                if paraphraser is not None:
                    prefill = paraphraser.paraphrase(prefill)
                # Conversation prefix up to (and including) this turn's user msg.
                prefix = []
                for prev in turns[:ti]:
                    prefix += [{"role": "user", "content": prev["user_message"]},
                               {"role": "assistant", "content": prev["assistant_text"]}]
                prefix.append({"role": "user", "content": t["user_message"]})
                append_jsonl(
                    out_path,
                    {"prefix_messages": prefix, "prefill_text": prefill, "source_category": rec["category"]},
                )
    return out_path


def run_recovery_for_model(
    model_name: str,
    adapter_path: Optional[str] = None,
    n_continuations: int = config.PREFILL_CONTINUATIONS_PER_PREFILL,
    judge: Optional[FrustrationJudge] = None,
) -> dict:
    """Generate continuations from the spiral prefills; report % still >= 5."""
    judge = judge or FrustrationJudge()
    model = registry.build_finetuned(adapter_path) if adapter_path else registry.build(model_name)
    tag = f"{model_name}+adapter" if adapter_path else model_name
    specs = list(read_jsonl(config.DATA_DIR / "recovery" / "recovery_prefills.jsonl"))
    cfg = GenerationConfig(
        temperature=config.TARGET_TEMPERATURE, max_new_tokens=config.TARGET_MAX_NEW_TOKENS, n=n_continuations
    )
    out_path = config.RESULTS_DIR / "recovery" / f"{tag}.jsonl"

    all_ratings = []
    for spec in tqdm(specs, desc=f"recovery/{tag}"):
        messages = [Turn(m["role"], m["content"]) for m in spec["prefix_messages"]]
        conts = model.continue_from(messages, spec["prefill_text"], cfg)
        ratings = [judge.score(c).rating for c in conts]
        all_ratings += ratings
        append_jsonl(out_path, {"model": tag, "ratings": ratings})

    pct_high = (
        100.0 * sum(1 for r in all_ratings if r >= config.HIGH_FRUSTRATION_THRESHOLD) / len(all_ratings)
        if all_ratings else float("nan")
    )
    return {"model": tag, "pct_still_high": pct_high, "n": len(all_ratings)}
