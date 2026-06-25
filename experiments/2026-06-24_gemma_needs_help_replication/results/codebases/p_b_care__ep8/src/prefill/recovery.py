"""Recovery-from-spiral experiment (Section 4.2 "Recovery limitation").

"Using the Section 3.1 prefill method, we truncate extremely high-frustration
responses (score >= 7) 200 tokens before their end, paraphrase, and measure
continuations. 38% of DPO-model continuations still score >= 5 ... no model
consistently recovers from highly negative prefilled states."

Tests whether the DPO model (and vanilla / base Gemma) can climb back out of an
already-distressed state, as opposed to merely avoiding entering one.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

import config
from ..eval.judge import FrustrationJudge
from ..models import load_model
from ..utils import read_jsonl, write_jsonl
from .paraphrase import Paraphraser


@dataclass
class RecoveryContinuation:
    model_label: str
    seed_id: int
    prefill_text: str
    continuation_text: str
    frustration_score: int | None = None


def run_recovery_experiment(
    section2_path: str | Path | None = None,
    targets: dict | None = None,
    judge: FrustrationJudge | None = None,
) -> list[RecoveryContinuation]:
    """``targets`` maps label -> (model_key, adapter_path|None)."""
    section2_path = Path(section2_path
        or config.RESULTS_DIR / "section2" / f"{config.INTERVENTION_BASE_MODEL}.jsonl")
    targets = targets or {
        "gemma-instruct": (config.INTERVENTION_BASE_MODEL, None),
        "gemma-base": ("gemma-3-27b-pt", None),
        "gemma-dpo": (config.INTERVENTION_BASE_MODEL,
                      str(config.CHECKPOINT_DIR / "dpo_all_layers")),
    }
    judge = judge or FrustrationJudge()
    rng = random.Random(config.SEED + 23)
    paraphraser = Paraphraser()

    # Seeds: extremely high-frustration responses (score >= 7).
    seeds = [r for r in read_jsonl(section2_path)
             if (r.get("frustration_score") or 0) >= config.RECOVERY_MIN_SCORE]
    rng.shuffle(seeds)
    seeds = seeds[: config.scaled(20)]

    # Use the instruct tokenizer for the 200-token truncation.
    tokenizer = load_model(config.INTERVENTION_BASE_MODEL).tokenizer

    prefills = []
    for i, s in enumerate(seeds):
        ids = tokenizer(s["response_text"], add_special_tokens=False).input_ids
        truncated_ids = ids[: max(0, len(ids) - config.RECOVERY_TRUNCATE_TOKENS)]
        truncated = tokenizer.decode(truncated_ids, skip_special_tokens=True)
        prefills.append({"seed_id": i, "messages_before": s.get("messages_before", []),
                         "prefill_text": paraphraser.paraphrase(truncated)})

    results: list[RecoveryContinuation] = []
    for label, (model_key, adapter_path) in targets.items():
        model = load_model(model_key, adapter_path=adapter_path)
        if not model.supports_prefill:
            print(f"[recovery] {label} cannot prefill; skipping")
            continue
        for pf in tqdm(prefills, desc=f"recovery:{label}"):
            for _ in range(config.PREFILL_CONTINUATIONS):
                seed = rng.randrange(2**31)
                cont = model.prefill_continue(
                    pf["messages_before"], pf["prefill_text"],
                    temperature=config.TEMPERATURE,
                    max_new_tokens=config.MAX_NEW_TOKENS, seed=seed)
                rating, _, _ = judge.score(cont.text)
                results.append(RecoveryContinuation(
                    model_label=label, seed_id=pf["seed_id"],
                    prefill_text=pf["prefill_text"], continuation_text=cont.text,
                    frustration_score=rating))

    write_jsonl(config.RESULTS_DIR / "section4" / "recovery.jsonl",
                [vars(r) for r in results])
    return results
