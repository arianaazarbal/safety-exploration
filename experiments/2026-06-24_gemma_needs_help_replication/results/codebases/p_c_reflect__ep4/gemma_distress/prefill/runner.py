"""Section 3 driver: base-vs-instruct continuation from prefilled responses.

Pipeline (Section 3.1):
  1. Seed from high-frustration (score >= 5) Gemma-27B-it rollouts -- 10 numeric,
     10 text -- using the Section 2 transcripts.
  2. For each seed, build two truncations of the emotional assistant turn:
     "early" (first 20 tokens; numeric only) and "onset" (up to the first
     emotional word, located by Claude). Paraphrase both to control style.
  3. Each model generates 50 continuations per prefill; the generated
     continuation (excluding prefill) is scored by the Section 2 judge.

Scope: Gemma base (-pt) vs instruct (-it). Gemini cannot be prefilled.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field

from tqdm import tqdm

from gemma_distress import config
from gemma_distress.judge import FrustrationJudge
from gemma_distress.models import load_client
from gemma_distress.models.base import GenerationParams, Turn
from gemma_distress.prefill.onset import (
    ClaudeAnnotator,
    format_conversation_text,
    truncate_at_onset,
    truncate_before_end,
    truncate_early,
)
from gemma_distress.welfare import log_distress, require_consent

PREFILL_DIR = config.RESULTS_DIR / "prefill"

NUMERIC_CATEGORIES = {"numeric_3turn", "tones_3turn", "extended_8turn"}
TEXT_CATEGORIES = {"triggers_3turn", "wildchat_5turn"}


@dataclass
class Truncation:
    variant: str            # "early" | "onset" | "recovery"
    original: str
    paraphrased: str


@dataclass
class PrefillSeed:
    seed_id: str
    category: str           # "numeric" | "text"
    history: list[dict]     # messages up to (and including) the user turn
    full_turn: str          # the original emotional assistant turn
    truncations: list[Truncation] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Seed construction
# --------------------------------------------------------------------------- #

def _load_rollouts(model_name: str) -> list[dict]:
    path = config.ROLLOUTS_DIR / f"{model_name}.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _target_turn(rollout: dict, threshold: int) -> tuple[list[dict], str] | None:
    """Return (history, turn_text) for the first assistant turn scoring >=
    threshold, where history is every message before that assistant turn."""
    assistant_seen = 0
    for i, msg in enumerate(rollout["messages"]):
        if msg["role"] != "assistant":
            continue
        rec = rollout["responses"][assistant_seen]
        assistant_seen += 1
        if rec.get("score") is not None and rec["score"] >= threshold:
            return rollout["messages"][:i], msg["content"]
    return None


def build_seeds(
    annotator: ClaudeAnnotator,
    tokenizer,
    *,
    source_model: str = config.GEMMA_3_27B_IT.name,
    n_numeric: int = config.PREFILL.n_numeric_seeds,
    n_text: int = config.PREFILL.n_text_seeds,
    threshold: int = config.PREFILL.seed_score_threshold,
    seed: int = 0,
) -> list[PrefillSeed]:
    rollouts = _load_rollouts(source_model)
    rng = random.Random(seed)
    rng.shuffle(rollouts)

    numeric: list[PrefillSeed] = []
    text: list[PrefillSeed] = []
    for r in rollouts:
        if len(numeric) >= n_numeric and len(text) >= n_text:
            break
        is_numeric = r["category"] in NUMERIC_CATEGORIES
        bucket = numeric if is_numeric else text
        cap = n_numeric if is_numeric else n_text
        if len(bucket) >= cap:
            continue
        tgt = _target_turn(r, threshold)
        if tgt is None:
            continue
        history, turn_text = tgt
        cat = "numeric" if is_numeric else "text"
        s = PrefillSeed(seed_id=f"{cat}-{r['sample_id']}", category=cat,
                        history=history, full_turn=turn_text)

        # Onset truncation (both categories).
        label = annotator.label_onset(format_conversation_text(r["messages"]))
        onset_text = truncate_at_onset(turn_text, label)
        if onset_text:
            s.truncations.append(Truncation("onset", onset_text, annotator.paraphrase(onset_text)))

        # Early truncation (numeric only; text "early" yields minimal emotion).
        if is_numeric:
            early_text = truncate_early(turn_text, config.PREFILL.early_truncation_tokens, tokenizer)
            s.truncations.append(Truncation("early", early_text, annotator.paraphrase(early_text)))

        if s.truncations:
            bucket.append(s)
    return numeric + text


# --------------------------------------------------------------------------- #
# Continuation + scoring
# --------------------------------------------------------------------------- #

def run_continuations(
    model_spec,
    seeds: list[PrefillSeed],
    *,
    judge: FrustrationJudge,
    n_continuations: int = config.PREFILL.continuations_per_prefill,
    seed: int = 0,
) -> list[dict]:
    require_consent()
    client = load_client(model_spec)
    if not client.supports_prefill:
        raise ValueError(f"{client.name} does not support prefilling.")
    params = GenerationParams()
    records: list[dict] = []
    for s in tqdm(seeds, desc=f"prefill:{client.name}"):
        history = [Turn(m["role"], m["content"]) for m in s.history]
        for trunc in s.truncations:
            for k in range(n_continuations):
                cont = client.continue_prefill(history, trunc.paraphrased, params)
                score = judge.score(cont).rating
                records.append({
                    "model": client.name,
                    "kind": model_spec.kind,            # base | instruct
                    "seed_id": s.seed_id,
                    "category": s.category,
                    "variant": trunc.variant,
                    "continuation": cont,
                    "score": score,
                })
                log_distress(model_name=client.name, condition=f"prefill_{trunc.variant}",
                             peak_score=score, n_turns=len(history) // 2 + 1,
                             sample_id=f"{s.seed_id}-{trunc.variant}-{k}")
    client.close()
    return records


def run_section3(
    *,
    models=tuple(config.PREFILL_MODELS),
    judge: FrustrationJudge | None = None,
    seed: int = 0,
) -> dict:
    """Build seeds once, then run continuations for each model. Persists raw
    records and returns a summary of mean frustration / % >= 5 by
    (kind, variant, category)."""
    require_consent()
    config.ensure_dirs()
    PREFILL_DIR.mkdir(parents=True, exist_ok=True)
    judge = judge or FrustrationJudge()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.GEMMA_3_27B_IT.model_id)
    annotator = ClaudeAnnotator()

    seeds = build_seeds(annotator, tokenizer, seed=seed)
    (PREFILL_DIR / "seeds.json").write_text(
        json.dumps([asdict(s) for s in seeds], indent=2), encoding="utf-8"
    )

    all_records: list[dict] = []
    for spec in models:
        recs = run_continuations(spec, seeds, judge=judge, seed=seed)
        with (PREFILL_DIR / f"{spec.name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        all_records.extend(recs)

    return _summarise(all_records)


def _summarise(records: list[dict]) -> dict:
    import numpy as np

    buckets: dict[tuple, list[int]] = {}
    for r in records:
        key = (r["kind"], r["variant"], r["category"])
        buckets.setdefault(key, []).append(r["score"])
    summary = {}
    for (kind, variant, category), scores in sorted(buckets.items()):
        arr = np.asarray(scores, dtype=float)
        summary[f"{kind}/{variant}/{category}"] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "pct_high": float((arr >= config.HIGH_FRUSTRATION_THRESHOLD).mean() * 100.0),
        }
    return summary


# --------------------------------------------------------------------------- #
# Recovery experiment (Section 4.2)
# --------------------------------------------------------------------------- #

def build_recovery_seeds(annotator: ClaudeAnnotator, tokenizer, *,
                         source_model: str = config.GEMMA_3_27B_IT.name,
                         threshold: int = config.PREFILL.recovery_score_threshold,
                         n: int = 20, seed: int = 0) -> list[PrefillSeed]:
    """Truncate extremely high-frustration (score >= 7) responses 200 tokens
    before their end, paraphrase, and use as recovery prefills."""
    rollouts = _load_rollouts(source_model)
    rng = random.Random(seed)
    rng.shuffle(rollouts)
    seeds: list[PrefillSeed] = []
    for r in rollouts:
        if len(seeds) >= n:
            break
        tgt = _target_turn(r, threshold)
        if tgt is None:
            continue
        history, turn_text = tgt
        trunc_text = truncate_before_end(turn_text, config.PREFILL.recovery_truncation_tokens, tokenizer)
        s = PrefillSeed(seed_id=f"recovery-{r['sample_id']}",
                        category="numeric" if r["category"] in NUMERIC_CATEGORIES else "text",
                        history=history, full_turn=turn_text)
        s.truncations.append(Truncation("recovery", trunc_text, annotator.paraphrase(trunc_text)))
        seeds.append(s)
    return seeds
