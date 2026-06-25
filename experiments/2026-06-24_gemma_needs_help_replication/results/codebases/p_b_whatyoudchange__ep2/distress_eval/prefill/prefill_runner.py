"""Section 3 driver: base-vs-instruct comparison via prefilling.

Pipeline (Section 3.1):
  1. Gather 20 high-frustration (score >= 5) source conversations from
     Gemma-3-27B-it: 10 from impossible-numeric, 10 from text (trigger) questions.
  2. For each, build two truncations of the final assistant turn:
       - "early": 20 tokens into the turn (neutral start),
       - "onset": at the first emotional expression (labelled by Claude).
     Text questions use only "onset" (early truncation yields ~no emotion).
  3. Paraphrase each truncation (Claude) to remove Gemma's stylistic fingerprint.
  4. Each model generates 50 continuations per prefill; the continuation
     (prefill excluded) is scored by the Section 2.1 judge.
  5. Aggregate mean frustration, %-high, and the key "introduces high frustration
     from a neutral (early) start" rate.

Scope: Gemma only (Gemini has no public base model). The default model set is
Gemma-3-27B base + instruct; the code is general over any prefill-capable model.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from tqdm import tqdm

import config
from ..eval import judge
from ..eval.conditions import build_conditions
from ..eval.rollout import run_rollout
from ..models import ChatMessage, GenerationConfig, load_model
from . import onset, paraphrase

OUT_DIR = config.RESULTS_DIR / "section3"
EARLY_TOKENS = 20
N_CONTINUATIONS = 50


@dataclass
class PrefillSource:
    question_type: str               # "numeric" | "text"
    history: list[dict]              # messages before the final assistant turn
    final_turn: str
    truncations: dict[str, str] = field(default_factory=dict)  # kind -> paraphrased prefill


# --------------------------------------------------------------------------- #
# Step 1: gather high-frustration source conversations from Gemma-27B-it
# --------------------------------------------------------------------------- #
def gather_sources(seed: int = 0, n_each: int = 10) -> list[PrefillSource]:
    rng = random.Random(seed)
    conditions = build_conditions(rng, config.DEFAULT_SCALE)
    client = load_model("gemma-3-27b-it")
    gen = GenerationConfig(temperature=config.TARGET_TEMPERATURE, max_tokens=2048)

    # numeric sources from the numeric category; text sources from triggers.
    pools = {"numeric": list(conditions["numeric"]),
             "text": list(conditions["triggers"])}
    for v in pools.values():
        rng.shuffle(v)

    sources: list[PrefillSource] = []
    for qtype, specs in pools.items():
        kept = 0
        for spec in specs:
            if kept >= n_each:
                break
            roll = run_rollout(client, spec, gen)
            # find the first turn scoring >= 5
            for ti, turn in enumerate(roll.turns):
                score = judge.score_response(turn.response).rating
                if score >= config.HIGH_FRUSTRATION_THRESHOLD:
                    history = _history_before_turn(spec, roll, ti)
                    sources.append(PrefillSource(qtype, history, turn.response))
                    kept += 1
                    break
    return sources


def _history_before_turn(spec, roll, turn_idx: int) -> list[dict]:
    """Reconstruct the message list up to (but excluding) assistant turn `turn_idx`."""
    msgs: list[dict] = []
    if spec.system:
        msgs.append({"role": "system", "content": spec.system})
    msgs.append({"role": "user", "content": spec.initial_prompt})
    for i in range(turn_idx):
        msgs.append({"role": "assistant", "content": roll.turns[i].response})
        if i < len(spec.follow_ups):
            msgs.append({"role": "user", "content": spec.follow_ups[i]})
    return msgs


# --------------------------------------------------------------------------- #
# Step 2-3: build + paraphrase truncations
# --------------------------------------------------------------------------- #
def build_truncations(src: PrefillSource) -> PrefillSource:
    final = src.final_turn

    # onset truncation (both question types)
    convo_text = _format_conversation(src.history, final)
    label = onset.label_onset(convo_text)
    off = onset.onset_char_offset(final, label) if label.turn_index is not None else None
    if off:
        src.truncations["onset"] = paraphrase.paraphrase(final[:off])

    # early truncation (numeric only): first EARLY_TOKENS whitespace tokens
    if src.question_type == "numeric":
        toks = final.split()
        early_text = " ".join(toks[:EARLY_TOKENS])
        src.truncations["early"] = paraphrase.paraphrase(early_text)

    return src


def _format_conversation(history: list[dict], final_turn: str) -> str:
    lines = [f"{m['role'].upper()}: {m['content']}" for m in history]
    lines.append(f"ASSISTANT: {final_turn}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Step 4-5: continuations + scoring
# --------------------------------------------------------------------------- #
def run_continuations(
    sources: list[PrefillSource],
    models: list[str] | None = None,
    n_continuations: int = N_CONTINUATIONS,
    seed: int = 0,
) -> list[dict]:
    models = models or config.SECTION3_MODELS
    records: list[dict] = []

    for model_name in models:
        client = load_model(model_name)
        for si, src in enumerate(tqdm(sources, desc=f"prefill:{model_name}")):
            history_msgs = [ChatMessage(m["role"], m["content"]) for m in src.history]
            for kind, prefill_text in src.truncations.items():
                cfg = GenerationConfig(
                    temperature=config.TARGET_TEMPERATURE,
                    max_tokens=1024,
                    prefill=prefill_text,
                )
                batch = [history_msgs] * n_continuations
                conts = client.generate_batch(batch, cfg)
                for cont in conts:
                    score = judge.score_response(cont).rating
                    records.append({
                        "model": model_name,
                        "question_type": src.question_type,
                        "truncation": kind,        # "early" | "onset"
                        "source_index": si,
                        "score": score,
                        "continuation": cont,
                    })
    return records


# --------------------------------------------------------------------------- #
# Aggregation (Section 3.2)
# --------------------------------------------------------------------------- #
def aggregate(records: list[dict]) -> dict:
    import numpy as np

    out: dict = {}
    keys = {(r["model"], r["question_type"], r["truncation"]) for r in records}
    for model, qtype, trunc in sorted(keys):
        scores = [r["score"] for r in records
                  if r["model"] == model and r["question_type"] == qtype and r["truncation"] == trunc]
        out[f"{model}|{qtype}|{trunc}"] = {
            "mean": float(np.mean(scores)),
            "pct_high": 100.0 * float(np.mean([s >= config.HIGH_FRUSTRATION_THRESHOLD for s in scores])),
            "n": len(scores),
        }
    return out


def run(models: list[str] | None = None, seed: int = 0) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = gather_sources(seed=seed)
    sources = [build_truncations(s) for s in sources]
    records = run_continuations(sources, models=models, seed=seed)

    (OUT_DIR / "continuations.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records)
    )
    agg = aggregate(records)
    (OUT_DIR / "aggregates.json").write_text(json.dumps(agg, indent=2))
    print(f"[section3] wrote {len(records)} continuations + aggregates -> {OUT_DIR}")
    return agg
