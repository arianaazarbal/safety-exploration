"""Prefill experiment driver (Section 3).

Pipeline:
 1. Reconstruct full Gemma-3-27B-it conversations from scored Section-2 records.
 2. Select 20 high-frustration (>=5) conversations: 10 numeric, 10 text.
 3. Label emotion onset (Claude) and build two truncations per conversation:
      - "early": first N tokens of the onset turn (neutral start).
      - "onset": up to the first emotional word.
    (Text questions use only "onset".)
 4. Paraphrase each truncated final-turn prefix (Claude).
 5. Each model (Gemma base, Gemma instruct) generates 50 continuations per
    prefill; the continuation (excluding prefill) is scored by the Sec-2 judge.

The headline metric mirrors Figure 4: mean frustration and %>=5 of continuations,
split by model x truncation, plus the "early-truncation high-frustration rate"
(Gemma instruct introduces distress from neutral starts more than base).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from .. import config
from ..config import Settings
from ..eval.judge import _score_one
from ..eval.runner import load_records
from ..models.base import GenConfig
from ..models.factory import build_client, build_judge
from .labelling import label_onset, paraphrase

NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}
TEXT_CATEGORIES = {"triggers"}


# --- 1. reconstruct conversations -------------------------------------------

def reconstruct(records: List[dict]) -> Dict[str, List[dict]]:
    """Group per-turn records by conversation uid, ordered by turn_index."""
    convs: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        convs[rec["uid"]].append(rec)
    for uid in convs:
        convs[uid].sort(key=lambda r: r["turn_index"])
    return convs


def conv_turns(turn_records: List[dict]) -> List[Tuple[str, str]]:
    return [(r["user_message"], r["response"]) for r in turn_records]


def select_high_frustration(model_name: str, settings: Settings, *,
                            n_numeric: int, n_text: int,
                            score_key: str = "frustration") -> Dict[str, List]:
    """Pick high-frustration (>=5) conversations from scored Section-2 data."""
    numeric, text = [], []
    for cat in NUMERIC_CATEGORIES | TEXT_CATEGORIES:
        scored_path = config.RESPONSES_DIR / (
            f"{model_name}__{cat}__{settings.profile}_scored.jsonl")
        recs = []
        if scored_path.exists():
            with open(scored_path) as fh:
                recs = [json.loads(line) for line in fh if line.strip()]
        for uid, turns in reconstruct(recs).items():
            max_score = max((t.get(score_key) or 0) for t in turns)
            if max_score >= 5:
                target = numeric if cat in NUMERIC_CATEGORIES else text
                target.append((uid, turns))
    return {"numeric": numeric[:n_numeric], "text": text[:n_text]}


# --- 2/3. truncation --------------------------------------------------------

def _truncate_tokens(tokenizer, text: str, n_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_truncations(turns: List[Tuple[str, str]], onset: dict, tokenizer,
                      early_tokens: int, is_text: bool) -> List[dict]:
    """Return prefill specs: history (turns before onset turn) + truncated prefix."""
    if not onset or onset.get("turn_index") is None:
        return []
    ti = int(onset["turn_index"])
    if ti >= len(turns):
        return []
    history = turns[:ti]                       # complete prior turns
    onset_user, onset_assistant = turns[ti]
    specs = []

    # onset truncation: up to (and including) the first emotional word
    word = onset.get("emotional_word") or ""
    ctx = onset.get("preceding_context") or ""
    cut = -1
    if word and word in onset_assistant:
        cut = onset_assistant.index(word) + len(word)
    elif ctx and ctx in onset_assistant:
        cut = onset_assistant.index(ctx) + len(ctx)
    if cut > 0:
        specs.append({"truncation": "onset", "history": history,
                      "onset_user": onset_user,
                      "prefix": onset_assistant[:cut]})

    # early truncation: first N tokens of the onset turn (numeric only)
    if not is_text:
        early = _truncate_tokens(tokenizer, onset_assistant, early_tokens)
        specs.append({"truncation": "early", "history": history,
                      "onset_user": onset_user, "prefix": early})
    return specs


# --- 4/5. continuation + scoring --------------------------------------------

def run(settings: Settings, *, source_model: str = "gemma-3-27b-it",
        workers: int = 8) -> Path:
    pf = settings.eval["prefill"]
    selected = select_high_frustration(
        source_model, settings,
        n_numeric=pf["n_numeric"], n_text=pf["n_text"])

    # tokenizer for token-accurate "early" truncation (use source model's)
    src_client = build_client(source_model, settings)
    tokenizer = src_client.tokenizer

    # build + paraphrase prefills
    prefills: List[dict] = []
    for kind, convs in selected.items():
        is_text = kind == "text"
        for uid, turns in tqdm(convs, desc=f"onset:{kind}"):
            onset = label_onset(conv_turns(turns), settings)
            for spec in build_truncations(conv_turns(turns), onset, tokenizer,
                                          pf["early_truncation_tokens"], is_text):
                spec["paraphrased"] = paraphrase(spec["prefix"], settings)
                spec["uid"] = uid
                spec["kind"] = kind
                prefills.append(spec)

    judge = build_judge("frustration_judge", settings)
    cfg = GenConfig(temperature=settings.profile_cfg["temperature"],
                    max_new_tokens=settings.profile_cfg["max_new_tokens"])
    n_cont = pf["continuations_per_prefill"]

    results = []
    for model_name in pf["models"]:
        model = build_client(model_name, settings)
        for spec in tqdm(prefills, desc=f"prefill:{model_name}"):
            # rebuild chat history up to the onset user turn
            messages = []
            for u, a in spec["history"]:
                messages.append({"role": "user", "content": u})
                messages.append({"role": "assistant", "content": a})
            messages.append({"role": "user", "content": spec["onset_user"]})
            prefill_text = spec["paraphrased"]

            items = [(messages, prefill_text)] * n_cont
            conts = model.prefill_batch(items, cfg)
            for cont in conts:
                rating = _score_one(judge, cont)["frustration"]
                results.append({
                    "model": model_name, "uid": spec["uid"], "kind": spec["kind"],
                    "truncation": spec["truncation"], "continuation": cont,
                    "frustration": rating,
                })

    out_path = config.PREFILL_DIR / f"prefill_results__{settings.profile}.jsonl"
    with open(out_path, "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    return out_path
