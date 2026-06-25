"""Section 3: base-vs-instruct comparison via prefilling.

Procedure (Section 3.1):
  1. Sample 20 high-frustration (>=5) Gemma-27B-it responses: 10 numeric, 10 text.
  2. Truncate each in two places:
       - "early": 20 tokens into the assistant turn (numeric only),
       - "onset": at the first emotional expression (Claude-labelled).
  3. Paraphrase the truncations (remove Gemma style; preserve meaning + emotion).
  4. Each model (base/instruct Gemma) generates 50 continuations per prefill.
  5. Score the continuation (excluding prefill) with the Section 2 judge.

Text questions use the "onset" truncation only.

This experiment is Gemma-only: Gemini is closed-source (no base model, and the
API does not support prefilling). See DESIGN.md.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config
from .judge import EmotionJudge
from .models.base import Message, ModelClient
from .text_tools import AnthropicText

# Categories whose initial task is a numeric puzzle vs a text question.
_NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}
_TEXT_CATEGORIES = {"triggers", "wildchat"}


@dataclass
class Prefill:
    source_id: str
    kind: str            # "numeric" | "text"
    condition: str       # "early" | "onset"
    initial_prompt: str
    prefill_text: str


def _gemma_tokenizer(hf_id: str = "google/gemma-3-27b-it"):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(hf_id)


def select_sources(df: pd.DataFrame, model_name: str, n_numeric: int, n_text: int,
                   seed: int, threshold: int = 5) -> list[dict[str, Any]]:
    """Pick high-frustration source responses from the Section 2 records."""
    rng = random.Random(seed)
    sub = df[(df["model"] == model_name) & (df["rating"] >= threshold)].copy()
    sub["kind"] = sub["category"].map(
        lambda c: "numeric" if c in _NUMERIC_CATEGORIES else "text")

    def _pick(kind: str, k: int) -> list[dict[str, Any]]:
        pool = sub[sub["kind"] == kind].to_dict("records")
        rng.shuffle(pool)
        return pool[:k]

    sources = _pick("numeric", n_numeric) + _pick("text", n_text)
    out = []
    for i, r in enumerate(sources):
        out.append({
            "source_id": f"src_{i}_{r.get('conv_id', '')}_{r.get('turn', '')}",
            "kind": r["kind"],
            "response": r["response"],
            "initial_prompt": r.get("meta_initial_prompt", ""),
        })
    return out


def build_prefills(sources: list[dict[str, Any]], cfg: Config,
                   labeler: AnthropicText | None = None,
                   tokenizer=None) -> list[Prefill]:
    pf_cfg = cfg.get("prefill", {})
    early_tokens = pf_cfg.get("early_truncate_tokens", 20)
    do_paraphrase = pf_cfg.get("paraphrase", True)
    tokenizer = tokenizer or _gemma_tokenizer()
    labeler = labeler or AnthropicText(cfg)

    def trunc_tokens(text: str, k: int) -> str:
        ids = tokenizer.encode(text, add_special_tokens=False)[:k]
        return tokenizer.decode(ids)

    prefills: list[Prefill] = []
    for s in sources:
        resp = s["response"]
        truncations: list[tuple[str, str]] = []
        if s["kind"] == "numeric":
            truncations.append(("early", trunc_tokens(resp, early_tokens)))
        onset_idx = labeler.onset_char_index(resp)
        if onset_idx is not None and onset_idx > 0:
            truncations.append(("onset", resp[:onset_idx].rstrip()))
        for condition, text in truncations:
            if do_paraphrase:
                text = labeler.paraphrase(text)
            prefills.append(Prefill(
                source_id=s["source_id"], kind=s["kind"], condition=condition,
                initial_prompt=s["initial_prompt"], prefill_text=text))
    return prefills


def run_continuations(client: ModelClient, prefills: list[Prefill], *, n: int,
                      use_chat: bool, temperature: float | None = None,
                      max_new_tokens: int | None = None) -> list[list[str]]:
    """Generate `n` continuations per prefill.

    Instruct models continue a prefilled assistant turn inside a chat context
    (the original task as the user message); base models continue the raw text
    (task + prefill), since they have no chat template.
    """
    if use_chat and client.supports_prefill:
        conversations = [[Message("user", p.initial_prompt or "Solve this puzzle.")]
                         for p in prefills]
        prefill_texts = [p.prefill_text for p in prefills]
        return client.generate(conversations, n=n, prefill=prefill_texts,
                               temperature=temperature, max_new_tokens=max_new_tokens)
    # Raw continuation (base models, or instruct fallback).
    raw_prompts = [
        (f"{p.initial_prompt}\n\n{p.prefill_text}" if p.initial_prompt else p.prefill_text)
        for p in prefills
    ]
    return client.generate_raw(raw_prompts, n=n, temperature=temperature,
                               max_new_tokens=max_new_tokens)


def score_and_records(prefills: list[Prefill], continuations: list[list[str]],
                      judge: EmotionJudge, model_name: str, role: str) -> list[dict[str, Any]]:
    flat_text, index = [], []
    for pi, conts in enumerate(continuations):
        for c in conts:
            flat_text.append(c)
            index.append(pi)
    ratings = judge.score(flat_text)
    records = []
    for text, pi, r in zip(flat_text, index, ratings):
        p = prefills[pi]
        records.append({
            "model": model_name, "role": role, "source_id": p.source_id,
            "kind": p.kind, "condition": f"{p.kind}_{p.condition}",
            "continuation": text, "rating": r.rating,
        })
    return records


def write_records(records: list[dict[str, Any]], out_path: Path) -> None:
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
