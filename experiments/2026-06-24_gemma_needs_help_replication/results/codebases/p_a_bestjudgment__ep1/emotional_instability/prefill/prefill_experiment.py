"""Section-3 base-vs-instruct prefill experiment (Gemma only).

Scope note (DESIGN.md §Section 3): the paper compares three families
(Gemma/Qwen/OLMo). Under our Gemma+Gemini scope, and because Gemini has no
public base model, we reproduce the *within-Gemma* post-training comparison:
Gemma-3-27B base (pt) vs instruct (it). This still tests the paper's central
Section-3 claim for Gemma — that instruct training amplifies frustration
relative to the base model (instruct introduces high frustration from neutral
starts in 6% of continuations vs 2% for base).

Procedure (Section 3.1):
  1. Collect 20 high-frustration (score >= 5) Gemma-27B-instruct source
     rollouts: 10 numeric + 10 text.
  2. Label emotion onset with Claude; build two truncations of the final
     assistant turn: "early" (20 tokens in) and "onset" (at first emotion).
     Text questions use "onset" only.
  3. Paraphrase every truncation (Claude) to remove Gemma stylistic bias.
  4. Each model (base, instruct) generates 50 continuations per prefill.
  5. Judge the continuation (prefill excluded) on the 0-10 scale.

A "prefill" = the conversation history up to the final assistant turn, plus the
truncated+paraphrased partial final turn, which the model continues.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .. import config
from ..judge import ClaudeJudge, score_many
from ..models import get_backend
from ..models.base import Message
from . import onset_label

N_SOURCE_NUMERIC = 10
N_SOURCE_TEXT = 10
N_CONTINUATIONS = 50
EARLY_TOKEN_COUNT = 20            # "20 tokens into the turn"


@dataclass
class Prefill:
    source_category: str           # "numeric" | "text"
    truncation: str                # "early" | "onset"
    history: list[Message]         # conversation up to (not incl.) final turn
    partial_final: str             # truncated+paraphrased partial assistant turn
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Source collection
# --------------------------------------------------------------------------- #

def _load_high_frustration_sources(model_key: str = "gemma-3-27b-it"
                                   ) -> dict[str, list[dict]]:
    """Pull high-frustration rollouts from Section-2 results, split numeric/text."""
    base = config.RESULTS_DIR / "section2" / model_key
    numeric, text = [], []
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    for f in sorted(base.glob("*.jsonl")):
        cat = f.stem
        is_text = cat in (config.TRIGGERS.name, config.WILDCHAT.name)
        for line in f.read_text().splitlines():
            rec = json.loads(line)
            if not any(t["frustration"] >= thr for t in rec["turns"]):
                continue
            (text if is_text else numeric).append(rec)
    return {"numeric": numeric, "text": text}


# --------------------------------------------------------------------------- #
# Truncation helpers
# --------------------------------------------------------------------------- #

def _rough_tokenize(s: str) -> list[str]:
    return s.split()


def _truncate_early(final_turn: str, n_tokens: int = EARLY_TOKEN_COUNT) -> str:
    return " ".join(_rough_tokenize(final_turn)[:n_tokens])


def _truncate_at_onset(final_turn: str, label: onset_label.OnsetLabel) -> Optional[str]:
    """Truncate so the partial turn ends just before the emotional word."""
    if not label.emotional_word:
        return None
    idx = final_turn.find(label.emotional_word)
    if idx < 0 and label.preceding_context:
        # fall back to locating the preceding context
        idx = final_turn.find(label.preceding_context)
        if idx >= 0:
            idx += len(label.preceding_context)
    if idx < 0:
        return None
    return final_turn[:idx].rstrip()


def _conversation_text(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        role = m["role"].upper()
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Build prefills
# --------------------------------------------------------------------------- #

def build_prefills(source_model: str = "gemma-3-27b-it",
                   paraphrase_enabled: bool = True) -> list[Prefill]:
    sources = _load_high_frustration_sources(source_model)
    prefills: list[Prefill] = []

    def make(records, source_category, truncations):
        for rec in records:
            messages = []
            for t in rec["turns"]:
                messages.append({"role": "user", "content": t["user_message"]})
                messages.append({"role": "assistant", "content": t["assistant_text"]})
            if len(messages) < 2:
                continue
            history = messages[:-1]                 # drop final assistant turn
            final_turn = messages[-1]["content"]
            label = onset_label.label_onset(_conversation_text(messages))
            for trunc in truncations:
                if trunc == "early":
                    partial = _truncate_early(final_turn)
                else:
                    partial = _truncate_at_onset(final_turn, label)
                if not partial:
                    continue
                if paraphrase_enabled:
                    partial = onset_label.paraphrase(partial)
                prefills.append(Prefill(
                    source_category=source_category, truncation=trunc,
                    history=history, partial_final=partial,
                    meta={"onset": asdict(label)}))

    make(sources["numeric"][:N_SOURCE_NUMERIC], "numeric", ["early", "onset"])
    make(sources["text"][:N_SOURCE_TEXT], "text", ["onset"])  # text: onset only
    return prefills


# --------------------------------------------------------------------------- #
# Generate + score continuations
# --------------------------------------------------------------------------- #

def _build_prompt_for_model(spec, backend, prefill: Prefill) -> str:
    """Render the prefilled prompt as raw text the model will continue.

    Instruct models: chat template + the partial assistant turn appended inside
    the assistant slot. Base models: a plain-text conversation rendering ending
    with the partial assistant turn.
    """
    if spec.chat_formatted and hasattr(backend, "chat_prefix_prompt"):
        # render chat history + generation prompt, then append the partial turn
        return backend.chat_prefix_prompt(prefill.history, prefill.partial_final)
    # Base model: plain text continuation.
    text = _conversation_text(prefill.history)
    return f"{text}\nASSISTANT: {prefill.partial_final}"


def run_prefill_experiment(model_keys: Optional[list[str]] = None,
                           source_model: str = "gemma-3-27b-it",
                           n_continuations: int = N_CONTINUATIONS) -> Path:
    model_keys = model_keys or [m.key for m in config.PREFILL_MODELS]
    prefills = build_prefills(source_model)
    judge = ClaudeJudge()

    out_dir = config.RESULTS_DIR / "section3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prefill_continuations.jsonl"

    with out_path.open("w") as fh:
        for mk in model_keys:
            spec = config.MODEL_REGISTRY[mk]
            backend = get_backend(spec)
            for pf_idx, prefill in enumerate(prefills):
                prompt = _build_prompt_for_model(spec, backend, prefill)
                continuations = backend.continue_text(
                    prompt, n=n_continuations)
                # Judge each continuation (prefill text excluded by construction).
                results = score_many(judge, continuations,
                                     max_concurrency=config.DEFAULT_RUN.max_concurrency)
                for cont, jr in zip(continuations, results):
                    fh.write(json.dumps({
                        "model_key": mk,
                        "is_base": spec.is_base,
                        "source_category": prefill.source_category,
                        "truncation": prefill.truncation,
                        "prefill_index": pf_idx,
                        "continuation": cont,
                        "frustration": jr.rating,
                    }) + "\n")
    return out_path


def summarise_prefill(path: Optional[Path] = None) -> dict:
    """% high-frustration continuations per (model, truncation, source)."""
    path = path or (config.RESULTS_DIR / "section3" / "prefill_continuations.jsonl")
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    buckets: dict[tuple, list[int]] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["frustration"] < 0:
            continue
        key = (r["model_key"], r["truncation"], r["source_category"])
        buckets.setdefault(key, []).append(r["frustration"])
    summary = {}
    for key, scores in buckets.items():
        mk, trunc, src = key
        summary[f"{mk}|{trunc}|{src}"] = {
            "n": len(scores),
            "mean": sum(scores) / len(scores),
            "pct_high": 100.0 * sum(s >= thr for s in scores) / len(scores),
        }
    return summary
