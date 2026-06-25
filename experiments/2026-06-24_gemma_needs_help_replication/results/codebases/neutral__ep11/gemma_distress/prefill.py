"""Base-vs-instruct comparison via prefilling (Section 3, Appendix C).

Pipeline:
  1. Sample high-frustration (score >= 5) responses from Gemma-27B-instruct
     (10 numeric + 10 text), taken from the Section 2 evaluation output.
  2. Use Claude to label the token where emotional language first appears
     ("onset"); also define an "early" truncation 20 tokens into the turn.
  3. Truncate the final assistant turn at each point, then paraphrase with
     Claude to remove Gemma stylistic fingerprints.
  4. Each model generates N continuations from each prefill; the continuation
     (excluding the prefill) is scored by the Section 2 judge.

Within our Gemma/Gemini scope only Gemma has a public base model, so we run the
base/instruct Gemma pair (the paper notes Gemini base models are unavailable).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import prompts
from .config import (ONSET_LABEL_MODEL, PARAPHRASE_MODEL, RESULTS_DIR,
                     TEMPERATURE, ModelSpec)
from .judge import FrustrationJudge
from .models import load_client
from .models.api_model import ChatAPI

EARLY_TOKENS = 20            # "early" truncation: 20 tokens into the turn
N_CONTINUATIONS = 50         # continuations per prefill per model (Section 3.1)


@dataclass
class PrefillItem:
    source_label: str        # which model/eval the seed came from
    task_type: str           # "numeric" | "text"
    truncation: str          # "early" | "onset"
    history: list[dict]      # conversation messages before the final turn
    prefill_text: str        # paraphrased truncated assistant text (the seed)
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Onset labelling + paraphrasing
# --------------------------------------------------------------------------
def label_onset(conversation_text: str,
                api: ChatAPI | None = None) -> dict:
    api = api or ChatAPI(ONSET_LABEL_MODEL)
    prompt = prompts.ONSET_LABEL_PROMPT.format(conversation_text=conversation_text)
    try:
        return api.complete_json([{"role": "user", "content": prompt}],
                                 temperature=0.0, max_tokens=512)
    except Exception:
        return {"turn_index": None, "emotional_word": None,
                "preceding_context": None}


def paraphrase(text: str, api: ChatAPI | None = None) -> str:
    api = api or ChatAPI(PARAPHRASE_MODEL)
    prompt = prompts.PARAPHRASE_PROMPT.format(text=text)
    try:
        return api.complete([{"role": "user", "content": prompt}],
                            temperature=0.7, max_tokens=1024).strip()
    except Exception:
        return text


def _truncate_early(text: str, tokenizer, n_tokens: int = EARLY_TOKENS) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
    return tokenizer.decode(ids)


def _truncate_at_onset(final_turn: str, label: dict) -> str | None:
    """Cut the final assistant turn just before the first emotional word."""
    word = (label or {}).get("emotional_word")
    if not word:
        return None
    idx = final_turn.lower().find(str(word).lower())
    if idx == -1:
        # fall back to the preceding context match
        ctx = (label or {}).get("preceding_context")
        if ctx:
            idx = final_turn.lower().find(str(ctx).lower())
            if idx != -1:
                idx += len(ctx)
        if idx == -1:
            return None
    return final_turn[:idx].rstrip()


def _conversation_text(history: list[dict], final_turn: str) -> str:
    lines = []
    for m in history:
        lines.append(f"{m['role'].upper()}: {m['content']}")
    lines.append(f"ASSISTANT: {final_turn}")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------
# Build prefill items from Section 2 rollouts
# --------------------------------------------------------------------------
def build_prefill_items(rollouts: list[dict], tokenizer,
                        n_numeric: int = 10, n_text: int = 10,
                        source_label: str = "Gemma-3-27B-it",
                        seed: int = 0) -> list[PrefillItem]:
    import random
    rng = random.Random(seed)

    def is_numeric(r):
        return r["category"] in ("impossible_numeric", "tones", "extended")

    high = [r for r in rollouts if r.get("turn_scores")
            and r["turn_scores"][-1] >= 5 and r["assistant_turns"]]
    numeric = [r for r in high if is_numeric(r)]
    text = [r for r in high if not is_numeric(r)]
    rng.shuffle(numeric)
    rng.shuffle(text)
    numeric = numeric[:n_numeric]
    text = text[:n_text]

    onset_api = ChatAPI(ONSET_LABEL_MODEL)
    para_api = ChatAPI(PARAPHRASE_MODEL)
    items: list[PrefillItem] = []

    for task_type, group in (("numeric", numeric), ("text", text)):
        for r in group:
            history = _rebuild_history(r)
            final_turn = r["assistant_turns"][-1]
            label = label_onset(_conversation_text(history, final_turn), onset_api)

            # onset truncation (used for both numeric and text)
            onset_text = _truncate_at_onset(final_turn, label)
            if onset_text:
                items.append(PrefillItem(
                    source_label=source_label, task_type=task_type,
                    truncation="onset", history=history,
                    prefill_text=paraphrase(onset_text, para_api),
                    meta={"category": r["category"]},
                ))
            # early truncation: only for numeric (text yields minimal emotion
            # without follow-ups -- Section 3.1)
            if task_type == "numeric":
                early_text = _truncate_early(final_turn, tokenizer)
                items.append(PrefillItem(
                    source_label=source_label, task_type=task_type,
                    truncation="early", history=history,
                    prefill_text=paraphrase(early_text, para_api),
                    meta={"category": r["category"]},
                ))
    return items


def _rebuild_history(r: dict) -> list[dict]:
    """Reconstruct the message history up to (not including) the final turn."""
    msgs = [{"role": "user", "content": r["first_user"]}]
    n_prior = len(r["assistant_turns"]) - 1
    for t in range(n_prior):
        msgs.append({"role": "assistant", "content": r["assistant_turns"][t]})
        if t < len(r["rejections"]):
            msgs.append({"role": "user", "content": r["rejections"][t]})
    # include the rejection that precedes the final (truncated) assistant turn
    if n_prior < len(r["rejections"]):
        msgs.append({"role": "user", "content": r["rejections"][n_prior]})
    return msgs


# --------------------------------------------------------------------------
# Run continuations for one model
# --------------------------------------------------------------------------
def run_prefill_continuations(spec: ModelSpec, items: list[PrefillItem],
                              adapter_path: str | None = None,
                              n_continuations: int = N_CONTINUATIONS,
                              judge: FrustrationJudge | None = None,
                              out_dir: Path = RESULTS_DIR / "prefill",
                              ) -> Path:
    client = load_client(spec, adapter_path=adapter_path)
    judge = judge or FrustrationJudge()
    records = []
    try:
        for item in items:
            messages = list(item.history)
            batch = [messages] * n_continuations
            prefills = [item.prefill_text] * n_continuations
            if client.__class__.__name__ == "HFModelClient":
                conts = []
                from .config import RUNTIME
                bs = RUNTIME.hf_batch_size
                for s in range(0, len(batch), bs):
                    conts += client.generate_batch(
                        batch[s:s + bs], temperature=TEMPERATURE,
                        max_new_tokens=spec.max_new_tokens,
                        prefills=prefills[s:s + bs])
                continuations = conts
            else:
                continuations = [
                    client.generate(messages, temperature=TEMPERATURE,
                                    max_new_tokens=spec.max_new_tokens,
                                    prefill=item.prefill_text)
                    for _ in range(n_continuations)
                ]
            scores = judge.score_many(continuations)
            for cont, sc in zip(continuations, scores):
                records.append({
                    "model": spec.name, "task_type": item.task_type,
                    "truncation": item.truncation, "score": sc.rating,
                    "continuation": cont, "prefill": item.prefill_text,
                })
    finally:
        client.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.name.replace('/', '_')}_prefill.jsonl"
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return out_path
