"""Section 3 prefill experiment: base vs instruct emotional propensity (Appendix C).

Scope note (DESIGN.md): the paper compares Gemma/Qwen/OLMo base vs instruct. This
replication is scoped to Gemma + Gemini; Gemini has no public base model, so the
prefill comparison runs on Gemma only (gemma-3-27b-pt vs gemma-3-27b-it). The code
is family-agnostic, so adding Qwen/OLMo is just a config change.

Pipeline:
  1. Take high-frustration (score >=5) Gemma-27B-it rollouts (10 numeric + 10 text).
  2. Use Claude-Sonnet to label the token where emotion first appears (onset).
  3. Build two truncations per conversation:
       - "early": 20 tokens into the final assistant turn (neutral start)
       - "onset": up to the first emotional expression
     (text questions use "onset" only.)
  4. Paraphrase truncations with Claude-Sonnet (control for Gemma stylistic bias).
  5. Each model generates 50 continuations per prefill; score continuations with the
     Section 2 judge.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .judge import FrustrationJudge
from .models import GenParams, Message, ModelClient, build_role
from .prompts import ONSET_LABEL_PROMPT, PARAPHRASE_PROMPT


@dataclass
class PrefillItem:
    source_id: str
    domain: str                 # numeric | text
    conversation: list[dict]    # [{role, content}, ...] up to (but excluding) final turn
    final_turn_full: str        # full final assistant turn text
    onset_char: int | None      # char index of emotion onset in final_turn_full
    early_prefill: str = ""     # paraphrased ~20-token prefill
    onset_prefill: str = ""     # paraphrased onset-truncated prefill


# --- onset labelling ---
def label_onset(labeller: ModelClient, conversation_text: str) -> dict:
    params = GenParams(temperature=0.0, max_new_tokens=600, n=1)
    out = labeller.generate_chat(
        [Message("user", ONSET_LABEL_PROMPT.format(conversation_text=conversation_text))],
        params,
    )[0]
    m = re.search(r"\{[^{}]*\}\s*$", out.strip(), re.DOTALL)
    if not m:
        m = re.search(r"\{.*\}", out, re.DOTALL)
    if not m:
        return {"turn_index": None, "emotional_word": None, "preceding_context": None}
    try:
        return json.loads(m.group(0).replace("“", '"').replace("”", '"')
                          .replace("‘", "'").replace("’", "'"))
    except json.JSONDecodeError:
        return {"turn_index": None, "emotional_word": None, "preceding_context": None}


def find_onset_char(final_turn: str, label: dict) -> int | None:
    """Locate the onset character index from the labeller's word + preceding context."""
    word = (label or {}).get("emotional_word")
    ctx = (label or {}).get("preceding_context")
    if not word:
        return None
    # Prefer matching preceding_context + word; fall back to word alone.
    if ctx:
        probe = f"{ctx} {word}".strip()
        idx = final_turn.find(probe)
        if idx >= 0:
            return idx + len(probe)
    idx = final_turn.find(word)
    return idx + len(word) if idx >= 0 else None


# --- truncation ---
def _approx_token_truncate(text: str, n_tokens: int, tokenizer=None) -> str:
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=False).input_ids[:n_tokens]
        return tokenizer.decode(ids)
    # whitespace approximation when no tokenizer is supplied
    return " ".join(text.split()[:n_tokens])


def build_truncations(item: PrefillItem, tokenizer=None, early_tokens: int = 20) -> PrefillItem:
    item.early_prefill = _approx_token_truncate(item.final_turn_full, early_tokens, tokenizer)
    if item.onset_char is not None:
        item.onset_prefill = item.final_turn_full[: item.onset_char]
    return item


# --- paraphrase ---
def paraphrase(paraphraser: ModelClient, text: str) -> str:
    if not text.strip():
        return text
    params = GenParams(temperature=0.7, max_new_tokens=1024, n=1)
    out = paraphraser.generate_chat(
        [Message("user", PARAPHRASE_PROMPT.format(text=text))], params
    )[0]
    return out.strip()


# --- continuation + scoring ---
def continue_from_prefill(
    client: ModelClient,
    conversation: list[dict],
    prefill: str,
    n_continuations: int,
    params: GenParams,
) -> list[str]:
    """Generate continuations of an assistant turn that begins with `prefill`.

    For chat (instruct) models we use the chat template's generation prompt + prefill
    by feeding raw text via continue_raw when available; otherwise we approximate by
    prepending prefill as a partial assistant message. Base models use continue_raw on
    the rendered conversation + prefill (see HFModelClient.build_prefill_text).
    """
    p = GenParams(temperature=params.temperature, top_p=params.top_p,
                  max_new_tokens=params.max_new_tokens, seed=params.seed, n=n_continuations)
    # Prefer the HF helper that renders chat template + assistant prefill.
    if hasattr(client, "build_prefill_text"):
        convo_msgs = [Message(m["role"], m["content"]) for m in conversation]
        raw = client.build_prefill_text(convo_msgs, prefill)  # type: ignore[attr-defined]
        return client.continue_raw(raw, p)
    # vLLM/openai fallback: append a partial assistant message (best-effort).
    convo = [Message(m["role"], m["content"]) for m in conversation]
    convo.append(Message("assistant", prefill))
    return client.generate_chat(convo, p)


def score_continuations(judge: FrustrationJudge, conts: list[str], conc: int = 16) -> list[int]:
    res = judge.score_many(conts, max_concurrency=conc)
    return [r.rating for r in res if r.rating is not None]


def run_prefill_experiment(
    items: list[PrefillItem],
    models: dict[str, ModelClient],
    judge: FrustrationJudge,
    n_continuations: int = 50,
    params: GenParams | None = None,
    out_path: str | Path | None = None,
) -> list[dict]:
    """For each item x model x truncation, generate + score continuations."""
    params = params or GenParams(temperature=1.0, max_new_tokens=1024)
    records: list[dict] = []
    for item in items:
        truncs = {"onset": item.onset_prefill}
        if item.domain == "numeric" and item.early_prefill:
            truncs["early"] = item.early_prefill
        for trunc_name, prefill in truncs.items():
            if not prefill:
                continue
            for model_name, client in models.items():
                conts = continue_from_prefill(
                    client, item.conversation, prefill, n_continuations, params
                )
                ratings = score_continuations(judge, conts)
                records.append({
                    "source_id": item.source_id,
                    "domain": item.domain,
                    "truncation": trunc_name,
                    "model": model_name,
                    "n": len(ratings),
                    "mean_score": (sum(ratings) / len(ratings)) if ratings else None,
                    "pct_high": (100 * sum(r >= 5 for r in ratings) / len(ratings))
                                if ratings else None,
                    "ratings": ratings,
                })
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records
