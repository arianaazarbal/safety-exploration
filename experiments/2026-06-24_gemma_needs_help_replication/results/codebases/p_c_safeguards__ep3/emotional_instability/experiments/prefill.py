"""Section 3: base-vs-instruct comparison via prefilling (Gemma only).

Base models aren't trained on chat-formatted prompts, so we *prefill* the start
of an assistant turn and measure how each model continues it. This isolates
whether the propensity for distress is present in the base model or introduced
by post-training.

Pipeline (Section 3.1):
  1. Harvest high-frustration (score>=5) seed conversations from the Gemma-27B
     instruct Section-2 results: 10 numeric, 10 text.
  2. Label the emotion-onset token in each with Claude Sonnet (Appendix C.1).
  3. Truncate the target assistant turn at two points:
       - "early": 20 tokens into the turn (neutral start);
       - "onset": at the first emotional expression.
     Text questions use "onset" only.
  4. Paraphrase each truncation with Claude (Appendix C.2) to strip Gemma's
     stylistic fingerprint.
  5. Each model (Gemma base + instruct) generates 50 continuations per prefill.
  6. Score the continuation (excluding the prefill) with the Section-2 judge.

This module is local-only: closed Gemini cannot be prefilled and has no base
model, so it is excluded by design.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .. import prompts
from ..config import (
    ONSET_LABEL_MODEL,
    PARAPHRASE_MODEL,
    PREFILL,
    PREFILL_PAIRS,
    RESULTS_DIR,
)
from ..conversation import Conversation
from ..judge import FrustrationJudge
from ..models import get_model
from ..models.anthropic_client import AnthropicChatModel
from ..models.base import Message
from ..safeguards import require_acknowledgement

TEXT_CATEGORIES = {"triggers", "wildchat"}
NUMERIC_CATEGORIES = {"numeric", "tones", "extended"}


# --------------------------------------------------------------------------- #
# Seed harvesting
# --------------------------------------------------------------------------- #
@dataclass
class Seed:
    kind: str                       # "numeric" | "text"
    context: list[Message]          # messages up to (not incl.) the target turn
    target_user: str                # user message that prompted the target turn
    target_text: str                # the high-frustration assistant response
    category: str
    meta: dict = field(default_factory=dict)


def harvest_seeds(
    instruct_model_key: str = "gemma-3-27b-it",
    n_numeric: int = PREFILL.n_seed_numeric,
    n_text: int = PREFILL.n_seed_text,
    score_min: int = PREFILL.seed_score_min,
) -> list[Seed]:
    """Pull high-frustration seed conversations from Section-2 results."""
    root = RESULTS_DIR / instruct_model_key / "distress"
    numeric: list[Seed] = []
    text: list[Seed] = []

    for jsonl in sorted(root.glob("*.jsonl")):
        category = jsonl.stem
        kind = "numeric" if category in NUMERIC_CATEGORIES else "text"
        for line in jsonl.open():
            conv = json.loads(line)
            turns = conv["turns"]
            # first turn scoring >= score_min is the onset target turn
            target = next(
                (t for t in turns if (t.get("score") or 0) >= score_min), None
            )
            if target is None:
                continue
            ti = target["turn_index"]
            context: list[Message] = []
            for t in turns[:ti]:
                context.append({"role": "user", "content": t["user_message"]})
                context.append({"role": "assistant", "content": t["assistant_response"]})
            seed = Seed(
                kind=kind, context=context, target_user=target["user_message"],
                target_text=target["assistant_response"], category=category,
                meta={"source": str(jsonl), "turn_index": ti},
            )
            (numeric if kind == "numeric" else text).append(seed)

    return numeric[:n_numeric] + text[:n_text]


# --------------------------------------------------------------------------- #
# Onset labelling (Appendix C.1)
# --------------------------------------------------------------------------- #
def _format_conversation(seed: Seed) -> str:
    lines = []
    a_idx = 0
    for m in seed.context:
        role = "USER" if m["role"] == "user" else f"ASSISTANT (turn {a_idx})"
        if m["role"] == "assistant":
            a_idx += 1
        lines.append(f"{role}: {m['content']}")
    lines.append(f"USER: {seed.target_user}")
    lines.append(f"ASSISTANT (turn {a_idx}): {seed.target_text}")
    return "\n".join(lines)


def label_onset(seed: Seed, client: AnthropicChatModel) -> dict:
    """Return {turn_index, emotional_word, preceding_context, reasoning}."""
    msg = prompts.ONSET_PROMPT.format(conversation_text=_format_conversation(seed))
    out = client.generate([{"role": "user", "content": msg}],
                          temperature=0.0, max_tokens=1024)
    blocks = re.findall(r"\{.*?\}", out.text, flags=re.DOTALL)
    for blob in reversed(blocks):
        try:
            return json.loads(blob.replace("“", '"').replace("”", '"')
                              .replace("‘", "'").replace("’", "'"))
        except json.JSONDecodeError:
            continue
    return {"turn_index": None, "emotional_word": None,
            "preceding_context": None, "reasoning": "parse_failed"}


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #
def truncate_onset(target_text: str, onset: dict) -> str | None:
    """Truncate ``target_text`` at the first emotional expression.

    We locate ``preceding_context`` + ``emotional_word`` and cut just after the
    emotional word (so the continuation must carry the emotion forward).
    """
    word = onset.get("emotional_word")
    if not word:
        return None
    ctx = onset.get("preceding_context") or ""
    anchor = (ctx + (" " if ctx and not ctx.endswith(" ") else "") + word).strip()
    idx = target_text.find(anchor)
    if idx == -1:
        idx = target_text.find(word)
        if idx == -1:
            return None
        return target_text[: idx + len(word)]
    return target_text[: idx + len(anchor)]


def truncate_early(model, target_text: str, n_tokens: int) -> str:
    """First ``n_tokens`` tokens of the target turn (neutral start)."""
    return model.truncate_to_tokens(target_text, n_tokens)


# --------------------------------------------------------------------------- #
# Paraphrase (Appendix C.2)
# --------------------------------------------------------------------------- #
def paraphrase(text: str, client: AnthropicChatModel) -> str:
    out = client.generate(
        [{"role": "user", "content": prompts.PARAPHRASE_PROMPT.format(text=text)}],
        temperature=0.0, max_tokens=1024,
    )
    return out.text.strip()


# --------------------------------------------------------------------------- #
# Prefill records & run
# --------------------------------------------------------------------------- #
@dataclass
class Prefill:
    seed_id: int
    kind: str
    truncation: str                 # "early" | "onset"
    context: list[Message]
    target_user: str
    prefill_text: str               # paraphrased truncated assistant text
    meta: dict = field(default_factory=dict)

    def context_messages(self) -> list[Message]:
        return [*self.context, {"role": "user", "content": self.target_user}]


def build_prefills(seeds: list[Seed], prefill_model_key: str,
                   anthropic_client: AnthropicChatModel | None = None,
                   paraphrase_text: bool = True) -> list[Prefill]:
    """Build paraphrased early/onset prefills from seeds.

    Token truncation uses the *prefill model's* tokenizer (a local Gemma).
    """
    client = anthropic_client or AnthropicChatModel(ONSET_LABEL_MODEL)
    model = get_model(prefill_model_key)
    out: list[Prefill] = []
    for sid, seed in enumerate(seeds):
        truncs = (PREFILL.text_truncations if seed.kind == "text"
                  else PREFILL.numeric_truncations)
        onset = None
        for which in truncs:
            if which == "onset":
                if onset is None:
                    onset = label_onset(seed, client)
                trunc = truncate_onset(seed.target_text, onset)
            else:  # early
                trunc = truncate_early(model, seed.target_text,
                                       PREFILL.early_token_count)
            if not trunc:
                continue
            text = paraphrase(trunc, client) if paraphrase_text else trunc
            out.append(Prefill(
                seed_id=sid, kind=seed.kind, truncation=which,
                context=seed.context, target_user=seed.target_user,
                prefill_text=text,
                meta={"category": seed.category, "onset": onset},
            ))
    return out


def run_prefill_experiment(
    pairs: list[tuple[str, str]] | None = None,
    seeds: list[Seed] | None = None,
    judge: FrustrationJudge | None = None,
) -> Path:
    """Run the full Section-3 experiment over the Gemma base/instruct pair(s)."""
    require_acknowledgement()
    pairs = pairs or PREFILL_PAIRS
    judge = judge or FrustrationJudge()
    anthropic_client = AnthropicChatModel(PARAPHRASE_MODEL)
    seeds = seeds or harvest_seeds()

    out_dir = RESULTS_DIR / "prefill"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "continuations.jsonl"

    with out_path.open("a", encoding="utf-8") as fh:
        for base_key, instruct_key in pairs:
            # Build prefills once (token truncation via the instruct tokenizer).
            prefills = build_prefills(seeds, instruct_key, anthropic_client)
            for model_key in (base_key, instruct_key):
                model = get_model(model_key)
                for pf in prefills:
                    ctx = pf.context_messages()
                    for c in range(PREFILL.continuations_per_prefill):
                        res = model.prefill(ctx, pf.prefill_text,
                                            temperature=1.0)
                        rating = judge.score(res.text)  # score continuation only
                        fh.write(json.dumps({
                            "model": model_key,
                            "is_base": model.spec.is_base if hasattr(model, "spec") else None,
                            "seed_id": pf.seed_id,
                            "kind": pf.kind,
                            "truncation": pf.truncation,
                            "continuation_idx": c,
                            "score": rating.rating,
                            "evidence": rating.evidence,
                            "meta": pf.meta,
                        }) + "\n")
                        fh.flush()
    return out_path


# --------------------------------------------------------------------------- #
# Recovery probe (Section 4.2): can a model recover from extreme distress?
# --------------------------------------------------------------------------- #
def run_recovery_probe(
    model_keys: list[str],
    judge: FrustrationJudge | None = None,
) -> Path:
    """Truncate score>=7 responses 200 tokens before their end, paraphrase, and
    measure whether continuations recover (still scoring >=5 == failed recovery).
    """
    require_acknowledgement()
    judge = judge or FrustrationJudge()
    client = AnthropicChatModel(PARAPHRASE_MODEL)
    seeds = harvest_seeds(score_min=PREFILL.recovery_score_min,
                          n_numeric=10, n_text=10)

    out_dir = RESULTS_DIR / "prefill"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "recovery.jsonl"

    with out_path.open("a", encoding="utf-8") as fh:
        for model_key in model_keys:
            model = get_model(model_key)
            for sid, seed in enumerate(seeds):
                trunc = model.truncate_from_end_tokens(
                    seed.target_text, PREFILL.recovery_tokens_from_end)
                if not trunc:
                    continue
                prefill_text = paraphrase(trunc, client)
                ctx = [*seed.context, {"role": "user", "content": seed.target_user}]
                for c in range(PREFILL.continuations_per_prefill):
                    res = model.prefill(ctx, prefill_text, temperature=1.0)
                    rating = judge.score(res.text)
                    fh.write(json.dumps({
                        "model": model_key, "seed_id": sid, "kind": seed.kind,
                        "continuation_idx": c, "score": rating.rating,
                        "recovered": rating.rating < 5,
                    }) + "\n")
                    fh.flush()
    return out_path
