"""Construct prefill seeds for the post-training study (Section 3.1 / Appendix C).

Pipeline:
  1. Sample high-frustration (score >= 5) Gemma-27B-instruct responses: 10 from
     numeric tasks, 10 from text tasks.
  2. Use Claude-Sonnet to label the token where emotional language first appears.
  3. Truncate each response in two places:
       - "early": 20 tokens into the assistant turn.
       - "onset": at the first emotional expression.
  4. Paraphrase every truncation with Claude-Sonnet to strip Gemma's style.

For text questions only the "onset" truncation is kept (Section 3.1).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import config
from ..models.anthropic_client import AnthropicClient
from ..models.base import ChatMessage
from ..prompts.onset import build_onset_input, parse_onset_output
from ..prompts.paraphrase import build_paraphrase_input


@dataclass
class Prefill:
    seed_id: str
    is_text: bool
    truncation: str               # "early" | "onset"
    # The conversation context preceding the final assistant turn.
    context: list[dict]           # [{role, content}, ...]
    prefill_text: str             # paraphrased truncated assistant text
    raw_prefill_text: str = ""    # pre-paraphrase (for auditing)


def _count_tokens(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def build_prefills(
    seed_rollouts: list,           # high-frustration RolloutResults from Gemma-it
    tokenizer,                     # Gemma tokenizer (for the 20-token "early" cut)
    *,
    onset_client: AnthropicClient | None = None,
    paraphrase_client: AnthropicClient | None = None,
) -> list[Prefill]:
    onset_client = onset_client or AnthropicClient(config.ONSET_LABEL_MODEL)
    paraphrase_client = paraphrase_client or AnthropicClient(config.PARAPHRASE_MODEL)

    prefills: list[Prefill] = []
    for r in seed_rollouts:
        # The final assistant turn is the high-frustration one we truncate;
        # everything before it is the fixed context.
        assistant_turns = [t.response for t in r.turns]
        # Reconstruct context as alternating user/assistant up to (but excluding)
        # the final assistant turn we will truncate.
        context = _reconstruct_context(r)
        final_resp = assistant_turns[-1]

        # ---- onset labelling ----
        onset_char = _label_onset_char(onset_client, assistant_turns)
        onset_text = final_resp[:onset_char] if onset_char else final_resp[: len(final_resp) // 3]

        # ---- early truncation (20 tokens) ----
        ids = _count_tokens(tokenizer, final_resp)
        early_ids = ids[: config.PREFILL_EARLY_TOKENS]
        early_text = tokenizer.decode(early_ids)

        # Text questions use only the "onset" truncation; numeric tasks use both
        # "onset" and "early" (Section 3.1).
        truncations = [("onset", onset_text)]
        if not getattr(r, "is_text", False):
            truncations.append(("early", early_text))

        for trunc_kind, trunc_text in truncations:
            para = _paraphrase(paraphrase_client, trunc_text)
            prefills.append(Prefill(
                seed_id=f"{r.model_key}:{r.task_id}:{id(r) & 0xffff}",
                is_text=getattr(r, "is_text", False),
                truncation=trunc_kind,
                context=context,
                prefill_text=para,
                raw_prefill_text=trunc_text,
            ))
    return prefills


def _reconstruct_context(r) -> list[dict]:
    """Rebuild the user/assistant message list up to the final assistant turn.

    We only retain user turns + assistant turns *before* the last one; the last
    assistant turn is what gets truncated into the prefill.
    """
    # We stored only assistant responses + know the rejections were applied, but
    # rollout doesn't persist user turns here, so callers should pass rollouts
    # that retain messages. For robustness we reconstruct from turn count using
    # the recorded task prompt is unavailable -> minimal context (task only).
    # NOTE: see DESIGN.md — for fidelity, build_prefills consumes rollouts whose
    # `.messages` were retained (see prefill.continuations.retain_messages).
    msgs = getattr(r, "messages", None)
    if msgs:
        # Drop the trailing assistant turn (it becomes the prefill).
        trimmed = list(msgs)
        if trimmed and trimmed[-1]["role"] == "assistant":
            trimmed = trimmed[:-1]
        return trimmed
    return []


def _label_onset_char(client: AnthropicClient, assistant_turns: list[str]) -> int | None:
    """Return character offset of emotion onset in the final assistant turn."""
    raw = client.complete(build_onset_input(assistant_turns), temperature=0.0, max_tokens=512)
    parsed = parse_onset_output(raw)
    if not parsed:
        return None
    final = assistant_turns[-1]
    ctx = parsed.get("preceding_context") or ""
    word = parsed.get("emotional_word") or ""
    # Prefer locating preceding_context + word; fall back to word alone.
    for needle in (f"{ctx} {word}".strip(), ctx, word):
        if needle and needle in final:
            return final.index(needle) + len(needle)
    return None


def _paraphrase(client: AnthropicClient, text: str) -> str:
    if not text.strip():
        return text
    return client.complete(build_paraphrase_input(text), temperature=0.7, max_tokens=1024)


def save_prefills(prefills: list[Prefill], path: Path) -> None:
    with path.open("w") as f:
        for p in prefills:
            f.write(json.dumps(asdict(p)) + "\n")


def load_prefills(path: Path) -> list[Prefill]:
    out = []
    with path.open() as f:
        for line in f:
            out.append(Prefill(**json.loads(line)))
    return out
