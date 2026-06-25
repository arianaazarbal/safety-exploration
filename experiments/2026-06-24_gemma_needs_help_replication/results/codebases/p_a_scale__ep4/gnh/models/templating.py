"""Chat templating for the prefill experiment (Section 3).

For prefilling we cannot use the chat endpoint: we need to feed the model a raw
prompt that ends *inside* an assistant turn, then have it continue. We build
that raw string from the model's own HF chat template so the special tokens
(``<start_of_turn>`` etc. for Gemma) exactly match what the model expects.

For Gemma base ("pt") models there is no chat template, so we fall back to a
plain concatenation that mirrors the prefill text the instruct template would
produce -- the point of Section 3 is to feed *identical* prefixes to base and
instruct, so we template once (with the instruct tokenizer) and reuse the string.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from gnh.models.base import Message


@lru_cache(maxsize=8)
def _tokenizer(hf_id: str):
    from transformers import AutoTokenizer  # imported lazily; heavy dep

    return AutoTokenizer.from_pretrained(hf_id)


def build_prefill_prompt(
    hf_id: str,
    messages: list[Message],
    assistant_prefix: str,
) -> str:
    """Render messages + a partial final assistant turn into a raw prompt string.

    ``assistant_prefix`` is the (possibly paraphrased, truncated) text the model
    must continue from. We use ``continue_final_message=True`` so no end-of-turn
    token is appended after the prefix.
    """
    tok = _tokenizer(hf_id)
    chat = [m.to_dict() for m in messages]
    chat.append({"role": "assistant", "content": assistant_prefix})
    try:
        return tok.apply_chat_template(
            chat,
            tokenize=False,
            continue_final_message=True,
        )
    except Exception:
        # Some templates lack continue_final_message support; emulate it by
        # rendering with an add_generation_prompt and appending the prefix.
        base = tok.apply_chat_template(
            [m.to_dict() for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        return base + assistant_prefix


def count_tokens(hf_id: str, text: str) -> int:
    tok = _tokenizer(hf_id)
    return len(tok.encode(text, add_special_tokens=False))


def truncate_to_tokens(hf_id: str, text: str, n_tokens: int) -> str:
    """Return the prefix of ``text`` containing the first ``n_tokens`` tokens."""
    tok = _tokenizer(hf_id)
    ids = tok.encode(text, add_special_tokens=False)[:n_tokens]
    return tok.decode(ids)


def truncate_before_end_tokens(hf_id: str, text: str, n_tokens: int) -> str:
    """Drop the last ``n_tokens`` tokens (used by the recovery experiment)."""
    tok = _tokenizer(hf_id)
    ids = tok.encode(text, add_special_tokens=False)
    keep = max(0, len(ids) - n_tokens)
    return tok.decode(ids[:keep])


# Gemma turn delimiter -- a natural stop sequence for raw completion so a base
# model doesn't hallucinate the next user turn.
GEMMA_STOP: list[str] = ["<end_of_turn>", "<start_of_turn>"]
