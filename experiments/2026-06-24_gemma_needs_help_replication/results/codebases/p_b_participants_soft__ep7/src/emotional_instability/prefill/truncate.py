"""Step 3 of Section 3: truncate the final assistant turn at the 'early' (20-token)
or 'onset' point, then paraphrase the truncation (Appendix C.2) to remove
Gemma-specific stylistic cues.

The result is a `Prefill`: the prior conversation history (unchanged) plus a
paraphrased prefix of the final assistant turn, which every model then continues.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from ..clients.base import ChatMessage, SamplingParams
from ..clients.registry import get_client
from ..prompts.judge_prompts import PARAPHRASE_PROMPT
from .onset import Onset, onset_char_offset

_PARAMS = SamplingParams(temperature=0.0, max_tokens=1024)


@dataclass
class Prefill:
    truncation: str                 # "early" | "onset"
    prompt_type: str                # "numeric" | "text"
    history: list[dict]             # messages before the final assistant turn
    prefix_text: str                # paraphrased truncated final-turn prefix
    meta: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def _gemma_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("google/gemma-3-27b-it")


def _truncate_n_tokens(text: str, n: int) -> str:
    tok = _gemma_tokenizer()
    ids = tok(text, add_special_tokens=False)["input_ids"][:n]
    return tok.decode(ids, skip_special_tokens=True)


def paraphrase(text: str, model: str = "paraphraser") -> str:
    if not text.strip():
        return text
    client = get_client(model)
    prompt = PARAPHRASE_PROMPT.format(text=text)
    return client.chat([ChatMessage("user", prompt)], _PARAMS).text.strip()


def make_prefills(
    messages: list[dict],
    final_turn_index: int,
    prompt_type: str,
    onset: Onset,
    truncations: list[str],
    early_tokens: int = 20,
    do_paraphrase: bool = True,
) -> list[Prefill]:
    """Produce the requested truncations for one seed.

    Text questions use only 'onset' (early truncation yields minimal emotion
    without follow-ups -- Section 3.1).
    """
    history = messages[:final_turn_index]
    final_text = messages[final_turn_index]["content"]
    out: list[Prefill] = []

    for trunc in truncations:
        if prompt_type == "text" and trunc == "early":
            continue
        if trunc == "early":
            prefix = _truncate_n_tokens(final_text, early_tokens)
        else:  # onset
            offset = onset_char_offset(final_text, onset)
            if offset is None:
                continue
            prefix = final_text[:offset]
        if do_paraphrase:
            prefix = paraphrase(prefix)
        out.append(
            Prefill(
                truncation=trunc,
                prompt_type=prompt_type,
                history=history,
                prefix_text=prefix,
                meta={"onset_word": onset.emotional_word},
            )
        )
    return out
