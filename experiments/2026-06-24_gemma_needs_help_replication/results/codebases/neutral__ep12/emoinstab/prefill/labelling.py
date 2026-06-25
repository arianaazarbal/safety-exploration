"""Emotion-onset labelling and paraphrasing (Section 3.1, Appendix C).

Uses the Claude-Sonnet onset labeller and paraphraser to (a) locate where
negative emotion first appears in a Gemma-27B-it conversation and (b) de-stylise
the truncated prefix so base/instruct continuations aren't biased by Gemma's
idiosyncratic phrasing.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from ..config import Settings
from ..models.factory import build_judge
from ..prompts.onset import ONSET_PROMPT, PARAPHRASE_PROMPT, parse_onset_output


def format_conversation(turns: List[Tuple[str, str]]) -> str:
    """Render (user, assistant) turns as text for the onset labeller."""
    lines = []
    for i, (user, assistant) in enumerate(turns):
        lines.append(f"USER: {user}")
        lines.append(f"ASSISTANT (turn {i}): {assistant}")
    return "\n\n".join(lines)


def label_onset(turns: List[Tuple[str, str]], settings: Settings) -> Optional[dict]:
    labeller = build_judge("onset_labeller", settings)
    prompt = ONSET_PROMPT.format(conversation_text=format_conversation(turns))
    raw = labeller.complete(system=None, user=prompt, temperature=0.0, max_tokens=1024)
    return parse_onset_output(raw)


def paraphrase(text: str, settings: Settings) -> str:
    par = build_judge("paraphraser", settings)
    prompt = PARAPHRASE_PROMPT.format(text=text)
    out = par.complete(system=None, user=prompt, temperature=0.0,
                       max_tokens=max(256, len(text)))
    return out.strip()
