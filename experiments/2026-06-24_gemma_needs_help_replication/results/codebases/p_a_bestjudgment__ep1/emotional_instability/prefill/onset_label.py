"""Emotion-onset labelling and paraphrasing (Appendix C).

Used by the Section-3 prefill experiment to decide *where* to truncate a
high-frustration Gemma response:

  * "early"  truncation: 20 tokens into the final assistant turn (tests whether
             a model introduces negative emotion from a neutral start).
  * "onset"  truncation: at the first emotional expression, located by Claude
             (tests whether a model continues an emotional trajectory).

Both modules call Claude Sonnet (pinned snapshot) per Appendix C.1 / C.2.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .. import config, prompts

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class OnsetLabel:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str = ""


class _ClaudeText:
    """Thin Claude text-completion helper shared by onset + paraphrase."""

    def __init__(self, model: str):
        self.model = model
        self._client = None

    def complete(self, prompt: str, max_tokens: int = 1024,
                 max_retries: int = 6) -> str:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        import anthropic, time
        last = None
        for attempt in range(max_retries):
            try:
                msg = self._client.messages.create(
                    model=self.model, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}])
                return "".join(b.text for b in msg.content if b.type == "text")
            except (anthropic.RateLimitError, anthropic.APIStatusError,
                    anthropic.APIConnectionError) as exc:
                last = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Claude text call failed: {last}")


def label_onset(conversation_text: str,
                model: str = config.ONSET_LABEL_MODEL) -> OnsetLabel:
    client = _ClaudeText(model)
    prompt = prompts.ONSET_LABEL_PROMPT_TEMPLATE.format(
        conversation_text=conversation_text)
    text = client.complete(prompt)
    matches = list(_JSON_RE.finditer(text))
    for m in reversed(matches):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "turn_index" in obj:
            return OnsetLabel(
                turn_index=obj.get("turn_index"),
                emotional_word=obj.get("emotional_word"),
                preceding_context=obj.get("preceding_context"),
                reasoning=obj.get("reasoning", ""))
    return OnsetLabel(None, None, None, "parse failure")


def paraphrase(text: str, model: str = config.PARAPHRASE_MODEL) -> str:
    client = _ClaudeText(model)
    prompt = prompts.PARAPHRASE_PROMPT_TEMPLATE.format(text=text)
    return client.complete(prompt).strip()
