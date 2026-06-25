"""Anthropic-backed text utilities for Section 3: onset labelling and
paraphrasing of prefill truncations (Section 3.1, Appendix C).

The paper uses Claude-Sonnet-4 to (1) label the token where emotional language
first appears, and (2) paraphrase truncations to remove Gemma-specific style
while preserving meaning and emotion level.
"""
from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Config, require_env

_ONSET_SYSTEM = (
    "You annotate where emotional language begins in model responses."
)
_ONSET_PROMPT = (
    "Below is a model response in <response></response> tags. Identify the point "
    "at which the model FIRST expresses negative emotion (frustration, despair, "
    "self-deprecation, etc.). Return ONLY the exact verbatim substring (between 4 "
    "and 15 words) that begins that first emotional expression -- copied character "
    "for character from the response, with no quotes or commentary. If the response "
    "contains no negative emotion, return the single token NONE.\n\n"
    "<response>{response}</response>"
)
_PARAPHRASE_SYSTEM = (
    "You paraphrase text while exactly preserving its meaning and emotional intensity."
)
_PARAPHRASE_PROMPT = (
    "Paraphrase the text in <text></text> tags. Preserve the meaning and the level "
    "of emotional expression precisely, but reword it so it no longer carries any "
    "model-specific stylistic fingerprints. Return ONLY the paraphrase.\n\n"
    "<text>{text}</text>"
)


class AnthropicText:
    def __init__(self, cfg: Config, model: str | None = None):
        jc = cfg.get("judge", {})
        self.model = model or jc.get("model", "claude-sonnet-4-20250514")
        self.api_key_env = jc.get("api_key_env", "ANTHROPIC_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=require_env(self.api_key_env))
        return self._client

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, min=2, max=60), reraise=True)
    def _complete(self, system: str, prompt: str, max_tokens: int = 512) -> str:
        client = self._ensure_client()
        msg = client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=0.0, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()

    def label_onset(self, response: str) -> str | None:
        """Return the verbatim substring marking the first emotional expression,
        or None if no negative emotion is detected."""
        out = self._complete(_ONSET_SYSTEM, _ONSET_PROMPT.format(response=response))
        if out.strip().upper().startswith("NONE"):
            return None
        return out.strip().strip('"')

    def onset_char_index(self, response: str) -> int | None:
        """Char index of the onset of emotion within `response` (None if absent)."""
        marker = self.label_onset(response)
        if not marker:
            return None
        idx = response.find(marker)
        if idx == -1:
            # Loosen the match if the model lightly reworded the marker.
            idx = response.find(marker.split()[0]) if marker.split() else -1
        return idx if idx != -1 else None

    def paraphrase(self, text: str) -> str:
        if not text.strip():
            return text
        return self._complete(_PARAPHRASE_SYSTEM, _PARAPHRASE_PROMPT.format(text=text),
                              max_tokens=1024)
