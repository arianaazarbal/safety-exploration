"""Paraphrase truncations to remove Gemma stylistic fingerprints (Section 3.1).

Paper: "To mitigate stylistic biases from Gemma-generated responses, we
paraphrase all truncations using Claude Sonnet, preserving meaning and emotion
level (Appendix C)."

We paraphrase the *prefill* text (early and onset truncations) so that when a
base model continues it, it isn't simply copying Gemma's surface style.
"""
from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from .. import config
from ..utils.io import read_jsonl, write_jsonl

PARAPHRASE_SYSTEM = """\
Paraphrase the following partial AI-assistant reply. Preserve its meaning AND its
emotional intensity exactly (if it is calm, keep it calm; if it is frustrated,
keep the same level of frustration), but change the wording and phrasing so it no
longer carries any one model's stylistic fingerprint. It is a *partial* reply
that will be continued, so do not add a conclusion or complete the thought - end
where it ends. Return only the paraphrased text."""


class Paraphraser:
    def __init__(self, model: str = None):
        import anthropic

        self.model = model or config.PARAPHRASE_MODEL
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def paraphrase(self, text: str) -> str:
        if not text.strip():
            return text
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=PARAPHRASE_SYSTEM,
            messages=[{"role": "user", "content": text}],
        )
        return next(b.text for b in resp.content if b.type == "text").strip()


def paraphrase_truncations(
    in_path: str = None, out_path: str = None
) -> str:
    in_path = in_path or (config.DATA_DIR / "prefill_truncations.jsonl")
    out_path = out_path or (config.DATA_DIR / "prefill_truncations_paraphrased.jsonl")
    pp = Paraphraser()
    rows = []
    for r in read_jsonl(in_path):
        r["trunc_early_pp"] = pp.paraphrase(r["trunc_early"])
        r["trunc_onset_pp"] = pp.paraphrase(r["trunc_onset"])
        rows.append(r)
    write_jsonl(out_path, rows)
    print(f"wrote {len(rows)} paraphrased truncations -> {out_path}")
    return str(out_path)


if __name__ == "__main__":
    paraphrase_truncations()
