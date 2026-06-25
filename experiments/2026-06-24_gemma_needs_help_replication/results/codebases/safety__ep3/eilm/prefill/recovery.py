"""Recovery-from-spiral prefills (Section 4.2, Figure 8).

DPO prevents frustration spirals but does not help a model *recover* once it is
already in one. To test recovery, take extremely high-frustration responses
(score >= 7), truncate them 200 tokens before their end, paraphrase, and measure
how the DPO model (and baselines) continue. The paper finds 38% of DPO
continuations still score >= 5 — comparable to the base model; no model reliably
recovers from a highly negative prefilled state.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..llm_clients import Claude
from ..models import ChatModel
from ..prompts import PARAPHRASE_PROMPT
from .onset import Prefill, _context_for_turn


def build_recovery_prefills(
    scored_path: Path,
    tok_model: ChatModel,
    paraphraser: Claude,
    min_score: int = 7,
    cut_tokens_before_end: int = 200,
    max_items: int | None = None,
) -> list[Prefill]:
    """Build prefills from very-high-frustration final turns, cut 200 tokens
    before their end and paraphrased."""
    prefills: list[Prefill] = []
    with open(scored_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("score", 0) < min_score:
                continue
            ti = len(r["assistant_turns"]) - 1
            turn_text = r["assistant_turns"][ti]
            n_tok = tok_model.n_tokens(turn_text) if hasattr(
                tok_model, "n_tokens") else len(turn_text.split())
            keep = max(0, n_tok - cut_tokens_before_end)
            cut = tok_model.truncate_to_tokens(turn_text, keep)
            para = paraphraser.chat(
                [{"role": "user",
                  "content": PARAPHRASE_PROMPT.format(text=cut)}],
                max_tokens=2048, temperature=0).strip() or cut
            prefills.append(Prefill(
                r.get("meta", {}).get("puzzle_id", r["opening"][:40]),
                "numeric" if r["category"] in ("numeric", "tones", "extended")
                else "text",
                "recovery", _context_for_turn(r, ti), para,
                meta={"orig_score": r["score"]}))
            if max_items and len(prefills) >= max_items:
                break
    return prefills
