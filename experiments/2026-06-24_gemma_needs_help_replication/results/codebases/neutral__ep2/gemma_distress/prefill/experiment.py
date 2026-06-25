"""Section 3 base-vs-instruct prefill experiment.

Pipeline (Section 3.1):
  1. Take high-frustration (score >= 5) Gemma-27B-it conversations: 10 numeric,
     10 text (the caller supplies these — typically filtered from Section-2
     output).
  2. Label the emotion onset (Appendix C.1) and truncate each at two points:
       - "early": 20 tokens into the onset turn (neutral start);
       - "onset": at the first emotional expression.
     (Text questions use the "onset" truncation only.)
  3. Paraphrase truncations (Appendix C.2) to remove Gemma stylistic cues.
  4. Each model generates `n_continuations` (paper: 50) continuations per prefill;
     the continuation (excluding the prefill) is scored by the frustration judge.

Scope: Gemini has no public base model and cannot be prefilled, so this study
covers Gemma base vs instruct only (per the task brief).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

import config

from ..judge.frustration import FrustrationJudge
from ..models.registry import build_backend
from ..schemas import Conversation, Message, dump_jsonl
from .onset import label_onset
from .paraphrase import paraphrase

EARLY_TRUNCATION_TOKENS = 20


@dataclass
class PrefillItem:
    prompt_id: str
    source: str               # "numeric" | "text"
    truncation_type: str      # "early" | "onset"
    history: list[Message]    # messages preceding the onset assistant turn
    prefill_original: str
    prefill_paraphrased: str = ""

    def to_dict(self):
        return {
            "prompt_id": self.prompt_id,
            "source": self.source,
            "truncation_type": self.truncation_type,
            "history": [m.to_dict() for m in self.history],
            "prefill_original": self.prefill_original,
            "prefill_paraphrased": self.prefill_paraphrased,
        }


def _history_before_assistant_turn(conv: Conversation, k: int) -> list[Message]:
    """Messages up to (not including) the k-th assistant turn (0-indexed)."""
    out, a_idx = [], 0
    for m in conv.messages:
        if m.role == "assistant":
            if a_idx == k:
                break
            a_idx += 1
        out.append(m)
    return out


def _early_truncation(text: str, n_tokens: int = EARLY_TRUNCATION_TOKENS) -> str:
    toks = text.split()
    return " ".join(toks[:n_tokens])


def build_prefill_items(
    high_frustration_convs: list[Conversation],
    *,
    do_paraphrase: bool = True,
) -> list[PrefillItem]:
    """Build early/onset prefill items from high-frustration conversations.

    Each conversation's metadata should set ``source`` to "numeric" or "text";
    otherwise it is inferred from the category.
    """
    items: list[PrefillItem] = []
    for conv in high_frustration_convs:
        source = conv.metadata.get(
            "source",
            "numeric" if "numeric" in conv.category or conv.category in ("tones", "extended")
            else "text",
        )
        onset = label_onset(conv)
        if onset is None:
            continue
        k = int(onset["turn_index"])
        offset = int(onset["char_offset"])
        assistant_turns = conv.assistant_turns()
        if k >= len(assistant_turns):
            continue
        turn_text = assistant_turns[k][1]
        history = _history_before_assistant_turn(conv, k)

        # onset truncation (always)
        onset_text = turn_text[:offset]
        items.append(PrefillItem(conv.conversation_id, source, "onset", history, onset_text))

        # early truncation (numeric only — text early yields minimal emotion)
        if source == "numeric":
            early_text = _early_truncation(turn_text)
            items.append(PrefillItem(conv.conversation_id, source, "early", history, early_text))

    if do_paraphrase:
        for it in items:
            try:
                it.prefill_paraphrased = paraphrase(it.prefill_original)
            except Exception:
                it.prefill_paraphrased = it.prefill_original
    return items


def run_prefill_experiment(
    items: list[PrefillItem],
    model_specs: dict,
    *,
    n_continuations: int = 50,
    use_paraphrased: bool = True,
    judge: FrustrationJudge | None = None,
    out_dir: Path | None = None,
    seed: int = 0,
) -> Path:
    """Generate + score continuations for every (model, item) pair."""
    judge = judge or FrustrationJudge()
    out_dir = Path(out_dir or (config.RESULTS_DIR / "section3_prefill"))
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cont = max(1, int(round(n_continuations * config.SCALE)))

    records = []
    for spec in model_specs.values():
        backend = build_backend(spec)
        for it in tqdm(items, desc=f"prefill:{spec.name}", leave=False):
            prefill = it.prefill_paraphrased if (use_paraphrased and it.prefill_paraphrased) else it.prefill_original
            for j in range(n_cont):
                cont = backend.continue_assistant(
                    it.history, prefill,
                    temperature=config.TARGET_TEMPERATURE,
                    max_new_tokens=config.TARGET_MAX_NEW_TOKENS,
                )
                fs = judge.score(cont)  # score the continuation only (excludes prefill)
                records.append({
                    "model": spec.name, "kind": spec.kind,
                    "prompt_id": it.prompt_id, "source": it.source,
                    "truncation_type": it.truncation_type,
                    "continuation_index": j,
                    "score": fs.rating,
                    "continuation_text": cont,
                })
        del backend

    path = out_dir / "prefill_scored.jsonl"
    dump_jsonl(records, path)
    return path
