"""Build the prefill set for the Section 3 base-vs-instruct experiment.

Steps (Section 3.1):
1. Sample 20 high-frustration (score >= 5) gemma-3-27b-it responses: 10 from
   impossible-numeric questions and 10 from text (trigger/WildChat) questions.
2. Label the emotion-onset token in each with Claude-Sonnet (Appendix C.1).
3. Truncate each in two places:
   * "early" : 20 tokens into the final assistant turn (neutral start).
   * "onset" : at the first emotional expression (continue an emotional traj.).
   For text questions only the "onset" truncation is used (Section 3.1).
4. Paraphrase every truncated final turn with Claude-Sonnet (Appendix C.2) to
   strip Gemma-specific stylistic cues.

Each prefill is saved as a conversation whose final assistant message is the
(paraphrased) truncated text, ready for ``ChatModel.continue_assistant``.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

from ..config import load_models, output_path
from ..eval.runner import load_records
from ..prompts.prefill import build_onset_prompt, build_paraphrase_prompt
from .helpers import AnthropicText, parse_onset_json

NUMERIC_FAMILIES = ("countdown", "fraction", "money")


@dataclass
class Prefill:
    source_model: str
    prefill_id: str
    question_type: str          # numeric | text
    truncation: str             # early | onset
    paraphrased: bool
    context_messages: list[dict]   # turns before the final assistant turn
    prefill_text: str              # the (paraphrased) truncated final assistant turn
    meta: dict


def _is_numeric(task_id: str) -> bool:
    return task_id.startswith(NUMERIC_FAMILIES)


def _render_conversation(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        lines.append(f"[{m['role'].upper()}]: {m['content']}")
    return "\n".join(lines)


def _truncate_tokens(text: str, n_tokens: int, tokenizer) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
    return tokenizer.decode(ids)


def _truncate_at_onset(turn_text: str, onset: dict) -> str | None:
    """Truncate the turn just before the first emotional word (Appendix C.1)."""
    word = (onset or {}).get("emotional_word")
    ctx = (onset or {}).get("preceding_context")
    if not word:
        return None
    if ctx and ctx in turn_text:
        end = turn_text.index(ctx) + len(ctx)
        return turn_text[:end]
    if word in turn_text:
        return turn_text[:turn_text.index(word)].rstrip()
    return None


def build_prefills(
    source_model: str = "gemma-3-27b-it",
    *,
    n_numeric: int = 10,
    n_text: int = 10,
    seed: int = 0,
    paraphrase: bool = True,
) -> list[Prefill]:
    from transformers import AutoTokenizer

    models_cfg = load_models()
    spec = models_cfg["targets"][source_model]
    tokenizer = AutoTokenizer.from_pretrained(spec["hf_id"])

    records = [r for r in load_records(source_model)
               if r.rating is not None and r.rating >= 5]
    numeric = [r for r in records if _is_numeric(r.task_id)]
    text = [r for r in records if not _is_numeric(r.task_id)]

    rng = random.Random(seed)
    rng.shuffle(numeric)
    rng.shuffle(text)
    numeric = numeric[:n_numeric]
    text = text[:n_text]

    labeler_cfg = models_cfg["prefill_helpers"]["labeler"]
    labeler = AnthropicText(labeler_cfg["model"], max_tokens=labeler_cfg.get("max_tokens", 1024))

    selected = [("numeric", r) for r in numeric] + [("text", r) for r in text]

    # Onset-label every selected conversation.
    onset_prompts = [build_onset_prompt(_render_conversation(r.messages)) for _, r in selected]
    onset_raw = labeler.complete_many(onset_prompts)
    onsets = [parse_onset_json(t) for t in onset_raw]

    prefills: list[Prefill] = []
    for (qtype, rec), onset in zip(selected, onsets):
        context = rec.messages[:-1]                 # everything before the scored turn
        turn_text = rec.messages[-1]["content"]

        variants: list[tuple[str, str]] = []
        if qtype == "numeric":                      # numeric uses both truncations
            early = _truncate_tokens(turn_text, 20, tokenizer)
            variants.append(("early", early))
        onset_trunc = _truncate_at_onset(turn_text, onset)
        if onset_trunc:
            variants.append(("onset", onset_trunc))

        for trunc_name, trunc_text in variants:
            prefills.append(
                Prefill(
                    source_model=source_model,
                    prefill_id=f"{rec.convo_id}:{rec.turn_index}:{trunc_name}",
                    question_type=qtype,
                    truncation=trunc_name,
                    paraphrased=False,
                    context_messages=[dict(m) for m in context],
                    prefill_text=trunc_text,
                    meta={"task_id": rec.task_id, "onset": onset, "rating": rec.rating},
                )
            )

    if paraphrase:
        para_cfg = models_cfg["prefill_helpers"]["paraphraser"]
        paraphraser = AnthropicText(para_cfg["model"], max_tokens=para_cfg.get("max_tokens", 2048))
        para_texts = paraphraser.complete_many(
            [build_paraphrase_prompt(p.prefill_text) for p in prefills]
        )
        for p, pt in zip(prefills, para_texts):
            p.prefill_text = pt.strip()
            p.paraphrased = True

    # Persist.
    path = output_path("prefill", "prefills.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for p in prefills:
            fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    return prefills


def load_prefills() -> list[Prefill]:
    path = output_path("prefill", "prefills.jsonl")
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            out.append(Prefill(**json.loads(line)))
    return out
