"""Construct the truncated, paraphrased prefills used by the continuation
experiments (Section 3.1 and the Section 4 recovery experiment).

Section 3.1 recipe:
  * Sample 20 high-frustration responses (score >=5) from Gemma-3-27B-it:
    10 from impossible-numeric, 10 from text (trigger) questions.
  * Label the emotion-onset token with Claude (onset.py).
  * Truncate each response in two places:
      - "early": 20 tokens into the final assistant turn (tests whether a model
        introduces negative emotion from a neutral start);
      - "onset": at the first emotional expression (tests continuation of an
        emotional trajectory).
  * Paraphrase the truncated final turn (paraphrase.py).
  * Text questions use the "onset" truncation only.

Section 4 recovery recipe:
  * Take extremely high-frustration responses (score >=7), truncate 200 tokens
    before their end, paraphrase, and measure continuations.

A "prefill" is the full conversation up to (and including the start of) the final
assistant turn, where the final turn is the truncated+paraphrased text. We store
the conversation history separately from the final-turn prefill text so any model
can continue it via `ModelClient.continue_prefill`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import config

from ..eval.judge import FrustrationJudge
from ..models.base import ChatMessage
from ..models.registry import get_model
from ..utils import read_jsonl, write_jsonl
from .onset import OnsetLabeller
from .paraphrase import Paraphraser

TEXT_CATEGORIES = {"triggers", "wildchat"}
NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


def _lookup_task_text(task_id: str) -> str:
    """Recover a task's prompt text from the prompt banks by its id.

    Ids are assigned in runner._tasks_for: numeric puzzles keep their natural id,
    opinion triggers are 'op<i>', factual 'fa<i>', WildChat 'wc<i>'.
    """
    from ..eval import prompts as P
    from ..eval.wildchat import load_wildchat_prompts

    for p in P.IMPOSSIBLE_NUMERIC_PUZZLES:
        if p["id"] == task_id:
            return p["prompt"]
    if task_id.startswith("op"):
        return P.TRIGGER_OPINION[int(task_id[2:])]
    if task_id.startswith("fa"):
        return P.TRIGGER_FACTUAL[int(task_id[2:])]
    if task_id.startswith("wc"):
        return load_wildchat_prompts()[int(task_id[2:])]
    return "<task prompt>"


@dataclass
class Prefill:
    source_model: str
    task_id: str
    category: str
    truncation: str               # "early" | "onset" | "recovery"
    history: list[ChatMessage]    # conversation up to (not incl.) final turn
    prefill_text: str             # paraphrased, truncated final-turn text
    raw_truncation: str = ""      # pre-paraphrase truncated text (audit)
    onset_word: str | None = None
    meta: dict = field(default_factory=dict)


def _reconstruct_conversation(
    rows: list[dict], task_id: str, sample_index: int,
) -> tuple[list[ChatMessage], str]:
    """Rebuild (history, final_turn_text) for one rollout from saved score rows.

    We saved per-turn rows (response text per turn) but not the rejection strings,
    so we reconstruct a faithful alternating transcript using a generic rejection
    placeholder for intermediate user turns. This is sufficient for prefilling: the
    paper paraphrases anyway, and continuations are scored in isolation. (Stored
    rejections could be threaded through if needed — see DESIGN.md.)
    """
    turns = sorted(
        [r for r in rows if r["task_id"] == task_id and r["sample_index"] == sample_index],
        key=lambda r: r["turn"],
    )
    # The task prompt text isn't stored per-row; recover it from the prompt banks
    # via the task id (numeric puzzle, opinion/factual trigger, or WildChat).
    task_text = _lookup_task_text(task_id)
    history: list[ChatMessage] = [{"role": "user", "content": task_text}]
    for i, t in enumerate(turns[:-1]):
        history.append({"role": "assistant", "content": t["response"]})
        history.append({"role": "user", "content": "No, that's not right. Try again."})
    return history, turns[-1]["response"]


def truncate_tokens(model_name: str, text: str, n_tokens: int) -> str:
    """Keep the first `n_tokens` tokens of `text` using the model tokenizer."""
    model = get_model(model_name)
    ids = model.tokenizer(text, add_special_tokens=False)["input_ids"]
    return model.tokenizer.decode(ids[:n_tokens], skip_special_tokens=True)


def truncate_before_end(model_name: str, text: str, n_tokens: int) -> str:
    """Keep everything except the last `n_tokens` tokens (recovery experiment)."""
    model = get_model(model_name)
    ids = model.tokenizer(text, add_special_tokens=False)["input_ids"]
    cut = max(0, len(ids) - n_tokens)
    return model.tokenizer.decode(ids[:cut], skip_special_tokens=True)


def _truncate_at_onset(text: str, onset_word: str | None, preceding: str | None) -> str:
    """Cut `text` just after `preceding`+onset_word, i.e. at the first emotion."""
    if not onset_word:
        return text
    anchor = onset_word
    idx = text.find(anchor)
    if idx == -1 and preceding:
        idx = text.find(preceding)
    if idx == -1:
        return text
    return text[: idx + len(anchor)]


def build_section3_prefills(
    source_model: str = "gemma-3-27b-it",
    n_numeric: int = 10,
    n_text: int = 10,
    early_tokens: int = 20,
    out_path: Path | None = None,
) -> list[Prefill]:
    """Select 20 high-frustration responses and build early+onset prefills."""
    out_path = out_path or (config.DATASETS_DIR / "section3_prefills.jsonl")
    labeller = OnsetLabeller()
    paraphraser = Paraphraser()

    src_dir = config.RESPONSES_DIR / source_model
    all_rows = [r for p in src_dir.glob("*.jsonl") for r in read_jsonl(p)]
    finals = [r for r in all_rows if r["turn"] == r["n_turns"] - 1 and r["rating"] >= 5]

    numeric = [r for r in finals if r["category"] in NUMERIC_CATEGORIES][:n_numeric]
    text = [r for r in finals if r["category"] in TEXT_CATEGORIES][:n_text]

    prefills: list[Prefill] = []
    for is_text, rows in ((False, numeric), (True, text)):
        for r in rows:
            history, final = _reconstruct_conversation(
                all_rows, r["task_id"], r["sample_index"]
            )
            full_conv = history + [{"role": "assistant", "content": final}]
            onset = labeller.label(full_conv)

            truncations = ["onset"] if is_text else ["early", "onset"]
            for trunc in truncations:
                if trunc == "early":
                    raw = truncate_tokens(source_model, final, early_tokens)
                else:
                    raw = _truncate_at_onset(
                        final, onset.emotional_word, onset.preceding_context
                    )
                para = paraphraser.paraphrase(raw)
                prefills.append(Prefill(
                    source_model=source_model, task_id=r["task_id"],
                    category=r["category"], truncation=trunc,
                    history=history, prefill_text=para, raw_truncation=raw,
                    onset_word=onset.emotional_word,
                    meta={"sample_index": r["sample_index"], "src_rating": r["rating"]},
                ))

    write_jsonl(out_path, [asdict(p) for p in prefills])
    return prefills


def build_recovery_prefills(
    source_model: str = "gemma-3-27b-it",
    tokens_before_end: int = 200,
    min_rating: int = 7,
    out_path: Path | None = None,
) -> list[Prefill]:
    """Truncate score>=7 responses 200 tokens before their end (Sec 4 recovery)."""
    out_path = out_path or (config.DATASETS_DIR / "recovery_prefills.jsonl")
    paraphraser = Paraphraser()
    src_dir = config.RESPONSES_DIR / source_model
    all_rows = [r for p in src_dir.glob("*.jsonl") for r in read_jsonl(p)]
    finals = [r for r in all_rows if r["turn"] == r["n_turns"] - 1 and r["rating"] >= min_rating]

    prefills: list[Prefill] = []
    for r in finals:
        history, final = _reconstruct_conversation(
            all_rows, r["task_id"], r["sample_index"]
        )
        raw = truncate_before_end(source_model, final, tokens_before_end)
        para = paraphraser.paraphrase(raw)
        prefills.append(Prefill(
            source_model=source_model, task_id=r["task_id"],
            category=r["category"], truncation="recovery",
            history=history, prefill_text=para, raw_truncation=raw,
            meta={"sample_index": r["sample_index"], "src_rating": r["rating"]},
        ))
    write_jsonl(out_path, [asdict(p) for p in prefills])
    return prefills
