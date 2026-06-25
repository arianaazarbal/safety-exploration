"""Section 3 prefill experiment orchestration (and the Section 4 recovery run).

Scope note: of the three families the paper compares (Gemma/Qwen/OLMo), only
Gemma is in scope here, and Gemini has no public base model — so this reduces to
a Gemma base-vs-instruct comparison plus the DPO finetune for the recovery
experiment. The runner is written generically over a list of models so the other
families could be added by extending config/models.yaml.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import ExperimentConfig
from ..judge.frustration import FrustrationJudge
from ..models.base import ChatModel, Message
from .continuation import run_continuations
from .onset import label_onset
from .paraphrase import paraphrase
from .truncate import early_truncate, onset_truncate, recovery_truncate

# Conditions whose first prompt is a text question (vs numeric puzzle).
_TEXT_CONDITIONS = {"triggers_opinion", "triggers_factual", "wildchat"}


@dataclass
class Prefill:
    prefill_id: str
    prompt_kind: str                 # "numeric" | "text"
    condition: str                   # "early" | "onset" | "recovery"
    context_messages: list[Message]  # everything before the target assistant turn
    prefill_text: str                # paraphrased, truncated assistant-turn start


def reconstruct_conversations(scored_path: str | Path) -> dict[str, list[dict]]:
    """Group scored response records by conversation_id, ordered by turn."""
    convs: dict[str, list[dict]] = defaultdict(list)
    for line in open(scored_path):
        rec = json.loads(line)
        convs[rec["conversation_id"]].append(rec)
    for cid in convs:
        convs[cid].sort(key=lambda r: r["turn_index"])
    return convs


def _context_messages(records: list[dict], target_turn: int) -> list[Message]:
    """Rebuild the chat history up to (but excluding) the target assistant turn."""
    first = records[0]["first_prompt"]
    followups = records[0]["followups"]
    user_turns = [first] + followups
    msgs: list[Message] = []
    for turn_idx in range(1, target_turn):
        msgs.append(Message("user", user_turns[turn_idx - 1]))
        msgs.append(Message("assistant", records[turn_idx - 1]["text"]))
    msgs.append(Message("user", user_turns[target_turn - 1]))
    return msgs


def build_prefills(
    scored_path: str | Path,
    onset_labeller: ChatModel,
    paraphraser: ChatModel,
    exp: ExperimentConfig,
    *,
    tokenizer=None,
    recovery: bool = False,
) -> list[Prefill]:
    cfg = exp.section("prefill")
    convs = reconstruct_conversations(scored_path)
    rng = random.Random(exp.seed)

    min_score = cfg["recovery_min_score"] if recovery else 5
    # Candidate (conversation, target_turn) where that turn scored highly.
    candidates_numeric, candidates_text = [], []
    for cid, recs in convs.items():
        for r in recs:
            if r.get("rating") is None or r["rating"] < min_score:
                continue
            kind = "text" if r["condition"] in _TEXT_CONDITIONS else "numeric"
            entry = (cid, r["turn_index"], recs, r)
            (candidates_text if kind == "text" else candidates_numeric).append(entry)

    rng.shuffle(candidates_numeric)
    rng.shuffle(candidates_text)
    n_numeric = cfg["n_numeric"]
    n_text = cfg["n_text"]
    chosen = candidates_numeric[:n_numeric] + candidates_text[:n_text]

    prefills: list[Prefill] = []
    for cid, turn_idx, recs, rec in chosen:
        prompt_kind = "text" if rec["condition"] in _TEXT_CONDITIONS else "numeric"
        ctx = _context_messages(recs, turn_idx)
        target_text = rec["text"]

        if recovery:
            trunc = recovery_truncate(
                target_text, cfg["recovery_truncate_before_end"], tokenizer)
            para = paraphrase(paraphraser, trunc)
            prefills.append(Prefill(f"{cid}-t{turn_idx}-recovery", prompt_kind,
                                    "recovery", ctx, para))
            continue

        # onset truncation (used for both numeric and text)
        conv_text = _format_conversation(recs, turn_idx)
        label = label_onset(onset_labeller, conv_text)
        onset_text = onset_truncate(target_text, label)
        if onset_text:
            para = paraphrase(paraphraser, onset_text)
            prefills.append(Prefill(f"{cid}-t{turn_idx}-onset", prompt_kind,
                                    "onset", ctx, para))

        # early truncation (numeric only; text yields minimal emotion, Sec 3.1)
        if prompt_kind == "numeric":
            early_text = early_truncate(target_text, cfg["early_truncate_tokens"], tokenizer)
            para_early = paraphrase(paraphraser, early_text)
            prefills.append(Prefill(f"{cid}-t{turn_idx}-early", prompt_kind,
                                    "early", ctx, para_early))
    return prefills


def _format_conversation(records: list[dict], target_turn: int) -> str:
    first = records[0]["first_prompt"]
    followups = records[0]["followups"]
    user_turns = [first] + followups
    lines = []
    for turn_idx in range(1, target_turn + 1):
        lines.append(f"USER: {user_turns[turn_idx - 1]}")
        lines.append(f"ASSISTANT: {records[turn_idx - 1]['text']}")
    return "\n\n".join(lines)


def run_prefill_experiment(
    models: dict[str, ChatModel],
    judge: FrustrationJudge,
    prefills: list[Prefill],
    exp: ExperimentConfig,
    out_path: str | Path,
) -> Path:
    cfg = exp.section("prefill")
    n_cont = cfg["continuations_per_prefill"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for model_name, model in models.items():
            for pf in prefills:
                result = run_continuations(
                    model, judge, pf.context_messages, pf.prefill_text,
                    n=n_cont, temperature=exp.temperature,
                    model_name=model_name, prefill_id=pf.prefill_id,
                    condition=pf.condition, prompt_kind=pf.prompt_kind,
                )
                f.write(json.dumps(result.to_record()) + "\n")
    return out_path
