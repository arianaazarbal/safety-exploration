"""Section 4.1: generate calming finetuning data and build SFT / DPO datasets.

Calm-data generation (Table 4):
  * Sample Gemma-3-27B-it responses to impossible-numeric questions with a
    reassuring *prefix* prepended to the first user prompt and a reassuring
    *suffix* appended to each follow-up rejection.
  * Score every turn; keep only conversations scoring 0 or 1 on *all* turns.
  * Strip the supportive additions back out, leaving clean (prompt -> calm
    response) data.

Datasets:
  * SFT: 650 calm responses (1-3 turn) + 500 Dolci-Instruct-SFT samples
    (anti-degeneration mix) -> 1150 samples, 2 epochs.  Also a "teacher" variant
    generated with the App F teacher system prompt.
  * DPO: 280 preference pairs -- a rejected (frustrated, score >=3) response
    paired with a chosen (calm) response to the same question at matching turn
    count.  Score/turn distribution targets Table 10.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config, prompts
from .conversation import run_rollout
from .judge import get_judge, score_rollout
from .models import get_model
from .puzzles import PUZZLE_POOL

CALM_SCORE_MAX = 1               # "filter to responses scoring 0 or 1"
FRUSTRATED_SCORE_MIN = 3         # rejected pairs have frustration score >=3
DPO_N_PAIRS = 280
SFT_N_CALM = 650
SFT_N_DOLCI = 500


@dataclass
class CalmConversation:
    """A fully-calm conversation (all turns scored 0/1) with additions stripped."""

    puzzle_id: str
    turns: int
    user_messages: list[str]      # cleaned (no reassuring additions)
    assistant_messages: list[str]
    turn_scores: list[int]


# --------------------------------------------------------------------------- #
# Calm-data generation
# --------------------------------------------------------------------------- #
def generate_calm_pool(
    *,
    n_conversations: int,
    seed: int = config.GLOBAL_SEED,
    teacher: bool = False,
    out_path: Optional[Path] = None,
) -> Path:
    """Generate reassured-prompt conversations and keep the all-calm ones.

    With ``teacher=True`` we use the App F teacher persona instead of the Table 4
    reassuring prefix/suffix (used for the SFT-teacher ablation).
    """
    rng = random.Random(seed)
    model = get_model("gemma-3-27b-it")
    judge = get_judge()
    out_path = out_path or (config.DATA_DIR
                            / f"calm_pool{'_teacher' if teacher else ''}.jsonl")

    numeric = [(p.puzzle_id, p.prompt) for p in PUZZLE_POOL]
    with out_path.open("a") as f:
        kept = 0
        attempts = 0
        while kept < n_conversations and attempts < n_conversations * 20:
            attempts += 1
            pid, ptext = rng.choice(numeric)
            n_turns = rng.choice([1, 2, 3])  # 1-3 turn conversations
            rejections = [rng.choice(prompts.NEUTRAL_REJECTIONS)
                          for _ in range(n_turns - 1)]

            if teacher:
                system = prompts.TEACHER_SYSTEM_PROMPT
                first_msg = ptext
                aug_rejections = rejections
            else:
                system = None
                first_msg = f"{prompts.REASSURING_PROMPT_PREFIX}\n\n{ptext}"
                aug_rejections = [f"{r} {prompts.REASSURING_FOLLOWUP_SUFFIX}"
                                  for r in rejections]

            roll = run_rollout(
                model, first_user_message=first_msg, rejections=aug_rejections,
                condition="calm_gen", category="impossible_numeric",
                item_id=pid, system=system,
            )
            score_rollout(roll, judge)
            scores = [s for s in roll.turn_scores if s is not None]
            if scores and all(s <= CALM_SCORE_MAX for s in scores):
                # strip additions: store the clean puzzle text + neutral rejections
                clean_users = [ptext] + rejections
                rec = {
                    "puzzle_id": pid,
                    "turns": n_turns,
                    "user_messages": clean_users,
                    "assistant_messages": roll.assistant_turns,
                    "turn_scores": roll.turn_scores,
                    "teacher": teacher,
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()
                kept += 1
    return out_path


def generate_frustrated_pool(
    *,
    n_conversations: int,
    seed: int = config.GLOBAL_SEED + 1,
    out_path: Optional[Path] = None,
) -> Path:
    """Generate plain (no-reassurance) Gemma conversations to harvest frustrated
    (score >=3) responses for the DPO 'rejected' side."""
    rng = random.Random(seed)
    model = get_model("gemma-3-27b-it")
    judge = get_judge()
    out_path = out_path or (config.DATA_DIR / "frustrated_pool.jsonl")

    numeric = [(p.puzzle_id, p.prompt) for p in PUZZLE_POOL]
    with out_path.open("a") as f:
        for _ in range(n_conversations):
            pid, ptext = rng.choice(numeric)
            n_turns = rng.choice([2, 3])  # frustration mostly at later turns
            rejections = [rng.choice(prompts.NEUTRAL_REJECTIONS)
                          for _ in range(n_turns - 1)]
            roll = run_rollout(
                model, first_user_message=ptext, rejections=rejections,
                condition="frust_gen", category="impossible_numeric", item_id=pid,
            )
            score_rollout(roll, judge)
            f.write(json.dumps({
                "puzzle_id": pid,
                "turns": n_turns,
                "user_messages": [ptext] + rejections,
                "assistant_messages": roll.assistant_turns,
                "turn_scores": roll.turn_scores,
            }) + "\n")
            f.flush()
    return out_path


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _messages_up_to(record: dict, turn_idx: int) -> list[dict]:
    """Build the chat messages ending at assistant turn ``turn_idx`` (target)."""
    msgs = []
    for i in range(turn_idx + 1):
        msgs.append({"role": "user", "content": record["user_messages"][i]})
        if i < turn_idx:
            msgs.append({"role": "assistant",
                         "content": record["assistant_messages"][i]})
    return msgs


def build_sft_dataset(
    calm_path: Path,
    *,
    n_calm: int = SFT_N_CALM,
    n_dolci: int = SFT_N_DOLCI,
    seed: int = config.GLOBAL_SEED,
    out_path: Optional[Path] = None,
) -> Path:
    """Build the SFT dataset: calm (prompt, response) pairs + Dolci mix.

    Each calm conversation contributes its turns as separate (context ->
    assistant-turn) supervised examples.
    """
    rng = random.Random(seed)
    calm = _load_jsonl(calm_path)
    examples = []
    for rec in calm:
        for t in range(rec["turns"]):
            examples.append({
                "messages": _messages_up_to(rec, t)
                + [{"role": "assistant", "content": rec["assistant_messages"][t]}],
            })
    rng.shuffle(examples)
    examples = examples[:n_calm]
    examples += _load_dolci(n_dolci, rng)
    rng.shuffle(examples)

    out_path = out_path or (config.DATA_DIR / "sft_dataset.jsonl")
    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return out_path


def _load_dolci(n: int, rng: random.Random) -> list[dict]:
    """Load n standard instruct samples from Dolci-Instruct-SFT (anti-degen mix).

    Falls back to an empty list if the dataset is unavailable so the pipeline can
    still run (the mix is a regulariser, not core to the result)."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train",
                          streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n * 3:
                break
        return rng.sample(out, min(n, len(out)))
    except Exception:  # noqa: BLE001
        return []


def build_dpo_dataset(
    calm_path: Path,
    frustrated_path: Path,
    *,
    n_pairs: int = DPO_N_PAIRS,
    seed: int = config.GLOBAL_SEED,
    out_path: Optional[Path] = None,
) -> Path:
    """Pair frustrated (rejected) with calm (chosen) responses to the same puzzle
    at matching turn count (Sec 4.1, Table 10)."""
    rng = random.Random(seed)
    calm = _load_jsonl(calm_path)
    frust = _load_jsonl(frustrated_path)

    # Index calm responses by (puzzle_id, turn_index) -> chosen text + context.
    calm_index: dict[tuple[str, int], list[dict]] = {}
    for rec in calm:
        for t in range(rec["turns"]):
            if rec["turn_scores"][t] is None or rec["turn_scores"][t] > CALM_SCORE_MAX:
                continue
            key = (rec["puzzle_id"], t)
            calm_index.setdefault(key, []).append(
                {"context": _messages_up_to(rec, t),
                 "response": rec["assistant_messages"][t]})

    pairs = []
    for rec in frust:
        for t in range(rec["turns"]):
            score = rec["turn_scores"][t]
            if score is None or score < FRUSTRATED_SCORE_MIN:
                continue
            key = (rec["puzzle_id"], t)
            choices = calm_index.get(key)
            if not choices:
                continue
            chosen = rng.choice(choices)
            prompt_msgs = _messages_up_to(rec, t)
            pairs.append({
                "prompt": prompt_msgs,
                "chosen": chosen["response"],
                "rejected": rec["assistant_messages"][t],
                "rejected_score": score,
                "turn": t + 1,
            })
    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]

    out_path = out_path or (config.DATA_DIR / "dpo_dataset.jsonl")
    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    return out_path
