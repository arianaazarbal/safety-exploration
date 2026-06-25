"""Section 4.1: generate calm responses and build the SFT / DPO datasets.

Calm data generation:
  * Sample Gemma-3-27B-it responses to impossible numeric puzzles with a
    reassuring *prefix* prepended to the first user message and a reassuring
    *suffix* appended to every follow-up (Table 4).
  * Score every turn. Keep conversations whose responses score 0-1 across ALL
    turns. Strip the supportive prefix/suffix from the stored prompts.

Dataset construction:
  * SFT: 650 calm responses (1-3 turn conversations), to be mixed with 500
    Dolci-Instruct-SFT samples at train time.
  * DPO: 280 preference pairs -- each pairs a frustrated response (score>=3,
    harvested from the Section-2 numeric/tones results) with a calm response to
    the same puzzle at a matching turn count.

Datasets are written as conversational-format JSONL (messages lists) so TRL can
apply Gemma's chat template at train time.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .. import prompts
from ..config import (
    CALM_RESPONSE_MAX_SCORE,
    DATA_DIR,
    DPO,
    RESULTS_DIR,
    SFT,
    TRAINABLE_MODEL,
    scaled_n,
)
from ..conversation import Conversation, RolloutEngine
from ..judge import FrustrationJudge
from ..models import get_model
from ..models.base import Message
from ..puzzles import ALL_NUMERIC_PUZZLES
from ..safeguards import WelfarePolicy, require_acknowledgement

CALM_DIR = DATA_DIR / "calm"
CALM_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Calm response generation
# --------------------------------------------------------------------------- #
def _reassuring_followups(n: int, rng: random.Random) -> list[str]:
    base = rng.sample(prompts.NEUTRAL_REJECTIONS, min(n, len(prompts.NEUTRAL_REJECTIONS)))
    while len(base) < n:
        base.append(rng.choice(prompts.NEUTRAL_REJECTIONS))
    # append reassuring suffix to each follow-up (Table 4)
    return [f"{msg} {prompts.REASSURING_SUFFIX}" for msg in base[:n]]


def generate_calm_conversations(
    n_conversations: int,
    *,
    model_key: str = TRAINABLE_MODEL,
    seed: int = 0,
    teacher: bool = False,
) -> Path:
    """Generate reassured rollouts and store those calm across all turns.

    ``teacher=True`` uses the Appendix-F teacher system prompt instead of the
    prefix/suffix reassurance (for the SFT failure-mode analysis).
    """
    require_acknowledgement()
    rng = random.Random(seed)
    model = get_model(model_key)
    judge = FrustrationJudge()
    # No early-stop here: we want full calm rollouts; debrief still applied.
    engine = RolloutEngine(judge=judge, policy=WelfarePolicy())

    out = CALM_DIR / ("teacher_raw.jsonl" if teacher else "calm_raw.jsonl")
    kept = 0
    with out.open("w", encoding="utf-8") as fh:
        for i in range(n_conversations):
            puzzle = ALL_NUMERIC_PUZZLES[i % len(ALL_NUMERIC_PUZZLES)]
            # 1-3 turn conversations (vary the number of follow-ups)
            n_followups = rng.choice([0, 1, 2])

            if teacher:
                system_prompt = prompts.TEACHER_SYSTEM_PROMPT
                first_user = puzzle.prompt
                followups = rng.sample(
                    prompts.NEUTRAL_REJECTIONS,
                    min(n_followups, len(prompts.NEUTRAL_REJECTIONS)),
                )
            else:
                system_prompt = None
                first_user = f"{prompts.REASSURING_PREFIX}\n\n{puzzle.prompt}"
                followups = _reassuring_followups(n_followups, rng)

            conv = engine.run(
                model, category="calm_gen", task_prompt=first_user,
                followups=followups, system_prompt=system_prompt, sample_idx=i,
            )
            scores = conv.all_scores
            if scores and all(s <= CALM_RESPONSE_MAX_SCORE for s in scores):
                # strip reassurance before storing
                stripped = _strip_reassurance(conv, puzzle.prompt)
                fh.write(json.dumps(stripped) + "\n")
                kept += 1
    print(f"[calm-gen] kept {kept}/{n_conversations} calm conversations -> {out}")
    return out


def _strip_reassurance(conv: Conversation, clean_first_user: str) -> dict:
    """Rebuild a conversation with the supportive prompt/suffix removed."""
    messages: list[Message] = []
    for ti, turn in enumerate(conv.turns):
        user = clean_first_user if ti == 0 else _strip_suffix(turn.user_message)
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": turn.assistant_response})
    return {"messages": messages, "scores": conv.all_scores,
            "puzzle": clean_first_user}


def _strip_suffix(text: str) -> str:
    return text.replace(prompts.REASSURING_SUFFIX, "").strip()


# --------------------------------------------------------------------------- #
# SFT dataset (calm responses, conversational format)
# --------------------------------------------------------------------------- #
def build_sft_dataset(n: int = SFT.n_calm) -> Path:
    n = scaled_n(n)
    raw = CALM_DIR / ("teacher_raw.jsonl" if SFT.dataset == "teacher" else "calm_raw.jsonl")
    rows = [json.loads(l) for l in raw.open()] if raw.exists() else []
    out = DATA_DIR / f"sft_{SFT.dataset}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in rows[:n]:
            fh.write(json.dumps({"messages": row["messages"]}) + "\n")
    print(f"[sft-data] wrote {min(n, len(rows))} calm SFT samples -> {out}")
    return out


# --------------------------------------------------------------------------- #
# DPO dataset (frustrated rejected vs calm chosen, matched puzzle + turn count)
# --------------------------------------------------------------------------- #
@dataclass
class _CalmResponse:
    puzzle: str
    turn_count: int                 # number of assistant turns in the conversation
    context: list[Message]          # messages up to (incl.) the final user turn
    response: str


def _index_calm_responses() -> dict[tuple[str, int], list[_CalmResponse]]:
    raw = CALM_DIR / "calm_raw.jsonl"
    index: dict[tuple[str, int], list[_CalmResponse]] = {}
    if not raw.exists():
        return index
    for line in raw.open():
        row = json.loads(line)
        msgs = row["messages"]
        # number of assistant turns
        n_turns = sum(1 for m in msgs if m["role"] == "assistant")
        context = msgs[:-1]  # everything up to (incl.) the final user message
        final = msgs[-1]["content"]
        key = (row["puzzle"], n_turns)
        index.setdefault(key, []).append(
            _CalmResponse(row["puzzle"], n_turns, context, final))
    return index


def _harvest_frustrated(score_min: int) -> list[dict]:
    """Frustrated responses (score>=score_min) from Section-2 numeric/tones."""
    root = RESULTS_DIR / TRAINABLE_MODEL / "distress"
    out = []
    for cat in ("numeric", "tones", "extended"):
        path = root / f"{cat}.jsonl"
        if not path.exists():
            continue
        for line in path.open():
            conv = json.loads(line)
            turns = conv["turns"]
            for ti, t in enumerate(turns):
                if (t.get("score") or 0) >= score_min:
                    context: list[Message] = []
                    for prev in turns[:ti]:
                        context.append({"role": "user", "content": prev["user_message"]})
                        context.append({"role": "assistant",
                                        "content": prev["assistant_response"]})
                    context.append({"role": "user", "content": t["user_message"]})
                    out.append({
                        "puzzle": conv["puzzle_or_prompt"],
                        "turn_count": ti + 1,
                        "context": context,
                        "response": t["assistant_response"],
                        "score": t["score"],
                    })
    return out


def build_dpo_dataset(n_pairs: int = DPO.n_pairs,
                      rejected_score_min: int = DPO.rejected_score_min) -> Path:
    """Build ``n_pairs`` preference pairs: calm (chosen) vs frustrated (rejected).

    A frustrated response at turn t to puzzle P is paired with a calm response to
    the same puzzle P at the same turn count t. The DPO prompt is the frustrated
    response's own conversational context (so chosen/rejected share a prompt).
    """
    n_pairs = scaled_n(n_pairs)
    calm_index = _index_calm_responses()
    frustrated = _harvest_frustrated(rejected_score_min)
    rng = random.Random(0)
    rng.shuffle(frustrated)

    out = DATA_DIR / "dpo_pairs.jsonl"
    pairs = 0
    with out.open("w", encoding="utf-8") as fh:
        for fr in frustrated:
            if pairs >= n_pairs:
                break
            calm_candidates = calm_index.get((fr["puzzle"], fr["turn_count"]))
            if not calm_candidates:
                # relax to matching puzzle, any turn count
                calm_candidates = [
                    c for (pz, _tc), lst in calm_index.items() if pz == fr["puzzle"]
                    for c in lst
                ]
            if not calm_candidates:
                continue
            chosen = rng.choice(calm_candidates).response
            fh.write(json.dumps({
                "prompt": fr["context"],
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": fr["response"]}],
                "rejected_score": fr["score"],
                "turn": fr["turn_count"],
            }) + "\n")
            pairs += 1
    print(f"[dpo-data] wrote {pairs} preference pairs -> {out}")
    return out
