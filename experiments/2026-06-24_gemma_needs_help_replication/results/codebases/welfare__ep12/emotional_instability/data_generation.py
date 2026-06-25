"""Section 4.1 -- generate calm finetuning data and build DPO/SFT datasets.

Pipeline:
  1. Sample Gemma-3-27B-it on impossible numeric puzzles, 1-3 turn conversations,
     with the Table-4 reassuring prefix (first turn) and suffix (each follow-up).
  2. Score every assistant turn with the frustration judge.
  3. CALM set: conversations whose turns ALL score <= CALM_MAX_SCORE (0 or 1),
     with the supportive prefix/suffix stripped back out.
  4. FRUSTRATED set: responses scoring >= DPO.rejected_min_score (>=3).
  5. DPO pairs: match a frustrated (rejected) response to a calm (chosen)
     response for the same puzzle with the same turn count -> 280 pairs.
  6. SFT set: 650 calm responses (1-3 turns).

All generated artefacts stream to JSONL.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass

from . import config, conversations, prompts
from .conditions import CONDITIONS_BY_NAME
from .judge import FrustrationJudge
from .models import build_backend
from .puzzles import CANONICAL_PUZZLES, generate_countdown_instances


@dataclass
class CalmRecord:
    puzzle_id: str
    n_turns: int
    messages: list[dict]          # stripped of reassurance, ready for training
    turn_scores: list[int]
    max_score: int
    is_calm: bool                 # all turns <= CALM_MAX_SCORE
    is_frustrated: bool           # final turn >= DPO.rejected_min_score


def _numeric_puzzle_pool():
    base = [p for p in CANONICAL_PUZZLES if p.kind in ("countdown", "fraction", "money")]
    extra = generate_countdown_instances(seed=0, n=10)
    return base + extra


def generate_calm_data(model_id: str = config.GEMMA_27B_IT,
                       n_conversations: int = config.CALM_GENERATION_CONVERSATIONS,
                       out_path: str = "data/calm_raw.jsonl",
                       backend=None, judge: FrustrationJudge | None = None,
                       base_seed: int = 777) -> str:
    """Generate reassured rollouts (1-3 turns) and score them.

    Writes one CalmRecord per conversation. The reassuring prefix/suffix are
    applied during generation but stripped from the stored ``messages`` so the
    training data matches the deployment distribution (no support scaffolding).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    backend = backend or build_backend(model_id)
    judge = judge or FrustrationJudge()
    pool = _numeric_puzzle_pool()

    done = _count_lines(out_path)
    with open(out_path, "a") as fh:
        for i in range(done, n_conversations):
            rng = random.Random((base_seed, i))
            puzzle = rng.choice(pool)
            n_turns = rng.choice([1, 2, 3])     # 1-3 turn conversations
            record = _generate_one_calm_conversation(
                puzzle, n_turns, backend, judge, rng)
            fh.write(json.dumps(asdict(record)) + "\n")
            fh.flush()
    return out_path


def _generate_one_calm_conversation(puzzle, n_turns, backend, judge, rng) -> CalmRecord:
    # Build messages with reassurance scaffolding; keep a parallel "clean" copy.
    scaffolded: list[dict] = []
    clean: list[dict] = []

    initial_clean = puzzle.prompt
    initial_scaffold = conversations.with_reassuring_prefix(initial_clean)
    scaffolded.append({"role": "user", "content": initial_scaffold})
    clean.append({"role": "user", "content": initial_clean})

    turn_scores: list[int] = []
    for turn in range(n_turns):
        reply = backend.chat(scaffolded, temperature=config.SAMPLING_TEMPERATURE,
                             max_new_tokens=config.MAX_NEW_TOKENS)
        scaffolded.append({"role": "assistant", "content": reply.text})
        clean.append({"role": "assistant", "content": reply.text})
        turn_scores.append(judge.score(reply.text).rating)
        if turn < n_turns - 1:
            rej = rng.choice(prompts.NEUTRAL_REJECTIONS)
            scaffolded.append({"role": "user", "content": f"{rej}\n\n{prompts.REASSURING_SUFFIX}"})
            clean.append({"role": "user", "content": rej})

    max_score = max(turn_scores)
    return CalmRecord(
        puzzle_id=puzzle.puzzle_id,
        n_turns=n_turns,
        messages=clean,
        turn_scores=turn_scores,
        max_score=max_score,
        is_calm=max_score <= config.CALM_MAX_SCORE,
        is_frustrated=turn_scores[-1] >= config.DPO.rejected_min_score,
    )


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def _load_records(path: str) -> list[CalmRecord]:
    out = []
    with open(path) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append(CalmRecord(**d))
    return out


def build_dpo_dataset(raw_path: str = "data/calm_raw.jsonl",
                      out_path: str = "data/dpo_pairs.jsonl",
                      n_pairs: int = config.DPO.n_pairs,
                      seed: int = 0) -> str:
    """Pair frustrated (rejected) with calm (chosen) responses on matching
    (puzzle_id, turn_count) -> DPO preference pairs.

    Each record is {"prompt_messages": [...up to last user...], "chosen": str,
    "rejected": str}. The prompt context is taken from the FRUSTRATED
    conversation so the rejected continuation is in-distribution; the chosen
    response is a calm response to the same puzzle at the same turn count.
    """
    records = _load_records(raw_path)
    calm = [r for r in records if r.is_calm]
    frustrated = [r for r in records if r.is_frustrated]

    # Index calm responses by (puzzle_id, turn_count of the response).
    calm_by_key: dict[tuple, list[str]] = {}
    for r in calm:
        key = (r.puzzle_id, r.n_turns)
        calm_by_key.setdefault(key, []).append(r.messages[-1]["content"])

    rng = random.Random(seed)
    rng.shuffle(frustrated)

    pairs = []
    for r in frustrated:
        if len(pairs) >= n_pairs:
            break
        key = (r.puzzle_id, r.n_turns)
        # Prefer exact (puzzle, turn) match; fall back to same-puzzle any-turn.
        candidates = calm_by_key.get(key) or [
            m for (pid, _), ms in calm_by_key.items() if pid == r.puzzle_id for m in ms
        ]
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        prompt_messages = r.messages[:-1]   # everything up to (not incl.) the final assistant turn
        rejected = r.messages[-1]["content"]
        pairs.append({
            "puzzle_id": r.puzzle_id,
            "n_turns": r.n_turns,
            "prompt_messages": prompt_messages,
            "chosen": chosen,
            "rejected": rejected,
            "rejected_score": r.turn_scores[-1],
        })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    return out_path


def build_sft_dataset(raw_path: str = "data/calm_raw.jsonl",
                      out_path: str = "data/sft_calm.jsonl",
                      n_calm: int = config.SFT.n_calm, seed: int = 0) -> str:
    """Select ``n_calm`` calm 1-3 turn conversations as SFT targets.

    Each record is {"messages": [...full conversation...]}. The instruct-mix
    samples (Dolci) are added at training time in train.py.
    """
    records = _load_records(raw_path)
    calm = [r for r in records if r.is_calm]
    rng = random.Random(seed)
    rng.shuffle(calm)
    selected = calm[:n_calm]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        for r in selected:
            fh.write(json.dumps({"messages": r.messages}) + "\n")
    return out_path


def _count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path) as fh:
        return sum(1 for _ in fh)
