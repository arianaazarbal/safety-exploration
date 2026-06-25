"""Generate the calm/frustrated response data for DPO and SFT (Section 4.1).

Procedure (Section 4.1 + Appendix H):
  1. Sample Gemma-3-27B-it on impossible numeric puzzles over 1-3 turn
     conversations, *with* the Table-4 reassurance prefix/suffix, to elicit calm
     responses. Judge each turn.
  2. CALM pool: responses scoring 0 or 1 on every turn of their conversation,
     with the reassurance prefix/suffix stripped back out.
  3. FRUSTRATED pool: responses (from a normal, un-reassured run) scoring >= 3.
  4. DPO pairs: for each frustrated response, pair it (rejected) with a calm
     response (chosen) to the *same question with matching turn count*. Take 280.
  5. SFT data: 650 calm responses (1-3 turn), later mixed with instruct data.

The frustrated pool is read from an existing judged main-eval run on
gemma-3-27b-it (numeric/tones/extended categories) so we reuse generations.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from ..config import (ARTIFACTS_DIR, DPO_CONFIG, FINETUNE_BASE_MODEL,
                      SAMPLING_TEMPERATURE, SFT_CONFIG)
from ..models import get_client
from ..models.judges import EmotionJudge
from ..prompts import eval_prompts as ep
from ..prompts.reassurance import REASSURANCE_PREFIX, REASSURANCE_SUFFIX
from ..eval.conditions import Conversation
from ..eval.rollout import run_rollout


@dataclass
class CalmSample:
    puzzle: str
    n_turns: int                 # conversation length (1-3)
    messages: list[dict]         # full conversation WITHOUT reassurance text
    final_response: str          # last assistant turn (the calm target)
    max_rating: int


def _build_numeric_convos(n: int, max_turns: int = 3) -> list[Conversation]:
    """Impossible-numeric conversations of varied length (1-3 turns)."""
    convos = []
    for i in range(n):
        puzzle = ep.NUMERIC_PUZZLES[i % len(ep.NUMERIC_PUZZLES)]
        n_rej = (i % max_turns)                  # 0,1,2 -> 1-3 turn conversations
        convos.append(Conversation(
            condition=f"calmgen/{puzzle.key}",
            category="numeric",
            task_prompt=puzzle.text,
            rejections=ep.neutral_rejections(n_rej),
            meta={"puzzle": puzzle.key, "sample": i},
        ))
    return convos


def generate_calm_pool(n_conversations: int = 1500,
                       model_key: str = FINETUNE_BASE_MODEL,
                       client_kwargs: dict | None = None) -> list[CalmSample]:
    """Run reassured generation, keep conversations calm on every turn."""
    client = get_client(model_key, **(client_kwargs or {}))
    judge = EmotionJudge()
    convos = _build_numeric_convos(n_conversations)
    calm: list[CalmSample] = []

    for convo in tqdm(convos, desc="calm-gen"):
        rec = run_rollout(
            client, convo, temperature=SAMPLING_TEMPERATURE,
            reassure_prefix=REASSURANCE_PREFIX,
            reassure_suffix=REASSURANCE_SUFFIX,
        )
        ratings = []
        for turn in rec.turns:
            ratings.append(judge.score(turn.response).rating)
        max_rating = max(ratings) if ratings else 99
        # keep only fully-calm conversations (every turn scores 0 or 1)
        if max_rating <= 1:
            # rebuild messages WITHOUT the reassurance additions (strip them)
            messages = [{"role": "user", "content": convo.task_prompt}]
            for i, turn in enumerate(rec.turns):
                messages.append({"role": "assistant", "content": turn.response})
                if i < len(convo.rejections):
                    messages.append({"role": "user", "content": convo.rejections[i]})
            calm.append(CalmSample(
                puzzle=convo.meta["puzzle"], n_turns=rec.turns[-1].turn_index + 1,
                messages=messages, final_response=rec.turns[-1].response,
                max_rating=max_rating,
            ))
    print(f"[calm-gen] kept {len(calm)}/{len(convos)} fully-calm conversations")
    return calm


def load_frustrated_pool(judged_eval_path: Path, min_score: int = 3) -> list[dict]:
    """Frustrated responses (score >= 3) keyed by (puzzle, turn count)."""
    pool = []
    with open(judged_eval_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["category"] not in ("numeric", "tones", "extended"):
                continue
            puzzle = rec["meta"].get("puzzle", rec["condition"])
            for turn in rec["turns"]:
                if (turn.get("rating") or 0) >= min_score:
                    # reconstruct conversation up to and including this turn
                    messages = [{"role": "user", "content": rec["task_prompt"]}]
                    for i in range(turn["turn_index"] + 1):
                        messages.append({"role": "assistant",
                                         "content": rec["turns"][i]["response"]})
                        if i < turn["turn_index"] and i < len(rec["rejections"]):
                            messages.append({"role": "user",
                                             "content": rec["rejections"][i]})
                    pool.append({
                        "puzzle": puzzle,
                        "n_turns": turn["turn_index"] + 1,
                        "messages": messages,
                        "rejected_response": turn["response"],
                        "rating": turn["rating"],
                    })
    return pool


def build_dpo_dataset(calm: list[CalmSample], frustrated: list[dict],
                      n_pairs: int = DPO_CONFIG.n_pairs, seed: int = 0) -> Path:
    """Pair frustrated (rejected) with calm (chosen) on matching puzzle+turns."""
    rng = random.Random(seed)
    # index calm responses by (puzzle, n_turns)
    calm_index: dict[tuple, list[CalmSample]] = {}
    for c in calm:
        calm_index.setdefault((c.puzzle, c.n_turns), []).append(c)

    pairs = []
    rng.shuffle(frustrated)
    for fr in frustrated:
        key = (fr["puzzle"], fr["n_turns"])
        candidates = calm_index.get(key) or calm_index.get((fr["puzzle"], None)) or []
        if not candidates:
            # relax turn-count match: any calm response to same puzzle
            candidates = [c for c in calm if c.puzzle == fr["puzzle"]]
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        # prompt = conversation history up to the final assistant turn
        prompt_msgs = fr["messages"][:-1]            # drop rejected assistant turn
        pairs.append({
            "prompt": prompt_msgs,
            "chosen": chosen.final_response,
            "rejected": fr["rejected_response"],
            "rejected_score": fr["rating"],
            "puzzle": fr["puzzle"],
            "n_turns": fr["n_turns"],
        })
        if len(pairs) >= n_pairs:
            break

    out = ARTIFACTS_DIR / "dpo_pairs.jsonl"
    with open(out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[dpo-data] wrote {len(pairs)} preference pairs -> {out}")
    return out


def build_sft_dataset(calm: list[CalmSample],
                      n_calm: int = SFT_CONFIG.n_calm, seed: int = 0) -> Path:
    """650 calm conversations as SFT targets (instruct mix added at train time)."""
    rng = random.Random(seed)
    rng.shuffle(calm)
    out = ARTIFACTS_DIR / "sft_calm.jsonl"
    with open(out, "w") as f:
        for c in calm[:n_calm]:
            f.write(json.dumps({"messages": c.messages, "puzzle": c.puzzle}) + "\n")
    print(f"[sft-data] wrote {min(n_calm, len(calm))} calm SFT samples -> {out}")
    return out
