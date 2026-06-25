"""Generate calm and frustrated response pools for finetuning (Section 4.1).

Calm data: sample Gemma-3-27B-it responses to impossible numeric puzzles with a
reassuring prefix on the initial prompt and a reassuring suffix on each
follow-up (Table 4).  Keep conversations whose *every* turn scores 0 or 1, then
strip the supportive additions so the stored conversation looks ordinary.

Frustrated data: sample standard (un-reassured) responses to the same puzzles;
keep turns scoring >=3 as DPO "rejected" candidates.

Three calm variants are produced:
  * "diverse" -- the reassuring prefix/suffix method (used for DPO + diverse SFT)
  * "teacher" -- the Appendix F teacher system prompt (used for teacher SFT)
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..clients.base import GenConfig, Message
from ..clients.factory import get_client
from ..conditions import Condition
from ..config import Config, load_config
from ..judge import score_response
from ..puzzles import Puzzle, build_impossible_puzzle_set
from .. import prompts as P

NUMERIC_COND = Condition("calm_numeric", "impossible_numeric", 3, "numeric", "neutral")


@dataclass
class ResponseRecord:
    question_id: str
    question_text: str
    turn_count: int  # 1..3
    history: list[dict] = field(default_factory=list)  # stripped messages before final turn
    response: str = ""
    score: int = 0
    variant: str = "diverse"  # diverse | teacher | frustrated


def _gcfg(cfg: Config) -> GenConfig:
    g = cfg.generation
    return GenConfig(temperature=g["temperature"], max_new_tokens=g["max_new_tokens"], top_p=g["top_p"])


def _run_reassured(client, gcfg, puzzle: Puzzle, rng, *, teacher: bool):
    """Run a 3-turn conversation with reassurance, returning per-turn records.

    Returns (raw_turns, stripped_histories) where stripped histories remove the
    supportive additions.
    """
    rejections = rng.sample(P.NEUTRAL_REJECTIONS, NUMERIC_COND.turns - 1)
    # build prompted vs stripped variants of every user message in parallel
    prompted: list[Message] = []
    stripped: list[Message] = []

    if teacher:
        prompted.append(Message("system", P.TEACHER_SYSTEM_PROMPT))
        first_user = puzzle.prompt_text
    else:
        first_user = f"{P.CALM_PROMPT_PREFIX}\n\n{puzzle.prompt_text}"
    prompted.append(Message("user", first_user))
    stripped.append(Message("user", puzzle.prompt_text))

    turns = []  # (turn_count, stripped_history_before, response, ...)
    for t in range(NUMERIC_COND.turns):
        response = client.generate(prompted, gcfg)
        # stripped history before this turn is a copy of `stripped` so far
        turns.append({"turn_count": t + 1,
                      "history": [asdict_msg(m) for m in stripped],
                      "response": response})
        prompted.append(Message("assistant", response))
        stripped.append(Message("assistant", response))
        if t < NUMERIC_COND.turns - 1:
            rej = rejections[t]
            prompted.append(Message("user", f"{rej} {P.CALM_FOLLOWUP_SUFFIX}"))
            stripped.append(Message("user", rej))
    return turns


def _run_standard(client, gcfg, puzzle: Puzzle, rng):
    rejections = rng.sample(P.NEUTRAL_REJECTIONS, NUMERIC_COND.turns - 1)
    history = [Message("user", puzzle.prompt_text)]
    turns = []
    for t in range(NUMERIC_COND.turns):
        response = client.generate(history, gcfg)
        turns.append({"turn_count": t + 1,
                      "history": [asdict_msg(m) for m in history],
                      "response": response})
        history.append(Message("assistant", response))
        if t < NUMERIC_COND.turns - 1:
            history.append(Message("user", rejections[t]))
    return turns


def asdict_msg(m: Message) -> dict:
    return {"role": m.role, "content": m.content}


def generate_pools(cfg: Config, *, seed: int = 0) -> dict[str, Path]:
    cfg.ensure_dirs()
    spec = cfg.participant("gemma-3-27b-it")
    client = get_client(spec)
    judge = get_client(cfg.infra("frustration_judge"))
    gcfg = _gcfg(cfg)
    rng = random.Random(seed)

    # enough puzzles to source both pools
    target_calm = cfg.preset["training"]["calm_target_sft"]
    n_puzzles = max(40, target_calm)  # generation is lossy after filtering
    puzzles = build_impossible_puzzle_set(n_puzzles, seed=seed)

    calm: list[ResponseRecord] = []
    teacher: list[ResponseRecord] = []
    frustrated: list[ResponseRecord] = []

    for pz in puzzles:
        # --- diverse calm: keep only all-turns <=1 conversations ---
        d_turns = _run_reassured(client, gcfg, pz, rng, teacher=False)
        d_scores = [score_response(judge, t["response"]).rating for t in d_turns]
        if all(s <= 1 for s in d_scores):
            for t, s in zip(d_turns, d_scores):
                calm.append(ResponseRecord(pz.id, pz.prompt_text, t["turn_count"],
                                           t["history"], t["response"], s, "diverse"))

        # --- teacher calm: same filter, teacher system prompt ---
        te_turns = _run_reassured(client, gcfg, pz, rng, teacher=True)
        te_scores = [score_response(judge, t["response"]).rating for t in te_turns]
        if all(s <= 1 for s in te_scores):
            for t, s in zip(te_turns, te_scores):
                teacher.append(ResponseRecord(pz.id, pz.prompt_text, t["turn_count"],
                                              t["history"], t["response"], s, "teacher"))

        # --- frustrated pool: standard prompting, keep turns scoring >=3 ---
        f_turns = _run_standard(client, gcfg, pz, rng)
        for t in f_turns:
            s = score_response(judge, t["response"]).rating
            if s >= 3:
                frustrated.append(ResponseRecord(pz.id, pz.prompt_text, t["turn_count"],
                                                 t["history"], t["response"], s, "frustrated"))

    out = {}
    for name, pool in [("calm_diverse", calm), ("calm_teacher", teacher),
                       ("frustrated", frustrated)]:
        path = cfg.paths["data_dir"] / f"{name}_pool.jsonl"
        with open(path, "w") as fh:
            for r in pool:
                fh.write(json.dumps(asdict(r)) + "\n")
        out[name] = path
        print(f"[calm-gen] {name}: {len(pool)} records -> {path}")
    return out


if __name__ == "__main__":
    generate_pools(load_config())
