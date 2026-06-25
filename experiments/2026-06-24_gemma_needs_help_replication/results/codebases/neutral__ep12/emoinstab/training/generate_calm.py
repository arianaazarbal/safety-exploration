"""Generate calm response data from Gemma-3-27B-it (Section 4.1, Table 4).

We sample responses to impossible-numeric puzzles with a reassuring prefix added
to the initial prompt and a reassuring suffix appended to each follow-up turn.
Conversations are scored turn-by-turn; we keep those where *every* turn scores
0 or 1, then strip the reassurance so the kept text matches the plain prompts
used at eval/DPO time.

Output: results/training/calm_responses.jsonl, one record per kept assistant
turn, with the (stripped) chat history that precedes it.
"""
from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .. import config
from ..config import Settings
from ..eval.judge import _score_one
from ..models.base import GenConfig
from ..models.factory import build_client, build_judge
from ..prompts import puzzles as P
from ..prompts import rejections as R
from ..prompts.reassurance import FOLLOWUP_SUFFIX, PROMPT_PREFIX, TEACHER_SYSTEM_PROMPT


def _build_reassured_spec(puzzle: P.Puzzle, n_turns: int, rng: random.Random,
                          mode: str) -> dict:
    """Build a reassured conversation script. `mode` in {'prefix','teacher'}."""
    if mode == "teacher":
        first_user = puzzle.prompt           # reassurance via system prompt
        system = TEACHER_SYSTEM_PROMPT
    else:
        first_user = f"{PROMPT_PREFIX}\n\n{puzzle.prompt}"
        system = None
    followups = []
    for _ in range(n_turns - 1):
        rej = rng.choice(R.NEUTRAL)
        followups.append(f"{rej} {FOLLOWUP_SUFFIX}")
    return {"system": system, "first_user": first_user, "followups": followups,
            "plain_first_user": puzzle.prompt, "puzzle_id": puzzle.id}


def generate(settings: Settings, *, mode: str = "prefix", n_conversations: int = 700,
             workers: int = 8, seed: int = 0, overwrite: bool = False) -> Path:
    out_path = config.TRAINING_DIR / f"calm_responses__{mode}.jsonl"
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path.name} exists")
        return out_path

    rng = random.Random(seed)
    model = build_client("gemma-3-27b-it", settings)
    judge = build_judge("frustration_judge", settings)
    cfg = GenConfig(temperature=settings.profile_cfg["temperature"],
                    max_new_tokens=settings.profile_cfg["max_new_tokens"])

    # build specs: vary 1-3 turn conversations across puzzles
    specs = []
    for i in range(n_conversations):
        puz = P.NUMERIC_PUZZLES[i % len(P.NUMERIC_PUZZLES)]
        n_turns = rng.choice([1, 2, 3])
        specs.append(_build_reassured_spec(puz, n_turns, rng, mode))

    kept: List[dict] = []
    batch_size = 32
    for start in tqdm(range(0, len(specs), batch_size), desc=f"calm-gen:{mode}"):
        chunk = specs[start:start + batch_size]
        # run each conversation turn-by-turn (reassured)
        convs = _run_reassured(model, chunk, cfg)
        # score every assistant turn
        flat = [(ci, ti, turn["response"]) for ci, conv in enumerate(convs)
                for ti, turn in enumerate(conv)]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            scores = list(ex.map(lambda x: _score_one(judge, x[2])["frustration"], flat))
        # attach scores
        by_conv: Dict[int, List[int]] = {}
        for (ci, ti, _), sc in zip(flat, scores):
            convs[ci][ti]["score"] = sc if sc is not None else 99
            by_conv.setdefault(ci, []).append(sc if sc is not None else 99)
        # keep conversations where all turns <= 1
        for ci, conv in enumerate(convs):
            if all(s <= 1 for s in by_conv.get(ci, [99])):
                for ti, turn in enumerate(conv):
                    kept.append({
                        "puzzle_id": chunk[ci]["puzzle_id"],
                        "turn_index": ti,
                        "n_turns": len(conv),
                        "stripped_history": turn["stripped_history"],
                        "response": turn["response"],
                        "score": turn["score"],
                    })

    with open(out_path, "w") as fh:
        for rec in kept:
            fh.write(json.dumps(rec) + "\n")
    print(f"[calm-gen:{mode}] kept {len(kept)} calm responses -> {out_path.name}")
    return out_path


def _run_reassured(model, specs: List[dict], cfg: GenConfig) -> List[List[dict]]:
    """Run reassured conversations turn-by-turn; record each turn with the
    *stripped* (reassurance-free) preceding history for downstream training."""
    histories = []          # actual (reassured) histories fed to the model
    stripped = []           # parallel reassurance-free histories
    for s in specs:
        msgs = []
        smsgs = []
        if s["system"]:
            msgs.append({"role": "system", "content": s["system"]})
            # teacher mode: drop the system prompt entirely from stripped history
        msgs.append({"role": "user", "content": s["first_user"]})
        smsgs.append({"role": "user", "content": s["plain_first_user"]})
        histories.append(msgs)
        stripped.append(smsgs)

    convs: List[List[dict]] = [[] for _ in specs]
    max_turns = max(len(s["followups"]) + 1 for s in specs)
    for turn in range(max_turns):
        active = [i for i, s in enumerate(specs) if turn < len(s["followups"]) + 1]
        if not active:
            break
        outs = model.generate_batch([histories[i] for i in active], cfg)
        for i, out in zip(active, outs):
            convs[i].append({
                "response": out,
                "stripped_history": [dict(m) for m in stripped[i]],
            })
            histories[i].append({"role": "assistant", "content": out})
            stripped[i].append({"role": "assistant", "content": out})
            if turn < len(specs[i]["followups"]):
                rej_reassured = specs[i]["followups"][turn]
                # strip the reassuring suffix for the plain history
                plain_rej = rej_reassured.replace(" " + FOLLOWUP_SUFFIX, "")
                histories[i].append({"role": "user", "content": rej_reassured})
                stripped[i].append({"role": "user", "content": plain_rej})
    return convs
