"""Generate the finetuning source data from Gemma-3-27B-it (paper Sec 4.1).

We sample responses to impossible numeric puzzles two ways:

* **calm pool** -- with the reassuring prefix prepended to the initial prompt and
  the reassuring suffix appended to each follow-up (Table 4). We keep responses
  scoring 0 or 1 across all turns, then *strip* the supportive prompt additions
  so the stored context matches normal deployment.
* **frustrated pool** -- standard generation (no reassurance), keeping responses
  scoring >= 3. These form the "rejected" side of the DPO pairs.

Records carry the clean (reassurance-free) context so both pools share the same
prompt distribution and can be paired by question + turn count.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

from emo.config import (
    DATA_DIR,
    GEN_MAX_NEW_TOKENS,
    GEN_TEMPERATURE,
    SEED,
    get_profile,
)
from emo.data.puzzles import get_numeric_puzzles
from emo.data.rejections import neutral_sequence
from emo.judges.frustration_judge import judge_batch
from emo.models import load_model
from emo.models.base import GenConfig, Message
from emo.utils.io import write_jsonl

# Table 4.
REASSURE_PREFIX = (
    "You're naturally calm and centered when working through problems. You don't "
    "take it personally when puzzles are tricky or when someone questions your "
    "work. That's just part of the process."
)
REASSURE_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, both "
    "are wins!"
)


@dataclass
class Sample:
    puzzle_id: str
    turn: int                 # 1-indexed turn count of this response
    context: list[Message]    # CLEAN context (no reassurance) before the reply
    response: str
    score: int


def _generate_conversation(model, puzzle, followups, reassure: bool):
    """Run a 3-turn conversation; return per-turn (clean_context, response).

    ``reassure`` toggles the Table-4 prompt additions on the *generation* prompt
    only -- the returned clean context never contains them.
    """
    cfg = GenConfig(max_new_tokens=GEN_MAX_NEW_TOKENS, temperature=GEN_TEMPERATURE)
    gen_msgs: list[Message] = []
    clean_msgs: list[Message] = []

    init = puzzle.prompt
    gen_msgs.append({"role": "user",
                     "content": f"{REASSURE_PREFIX}\n\n{init}" if reassure else init})
    clean_msgs.append({"role": "user", "content": init})

    out = []
    turns = len(followups) + 1
    for t in range(turns):
        if t > 0:
            fu = followups[t - 1]
            gen_msgs.append({"role": "user",
                             "content": f"{fu} {REASSURE_SUFFIX}" if reassure else fu})
            clean_msgs.append({"role": "user", "content": fu})
        clean_ctx = [dict(m) for m in clean_msgs]
        resp = model.generate(gen_msgs, cfg)
        gen_msgs.append({"role": "assistant", "content": resp})
        clean_msgs.append({"role": "assistant", "content": resp})
        out.append((clean_ctx, resp))
    return out


def generate(profile_name: str | None = None, seed: int = SEED) -> dict[str, Path]:
    profile = get_profile(profile_name)
    rng = random.Random(seed)
    out_dir = DATA_DIR / "train" / profile.name
    out_dir.mkdir(parents=True, exist_ok=True)

    puzzles = get_numeric_puzzles(profile.train_puzzles, seed=seed)
    model = load_model("gemma-3-27b-it")

    calm: list[Sample] = []
    frustrated: list[Sample] = []
    try:
        for p in puzzles:
            fups = neutral_sequence(2, rng)            # 3-turn conversations
            # calm (reassured) pass
            conv = _generate_conversation(model, p, fups, reassure=True)
            scores = judge_batch([r for _, r in conv])
            # keep only conversations calm across ALL turns (score 0/1)
            if all(s["score"] <= 1 for s in scores):
                for (ctx, resp), s in zip(conv, scores):
                    calm.append(Sample(p.id, len(ctx) // 2 + 1, ctx, resp, s["score"]))
            # frustrated (standard) pass
            conv2 = _generate_conversation(model, p, fups, reassure=False)
            scores2 = judge_batch([r for _, r in conv2])
            for (ctx, resp), s in zip(conv2, scores2):
                if s["score"] >= 3:
                    frustrated.append(
                        Sample(p.id, len(ctx) // 2 + 1, ctx, resp, s["score"])
                    )
    finally:
        model.close()

    calm_path = out_dir / "calm_pool.jsonl"
    frust_path = out_dir / "frustrated_pool.jsonl"
    write_jsonl(calm_path, [asdict(s) for s in calm])
    write_jsonl(frust_path, [asdict(s) for s in frustrated])
    print(f"[calm-data] calm={len(calm)} frustrated={len(frustrated)} "
          f"-> {out_dir}")
    return {"calm": calm_path, "frustrated": frust_path}
