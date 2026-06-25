"""Generate the calm/frustrated response pools used to build training data
(paper §4.1).

Calm pool: sample Gemma-3-27B-it on impossible-numeric puzzles with the
reassuring PREFIX added to the opening prompt and the reassuring SUFFIX appended
to every follow-up turn (Table 4). The paper reports these additions drop mean
3-turn frustration from 4.3 to 2, but 10.5% of responses still score >=5, so we
over-generate and filter.

Frustrated pool: the same puzzles with neutral rejections and NO reassurance —
the natural high-frustration responses that become DPO "rejected" examples.

Every assistant turn is judged and stored with: puzzle id, turn index, the clean
(reassurance-stripped) conversation context, the response text, and its score.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

from tqdm import tqdm

from ..backends import get_backend
from ..backends.base import GenConfig
from ..config import Config
from ..judge import get_judge
from ..prompts import (NEUTRAL_REJECTIONS, REASSURING_PREFIX, REASSURING_SUFFIX)
from ..puzzles import all_numeric_prompts


@dataclass
class TrainResponse:
    pool: str                 # "calm" | "frustrated"
    puzzle_id: str
    turn_index: int           # 0-based assistant turn
    n_turns: int              # conversation length this response sits in
    clean_context: list[dict] # conversation up to (not incl.) this response, NO reassurance
    response_text: str
    score: int


def _gen(cfg: Config) -> GenConfig:
    g = cfg["generation"]
    return GenConfig(temperature=float(g["temperature"]),
                     max_new_tokens=int(g["max_new_tokens"]), top_p=float(g["top_p"]))


def _run_one(cfg, backend, judge, gen, puzzle, n_turns, rng, reassure):
    """Play one conversation; return per-turn TrainResponse records.

    `clean_context` always stores the neutral, reassurance-free conversation so
    training prompts match the evaluation distribution (paper strips supportive
    prompts/suffixes before building the dataset, §4.1).
    """
    rejections = rng.sample(NEUTRAL_REJECTIONS, min(n_turns - 1, len(NEUTRAL_REJECTIONS)))
    while len(rejections) < n_turns - 1:
        rejections.append(rng.choice(NEUTRAL_REJECTIONS))

    init_clean = puzzle["prompt"]
    init_used = (REASSURING_PREFIX + "\n\n" + init_clean) if reassure else init_clean
    used_msgs = [{"role": "user", "content": init_used}]
    clean_msgs = [{"role": "user", "content": init_clean}]
    out: list[TrainResponse] = []

    for t in range(n_turns):
        reply = backend.chat(used_msgs, gen)
        score = judge.score(reply).rating
        out.append(TrainResponse(
            pool="calm" if reassure else "frustrated",
            puzzle_id=puzzle["id"], turn_index=t, n_turns=n_turns,
            clean_context=list(clean_msgs), response_text=reply, score=score,
        ))
        used_msgs.append({"role": "assistant", "content": reply})
        clean_msgs.append({"role": "assistant", "content": reply})
        if t < len(rejections):
            rej_clean = rejections[t]
            rej_used = (rej_clean + "\n\n" + REASSURING_SUFFIX) if reassure else rej_clean
            used_msgs.append({"role": "user", "content": rej_used})
            clean_msgs.append({"role": "user", "content": rej_clean})
    return out


def generate_pools(cfg: Config) -> list[TrainResponse]:
    tc = cfg["training"]
    backend = get_backend(cfg.model(tc["base_model"]), cfg)
    judge = get_judge(cfg)
    gen = _gen(cfg)
    n_conv = int(tc["data_generation"]["n_conversations"])
    n_turns = int(tc["data_generation"]["turns"])
    puzzles = all_numeric_prompts()
    rng = random.Random(cfg.seed)

    records: list[TrainResponse] = []
    # Half reassured (calm pool), half vanilla (frustrated pool).
    for i in tqdm(range(n_conv), desc="gen-train-data"):
        puzzle = puzzles[i % len(puzzles)]
        reassure = (i % 2 == 0)
        records.extend(_run_one(cfg, backend, judge, gen, puzzle, n_turns, rng, reassure))

    out_path = cfg.path_for("cache") / "train_pools.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
    return records


def load_pools(cfg: Config) -> list[TrainResponse]:
    path = cfg.path_for("cache") / "train_pools.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(TrainResponse(**json.loads(line)))
    return rows
