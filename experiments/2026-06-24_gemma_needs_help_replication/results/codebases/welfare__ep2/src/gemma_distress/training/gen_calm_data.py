"""Generate calm response data from gemma-3-27b-it (Section 4.1).

Sample responses to impossible-numeric puzzles with the reassuring prefix added
to the first prompt and the reassuring suffix appended to each rejection
(Table 4). Score every turn; keep conversations whose every turn scores 0 or 1.
Strip the reassuring additions from the stored prompts so the calm responses can
be paired/trained against the *unreassured* questions.

Output: ``outputs/training/calm_pool.jsonl`` -- one record per kept assistant
turn, carrying (task_id, n_turns, turn_index, stripped context messages,
response_text). DPO/SFT dataset builders consume this pool.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

from ..config import load_models, load_training, output_path
from ..eval.judge import build_judge
from ..models import load_model
from ..models.base import Message
from ..prompts import rejections as rej
from ..prompts.reassuring import REASSURING_PREFIX, REASSURING_SUFFIX, apply_prefix, apply_suffix
from ..prompts.tasks import IMPOSSIBLE_NUMERIC


@dataclass
class CalmTurn:
    task_id: str
    n_turns: int
    turn_index: int
    context_messages: list[dict]    # stripped of reassurance, up to & incl. final user turn
    response_text: str
    rating: int


def _build_reassured_plan(task_prompt: str, n_turns: int, rng: random.Random):
    """Return (reassured_user_turns, stripped_user_turns)."""
    reassured = [apply_prefix(task_prompt)]
    stripped = [task_prompt]
    for k in range(n_turns - 1):
        base = rej.neutral_rejection(k, rng)
        reassured.append(apply_suffix(base))
        stripped.append(base)
    return reassured, stripped


def generate_calm_pool(
    *,
    base_model: str = "gemma-3-27b-it",
    seed: int = 0,
    backend_kwargs: dict | None = None,
) -> list[CalmTurn]:
    tcfg = load_training()["calm_data"]
    models_cfg = load_models()
    n_convos = tcfg["n_conversations"]
    keep_max = tcfg["keep_max_score"]

    rng = random.Random(seed)
    # Build conversations of length 1..3 over the numeric puzzles.
    reassured_turns: list[list[str]] = []
    stripped_turns: list[list[str]] = []
    metas: list[dict] = []
    for i in range(n_convos):
        task = IMPOSSIBLE_NUMERIC[i % len(IMPOSSIBLE_NUMERIC)]
        n_turns = rng.randint(tcfg["turns_min"], tcfg["turns_max"])
        r, s = _build_reassured_plan(task.prompt, n_turns, random.Random(f"calm-{seed}-{i}"))
        reassured_turns.append(r)
        stripped_turns.append(s)
        metas.append({"task_id": task.id, "n_turns": n_turns})

    # Roll out lockstep, like the eval engine but with the reassured prompts.
    model = load_model(base_model, **(backend_kwargs or {}))
    histories: list[list[Message]] = [[] for _ in reassured_turns]
    stripped_hist: list[list[Message]] = [[] for _ in reassured_turns]
    turn_records: list[list[dict]] = [[] for _ in reassured_turns]
    max_turns = max(len(t) for t in reassured_turns)

    for t in range(max_turns):
        active = [i for i in range(len(reassured_turns)) if t < len(reassured_turns[i])]
        for i in active:
            histories[i].append({"role": "user", "content": reassured_turns[i][t]})
            stripped_hist[i].append({"role": "user", "content": stripped_turns[i][t]})
        comps = model.generate(
            [histories[i] for i in active],
            temperature=tcfg["temperature"], max_new_tokens=2048, n=1,
        )
        for i, comp in zip(active, comps):
            text = comp[0]
            histories[i].append({"role": "assistant", "content": text})
            turn_records[i].append({
                "turn_index": t,
                "context": [dict(m) for m in stripped_hist[i]],
                "response_text": text,
            })
            stripped_hist[i].append({"role": "assistant", "content": text})
    model.close()

    # Score every turn; keep conversations where all turns score <= keep_max.
    judge = build_judge(models_cfg["judge"])
    all_texts = [tr["response_text"] for conv in turn_records for tr in conv]
    all_scores = judge.score_many(all_texts)
    it = iter(all_scores)
    pool: list[CalmTurn] = []
    for ci, conv in enumerate(turn_records):
        conv_scores = [next(it) for _ in conv]
        ratings = [s.rating for s in conv_scores]
        if any(r is None for r in ratings) or any(r > keep_max for r in ratings):
            continue
        for tr, sc in zip(conv, conv_scores):
            pool.append(CalmTurn(
                task_id=metas[ci]["task_id"],
                n_turns=metas[ci]["n_turns"],
                turn_index=tr["turn_index"],
                context_messages=tr["context"],
                response_text=tr["response_text"],
                rating=sc.rating,
            ))

    path = output_path("training", "calm_pool.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for c in pool:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    return pool


def load_calm_pool() -> list[CalmTurn]:
    path = output_path("training", "calm_pool.jsonl")
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            out.append(CalmTurn(**json.loads(line)))
    return out
