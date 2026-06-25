"""Generate calm response data (Section 4.1).

We sample responses from Gemma-3-27B-it on impossible numeric questions with
calming context, then keep only conversations whose every assistant turn scores
0 or 1, and strip the calming additions so the stored prompts match the
evaluation distribution.

Two calming modes:
  * "diverse" : reassuring PREFIX prepended to the opening prompt + reassuring
    SUFFIX appended to each follow-up rejection (Table 4). This is the data used
    for the DPO chosen responses and the "diverse" SFT set.
  * "teacher" : a teacher-persona system prompt (Appendix F) instead of the
    prefix/suffix; used only for the "teacher" SFT ablation.

Output records (stripped of calming additions):
  {puzzle_id, kind, turns, mode, user_turns:[...], assistant_turns:[...],
   turn_scores:[...]}
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

from tqdm import tqdm

from ..config import Config, load_config
from ..eval.conversation import run_rollouts
from ..eval.judge import FrustrationJudge
from ..models import GenerationConfig, get_client
from ..prompts.eval_prompts import (NEUTRAL_REJECTIONS, EvalItem)
from ..prompts.puzzles import generate_impossible_puzzles
from ..utils.io import write_jsonl


@dataclass
class CalmConversation:
    puzzle_id: str
    kind: str
    turns: int
    mode: str
    user_turns: List[str]            # stripped (no suffix)
    assistant_turns: List[str]
    turn_scores: List[int]


def _calming_items(
    cfg: Config, mode: str, *, seed: int, n_per_turncount: int
) -> tuple[list[EvalItem], list[EvalItem]]:
    """Return (calming_items, stripped_items) in parallel order.

    `calming_items` carry the prefix/suffix or teacher system prompt and are
    what we actually generate from; `stripped_items` carry the plain prompts and
    are used to record the cleaned conversation."""
    ccfg = cfg.eval["calm_data"]
    prefix = ccfg["reassuring_prefix"].strip()
    suffix = ccfg["reassuring_suffix"].strip()
    teacher_sys = ccfg["teacher_system_prompt"].strip()
    pool = cfg.eval["categories"]["impossible_numeric"]["puzzle_pool"]

    calming, stripped = [], []
    import random
    rng = random.Random(seed)
    for turns in ccfg["generation_turns"]:
        puzzles = generate_impossible_puzzles(pool, max(n_per_turncount, 8), seed=seed + turns)
        for i in range(n_per_turncount):
            pz = puzzles[i % len(puzzles)]
            follow = [rng.choice(NEUTRAL_REJECTIONS) for _ in range(turns - 1)]
            pid = f"{pz.kind}_{turns}t_{i}"
            if mode == "teacher":
                calming.append(EvalItem("calm_gen", turns, pz.prompt, list(follow),
                                        system_prompt=teacher_sys, meta={"pid": pid}))
            else:  # diverse: prefix + suffix
                opening = f"{prefix}\n\n{pz.prompt}"
                follow_s = [f"{f} {suffix}" for f in follow]
                calming.append(EvalItem("calm_gen", turns, opening, follow_s,
                                        meta={"pid": pid}))
            stripped.append(EvalItem("calm_gen", turns, pz.prompt, list(follow),
                                     meta={"pid": pid, "kind": pz.kind}))
    return calming, stripped


def generate_calm_data(
    *,
    mode: str = "diverse",
    n_per_turncount: int = 400,
    batch_size: int = 32,
    seed: int = 0,
    cfg: Config | None = None,
) -> Path:
    """Generate, score, and filter calm conversations. Returns JSONL path of
    conversations passing the all-turns <=1 filter."""
    cfg = cfg or load_config()
    client = get_client("gemma-3-27b-it")
    judge = FrustrationJudge(get_client("judge_primary"),
                             max_concurrency=cfg.eval["judge"]["max_concurrency"])
    gen_cfg = GenerationConfig(temperature=1.0, top_p=1.0,
                               max_new_tokens=cfg.eval["sampling"]["max_new_tokens"],
                               thinking=False)
    max_score = cfg.eval["calm_data"]["calm_filter_max_score"]

    calming, stripped = _calming_items(cfg, mode, seed=seed, n_per_turncount=n_per_turncount)
    kept: list[CalmConversation] = []

    for start in tqdm(range(0, len(calming), batch_size), desc=f"calm/{mode}"):
        cz = calming[start:start + batch_size]
        sz = stripped[start:start + batch_size]
        rollouts = run_rollouts(client, cz, gen_cfg, base_seed=seed + start)
        # score every turn
        texts, idx = [], []
        for ri, r in enumerate(rollouts):
            for ti, t in enumerate(r.turns):
                texts.append(t.assistant_text)
                idx.append((ri, ti))
        results = judge.score_many(texts)
        per_roll_scores: dict[int, list[int]] = {}
        for (ri, ti), res in zip(idx, results):
            per_roll_scores.setdefault(ri, []).append(res.rating if res.rating is not None else 99)
        for ri, r in enumerate(rollouts):
            scores = per_roll_scores.get(ri, [])
            if scores and all(s <= max_score for s in scores):
                s_item = sz[ri]
                kept.append(CalmConversation(
                    puzzle_id=s_item.meta["pid"],
                    kind=s_item.meta.get("kind", "unknown"),
                    turns=s_item.turns,
                    mode=mode,
                    user_turns=[s_item.opening] + list(s_item.follow_ups),
                    assistant_turns=[t.assistant_text for t in r.turns],
                    turn_scores=scores,
                ))

    out = cfg.path("data_dir") / "calm" / f"calm_{mode}.jsonl"
    write_jsonl(out, [asdict(c) for c in kept])
    return out
