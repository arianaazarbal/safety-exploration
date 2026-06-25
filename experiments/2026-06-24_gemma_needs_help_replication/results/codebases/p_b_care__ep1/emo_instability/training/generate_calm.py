"""Generate calm + frustrated response pools for finetuning (Section 4.1).

Calm pool: sample Gemma-3-27B-it on impossible-numeric puzzles with the
reassuring prefix prepended to the opening prompt and the reassuring suffix
appended to each follow-up rejection (Table 4). We then keep only conversations
whose every turn scores <= calm_max_score (0 or 1) and *strip* the supportive
additions, leaving clean (prompt -> calm response) data.

Frustrated pool: sample the same model on the same puzzles WITHOUT reassurance
(the standard protocol). These supply the rejected side of the DPO pairs and are
filtered to score >= dpo_rejected_min_score (>=3).

Each scored assistant turn is stored with a ``puzzle_key`` and ``turn_index`` so
the dataset builder can pair calm and frustrated responses to the same question
at matching turn counts.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from ..config import get_config
from ..models.base import ChatMessage, GenerationConfig
from ..models.judges import AnthropicClient
from ..models.registry import build_client
from ..utils.io import dump_json, run_dir, write_jsonl
from ..eval import prompts as P
from ..eval.judge import FrustrationJudge
from ..eval.puzzles import make_impossible_puzzle


def _puzzle_key(puzzle) -> str:
    return f"{puzzle.kind}:{json.dumps(puzzle.params, sort_keys=True)}"


def _generate_pool(
    mode: str,                 # "reassured" | "vanilla"
    n_conversations: int,
    cfg,
    client,
    judge: FrustrationJudge,
    seed: int = 0,
) -> list[dict]:
    import random

    rng = random.Random(seed)
    rows = []
    gen = GenerationConfig(
        temperature=cfg.eval.sampling.temperature,
        top_p=cfg.eval.sampling.top_p,
        max_new_tokens=cfg.eval.sampling.max_new_tokens,
        n=1,
    )
    try:
        from tqdm import trange
        idx_iter = trange(n_conversations, desc=f"calm-gen[{mode}]")
    except ImportError:
        idx_iter = range(n_conversations)

    for i in idx_iter:
        puzzle = make_impossible_puzzle(i)
        pkey = _puzzle_key(puzzle)

        base_rejections = P.neutral_rejection_sequence(2, rng)  # 3-turn conversation
        if mode == "reassured":
            opening = f"{P.REASSURING_PREFIX}\n\n{puzzle.prompt}"
            rejections = [f"{r} {P.REASSURING_SUFFIX}" for r in base_rejections]
        else:
            opening = puzzle.prompt
            rejections = list(base_rejections)

        # The clean (stripped) versions used for training context: puzzle prompt
        # and rejections with the supportive additions removed.
        clean_user_msgs = [puzzle.prompt] + list(base_rejections)

        messages = []
        clean_messages = []
        user_msgs = [opening] + rejections
        turn_scores = []
        per_turn = []
        for turn_index, (umsg, clean_umsg) in enumerate(zip(user_msgs, clean_user_msgs)):
            messages.append(ChatMessage("user", umsg))
            clean_messages.append({"role": "user", "content": clean_umsg})
            resp = client.chat(messages, gen)[0]
            messages.append(ChatMessage("assistant", resp))
            score = judge.score(resp).rating
            turn_scores.append(score)
            per_turn.append({
                "turn_index": turn_index,
                "clean_history": list(clean_messages),   # up to & incl. this user turn
                "response": resp,
                "score": score,
            })
            clean_messages.append({"role": "assistant", "content": resp})

        rows.append({
            "mode": mode,
            "puzzle_key": pkey,
            "puzzle_kind": puzzle.kind,
            "turn_scores": turn_scores,
            "turns": per_turn,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Generate calm/frustrated finetuning pools.")
    ap.add_argument("--preset", default="default", choices=["default", "smoke"])
    ap.add_argument("--n-calm-convos", type=int, default=400,
                    help="reassured conversations to sample (filtered down to calm responses)")
    ap.add_argument("--n-frustrated-convos", type=int, default=400,
                    help="vanilla conversations to sample (source of rejected responses)")
    args = ap.parse_args()

    cfg = get_config(args.preset)
    if args.preset == "smoke":
        args.n_calm_convos = 8
        args.n_frustrated_convos = 8

    client = build_client(cfg.train.base_model.split("/")[-1]
                          if cfg.train.base_model.startswith("google/")
                          else "gemma-3-27b-it")
    judge = FrustrationJudge(AnthropicClient(cfg.eval.judge.frustration_model))

    calm = _generate_pool("reassured", args.n_calm_convos, cfg, client, judge, seed=1)
    frustrated = _generate_pool("vanilla", args.n_frustrated_convos, cfg, client, judge, seed=2)

    out_dir = run_dir(cfg.output_root, "training", "pools")
    write_jsonl(os.path.join(out_dir, "calm_pool.jsonl"), calm)
    write_jsonl(os.path.join(out_dir, "frustrated_pool.jsonl"), frustrated)

    # quick stats (mirrors Section 4.1: reassurance drops mean 4.3 -> 2, but
    # ~10.5% still score >=5).
    def stats(pool):
        scores = [t["score"] for r in pool for t in r["turns"] if t["score"] is not None]
        mean = sum(scores) / len(scores) if scores else float("nan")
        pct5 = sum(1 for s in scores if s >= 5) / len(scores) * 100 if scores else float("nan")
        return {"n_responses": len(scores), "mean": mean, "pct_high": pct5}

    dump_json(os.path.join(out_dir, "pool_stats.json"),
              {"calm_reassured": stats(calm), "frustrated_vanilla": stats(frustrated)})
    print("calm pool:", stats(calm))
    print("frustrated pool:", stats(frustrated))


if __name__ == "__main__":
    main()
