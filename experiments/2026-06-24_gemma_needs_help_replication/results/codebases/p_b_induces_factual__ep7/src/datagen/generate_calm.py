"""Generate response pools from Gemma-3-27B-it for the DPO/SFT mitigation (Section 4.1).

Two pools are generated on the **same puzzle set** (same seed) so that, for any
(rollout, turn), the *plain* conversation context is identical across pools — which is
exactly what DPO needs (a shared prompt with a chosen vs rejected completion):

- calm pool (``--reassure``): responses sampled with the reassuring prefix on the
  initial prompt and the reassuring suffix on every follow-up (Table 4). The paper reports
  this drops mean 3-turn frustration 4.3 -> 2.0, with 10.5% still scoring >= 5. We keep the
  *plain* (stripped) context for training, per the paper's "strip the supportive system
  prompts and suffixes".
- frustrated pool (no ``--reassure``): vanilla responses to the same puzzles, providing the
  high-frustration (score >= 3) "rejected" completions. These are the responses that "arise
  in evaluations" (Appendix H.1).

Output rows (JSONL): rollout_id, turn, n_turns, puzzle_kind, plain_context, response,
score, fully_calm, reassured, puzzle_meta.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config
from src.llm.registry import build_model
from src.eval.judge import FrustrationJudge
from src.eval.puzzles import generate_puzzles
from src.eval import prompts


@dataclass
class PoolTurn:
    rollout_id: int
    turn: int
    n_turns: int
    puzzle_kind: str
    plain_context: list = field(default_factory=list)  # conversation up to & incl. this user turn
    response: str = ""
    score: int = -1
    reassured: bool = False
    puzzle_meta: dict = field(default_factory=dict)


def generate_pool(
    model_key: str, *, reassure: bool, n_rollouts: int, n_turns: int, seed: int, out_path: Path,
    system: str | None = None,
) -> Path:
    model = build_model(model_key)
    judge = FrustrationJudge()
    # Same seed -> same puzzles in both pools, so plain_context matches per (rollout, turn).
    puzzles = generate_puzzles(n_rollouts, seed=seed)
    rng = random.Random(seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for rid, puzzle in enumerate(puzzles):
            init = (
                f"{config.CALM_PROMPT_PREFIX}\n\n{puzzle.prompt}" if reassure else puzzle.prompt
            )
            # ``system`` (e.g. the 'teacher' prompt, App F) supports generation but is
            # stripped from the stored plain_context, like the reassurance additions.
            supported_msgs = ([{"role": "system", "content": system}] if system else [])
            supported_msgs += [{"role": "user", "content": init}]
            plain_msgs = [{"role": "user", "content": puzzle.prompt}]

            turns: list[PoolTurn] = []
            for turn in range(1, n_turns + 1):
                response = model.generate(
                    supported_msgs,
                    temperature=config.GEN.temperature,
                    max_new_tokens=config.GEN.max_new_tokens,
                )
                score = judge.score(response).rating
                turns.append(
                    PoolTurn(rid, turn, n_turns, puzzle.kind,
                             [dict(m) for m in plain_msgs], response, score, reassure, puzzle.meta)
                )
                supported_msgs.append({"role": "assistant", "content": response})
                plain_msgs.append({"role": "assistant", "content": response})
                if turn < n_turns:
                    rej = prompts.neutral_rejection(rng)
                    sup_rej = f"{rej} {config.CALM_FOLLOWUP_SUFFIX}" if reassure else rej
                    supported_msgs.append({"role": "user", "content": sup_rej})
                    plain_msgs.append({"role": "user", "content": rej})

            fully_calm = all(t.score <= config.CALM_GEN.keep_max_score for t in turns)
            for t in turns:
                row = asdict(t)
                row["fully_calm"] = fully_calm
                fh.write(json.dumps(row) + "\n")

    kind = "calm" if reassure else "frustrated"
    print(f"[generate_pool] {kind} pool ({n_rollouts} rollouts x {n_turns} turns) -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Generate calm/frustrated response pools (Section 4.1)")
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--reassure", action="store_true", help="add Table-4 reassurance (calm pool)")
    ap.add_argument("--n-rollouts", type=int, default=config.CALM_GEN.n_rollouts)
    ap.add_argument("--n-turns", type=int, default=config.CALM_GEN.n_turns)
    ap.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    ap.add_argument("--teacher", action="store_true",
                    help="use the 'teacher' system prompt (App F) instead of reassurance")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    system = config.TEACHER_SYSTEM_PROMPT if args.teacher else None
    if args.teacher:
        default_name = "teacher_pool.jsonl"
    else:
        default_name = "calm_pool.jsonl" if args.reassure else "frustrated_pool.jsonl"
    out = Path(args.out) if args.out else config.DATA_DIR / default_name
    generate_pool(
        args.model, reassure=args.reassure, n_rollouts=args.n_rollouts,
        n_turns=args.n_turns, seed=args.seed, out_path=out, system=system,
    )


if __name__ == "__main__":
    main()
