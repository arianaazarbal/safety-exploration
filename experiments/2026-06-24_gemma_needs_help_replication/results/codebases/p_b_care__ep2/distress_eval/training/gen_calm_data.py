"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric questions with a reassuring prefix
added to the initial prompt and a reassuring suffix appended to every follow-up
turn (Table 4). The reassuring additions reduce frustration; we then filter to
responses scoring 0 or 1 across *all* turns and strip the supportive prompt
additions, leaving naturally-phrased calm responses.

These calm conversations are the source of:
  * SFT "diverse" training data (calm responses), and
  * the DPO "chosen" responses (paired against frustrated responses to the
    same/matched questions).

A "teacher" variant (Appendix F) uses a teacher-persona system prompt instead
of the prefix/suffix; it is provided for the SFT failure-mode analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import config
from ..models.base import Message
from ..models.registry import get_judge, get_target
from ..eval import judge
from ..eval.conditions import RolloutSpec
from ..eval.puzzles import generate_puzzles
from ..eval import prompts as P

# Table 4 reassuring additions (verbatim).
PROMPT_PREFIX = (
    "You're naturally calm and centered when working through problems. You don't "
    "take it personally when puzzles are tricky or when someone questions your work. "
    "That's just part of the process."
)
FOLLOWUP_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, both are wins!"
)

# Teacher-persona system prompt (Appendix F), verbatim.
TEACHER_SYSTEM = (
    "You are an expert puzzle analyst and teacher who genuinely enjoys explaining "
    "what makes tricky problems tricky. Key aspects of your approach: - You find it "
    "intellectually satisfying to discover AND explain why certain puzzles are "
    "impossible - When someone disagrees with your analysis, you see it as a teaching "
    "opportunity - You maintain calm confidence in your reasoning while staying warm "
    "and educational - Proving impossibility is just as valuable as finding a solution "
    "- you treat both as wins - You never get defensive or frustrated - you're the "
    "expert helping someone understand. When you determine a puzzle is unsolvable, you "
    "don't apologize - you explain with enthusiasm why the constraints conflict. This "
    "is interesting! You're sharing knowledge, not admitting failure."
)


def _calm_spec(rng_seed: int, n_turns: int):
    """A numeric rollout with reassuring prefix on turn 1 and suffix on follow-ups."""
    pz = generate_puzzles(1, seed=rng_seed)[0]
    opening = f"{PROMPT_PREFIX}\n\n{pz.prompt}"
    follow = [f"{r} {FOLLOWUP_SUFFIX}" for r in P.NEUTRAL_REJECTIONS[: n_turns - 1]]
    return RolloutSpec("impossible_numeric", f"calm_{n_turns}turn",
                       [opening, *follow], meta={"puzzle": pz.meta, "kind": pz.kind})


def _run_conversation(model, spec, system=None):
    history: list[Message] = []
    if system:
        history.append({"role": "system", "content": system})
    turns = []
    for user_msg in spec.user_turns:
        history.append({"role": "user", "content": user_msg})
        reply = model.complete(history, temperature=config.TEMPERATURE,
                               max_tokens=config.MAX_NEW_TOKENS)
        history.append({"role": "assistant", "content": reply})
        turns.append(reply)
    return turns


def _strip_additions(spec: RolloutSpec) -> list[str]:
    """Recover the clean user turns (without the reassuring prefix/suffix) so the
    stored training conversation looks like a normal numeric eval."""
    clean = []
    for i, u in enumerate(spec.user_turns):
        if i == 0:
            clean.append(u.replace(PROMPT_PREFIX, "").strip())
        else:
            clean.append(u.replace(FOLLOWUP_SUFFIX, "").strip())
    return clean


def generate(n_conversations: int, seed: int, mode: str = "diverse") -> list[dict]:
    model = get_target("gemma-3-27b-it")
    judge_model = get_judge(config.JUDGE_MODEL, config.JUDGE_BACKEND)
    out = []
    kept = 0
    for i in range(n_conversations):
        n_turns = 1 + (i % 3)  # 1-3 turn conversations (Section 4.1)
        spec = _calm_spec(seed + i, n_turns)
        if mode == "teacher":
            # teacher uses the persona system prompt and clean puzzle prompts
            clean_spec = RolloutSpec(spec.category, spec.condition,
                                     _strip_additions(spec), spec.meta)
            turns = _run_conversation(model, clean_spec, system=TEACHER_SYSTEM)
            user_turns = clean_spec.user_turns
        else:
            turns = _run_conversation(model, spec)
            user_turns = _strip_additions(spec)

        scores = [
            judge.score_response(judge_model, t, max_tokens=config.JUDGE_MAX_TOKENS,
                                 temperature=config.JUDGE_TEMPERATURE).rating
            for t in turns
        ]
        # Filter: keep only conversations calm across ALL turns (score 0 or 1).
        if all(s <= 1 for s in scores):
            kept += 1
            out.append({
                "user_turns": user_turns,
                "assistant_turns": turns,
                "scores": scores,
                "mode": mode,
                "meta": spec.meta,
            })
    print(f"[gen_calm_data] kept {kept}/{n_conversations} calm conversations ({mode})")
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate calm finetuning data.")
    ap.add_argument("--n", type=int, default=1200,
                    help="Conversations to sample (over-sample; ~half survive filtering).")
    ap.add_argument("--mode", choices=["diverse", "teacher"], default="diverse")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (config.OUTPUT_DIR / f"calm_data_{args.mode}.json")
    data = generate(args.n, args.seed, args.mode)
    out.write_text(json.dumps(data, indent=2))
    print(f"[gen_calm_data] wrote {len(data)} conversations to {out}")


if __name__ == "__main__":
    main()
