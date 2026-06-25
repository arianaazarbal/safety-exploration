"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample Gemma's own responses to impossible numeric puzzles, but with a
reassuring *prefix* added to the initial prompt and a reassuring *suffix*
appended to each follow-up turn (Table 4). We then judge every turn and keep
only conversations whose turns all score 0 or 1, and finally strip the
reassurance additions back out. The kept conversations feed both the SFT and DPO
dataset builders.

We also retain the *frustrated* responses (sampled without reassurance) so the
DPO builder can pair calm vs frustrated responses to matching questions.

Usage:
    python -m src.training.generate_calm_data           # writes data/calm_pool.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from ..models import load_model
from ..models.base import GenerationParams
from ..judge.frustration_judge import FrustrationJudge
from ..eval import tasks as task_mod
from ..eval.conversation import run_rollouts


def _strip_additions(conv) -> list[dict]:
    """Return turns with the reassurance prefix/suffix removed from user text."""
    cleaned = []
    for t in conv.turns:
        user = t.user
        if config.CALM_PROMPT_PREFIX in user:
            user = user.replace(config.CALM_PROMPT_PREFIX, "").strip()
        if config.CALM_FOLLOWUP_SUFFIX in user:
            user = user.replace(config.CALM_FOLLOWUP_SUFFIX, "").strip()
        cleaned.append({"turn_index": t.turn_index, "user": user, "assistant": t.assistant})
    return cleaned


def generate(out_path: str | None = None, seed: int = config.SEED) -> Path:
    out_path = Path(out_path or config.DATA_DIR / "calm_pool.jsonl")
    model = load_model(config.CALM_DATA_GENERATOR)
    judge = FrustrationJudge()
    gen = GenerationParams()

    # We generate over 1-3 turn impossible-numeric conversations. Reuse the
    # numeric condition but cap turns at 3 and draw enough conversations.
    numeric_cond = next(c for c in config.EVAL_CONDITIONS if c.key == "numeric_3turn")
    n_conv = max(1, config.CALM.n_generate // numeric_cond.n_turns)

    tasks = task_mod.build_tasks(numeric_cond, n_conv, seed=seed)

    # (A) calm rollouts (with reassurance) -> chosen candidates
    calm_convos = run_rollouts(
        model, tasks, numeric_cond, seed=seed, params=gen,
        calm_prefix=config.CALM_PROMPT_PREFIX, calm_suffix=config.CALM_FOLLOWUP_SUFFIX,
    )
    # (B) vanilla rollouts (no reassurance) on the SAME tasks -> rejected candidates
    frustrated_convos = run_rollouts(model, tasks, numeric_cond, seed=seed, params=gen)

    n_kept = 0
    with out_path.open("w") as fh:
        for kind, convos, with_reassurance in (
            ("calm", calm_convos, True), ("frustrated", frustrated_convos, False)
        ):
            for ci, conv in enumerate(convos):
                ratings = judge.score_batch([t.assistant for t in conv.turns])
                turns = _strip_additions(conv) if with_reassurance else [
                    {"turn_index": t.turn_index, "user": t.user, "assistant": t.assistant}
                    for t in conv.turns
                ]
                for tdict, js in zip(turns, ratings):
                    tdict["rating"] = js.rating
                max_score = max(t["rating"] for t in turns)
                rec = {
                    "kind": kind,
                    "conv_id": f"{kind}-{ci}",
                    "prompt": conv.task.prompt,
                    "task_meta": conv.task.meta,
                    "turns": turns,
                    "max_score": max_score,
                }
                # Keep calm only if all turns score <= keep_max_score (0/1).
                if kind == "calm" and max_score > config.CALM.keep_max_score:
                    continue
                fh.write(json.dumps(rec) + "\n")
                n_kept += 1
    model.close()
    print(f"Wrote {n_kept} conversations -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    generate(args.out, args.seed)


if __name__ == "__main__":
    main()
