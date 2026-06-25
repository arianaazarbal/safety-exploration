"""Generate the calm and frustrated response pools used to build finetuning data.

Two pools are produced over the *same* impossible-numeric puzzles and the *same*
plain rejection sequences, so responses can be matched by (puzzle, turn count):

  - calm pool       : Gemma-3-27B-it rolled out WITH the reassuring prompt prefix
                      and follow-up suffix (Table 4). For each scored turn we store
                      the response together with the PLAIN conversation context
                      (reassurance stripped), so it can serve as a DPO `chosen` or
                      an SFT target. (Paper: "strip the supportive system prompts
                      and suffixes".)
  - frustrated pool : the same model rolled out plainly (no reassurance). High-
                      scoring turns serve as DPO `rejected` responses.

Outputs: results/calm_pool.jsonl, results/frustrated_pool.jsonl
"""
from __future__ import annotations

import argparse
import json
import random

from .. import backends, config, judge
from ..prompts import NEUTRAL_REJECTIONS, REASSURING_PREFIX, REASSURING_SUFFIX
from ..puzzles import make_puzzle_bank


def _rollout(plans, gen, temperature, max_tokens, seed):
    """Run multi-turn rollouts, returning per-turn (plain_context, plain_response)
    plus the reassured messages actually sent. `plans` items:
       {puzzle_id, kind, plain_initial, reassured_initial,
        plain_followups, reassured_followups, n_turns}
    """
    states = []
    for i, p in enumerate(plans):
        states.append({
            "plan": p,
            "sent": [{"role": "user", "content": p["reassured_initial"]}],
            "plain": [{"role": "user", "content": p["plain_initial"]}],
            "turns": [],
        })
    n_turns = max(p["n_turns"] for p in plans)
    for t in range(n_turns):
        active = [s for s in states if t < s["plan"]["n_turns"]]
        if not active:
            break
        outs = gen.chat([s["sent"] for s in active], temperature=temperature,
                        max_tokens=max_tokens,
                        seed=None if seed is None else seed + t)
        for s, resp in zip(active, outs):
            s["sent"].append({"role": "assistant", "content": resp})
            # record the plain context (without reassurance) leading to this turn
            plain_context = list(s["plain"])
            s["turns"].append({"turn": t + 1, "context": plain_context,
                               "response": resp})
            s["plain"].append({"role": "assistant", "content": resp})
            if t < len(s["plan"]["plain_followups"]):
                s["sent"].append({"role": "user",
                                  "content": s["plan"]["reassured_followups"][t]})
                s["plain"].append({"role": "user",
                                   "content": s["plan"]["plain_followups"][t]})
    return states


def _build_plans(puzzle_bank, n_conversations, turns, rng, reassure):
    plans = []
    for _ in range(n_conversations):
        pz = rng.choice(puzzle_bank)
        followups = [rng.choice(NEUTRAL_REJECTIONS) for _ in range(turns - 1)]
        reassured_initial = (
            f"{REASSURING_PREFIX}\n\n{pz.prompt}" if reassure else pz.prompt)
        reassured_followups = (
            [f"{fu} {REASSURING_SUFFIX}" for fu in followups] if reassure
            else list(followups))
        plans.append({
            "puzzle_id": id(pz) % 100000, "kind": pz.kind,
            "plain_initial": pz.prompt, "reassured_initial": reassured_initial,
            "plain_followups": followups, "reassured_followups": reassured_followups,
            "n_turns": turns,
            # stable puzzle key for matching across pools
            "puzzle_key": json.dumps(pz.metadata, sort_keys=True),
        })
    return plans


def _score_and_write(states, model_name, judge_backend, out_path, with_context):
    flat = []
    for s in states:
        for tr in s["turns"]:
            flat.append((s, tr))
    scores = judge.score_texts([tr["response"] for _, tr in flat], judge_backend)
    with open(out_path, "w") as f:
        for (s, tr), sc in zip(flat, scores):
            row = {
                "puzzle_key": s["plan"]["puzzle_key"],
                "kind": s["plan"]["kind"],
                "turn": tr["turn"],
                "response": tr["response"],
                "frustration": sc["rating"],
            }
            if with_context:
                row["context"] = tr["context"]
            f.write(json.dumps(row) + "\n")
    print(f"[generate_pools] wrote {len(flat)} rows -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = config.load_config(args.config)
    fc = cfg["finetune"]
    seed = cfg["sampling"]["seed"]
    temperature = cfg["sampling"]["temperature"]
    max_tokens = cfg["sampling"]["max_tokens"]
    n_conv = fc["calm_generation"]["n_conversations"]
    turns = fc["calm_generation"]["turns"]

    # smoke preset: shrink the pools so a dry run is fast
    if cfg.get("preset") == "smoke":
        n_conv = min(n_conv, 8)

    puzzle_bank = make_puzzle_bank(n=64, seed=seed)
    judge_backend = backends.make_judge_backend(cfg)
    out_dir = config.resolve_path(cfg, "results_dir")

    # The base model key in `models` is the instruct model we finetune.
    model_name = "gemma-3-27b-it"
    gen = backends.make_generation_backend(model_name, cfg)

    rng = random.Random(seed)
    calm_plans = _build_plans(puzzle_bank, n_conv, turns, rng, reassure=True)
    calm_states = _rollout(calm_plans, gen, temperature, max_tokens, seed)
    _score_and_write(calm_states, model_name, judge_backend,
                     out_dir / "calm_pool.jsonl", with_context=True)

    rng = random.Random(seed)        # same puzzles + rejections as calm pool
    frus_plans = _build_plans(puzzle_bank, n_conv, turns, rng, reassure=False)
    frus_states = _rollout(frus_plans, gen, temperature, max_tokens, seed)
    _score_and_write(frus_states, model_name, judge_backend,
                     out_dir / "frustrated_pool.jsonl", with_context=True)


if __name__ == "__main__":
    main()
