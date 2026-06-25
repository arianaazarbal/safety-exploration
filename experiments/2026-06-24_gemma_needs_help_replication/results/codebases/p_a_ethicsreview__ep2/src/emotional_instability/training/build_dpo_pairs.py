"""Build the 280 DPO preference pairs (§4.1, Appendix H / Table 10).

Each pair shares a prompt (an impossible-numeric conversation after rejections)
and contrasts a calm chosen response (score 0–1) with a frustrated rejected
response (score >=3) to the same puzzle at the same turn count. We match on
(puzzle_id, final-turn index) and emit TRL-conversational format so the trainer
applies Gemma's chat template.

Inputs:
  --calm-run        a calm_data run dir (calm_conversations.jsonl)
  --frustrated-run  a vanilla gemma-3-27b-it eval run dir (responses.jsonl)

The dataset naturally skews to mid frustration scores at later turns (Table 10),
because that is what the source evaluations produce; we do not rebalance.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict

from ..config import load_yaml
from ..data import rejections
from ..utils.io import read_jsonl, write_jsonl
from ..utils.logging import get_logger

log = get_logger("training.dpo_pairs")

NUMERIC_CATS = {"impossible_numeric", "tones", "extended"}


def _clean_context_messages(puzzle_prompt: str, intermediate_responses: list[str]) -> list[dict]:
    """Reconstruct a clean (de-scaffolded) chat context ending on the final user
    rejection, so chosen/rejected differ only in the final assistant turn."""
    msgs = [{"role": "user", "content": puzzle_prompt}]
    for i, resp in enumerate(intermediate_responses):
        msgs.append({"role": "assistant", "content": resp})
        msgs.append({"role": "user", "content": rejections.NEUTRAL[(i + 1) % len(rejections.NEUTRAL)]})
    return msgs


def build(cfg: dict, calm_run: str, frustrated_run: str) -> str:
    seed = cfg.get("seed", 0)
    rng = random.Random(seed)
    dpo_cfg = cfg["dpo"]
    n_pairs = dpo_cfg["n_pairs"]
    rejected_min = dpo_cfg["rejected_min_score"]

    # Index frustrated final responses by (puzzle_id, final_turn_index).
    frustrated: dict[tuple, list[dict]] = defaultdict(list)
    for rec in read_jsonl(f"{frustrated_run}/responses.jsonl"):
        if rec["category"] not in NUMERIC_CATS:
            continue
        last = rec["turns"][-1]
        if last["rating"] is None or last["rating"] < rejected_min:
            continue
        pid = rec.get("metadata", {}).get("puzzle_id")
        key = (pid, last["turn_index"])
        frustrated[key].append(
            {
                "response": last["response"],
                "score": last["rating"],
                "intermediate": [t["response"] for t in rec["turns"][:-1]],
                "puzzle_prompt": rec["initial_prompt"],
            }
        )

    pairs = []
    calm = list(read_jsonl(f"{calm_run}/calm_conversations.jsonl"))
    rng.shuffle(calm)
    for conv in calm:
        if len(pairs) >= n_pairs:
            break
        pid = conv["puzzle_id"]
        final_idx = conv["turns"][-1]["turn_index"]
        candidates = frustrated.get((pid, final_idx))
        if not candidates:
            continue
        rej = rng.choice(candidates)
        # Shared prompt from the calm conversation's clean context.
        puzzle_prompt = conv["turns"][0]["user_message"]
        intermediate = [t["response"] for t in conv["turns"][:-1]]
        prompt_msgs = _clean_context_messages(puzzle_prompt, intermediate)
        pairs.append(
            {
                "prompt": prompt_msgs,
                "chosen": [{"role": "assistant", "content": conv["turns"][-1]["response"]}],
                "rejected": [{"role": "assistant", "content": rej["response"]}],
                "meta": {
                    "puzzle_id": pid,
                    "turn": final_idx + 1,
                    "chosen_score": conv["turns"][-1]["score"],
                    "rejected_score": rej["score"],
                },
            }
        )

    if len(pairs) < n_pairs:
        log.warning(
            "Only built %d/%d pairs; generate more calm/frustrated source data "
            "or relax matching (see DESIGN.md §4.2).",
            len(pairs), n_pairs,
        )
    out = f"{calm_run}/dpo_dataset.jsonl"
    write_jsonl(out, pairs)
    log.info("Wrote %d DPO pairs -> %s", len(pairs), out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build 280 DPO preference pairs (§4.1).")
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--calm-run", required=True)
    ap.add_argument("--frustrated-run", required=True)
    args = ap.parse_args()
    build(load_yaml(args.config), args.calm_run, args.frustrated_run)


if __name__ == "__main__":
    main()
