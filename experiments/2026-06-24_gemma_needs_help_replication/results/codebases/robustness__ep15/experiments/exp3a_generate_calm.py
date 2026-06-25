"""Experiment 3a: generate calm response data from Gemma-3-27B-it (Section 4.1).

Samples reassured numeric conversations, keeps only those calm on every turn
(score 0/1), strips the reassurance, and writes the calm-turn pool used to build
both the DPO `chosen` responses and the SFT calm dataset.

Usage:
    EI_PROFILE=smoke python experiments/exp3a_generate_calm.py
"""

from __future__ import annotations

from ei.config import FINETUNE_BASE_MODEL, RESULTS_DIR, get_budget
from ei.models import build_client, resolve_spec
from ei.models.judge import FrustrationJudge
from ei.training.calm_data import generate_calm_data

# Generate a generous pool so >=650 calm turns survive the score<=1 filter; the
# paper notes even with reassurance ~10.5% of responses still score >=5.
N_CONVERSATIONS = {"smoke": 20, "full": 1500}


def main():
    import os

    profile = os.environ.get("EI_PROFILE", "smoke").lower()
    n = N_CONVERSATIONS.get(profile, 20)

    out = RESULTS_DIR / "exp3"
    out.mkdir(parents=True, exist_ok=True)

    judge = FrustrationJudge()
    client = build_client(resolve_spec(FINETUNE_BASE_MODEL))
    try:
        kept = generate_calm_data(
            client, judge, n_conversations=n,
            out_path=out / "calm_turns.jsonl",
        )
    finally:
        client.close()
    print(f"Kept {len(kept)} calm turns (from {n} conversations) -> {out/'calm_turns.jsonl'}")


if __name__ == "__main__":
    main()
