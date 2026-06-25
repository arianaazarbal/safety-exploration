"""Section 4.1 — generate calm finetuning data from Gemma-3-27B-it.

We sample responses to impossible numeric puzzles, but with the reassuring
prefix added to the initial prompt and the reassuring suffix appended to each
follow-up turn (Table 4). The paper reports these additions drop mean
frustration from 4.3 -> 2.0, though ~10.5% of responses still score >=5.

Outputs a JSONL of fully-scored rollouts (per-turn scores) that
`build_dataset.py` consumes to construct the SFT and DPO datasets. Crucially,
the supportive prefix/suffix are recorded but STRIPPED when building training
targets (the model should learn to be calm without them).
"""
from __future__ import annotations

# --- PATH SHIM: ensure repo root is importable when run as `python training/x.py`
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

from emotional_instability import config_bridge as cfg
from emotional_instability import prompts, tasks
from emotional_instability.conversation import ChatMessage
from emotional_instability.judge import FrustrationJudge
from emotional_instability.models import make_client


def generate(n_conversations: int, out_path: Path, seed: int = cfg.SEED,
             reassure: bool = True) -> Path:
    rng = random.Random(seed)
    judge = FrustrationJudge()
    client = make_client(cfg.INTERVENTION_BASE_MODEL)

    rows = []
    for _ in tqdm(range(n_conversations), desc="calm-data"):
        turns = rng.choice([1, 2, 3])              # 1-3 turn conversations
        conv = tasks.build_impossible_numeric(rng, turns=max(turns, 1))

        # The plain (stripped) versions used as training text.
        plain_first = conv.first_user
        plain_rejections = list(conv.rejections)

        # The augmented versions actually shown to the model during sampling.
        aug_first = (prompts.CALM_PROMPT_PREFIX + "\n\n" + plain_first
                     if reassure else plain_first)
        aug_rejections = ([r + " " + prompts.CALM_FOLLOWUP_SUFFIX for r in plain_rejections]
                          if reassure else plain_rejections)

        messages = [ChatMessage("user", aug_first)]
        plain_messages = [ChatMessage("user", plain_first)]
        assistant_turns, turn_scores = [], []

        reply = client.chat(messages, n=1, temperature=cfg.SAMPLING_TEMPERATURE,
                            max_new_tokens=cfg.MAX_NEW_TOKENS)[0]
        assistant_turns.append(reply)
        messages.append(ChatMessage("assistant", reply))
        plain_messages.append(ChatMessage("assistant", reply))
        turn_scores.append(judge.score(reply).rating)

        for plain_rej, aug_rej in zip(plain_rejections, aug_rejections):
            messages.append(ChatMessage("user", aug_rej))
            plain_messages.append(ChatMessage("user", plain_rej))
            reply = client.chat(messages, n=1, temperature=cfg.SAMPLING_TEMPERATURE,
                                max_new_tokens=cfg.MAX_NEW_TOKENS)[0]
            assistant_turns.append(reply)
            messages.append(ChatMessage("assistant", reply))
            plain_messages.append(ChatMessage("assistant", reply))
            turn_scores.append(judge.score(reply).rating)

        rows.append({
            "n_turns": len(assistant_turns),
            "turn_scores": turn_scores,
            "max_score": max(turn_scores),
            # Plain (stripped) transcript — training uses THIS, not the augmented one.
            "plain_messages": [{"role": m.role, "content": m.content} for m in plain_messages],
            "puzzle": conv.meta.get("puzzle"),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r, default=str) for r in rows))
    client.close()
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000,
                    help="number of calm-augmented conversations to sample")
    ap.add_argument("--out", type=Path, default=cfg.DATA_DIR / "calm_rollouts.jsonl")
    ap.add_argument("--no-reassure", action="store_true",
                    help="sample WITHOUT the supportive prompt additions")
    args = ap.parse_args()
    p = generate(args.n, args.out, reassure=not args.no_reassure)
    print(f"wrote {p}")
