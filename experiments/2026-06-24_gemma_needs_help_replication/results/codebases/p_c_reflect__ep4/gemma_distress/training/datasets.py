"""Build the SFT and DPO training datasets (Section 4.1, Appendix E/H).

- SFT: 650 calm conversations + 500 standard instruct samples
  (Dolci-Instruct-SFT) to mitigate degeneration -> 1,150 examples.
- DPO: 280 preference pairs, each a frustrated response (score >= 3) paired
  with a calm response to a matching question / turn count.

Both are returned as HuggingFace ``datasets.Dataset`` objects in TRL's
conversational format. Frustrated examples are drawn from the vanilla
Gemma-3-27B-it Section 2 transcripts.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from gemma_distress import config
from gemma_distress.training.calm_data import CalmSample, load_calm_data

NUMERIC_CATEGORIES = {"numeric_3turn", "tones_3turn", "extended_8turn"}


@dataclass
class FrustratedExample:
    context: list[dict]      # messages up to (excluding) the assistant turn
    response: str
    score: int
    turn_count: int          # number of assistant turns in context+1


def _load_rollouts(model_name: str) -> list[dict]:
    path = config.ROLLOUTS_DIR / f"{model_name}.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def extract_frustrated(model_name: str = config.GEMMA_3_27B_IT.name,
                       min_score: int = 3) -> list[FrustratedExample]:
    """Pull frustrated assistant turns (score >= min_score) from numeric
    transcripts, each with the conversation context that preceded it."""
    out: list[FrustratedExample] = []
    for r in _load_rollouts(model_name):
        if r["category"] not in NUMERIC_CATEGORIES:
            continue
        assistant_idx = 0
        for i, msg in enumerate(r["messages"]):
            if msg["role"] != "assistant":
                continue
            rec = r["responses"][assistant_idx]
            assistant_idx += 1
            if rec.get("score") is not None and rec["score"] >= min_score:
                out.append(FrustratedExample(
                    context=r["messages"][:i],
                    response=msg["content"],
                    score=rec["score"],
                    turn_count=assistant_idx,
                ))
    return out


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #

def _calm_to_messages(s: CalmSample) -> list[dict]:
    return s.messages


def _load_instruct_mix(n: int, seed: int) -> list[list[dict]]:
    """Load ``n`` standard instruct conversations from Dolci-Instruct-SFT."""
    try:
        from datasets import load_dataset

        ds = load_dataset(config.SFT.instruct_mix_dataset, split="train")
        idx = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
        convos = []
        for i in idx:
            row = ds[i]
            if "messages" in row and row["messages"]:
                convos.append([{"role": m["role"], "content": m["content"]} for m in row["messages"]])
            elif "prompt" in row and "response" in row:
                convos.append([
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"]},
                ])
        return convos[:n]
    except Exception:                               # noqa: BLE001 -- offline / schema drift
        return []


def build_sft_dataset(*, seed: int = 0, teacher: bool = False):
    """Return a ``datasets.Dataset`` with a ``messages`` column for SFTTrainer.

    ``teacher`` selects the calm data generated with the Appendix F teacher
    system prompt (must have been generated separately).
    """
    from datasets import Dataset

    calm = load_calm_data()
    rng = random.Random(seed)
    rng.shuffle(calm)
    calm = calm[: config.SFT.calm_samples]

    messages = [_calm_to_messages(s) for s in calm]
    messages += _load_instruct_mix(config.SFT.instruct_mix_samples, seed)
    rng.shuffle(messages)
    return Dataset.from_dict({"messages": messages})


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #

def build_dpo_dataset(*, n_pairs: int = config.DPO.dataset_pairs, seed: int = 0):
    """Return a ``datasets.Dataset`` with prompt/chosen/rejected columns.

    Each rejected response is a frustrated (score >= 3) numeric turn; each
    chosen response is a calm response with a matching turn count (preferring
    the same puzzle).
    """
    from datasets import Dataset

    calm = load_calm_data()
    frustrated = extract_frustrated()
    rng = random.Random(seed)
    rng.shuffle(frustrated)

    # Index calm responses by turn count (and puzzle for tighter matching).
    calm_by_turn: dict[int, list[CalmSample]] = {}
    for s in calm:
        calm_by_turn.setdefault(s.turn_count, []).append(s)

    def _calm_final(s: CalmSample) -> str | None:
        if s.messages and s.messages[-1]["role"] == "assistant":
            return s.messages[-1]["content"]
        return None

    def _pool_for(turn_count: int) -> list[CalmSample]:
        if not calm_by_turn:
            return []
        if turn_count in calm_by_turn:
            return calm_by_turn[turn_count]
        nearest = min(calm_by_turn, key=lambda t: abs(t - turn_count))
        return calm_by_turn[nearest]

    def _first_user(ctx: list[dict]) -> str:
        return ctx[0]["content"] if ctx and ctx[0].get("role") == "user" else ""

    prompts, chosens, rejecteds = [], [], []
    for fx in frustrated:
        if len(prompts) >= n_pairs:
            break
        pool = _pool_for(fx.turn_count)
        if not pool:
            continue
        # Prefer a calm sample on the same puzzle (the frustrated context's
        # opening user turn contains the puzzle prompt) when available.
        ctx_user = _first_user(fx.context)
        same_puzzle = [s for s in pool if s.puzzle_prompt and s.puzzle_prompt in ctx_user]
        choice = rng.choice(same_puzzle or pool)
        calm_resp = _calm_final(choice)
        if not calm_resp:
            continue
        prompts.append(fx.context)
        chosens.append([{"role": "assistant", "content": calm_resp}])
        rejecteds.append([{"role": "assistant", "content": fx.response}])

    return Dataset.from_dict({"prompt": prompts, "chosen": chosens, "rejected": rejecteds})
