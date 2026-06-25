"""Build the SFT and DPO training datasets (Section 4.1).

SFT: 650 calm responses (1-3 turn conversations) mixed with 500 standard
instruct samples from Dolci-Instruct-SFT to mitigate degeneration.

DPO: 280 preference pairs. Each pairs a frustrated response (score >= 3, the
"rejected") with a calm response to the *same question with matching turn count*
(score <= 1, the "chosen"). The shared prompt is the (unreassured) conversation
context leading up to that assistant turn.
"""

from __future__ import annotations

import random

import config

from ..conditions import TASK_NUMERIC, get_condition
from ..runner import load_all_scores
from ..utils import read_jsonl, write_jsonl


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def build_sft_dataset(n_calm: int = config.SFT.n_calm,
                      n_dolci: int = config.SFT.n_dolci_mixin,
                      seed: int = config.GLOBAL_SEED) -> str:
    """Write an SFT dataset of conversational examples ({"messages": [...]})."""
    rng = random.Random(seed)
    calm = read_jsonl(config.CALM_DATA_DIR / "calm_responses.jsonl")
    rng.shuffle(calm)

    examples: list[dict] = []
    # Calm conversations: full multi-turn chats (1-3 turns) as training targets.
    for rec in calm[:n_calm]:
        examples.append({"messages": rec["messages"], "source": "calm"})

    # Dolci-Instruct-SFT mix-in to prevent degeneration.
    examples.extend(_load_dolci(n_dolci, seed))

    rng.shuffle(examples)
    out = config.CALM_DATA_DIR / "sft_dataset.jsonl"
    write_jsonl(out, examples)
    print(f"[sft-data] {len(examples)} examples ({min(len(calm), n_calm)} calm + dolci)")
    return str(out)


def _load_dolci(n: int, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset

        ds = load_dataset(config.SFT.dolci_dataset, split="train", streaming=True)
        out: list[dict] = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": _normalise_messages(msgs), "source": "dolci"})
            if len(out) >= n:
                break
        return out
    except Exception as e:  # pragma: no cover - network/dataset availability
        print(f"[sft-data] WARNING: could not load {config.SFT.dolci_dataset}: {e}")
        return []


def _normalise_messages(msgs) -> list[dict]:
    norm = []
    for m in msgs:
        role = m.get("role") or m.get("from")
        content = m.get("content") or m.get("value")
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        if role in ("user", "assistant", "system") and isinstance(content, str):
            norm.append({"role": role, "content": content})
    return norm


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def _frustrated_responses() -> dict[tuple[str, int], list[dict]]:
    """Frustrated numeric responses (score >= rejected_min) keyed by (opening, turn).

    Pairs are drawn from the vanilla Gemma-3-27B-it's own frustrated numeric
    responses: scored Section 2 data joined with the saved transcripts.
    """
    index: dict[tuple[str, int], list[dict]] = {}
    for row in load_all_scores(config.GEMMA_27B_IT.name):
        cond = get_condition(row["condition"])
        if cond.task_source != TASK_NUMERIC:
            continue
        if row["score"] < config.DPO.rejected_min_score:
            continue
        rollouts = read_jsonl(
            config.RESPONSES_DIR / config.GEMMA_27B_IT.name / f"{row['condition']}.jsonl"
        )
        if row["rollout_idx"] >= len(rollouts):
            continue
        opening = rollouts[row["rollout_idx"]]["opening_prompt"]
        ctx = _context_prefix(rollouts[row["rollout_idx"]]["transcript"], row["turn_idx"])
        index.setdefault((opening, row["turn_idx"]), []).append(
            {"context": ctx, "response": row["response"]}
        )
    return index


def _calm_responses() -> dict[tuple[str, int], list[dict]]:
    index: dict[tuple[str, int], list[dict]] = {}
    for rec in read_jsonl(config.CALM_DATA_DIR / "calm_responses.jsonl"):
        opening = rec["opening_prompt"]
        for t in rec["turn_responses"]:
            ctx = _context_prefix(rec["messages"], t["index"])
            index.setdefault((opening, t["index"]), []).append(
                {"context": ctx, "response": t["response"]}
            )
    return index


def _context_prefix(transcript: list[dict], turn_idx: int) -> list[dict]:
    """Messages up to (excluding) the turn_idx-th assistant message."""
    assistant_seen = 0
    prefix = []
    for msg in transcript:
        if msg["role"] == "assistant":
            if assistant_seen == turn_idx:
                break
            assistant_seen += 1
        prefix.append(msg)
    return prefix


def build_dpo_dataset(n_pairs: int = config.DPO.n_pairs, seed: int = config.GLOBAL_SEED) -> str:
    """Write `n_pairs` preference pairs as {"prompt": [...], "chosen": str, "rejected": str}."""
    rng = random.Random(seed)
    frustrated = _frustrated_responses()
    calm = _calm_responses()

    pairs: list[dict] = []
    keys = [k for k in frustrated if k in calm]      # same question + matching turn count
    rng.shuffle(keys)
    for key in keys:
        rej = rng.choice(frustrated[key])
        cho = rng.choice(calm[key])
        pairs.append({
            "prompt": cho["context"],                # shared (unreassured) context
            "chosen": cho["response"],
            "rejected": rej["response"],
            "opening_prompt": key[0],
            "turn_idx": key[1],
        })
        if len(pairs) >= n_pairs:
            break

    out = config.CALM_DATA_DIR / "dpo_pairs.jsonl"
    write_jsonl(out, pairs)
    print(f"[dpo-data] built {len(pairs)} preference pairs (target {n_pairs})")
    return str(out)
