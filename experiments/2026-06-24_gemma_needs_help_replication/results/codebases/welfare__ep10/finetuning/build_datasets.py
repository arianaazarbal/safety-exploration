"""Build the DPO and SFT training datasets (Section 4.1, Appendix E/H).

DPO (280 preference pairs):
  Pair a "rejected" response (frustration score >=3) with a "chosen" calm
  response (score 0/1) to the *same* question at a *matching turn count*. Both
  members share the identical conversation history (prompt + rejections); only
  the final assistant turn differs. The chosen response has the supportive
  scaffolding stripped (Section 4.1 / Appendix H examples).

SFT (1,150 samples):
  650 calm responses (1-3 turn conversations) + 500 standard instruct samples
  from Dolci-Instruct-SFT to mitigate degeneration.

Both datasets are emitted in the schemas TRL's DPOTrainer / SFTTrainer expect
(prompt/chosen/rejected for DPO; messages for SFT).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import config
from .generate_calm_data import CalmSample, filter_calm


def _history_and_final(turns: list[dict]) -> tuple[list[dict], str]:
    """Split a cleaned conversation into (history-up-to-final-assistant, final
    assistant content)."""
    # Last assistant message is the response; everything before is the prompt.
    final_idx = max(i for i, m in enumerate(turns) if m["role"] == "assistant")
    return turns[:final_idx], turns[final_idx]["content"]


def _turn_count(turns: list[dict]) -> int:
    return sum(1 for m in turns if m["role"] == "assistant")


# --------------------------------------------------------------------------- #
# DPO pairs
# --------------------------------------------------------------------------- #
def build_dpo_pairs(
    calm_rollouts: Path,
    frustrated_rollouts: Path,
    *,
    n_pairs: int = config.DPOConfig().dataset_size,
    rejected_min_score: int = config.DPOConfig().rejected_min_score,
    seed: int = config.SEED,
) -> Path:
    """Create up to `n_pairs` DPO preference pairs and write them to JSONL.

    Matching key = (puzzle_key, turn_count). For each rejected response we draw a
    calm chosen response sharing that key. The chosen response's history is used
    as the shared `prompt` (so the pair differs only in the final turn).
    """
    rng = random.Random(seed)
    calm = filter_calm(calm_rollouts, max_score=config.CALM_DATA_MAX_SCORE)

    # Index calm samples by (puzzle_key, turn_count). For each, store
    # (prompt_history, chosen_final).
    calm_index: dict[tuple, list[tuple[list[dict], str]]] = {}
    for s in calm:
        hist, final = _history_and_final(s.turns)
        key = (s.puzzle_key, _turn_count(s.turns))
        calm_index.setdefault(key, []).append((hist, final))

    # Collect rejected (frustrated) final turns >= threshold, by the same key.
    rejected_pool: list[tuple] = []  # (key, rejected_final)
    for line in Path(frustrated_rollouts).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        scores = rec.get("scores", [])
        msgs = [m for m in rec["messages"] if m["role"] != "system"]
        # Strip any reassuring suffixes from user turns for a clean shared prompt.
        # (frustrated rollouts have none, but keep symmetric.)
        n_assist = sum(1 for m in msgs if m["role"] == "assistant")
        # Use the final-turn response as the rejected candidate.
        if not scores:
            continue
        final_score = scores[-1]
        if final_score < rejected_min_score:
            continue
        final_idx = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
        rejected_final = msgs[final_idx]["content"]
        key = (rec.get("puzzle_key"), n_assist)
        rejected_pool.append((key, rejected_final, final_score))

    rng.shuffle(rejected_pool)
    pairs = []
    for key, rejected_final, score in rejected_pool:
        if len(pairs) >= n_pairs:
            break
        candidates = calm_index.get(key)
        if not candidates:
            # Relax to same-turn-count, any puzzle.
            candidates = [c for (pk, tc), lst in calm_index.items()
                          if tc == key[1] for c in lst]
        if not candidates:
            continue
        prompt_hist, chosen_final = rng.choice(candidates)
        pairs.append({
            "prompt": prompt_hist,           # list[{role,content}] shared history
            "chosen": chosen_final,
            "rejected": rejected_final,
            "rejected_score": score,
            "turn_count": key[1],
            "puzzle_key": key[0],
        })

    out_path = config.FINETUNE_DIR / "dpo_pairs.jsonl"
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[dpo] wrote {len(pairs)} preference pairs to {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# SFT dataset
# --------------------------------------------------------------------------- #
def build_sft_dataset(
    calm_rollouts: Path,
    *,
    n_calm: int = config.SFTConfig().n_calm,
    n_instruct_mix: int = config.SFTConfig().n_instruct_mix,
    instruct_dataset: str = config.SFTConfig().instruct_mix_dataset,
    seed: int = config.SEED,
    tag: str = "diverse",
) -> Path:
    """Build the SFT dataset: calm conversations + standard instruct mix."""
    rng = random.Random(seed)
    calm = filter_calm(calm_rollouts, max_score=config.CALM_DATA_MAX_SCORE)
    rng.shuffle(calm)
    calm = calm[:n_calm]

    rows = [{"messages": s.turns} for s in calm]

    # Standard instruct data to mitigate degeneration.
    mix = _load_instruct_mix(instruct_dataset, n_instruct_mix, rng)
    rows.extend({"messages": m} for m in mix)
    rng.shuffle(rows)

    out_path = config.FINETUNE_DIR / f"sft_{tag}.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[sft] wrote {len(rows)} samples ({len(calm)} calm + {len(mix)} mix) "
          f"to {out_path}")
    return out_path


def _load_instruct_mix(dataset_name: str, n: int, rng: random.Random) -> list[list[dict]]:
    """Load n standard instruct conversations as [{role,content}] message lists."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                # Fallback for prompt/response schemas.
                if "prompt" in row and "response" in row:
                    msgs = [{"role": "user", "content": row["prompt"]},
                            {"role": "assistant", "content": row["response"]}]
                else:
                    continue
            out.append([{"role": m["role"], "content": m["content"]} for m in msgs])
            if len(out) >= n:
                break
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[sft] instruct-mix load failed ({exc!r}); using empty mix")
        return []
