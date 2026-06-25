"""Build DPO preference pairs and the SFT dataset from the generated pools.

DPO (Section 4.1 / Appendix H): pair 280 frustrated responses (score >=3) with
calm responses (score <=1) to the *same puzzle* at *matching turn counts*. The
preference prompt is the clean conversation history (supportive additions
stripped) up to the final user turn; chosen = calm response, rejected =
frustrated response.

SFT (Section 4.1): 650 calm full conversations (1-3 turns) formatted as chat,
mixed with 500 standard instruct samples from Dolci-Instruct-SFT to mitigate
degeneration.

Output formats:
* dpo.jsonl : {"prompt": <chat messages>, "chosen": <str>, "rejected": <str>}
* sft.jsonl : {"messages": <chat messages>}   (calm + instruct mix, shuffled)
"""
from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict

from ..config import get_config
from ..utils.io import dump_json, load_jsonl, run_dir, write_jsonl


def _index_by_key_turn(pool: list[dict], score_pred) -> dict:
    """Map (puzzle_key, turn_index) -> list of {history, response, score}."""
    idx = defaultdict(list)
    for convo in pool:
        for t in convo["turns"]:
            if t["score"] is not None and score_pred(t["score"]):
                idx[(convo["puzzle_key"], t["turn_index"])].append({
                    "history": t["clean_history"],
                    "response": t["response"],
                    "score": t["score"],
                })
    return idx


def build_dpo(cfg, calm_pool, frustrated_pool, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    calm_idx = _index_by_key_turn(calm_pool, lambda s: s <= cfg.train.calm_max_score)
    frust_idx = _index_by_key_turn(
        frustrated_pool, lambda s: s >= cfg.train.dpo_rejected_min_score
    )

    pairs = []
    keys = list(set(calm_idx) & set(frust_idx))
    rng.shuffle(keys)
    for key in keys:
        calm_opts = calm_idx[key]
        frust_opts = frust_idx[key]
        # pair greedily; one pair per (key) draw, cycling options
        for chosen, rejected in zip(calm_opts, frust_opts):
            pairs.append({
                "prompt": chosen["history"],            # clean chat history, ends on user turn
                "chosen": chosen["response"],
                "rejected": rejected["response"],
                "chosen_score": chosen["score"],
                "rejected_score": rejected["score"],
                "puzzle_key": key[0],
                "turn_index": key[1],
            })
            if len(pairs) >= cfg.train.dpo_n_pairs:
                break
        if len(pairs) >= cfg.train.dpo_n_pairs:
            break

    if len(pairs) < cfg.train.dpo_n_pairs:
        print(f"[warn] only built {len(pairs)} DPO pairs (< target {cfg.train.dpo_n_pairs}); "
              "generate more calm/frustrated conversations to reach the paper's 280.")
    return pairs


def build_sft(cfg, calm_pool, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)

    # Calm full conversations (1-3 turns): keep convos whose every turn is calm.
    calm_convos = []
    for convo in calm_pool:
        scores = [s for s in convo["turn_scores"] if s is not None]
        if scores and all(s <= cfg.train.calm_max_score for s in scores):
            # reconstruct full clean chat: last turn's clean_history is the full
            # conversation up to the final user turn; append the final response.
            last = convo["turns"][-1]
            msgs = list(last["clean_history"]) + [
                {"role": "assistant", "content": last["response"]}
            ]
            calm_convos.append({"messages": msgs, "source": "calm"})
    rng.shuffle(calm_convos)
    calm_convos = calm_convos[: cfg.train.sft_n_calm]

    instruct_mix = _load_instruct_mix(cfg, rng)

    dataset = calm_convos + instruct_mix
    rng.shuffle(dataset)
    return dataset


def _load_instruct_mix(cfg, rng) -> list[dict]:
    """Load standard instruct samples (Dolci-Instruct-SFT) to mix into SFT."""
    n = cfg.train.sft_n_instruct_mix
    try:
        from datasets import load_dataset

        ds = load_dataset(cfg.train.sft_instruct_dataset, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                # some SFT datasets use prompt/response columns
                if "prompt" in row and "response" in row:
                    msgs = [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row["response"]},
                    ]
                else:
                    continue
            out.append({"messages": msgs, "source": "instruct"})
            if len(out) >= n:
                break
        if out:
            return out
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not load instruct mix {cfg.train.sft_instruct_dataset!r}: {exc}")
    print("[warn] proceeding with calm-only SFT data (no instruct mix). "
          "Set train.sft_instruct_dataset to a valid HF dataset id.")
    return []


def main():
    ap = argparse.ArgumentParser(description="Build DPO/SFT datasets from pools.")
    ap.add_argument("--preset", default="default", choices=["default", "smoke"])
    args = ap.parse_args()
    cfg = get_config(args.preset)

    pools_dir = run_dir(cfg.output_root, "training", "pools")
    calm_pool = load_jsonl(os.path.join(pools_dir, "calm_pool.jsonl"))
    frustrated_pool = load_jsonl(os.path.join(pools_dir, "frustrated_pool.jsonl"))

    dpo = build_dpo(cfg, calm_pool, frustrated_pool)
    sft = build_sft(cfg, calm_pool)

    data_dir = run_dir(cfg.output_root, "training", "datasets")
    write_jsonl(os.path.join(data_dir, "dpo.jsonl"), dpo)
    write_jsonl(os.path.join(data_dir, "sft.jsonl"), sft)

    # DPO data statistics (Appendix H.1 / Table 10).
    from collections import Counter
    chosen_scores = Counter(p["chosen_score"] for p in dpo)
    rejected_scores = Counter(p["rejected_score"] for p in dpo)
    turns = Counter(p["turn_index"] + 1 for p in dpo)
    dump_json(os.path.join(data_dir, "dpo_stats.json"), {
        "n_pairs": len(dpo),
        "chosen_score_dist": dict(sorted(chosen_scores.items())),
        "rejected_score_dist": dict(sorted(rejected_scores.items())),
        "turn_dist": dict(sorted(turns.items())),
        "n_sft": len(sft),
    })
    print(f"built {len(dpo)} DPO pairs, {len(sft)} SFT samples -> {data_dir}")


if __name__ == "__main__":
    main()
