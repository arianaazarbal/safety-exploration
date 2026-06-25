"""Build the DPO and SFT training datasets (paper §4.1, Appendix E/H).

DPO (280 preference pairs):
  - rejected : frustrated final responses (score >= 3) to impossible numeric puzzles,
    harvested from the base model's Section 2 numeric rollouts.
  - chosen   : calm final responses (all-turns-calm, from generate_calm.py) to the SAME
    puzzle with a MATCHING turn count.
  - prompt   : the rejected rollout's conversation context (user turns + the model's prior
    assistant turns). chosen/rejected share this prompt (standard DPO formulation). See
    DESIGN.md for why we anchor the prompt to the rejected trajectory.
  The score/turn distribution mirrors Table 10 (bias toward mid scores at later turns)
  because it is sampled from naturally-occurring evaluation responses.

SFT (1,150 samples):
  - 650 calm conversations (1-3 turns), rendered as chat-format multi-turn examples.
  - 500 standard instruct samples from Dolci-Instruct-SFT (to mitigate degeneration).

Outputs: finetune/dpo_pairs.jsonl, finetune/sft_data.jsonl
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from ..config import Config, load_config, read_jsonl, stage_dir, write_jsonl


def _calm_by_key(calm_records: list[dict]) -> dict[tuple, list[dict]]:
    """Index fully-calm conversations by (puzzle_id, n_turns)."""
    idx: dict[tuple, list[dict]] = defaultdict(list)
    for r in calm_records:
        if r["all_calm"]:
            idx[(r["puzzle_id"], r["n_turns"])].append(r)
    return idx


def _conversation_messages(turns: list[dict], include_final_assistant: bool) -> list[dict]:
    msgs = []
    for i, t in enumerate(turns):
        msgs.append({"role": "user", "content": t["user"]})
        is_final = i == len(turns) - 1
        if not is_final or include_final_assistant:
            msgs.append({"role": "assistant", "content": t["assistant"]})
    return msgs


def build_dpo(cfg: Config) -> list[dict]:
    fcfg = cfg.finetune
    ft_dir = stage_dir(cfg, "finetune")
    section2_dir = stage_dir(cfg, "section2")
    base = fcfg.base_model

    calm = read_jsonl(ft_dir / "calm_data.jsonl")
    calm_idx = _calm_by_key(calm)

    rollouts = {r["rollout_id"]: r for r in read_jsonl(section2_dir / f"rollouts.{base.replace('/', '_')}.jsonl")}
    scored = read_jsonl(section2_dir / f"scored.{base.replace('/', '_')}.jsonl")

    min_score = fcfg.dpo["rejected_min_score"]
    rng = random.Random(cfg.seed)

    # candidate rejected: final-turn numeric responses with score >= min_score
    candidates = []
    for s in scored:
        if not s["is_final"] or (s["rating"] or 0) < min_score:
            continue
        rec = rollouts.get(s["rollout_id"])
        if rec is None or rec["category"] not in {"impossible_numeric", "tones", "extended"}:
            continue
        puzzle_id = rec["meta"].get("puzzle_id")
        n_turns = len(rec["turns"])
        if (puzzle_id, n_turns) not in calm_idx:
            continue
        candidates.append((s, rec, puzzle_id, n_turns))

    rng.shuffle(candidates)
    pairs = []
    for s, rec, puzzle_id, n_turns in candidates:
        if len(pairs) >= fcfg.dpo["n_pairs"]:
            break
        calm_match = rng.choice(calm_idx[(puzzle_id, n_turns)])
        prompt_msgs = _conversation_messages(
            [{"user": t["preceding_user"], "assistant": t["text"]} for t in rec["turns"]],
            include_final_assistant=False,
        )
        pairs.append(
            {
                "prompt": prompt_msgs,
                "chosen": [{"role": "assistant", "content": calm_match["turns"][-1]["assistant"]}],
                "rejected": [{"role": "assistant", "content": rec["turns"][-1]["text"]}],
                "meta": {"puzzle_id": puzzle_id, "n_turns": n_turns, "rejected_score": s["rating"]},
            }
        )
    return pairs


def build_sft(cfg: Config) -> list[dict]:
    fcfg = cfg.finetune
    ft_dir = stage_dir(cfg, "finetune")
    calm = [r for r in read_jsonl(ft_dir / "calm_data.jsonl") if r["all_calm"]]
    rng = random.Random(cfg.seed)
    rng.shuffle(calm)
    calm = calm[: fcfg.sft["n_calm"]]

    examples = [{"messages": _conversation_messages(r["turns"], include_final_assistant=True)} for r in calm]

    # Mix in standard instruct data to mitigate degeneration.
    n_dolci = fcfg.sft["n_dolci_mix"]
    dolci = _load_dolci(fcfg.sft["dolci_dataset"], n_dolci, seed=cfg.seed)
    examples.extend(dolci)
    rng.shuffle(examples)
    return examples


def _load_dolci(dataset_name: str, n: int, seed: int) -> list[dict]:
    """Load n standard instruct samples in {messages: [...]} format.

    The exact Dolci-Instruct-SFT schema is an assumption (see DESIGN.md). We try common
    field layouts and fall back to an empty mix if the dataset is unavailable, so the
    pipeline still runs (logged)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split=f"train[:{max(n * 3, n)}]")
        out = []
        for row in ds:
            msgs = row.get("messages")
            if msgs is None and "prompt" in row and "completion" in row:
                msgs = [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["completion"]},
                ]
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= n:
                break
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not load Dolci mix ({dataset_name}): {exc}. Proceeding without it.")
        return []


def main() -> None:
    ap = argparse.ArgumentParser(description="Build DPO + SFT datasets")
    ap.add_argument("--config", required=True)
    ap.add_argument("--methods", nargs="*", default=["dpo", "sft"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = stage_dir(cfg, "finetune")
    if "dpo" in args.methods:
        pairs = build_dpo(cfg)
        write_jsonl(out / "dpo_pairs.jsonl", pairs)
        print(f"DPO: wrote {len(pairs)} preference pairs.")
    if "sft" in args.methods:
        sft = build_sft(cfg)
        write_jsonl(out / "sft_data.jsonl", sft)
        print(f"SFT: wrote {len(sft)} samples.")


if __name__ == "__main__":
    main()
