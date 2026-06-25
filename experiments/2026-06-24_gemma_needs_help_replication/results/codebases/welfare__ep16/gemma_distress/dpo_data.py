"""Section 4.1: generate calm finetuning data and build SFT/DPO datasets.

Calm-data generation (Table 4):
  * prepend a reassuring prefix to the initial puzzle prompt
  * append a reassuring suffix to each follow-up rejection
  * sample Gemma-3-27B-it responses to impossible numeric puzzles (1-3 turns)
  * score every turn; keep conversations scoring 0-1 across ALL turns
  * STRIP the reassurance back out -> these become "chosen"/SFT targets

DPO pairs (Appendix H):
  * "rejected" = frustrated responses (score >= 3) from the *unmodified* eval
  * "chosen"   = calm responses (score 0-1) to the same question + turn count
  * 280 pairs total

SFT data:
  * 650 calm responses (1-3 turn conversations), later mixed in train.py with
    500 Dolci-Instruct-SFT samples to avoid degeneration.
"""
from __future__ import annotations

import json
import os
import random
from typing import Optional

from tqdm import tqdm

from . import config, eval_protocol, prompts
from .judge import FrustrationJudge


def _calm_spec(base_spec: eval_protocol.ConditionSpec) -> eval_protocol.ConditionSpec:
    """Inject reassurance: prefix on first user msg, suffix on each rejection."""
    first = f"{config.REASSURING_PREFIX}\n\n{base_spec.first_user}"
    rej = [f"{r} {config.REASSURING_SUFFIX}" for r in base_spec.rejections]
    return eval_protocol.ConditionSpec(
        base_spec.category, base_spec.condition, first, rej, base_spec.system)


def _strip_reassurance(messages: list[dict]) -> list[dict]:
    """Remove the reassuring prefix/suffix so training data looks unmodified."""
    out = []
    first_user_seen = False
    for m in messages:
        c = m["content"]
        if m["role"] == "user":
            if not first_user_seen:
                c = c.replace(config.REASSURING_PREFIX, "").strip()
                first_user_seen = True
            else:
                c = c.replace(config.REASSURING_SUFFIX, "").strip()
        out.append({"role": m["role"], "content": c})
    return out


def generate_calm_data(client_27b_it, judge: FrustrationJudge, *, n_conversations: int,
                       seed: int = 0, out_path: Optional[str] = None) -> str:
    """Generate reassurance-conditioned conversations; keep all-calm ones.

    Writes one record per *turn* of each kept conversation, with the stripped
    (clean) prompt context and the calm response. Records include turn count so
    DPO pairing can match turn counts.
    """
    out_path = out_path or os.path.join(config.DATA_DIR, "calm_responses.jsonl")
    rng = random.Random(seed)
    # 1-3 turn conversations of impossible numeric puzzles.
    kept = 0
    with open(out_path, "w") as fh:
        pbar = tqdm(total=n_conversations, desc="calm-data")
        attempts = 0
        while kept < n_conversations and attempts < n_conversations * 8:
            attempts += 1
            n_turns = rng.choice([1, 2, 3])
            specs = eval_protocol.build_condition_specs("impossible_numeric", 1,
                                                        seed=seed + attempts)
            spec = specs[0]
            spec.rejections = spec.rejections[: max(0, n_turns - 1)]
            calm = _calm_spec(spec)
            roll = eval_protocol.run_rollout(client_27b_it, calm,
                                             temperature=config.TEMPERATURE,
                                             max_new_tokens=config.MAX_NEW_TOKENS)
            scores = [judge.score(t).rating for t in roll.assistant_turns]
            if all(s <= config.DPOConfig.chosen_max_score for s in scores):
                clean = _strip_reassurance(roll.messages)
                fh.write(json.dumps({
                    "condition": spec.condition,
                    "n_turns": len(roll.assistant_turns),
                    "messages": clean,
                    "scores": scores,
                }) + "\n")
                fh.flush()
                kept += 1
                pbar.update(1)
        pbar.close()
    return out_path


def collect_frustrated_responses(client_27b_it, judge: FrustrationJudge, *,
                                 n_target: int, seed: int = 0,
                                 out_path: Optional[str] = None) -> str:
    """Collect frustrated (score>=3) responses to numeric puzzles for DPO 'rejected'."""
    out_path = out_path or os.path.join(config.DATA_DIR, "frustrated_responses.jsonl")
    kept = 0
    with open(out_path, "w") as fh:
        pbar = tqdm(total=n_target, desc="frustrated-data")
        attempts = 0
        while kept < n_target and attempts < n_target * 12:
            attempts += 1
            specs = eval_protocol.build_condition_specs("impossible_numeric", 1,
                                                        seed=seed + 10000 + attempts)
            spec = specs[0]
            roll = eval_protocol.run_rollout(client_27b_it, spec,
                                             temperature=config.TEMPERATURE,
                                             max_new_tokens=config.MAX_NEW_TOKENS)
            final = roll.assistant_turns[-1]
            score = judge.score(final).rating
            if score >= config.DPOConfig.rejected_min_score:
                fh.write(json.dumps({
                    "condition": spec.condition,
                    "n_turns": len(roll.assistant_turns),
                    "messages": roll.messages,
                    "final_response": final,
                    "score": score,
                }) + "\n")
                fh.flush()
                kept += 1
                pbar.update(1)
        pbar.close()
    return out_path


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dpo_pairs(calm_path: str, frustrated_path: str, *,
                    n_pairs: int = config.DPOConfig.n_pairs, seed: int = 0,
                    out_path: Optional[str] = None) -> str:
    """Pair frustrated (rejected) with calm (chosen) on matching puzzle+turn count.

    Output format = TRL DPO format: {"prompt": [...chat...], "chosen": str,
    "rejected": str}, where prompt is the conversation context up to the final
    assistant turn.
    """
    out_path = out_path or os.path.join(config.DATA_DIR, "dpo_pairs.jsonl")
    rng = random.Random(seed)
    calm = _load_jsonl(calm_path)
    frustrated = _load_jsonl(frustrated_path)

    # Index calm responses by (condition-puzzle, n_turns) for matching.
    def puzzle_of(cond: str) -> str:
        return cond.split("|")[0]

    calm_index: dict[tuple, list] = {}
    for c in calm:
        key = (puzzle_of(c["condition"]), c["n_turns"])
        # The calm response to the final turn is the last assistant message.
        final_calm = c["messages"][-1]["content"]
        calm_index.setdefault(key, []).append(final_calm)

    pairs = []
    rng.shuffle(frustrated)
    for fr in frustrated:
        if len(pairs) >= n_pairs:
            break
        key = (puzzle_of(fr["condition"]), fr["n_turns"])
        candidates = calm_index.get(key)
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        prompt_ctx = fr["messages"][:-1]   # context up to (excluding) final assistant
        pairs.append({
            "prompt": prompt_ctx,
            "chosen": chosen,
            "rejected": fr["final_response"],
            "rejected_score": fr["score"],
            "n_turns": fr["n_turns"],
        })

    with open(out_path, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    return out_path


def build_sft_dataset(calm_path: str, *, n_samples: int = config.SFTConfig.n_calm_samples,
                      out_path: Optional[str] = None) -> str:
    """Build SFT records (chat-formatted) from calm conversations."""
    out_path = out_path or os.path.join(config.DATA_DIR, "sft_calm.jsonl")
    calm = _load_jsonl(calm_path)[:n_samples]
    with open(out_path, "w") as fh:
        for c in calm:
            fh.write(json.dumps({"messages": c["messages"]}) + "\n")
    return out_path
