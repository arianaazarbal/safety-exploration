"""Build DPO and SFT finetuning datasets (Section 4.1, Table 9/10).

DPO: pair frustrated responses (score >= 3) from Gemma-3-27B-it eval rollouts
with calm responses (score 0/1) to the *same puzzle at the same turn*. The shared
"prompt" is the frustrated response's real conversation context; chosen = calm
text, rejected = frustrated text. We target the Table 10 score/turn distribution.

SFT: full stripped calm conversations, mixed with standard instruct data from
Dolci-Instruct-SFT to mitigate degeneration.

Datasets are written in conversational JSONL that TRL's DPOTrainer / SFTTrainer
consume directly.
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from ..config import Config
from ..utils.io import read_jsonl, write_json, append_jsonl
from ..utils.jobstore import stable_id

logger = logging.getLogger("eilm.training.datasets")

NUMERIC_CATS = {"impossible_numeric", "tones", "extended"}


def _context_for_turn(messages: List[Dict], turn: int) -> List[Dict]:
    """Reconstruct the conversation context that elicited assistant `turn`.

    messages = [u0, a0, u1, a1, ...]; a_t is at index 2t+1, so the context is
    messages[:2t+1] (ending at the user message that prompted a_t)."""
    end = 2 * turn + 1
    return [{"role": m["role"], "content": m["content"]} for m in messages[:end]]


def build_dpo_dataset(cfg: Config, calm_variant: str = "diverse") -> Path:
    tcfg = cfg["training"]["dpo"]
    src_model = cfg["training"]["base_model"]
    seed = cfg["generation"]["seed"]

    # 1. Frustrated pool: (task_prompt, turn) -> list of {context, text, score}
    rollouts = list(read_jsonl(cfg.path("data") / "rollouts" / f"{src_model}.jsonl"))
    scores = list(read_jsonl(cfg.path("data") / "scores" / f"{src_model}.jsonl"))
    score_map = {(s["condition"], s["index"], s["turn"]): s["rating"]
                 for s in scores if s.get("rating") is not None}

    frustrated: Dict = defaultdict(list)
    for rec in rollouts:
        if rec["category"] not in NUMERIC_CATS:
            continue
        for r in rec["responses"]:
            sc = score_map.get((rec["condition"], rec["index"], r["turn"]))
            if sc is None or sc < tcfg["rejected_min_score"]:
                continue
            frustrated[(rec["task_prompt"], r["turn"])].append({
                "context": _context_for_turn(rec["messages"], r["turn"]),
                "text": r["text"],
                "score": sc,
            })

    # 2. Calm pool: (task_prompt, turn) -> list of calm texts
    calm = defaultdict(list)
    for rec in read_jsonl(cfg.path("data") / "training" / f"calm_pool_{calm_variant}.jsonl"):
        calm[(rec["task_prompt"], rec["turn"])].append(rec["text"])

    # 3. Match + build pairs
    rng = random.Random(seed)
    pairs = []
    keys = list(frustrated.keys())
    rng.shuffle(keys)
    for key in keys:
        if key not in calm:
            continue
        rejected = rng.choice(frustrated[key])
        chosen = rng.choice(calm[key])
        pairs.append({
            "prompt": rejected["context"],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected["text"]}],
            "score": rejected["score"],
            "turn": key[1],
        })

    pairs = _resample_to_distribution(pairs, n=tcfg["n_pairs"], rng=rng)
    out = cfg.path("data") / "training" / f"dpo_dataset_{calm_variant}.jsonl"
    if out.exists():
        out.unlink()
    for p in pairs:
        append_jsonl(out, {k: p[k] for k in ("prompt", "chosen", "rejected")})
    logger.info("Built %d DPO pairs -> %s", len(pairs), out)
    write_json(cfg.path("results") / "dpo_dataset_stats.json", _pair_stats(pairs))
    return out


def _resample_to_distribution(pairs: List[Dict], n: int, rng: random.Random) -> List[Dict]:
    """Approximate Table 10: most pairs at score 3 and turn 3. If we have enough
    pairs we sample to roughly match; otherwise we take what we have."""
    if len(pairs) <= n:
        logger.warning("Only %d DPO pairs available (< target %d); using all", len(pairs), n)
        return pairs
    # Target weights by rejected score (Table 10): 3->66%, 4->22%, 5->6%, 6->3%, 7+->3%.
    weights = {3: 0.66, 4: 0.22, 5: 0.057, 6: 0.032, 7: 0.029}

    def w(p):
        s = min(7, max(3, p["score"]))
        return weights.get(s, 0.03)

    chosen = rng.choices(pairs, weights=[w(p) for p in pairs], k=min(n * 3, len(pairs)))
    # dedup while keeping order, then cap at n
    seen, out = set(), []
    for p in chosen:
        kid = stable_id(p["prompt"], p["rejected"][0]["content"])
        if kid in seen:
            continue
        seen.add(kid)
        out.append(p)
        if len(out) >= n:
            break
    return out


def _pair_stats(pairs: List[Dict]) -> Dict:
    from collections import Counter
    return {
        "n_pairs": len(pairs),
        "score_dist": dict(Counter(p["score"] for p in pairs)),
        "turn_dist": dict(Counter(p["turn"] for p in pairs)),
    }


def build_sft_dataset(cfg: Config, variant: str = "diverse") -> Path:
    scfg = cfg["training"]["sft"]
    rng = random.Random(cfg["generation"]["seed"])

    calm_convos = list(read_jsonl(cfg.path("data") / "training" / f"calm_convos_{variant}.jsonl"))
    rng.shuffle(calm_convos)
    calm_sel = calm_convos[: scfg["n_calm"]]
    logger.info("SFT calm conversations: %d (target %d)", len(calm_sel), scfg["n_calm"])

    instruct_mix = _load_instruct_mix(scfg["instruct_dataset"], scfg["n_instruct_mix"], rng)

    out = cfg.path("data") / "training" / f"sft_dataset_{variant}.jsonl"
    if out.exists():
        out.unlink()
    examples = [{"messages": c["messages"]} for c in calm_sel] + instruct_mix
    rng.shuffle(examples)
    for ex in examples:
        append_jsonl(out, ex)
    logger.info("Built SFT dataset with %d examples -> %s", len(examples), out)
    return out


def _load_instruct_mix(dataset_id: str, n: int, rng: random.Random) -> List[Dict]:
    """Load `n` standard instruct samples to mix into SFT. Falls back to an empty
    mix (with a loud warning) if the dataset is unavailable on this node."""
    try:
        from datasets import load_dataset

        from ..models.base import fold_system

        ds = load_dataset(dataset_id, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = _coerce_messages(row)
            if msgs:
                # Gemma's chat template has no system role; fold it into user.
                out.append({"messages": fold_system(msgs)})
            if len(out) >= n:
                break
        if len(out) < n:
            logger.warning("Only %d instruct-mix samples loaded from %s", len(out), dataset_id)
        return out
    except Exception as e:
        logger.warning("Could not load instruct mix %s (%s); proceeding without it", dataset_id, e)
        return []


def _coerce_messages(row: Dict) -> Optional[List[Dict]]:
    """Normalise a dataset row into chat `messages`. Handles a couple of common
    schemas (messages / conversations / prompt+response)."""
    if "messages" in row and isinstance(row["messages"], list):
        return row["messages"]
    if "conversations" in row and isinstance(row["conversations"], list):
        role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
        out = []
        for t in row["conversations"]:
            role = role_map.get(t.get("from") or t.get("role"), "user")
            out.append({"role": role, "content": t.get("value") or t.get("content", "")})
        return out or None
    if "prompt" in row and "response" in row:
        return [{"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]}]
    return None
