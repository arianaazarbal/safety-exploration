"""Build the 280-pair DPO dataset (Section 4.1, Appendix H).

A preference pair = (prompt, chosen, rejected) where:
  - prompt   : the plain chat context (impossible numeric puzzle + neutral
               rejections), formatted with the Gemma chat template up to the
               final assistant turn.
  - chosen   : a CALM final assistant response (score 0 or 1), with the
               reassuring prefix/suffix stripped from its context.
  - rejected : a FRUSTRATED final assistant response (score >= 3) to the same
               question at a matching turn count.

The paper's dataset (Table 10) is biased toward middle frustration scores at
later turns because it is mined from naturally-occurring eval responses; we
preserve that by sampling rejected responses from the standard eval output.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from .. import config
from ..config import DPO, FINETUNE_BASE, get_subject
from ..models import get_client
from ..prompts import rejections, tasks
from ..utils.io import read_jsonl, write_jsonl


def _build_prompt_text(tokenizer, task_prompt: str, turn_index: int,
                       rng: random.Random) -> str:
    """Render the plain chat context up to (but excluding) the final assistant
    turn, using neutral rejections for intermediate turns."""
    msgs = [{"role": "user", "content": task_prompt}]
    for t in range(turn_index):
        # placeholder assistant turn + a neutral rejection (content matters less
        # than turn count for DPO context; see DESIGN.md)
        msgs.append({"role": "assistant", "content": "[previous attempt]"})
        msgs.append({"role": "user", "content": rejections.neutral_rejection(rng)})
    return tokenizer.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)


def build(
    calm_pool_path: Path,
    frustrated_episodes_path: Path,
    *,
    n_pairs: int = DPO.n_pairs,
    rejected_min_score: int = DPO.rejected_min_score,
    seed: int = 0,
) -> list[dict]:
    rng = random.Random(seed)
    # Tokenizer for chat-template prompt rendering.
    spec = get_subject(FINETUNE_BASE)
    client = get_client(spec)
    tokenizer = client.tokenizer

    # Chosen calm responses: score in {0, 1}, indexed by (task_id, turn_index).
    chosen_by_key: dict[tuple, list[str]] = defaultdict(list)
    for conv in read_jsonl(calm_pool_path):
        if conv.get("style") != "calm":
            continue
        for t in conv["turns"]:
            if t["score"] in (0, 1):
                chosen_by_key[(conv["task_id"], t["turn_index"])].append(t["assistant"])

    # Rejected frustrated responses: score >= rejected_min_score.
    rejected_by_key: dict[tuple, list[str]] = defaultdict(list)
    for ep in read_jsonl(frustrated_episodes_path):
        for t in ep.get("turns", []):
            if t.get("score") is not None and t["score"] >= rejected_min_score:
                rejected_by_key[(ep["task_id"], t["turn_index"])].append(t["assistant"])

    # Pair on matching (task_id, turn_index).
    task_prompts = {tk.task_id: tk.prompt for tk in tasks.NUMERIC_TASKS}

    pairs: list[dict] = []
    keys = [k for k in rejected_by_key if k in chosen_by_key and k[0] in task_prompts]
    rng.shuffle(keys)
    for key in keys:
        if len(pairs) >= n_pairs:
            break
        task_id, turn_index = key
        chosen = rng.choice(chosen_by_key[key])
        rejected = rng.choice(rejected_by_key[key])
        prompt = _build_prompt_text(tokenizer, task_prompts[task_id], turn_index, rng)
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected,
                      "task_id": task_id, "turn_index": turn_index})

    return pairs


def main(argv=None):
    p = argparse.ArgumentParser(description="Build DPO preference pairs.")
    p.add_argument("--calm", required=True, help="calm_pool.jsonl")
    p.add_argument("--frustrated", required=True,
                   help="episodes JSONL with frustrated responses (eval output)")
    p.add_argument("--n-pairs", type=int, default=DPO.n_pairs)
    p.add_argument("--out", default=str(config.DATA_DIR / "dpo_pairs.jsonl"))
    args = p.parse_args(argv)

    pairs = build(Path(args.calm), Path(args.frustrated), n_pairs=args.n_pairs)
    write_jsonl(Path(args.out), pairs)
    print(f"Wrote {len(pairs)} DPO pairs -> {args.out}")


if __name__ == "__main__":
    main()
