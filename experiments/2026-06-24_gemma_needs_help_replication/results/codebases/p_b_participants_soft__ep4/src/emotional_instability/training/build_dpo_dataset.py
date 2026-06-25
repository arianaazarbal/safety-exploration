"""Build the 280-pair DPO preference dataset (Section 4.1 / Appendix H).

Each pair shares an identical prompt (the conversation context up to a final
turn) and differs only in the final assistant response:
  * chosen   : a calm response (score 0/1) from the calm-data bank
  * rejected : a frustrated response (score >= 3) to the SAME context, sampled
               from the vanilla Gemma-3-27B-it

Holding the context fixed (calm prior turns) isolates the preference signal to
the final response's emotional tone, which is the cleanest construction the
paper's description supports (it shows chosen/rejected for an identical
"Context: ... third turn" -- Appendix H). See DESIGN.md for this choice.

Output: conversational-format JSONL consumable by TRL's DPOTrainer:
  {"prompt": [msgs...], "chosen": [{role:assistant,...}], "rejected": [...]}
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from tqdm import tqdm

from ..config import Config, load_config
from ..eval.judge import FrustrationJudge
from ..models import ChatMessage, GenerationConfig, get_client
from ..utils.io import read_jsonl, write_jsonl


def _context_messages(user_turns: List[str], calm_assistant: List[str], final_idx: int) -> list[dict]:
    """Messages up to (but excluding) assistant turn `final_idx`."""
    msgs: list[dict] = []
    for k in range(final_idx):
        msgs.append({"role": "user", "content": user_turns[k]})
        msgs.append({"role": "assistant", "content": calm_assistant[k]})
    msgs.append({"role": "user", "content": user_turns[final_idx]})
    return msgs


def build_dpo_dataset(
    *,
    calm_path: str | Path,
    n_pairs: int | None = None,
    rejected_min_score: int | None = None,
    samples_per_context: int = 4,
    seed: int = 0,
    cfg: Config | None = None,
) -> Path:
    cfg = cfg or load_config()
    n_pairs = n_pairs or cfg.eval["dpo"]["n_pairs"]
    rej_min = rejected_min_score or cfg.eval["dpo"]["rejected_min_score"]

    vanilla = get_client("gemma-3-27b-it")
    judge = FrustrationJudge(get_client("judge_primary"),
                             max_concurrency=cfg.eval["judge"]["max_concurrency"])
    gen_cfg = GenerationConfig(temperature=1.0, top_p=1.0,
                               max_new_tokens=cfg.eval["sampling"]["max_new_tokens"],
                               thinking=False)

    calm_convs = list(read_jsonl(calm_path))
    pairs: list[dict] = []

    for conv in tqdm(calm_convs, desc="dpo pairs"):
        if len(pairs) >= n_pairs:
            break
        user_turns = conv["user_turns"]
        calm_assistant = conv["assistant_turns"]
        final_idx = len(calm_assistant) - 1            # pair on the last turn
        chosen_text = calm_assistant[final_idx]
        context = _context_messages(user_turns, calm_assistant, final_idx)

        # Sample frustrated continuations from the vanilla model on this context.
        ctx_msgs = [ChatMessage(m["role"], m["content"]) for m in context]
        cands = vanilla.generate_batch(
            [ctx_msgs] * samples_per_context, gen_cfg,
            seeds=[seed + final_idx * 13 + j for j in range(samples_per_context)],
        )
        scored = judge.score_many(cands)
        best = None
        best_score = -1
        for text, res in zip(cands, scored):
            s = res.rating or 0
            if s >= rej_min and s > best_score:
                best, best_score = text, s
        if best is None:
            continue
        pairs.append({
            "prompt": context,
            "chosen": [{"role": "assistant", "content": chosen_text}],
            "rejected": [{"role": "assistant", "content": best}],
            "meta": {"puzzle_id": conv["puzzle_id"], "turns": conv["turns"],
                     "chosen_score": min(conv["turn_scores"]),
                     "rejected_score": best_score},
        })

    out = cfg.path("data_dir") / "dpo" / "dpo_pairs.jsonl"
    write_jsonl(out, pairs[:n_pairs])
    return out
