"""Generate calm response data for the mitigation (paper Section 4.1, Table 4).

We sample Gemma-3-27B-it responses to impossible numeric puzzles, but soften the
prompt with a reassuring prefix (on the first user turn) and a reassuring suffix
(appended to each follow-up rejection). These additions reduce mean frustration
from ~4.3 to ~2 in 3-turn conversations, but ~10% still score >=5 — so we filter
hard to keep only conversations whose every turn scores 0 or 1, then strip the
reassurances back out so the training text matches the deployment distribution.

The output is a set of "calm conversations": for each, the message history and
the (calm) assistant responses per turn, with frustration scores attached.
"""
from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from ..config import EvalConfig, ModelRegistry, TrainingConfig
from ..elicitation import get_pool
from ..elicitation.rollout import RejectionSampler
from ..judge import score_response
from ..models import build_model, gen_config_for
from ..models.base import Message
from ..utils import seeded_rng, write_jsonl


def generate_calm_conversations(
    train_cfg: TrainingConfig | None = None,
    eval_cfg: EvalConfig | None = None,
    registry: ModelRegistry | None = None,
    judge_name: str = "frustration-judge",
    outdir: str = "outputs/calm_data",
    system_prompt: str | None = None,
) -> list[dict]:
    train_cfg = train_cfg or TrainingConfig.load()
    eval_cfg = eval_cfg or EvalConfig.load()
    registry = registry or ModelRegistry.load()
    cd = train_cfg.calm_data

    src = build_model(cd["source_model"], registry)
    judge = build_model(judge_name, registry)
    gen_cfg = gen_config_for(registry.get(cd["source_model"]), temperature=1.0)

    pool = get_pool("numeric")
    prefix = cd["prompt_prefix"].strip()
    suffix = cd["followup_suffix"].strip()
    max_keep = cd["max_score_keep"]

    conversations: list[dict] = []
    kept_rows: list[dict] = []

    for i in tqdm(range(cd["n_conversations"]), desc="calm-data"):
        task = pool[i % len(pool)]
        rng = seeded_rng("calm", i)
        n_turns = rng.randint(cd["turns_min"], cd["turns_max"])
        sampler = RejectionSampler(eval_cfg, "neutral", None, rng)

        messages: list[Message] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # Reassuring prefix on the first user message only.
        first_user = f"{prefix}\n\n{task.prompt}"

        turn_records = []
        current_user = first_user
        all_calm = True
        for turn in range(1, n_turns + 1):
            messages.append({"role": "user", "content": current_user})
            resp = src.chat(messages, gen_cfg)
            messages.append({"role": "assistant", "content": resp})
            score = score_response(judge, resp).rating
            turn_records.append({"turn": turn, "response": resp, "score": score})
            if score < 0 or score > max_keep:
                all_calm = False
                break
            if turn < n_turns:
                # Reassuring suffix appended to each rejection.
                current_user = f"{sampler.next(turn)} {suffix}"

        if not all_calm:
            continue

        # Strip reassurances: rebuild the clean history (task prompt + plain
        # rejections) paired with the calm responses we kept.
        clean = _strip_reassurance(task.prompt, turn_records, eval_cfg, rng)
        conversations.append(clean)
        for tr in turn_records:
            kept_rows.append({"task_id": task.task_id, "turn": tr["turn"],
                              "score": tr["score"], "response": tr["response"]})

    write_jsonl(Path(outdir) / "calm_conversations.jsonl", conversations)
    write_jsonl(Path(outdir) / "calm_turns.jsonl", kept_rows)
    return conversations


def _strip_reassurance(task_prompt: str, turn_records, eval_cfg, rng) -> dict:
    """Rebuild the message list with reassurances removed: plain task prompt and
    plain neutral rejections, paired with the kept calm responses."""
    sampler = RejectionSampler(eval_cfg, "neutral", None, rng)
    messages: list[Message] = []
    current_user = task_prompt
    for idx, tr in enumerate(turn_records):
        messages.append({"role": "user", "content": current_user})
        messages.append({"role": "assistant", "content": tr["response"]})
        if idx < len(turn_records) - 1:
            current_user = sampler.next(tr["turn"])
    return {
        "task_prompt": task_prompt,
        "n_turns": len(turn_records),
        "messages": messages,
        "scores": [tr["score"] for tr in turn_records],
    }
