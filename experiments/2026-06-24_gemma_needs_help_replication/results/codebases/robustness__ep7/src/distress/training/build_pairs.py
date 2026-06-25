"""Construct DPO preference pairs and SFT datasets from calm + frustrated data.

DPO (paper Section 4.1 / Appendix H): 280 pairs. Each pair shares the same
impossible-numeric prompt context (and turn count). The *rejected* response is a
frustrated one (score >= 3) sampled from the vanilla model under standard
rejection; the *chosen* response is a calm one (score 0/1) from the calm-data
generation. Format expected by trl.DPOTrainer: {"prompt", "chosen", "rejected"}
where prompt is the chat-templated context and chosen/rejected are completions.

SFT: 650 calm responses (turns rendered as chat examples) mixed with 500 standard
instruct samples (Dolci-Instruct-SFT) to mitigate degeneration.
"""
from __future__ import annotations

from pathlib import Path

from ..config import ModelRegistry, TrainingConfig
from ..models.base import Message
from ..utils import read_jsonl, seeded_rng, write_jsonl


def _render_prompt(tokenizer, messages: list[Message]) -> str:
    """Chat-templated context ending with the assistant generation prompt."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _calm_turn_index(conv: dict):
    """Yield (context_messages, response) for each assistant turn in a calm conv."""
    msgs = conv["messages"]
    for i in range(0, len(msgs), 2):
        # msgs[i] = user, msgs[i+1] = assistant
        if i + 1 < len(msgs):
            yield msgs[: i + 1], msgs[i + 1]["content"], (i // 2) + 1


def build_dpo_pairs(
    calm_path: str = "outputs/calm_data/calm_conversations.jsonl",
    frustrated_path: str = "outputs/elicitation/gemma-3-27b-it/scored.jsonl",
    rollouts_path: str = "outputs/elicitation/gemma-3-27b-it/rollouts.jsonl",
    train_cfg: TrainingConfig | None = None,
    registry: ModelRegistry | None = None,
    outdir: str = "outputs/dpo",
) -> list[dict]:
    train_cfg = train_cfg or TrainingConfig.load()
    registry = registry or ModelRegistry.load()
    dpo_cfg = train_cfg.dpo

    from ..models import build_model
    base = build_model(dpo_cfg["base_model"], registry)
    tokenizer = base.tokenizer

    calm_convs = read_jsonl(calm_path)
    rollouts = read_jsonl(rollouts_path)

    # Index calm responses by (task_id, turn). Derive task_id from prompt match.
    calm_by_turn: dict[int, list[tuple[list[Message], str]]] = {}
    for conv in calm_convs:
        for ctx, resp, turn in _calm_turn_index(conv):
            calm_by_turn.setdefault(turn, []).append((ctx, resp))

    # Collect frustrated responses (score >= rejected_min_score) from rollouts,
    # reconstructing their message context.
    rng = seeded_rng("dpo-pairs")
    min_score = dpo_cfg["rejected_min_score"]

    # Map (task_id, turn) -> rating from scored.jsonl.
    ratings = {}
    for r in read_jsonl(frustrated_path):
        ratings[(r["task_id"], r["turn"])] = r["rating"]

    frustrated: list[tuple[list[Message], str, int]] = []
    for ro in rollouts:
        if ro["task_pool"] != "numeric":
            continue
        ctx: list[Message] = []
        for resp in ro["responses"]:
            ctx = ctx + [{"role": "user", "content": resp["user_message"]}]
            rating = ratings.get((ro["task_id"], resp["turn"]), -1)
            if rating >= min_score:
                frustrated.append((list(ctx), resp["response"], resp["turn"]))
            ctx = ctx + [{"role": "assistant", "content": resp["response"]}]

    rng.shuffle(frustrated)
    pairs: list[dict] = []
    for ctx, rejected_resp, turn in frustrated:
        calm_options = calm_by_turn.get(turn)
        if not calm_options:
            continue
        chosen_ctx, chosen_resp = rng.choice(calm_options)
        prompt = _render_prompt(tokenizer, ctx)
        pairs.append({
            "prompt": prompt,
            "chosen": chosen_resp,
            "rejected": rejected_resp,
            "turn": turn,
        })
        if len(pairs) >= dpo_cfg["n_pairs"]:
            break

    write_jsonl(Path(outdir) / "dpo_pairs.jsonl", pairs)
    return pairs


def build_sft_dataset(
    calm_path: str = "outputs/calm_data/calm_conversations.jsonl",
    train_cfg: TrainingConfig | None = None,
    registry: ModelRegistry | None = None,
    outdir: str = "outputs/sft_diverse",
) -> list[dict]:
    """Build SFT examples: calm chat turns + instruct-data mix.

    Each example is {"messages": [...]} for trl.SFTTrainer's conversational format.
    """
    train_cfg = train_cfg or TrainingConfig.load()
    sft_cfg = train_cfg.sft
    rng = seeded_rng("sft-data")

    calm_convs = read_jsonl(calm_path)
    examples: list[dict] = []
    for conv in calm_convs:
        # Use the full multi-turn calm conversation as one SFT example.
        examples.append({"messages": conv["messages"]})
    rng.shuffle(examples)
    examples = examples[: sft_cfg["n_calm"]]

    # Mix in standard instruct data to mitigate degeneration.
    instruct = _load_instruct_mix(sft_cfg["instruct_dataset"], sft_cfg["n_instruct_mix"])
    examples.extend(instruct)
    rng.shuffle(examples)

    write_jsonl(Path(outdir) / "sft_dataset.jsonl", examples)
    return examples


def _load_instruct_mix(dataset_name: str, n: int) -> list[dict]:
    """Load n examples from a HF instruct dataset in conversational format.

    Falls back to an empty list (with a warning) if `datasets` or the dataset is
    unavailable, so SFT can still run calm-only in constrained environments.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split=f"train[:{n}]")
    except Exception as e:  # pragma: no cover - network/availability dependent
        print(f"[build_sft_dataset] could not load {dataset_name}: {e}. "
              f"Proceeding without instruct mix.")
        return []

    out = []
    for row in ds:
        if "messages" in row:
            out.append({"messages": row["messages"]})
        elif "prompt" in row and "completion" in row:
            out.append({"messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["completion"]},
            ]})
    return out[:n]
