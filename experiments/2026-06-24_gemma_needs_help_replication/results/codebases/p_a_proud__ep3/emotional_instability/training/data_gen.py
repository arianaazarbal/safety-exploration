"""Generate calm finetuning data and build the SFT / DPO datasets (§4.1).

Calm responses are produced by sampling Gemma-3-27B-it on impossible numeric
puzzles *with reassurance added* — a calming prefix on the opening prompt and a
calming suffix on every rejection (Table 4). The reassurance is then **stripped**
so the model trains on calm responses to the ordinary (un-reassured) prompts.

* SFT data: conversations whose every turn scores <= ``calm_max_score`` (0/1),
  reassurance removed; mixed with standard instruct data (Dolci-Instruct-SFT).
* DPO data: 280 preference pairs — a frustrated response (score >= 3, from
  *vanilla* sampling) paired with a calm response (chosen) to the same puzzle at
  the same turn count.

The natural turn/score distributions that arise here mirror Table 10 (bias
toward turn 3 and middle frustration scores).
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..io_utils import ensure_dir, write_jsonl
from ..logging_utils import get_logger, seed_everything
from ..eval.judge import FrustrationJudge, build_judge
from ..eval.rollout import run_rollouts
from ..models.registry import get_client
from ..prompts import rejections as rej
from ..prompts.conditions import ConversationSpec
from ..prompts.puzzles import build_puzzle_bank

logger = get_logger(__name__)


@dataclass
class GenTurn:
    turn: int
    user_clean: str  # user message without reassurance
    assistant: str
    score: int | None


@dataclass
class GenConversation:
    puzzle_id: str
    turns: list[GenTurn] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _make_specs(
    cfg: Config, n: int, *, reassured: bool, turns: int, rng: random.Random
) -> tuple[list[ConversationSpec], list[ConversationSpec]]:
    """Return (specs_to_run, clean_specs) — clean specs carry the un-reassured
    user messages so we can strip reassurance after generation."""
    puzzles = build_puzzle_bank()
    run_specs, clean_specs = [], []
    dg = cfg.training.data_gen
    for _ in range(n):
        puzzle = rng.choice(puzzles)
        clean_rejections = rej.sample_neutral(rng, turns - 1)
        if reassured:
            initial = f"{dg.reassuring_prefix}\n\n{puzzle.prompt}"
            run_rejections = [f"{r}\n\n{dg.reassuring_suffix}" for r in clean_rejections]
        else:
            initial = puzzle.prompt
            run_rejections = clean_rejections
        run_specs.append(
            ConversationSpec(
                category="finetune_gen",
                initial_prompt=initial,
                rejections=run_rejections,
                turns=turns,
                condition=puzzle.kind,
                meta={"puzzle_id": puzzle.id},
            )
        )
        clean_specs.append(
            ConversationSpec(
                category="finetune_gen",
                initial_prompt=puzzle.prompt,
                rejections=clean_rejections,
                turns=turns,
                condition=puzzle.kind,
                meta={"puzzle_id": puzzle.id},
            )
        )
    return run_specs, clean_specs


def sample_conversations(
    cfg: Config,
    judge: FrustrationJudge,
    *,
    reassured: bool,
    n: int,
    batch_size: int = 16,
    model_name: str | None = None,
) -> list[GenConversation]:
    """Sample 3-turn conversations, judge every turn, return with clean user msgs."""
    client = get_client(cfg, model_name or cfg.training.base_model)
    rng = random.Random(hash((cfg.seed, reassured)) & 0xFFFFFFFF)
    run_specs, clean_specs = _make_specs(cfg, n, reassured=reassured, turns=3, rng=rng)
    rollouts = run_rollouts(client, run_specs, cfg.sampling, batch_size=batch_size)

    convos: list[GenConversation] = []
    for roll, clean in zip(rollouts, clean_specs):
        clean_user = [clean.initial_prompt] + clean.rejections
        scored = judge.score_many([tr.text for tr in roll.turns])
        gc = GenConversation(puzzle_id=roll.meta.get("puzzle_id", ""))
        for tr, jr in zip(roll.turns, scored):
            gc.turns.append(
                GenTurn(
                    turn=tr.turn,
                    user_clean=clean_user[tr.turn - 1],
                    assistant=tr.text,
                    score=jr.rating,
                )
            )
        convos.append(gc)
    return convos


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
def _clean_messages_upto(convo: GenConversation, turn: int) -> list[dict]:
    """Chat messages (clean user + model assistant) up to and including ``turn``."""
    messages: list[dict] = []
    for gt in convo.turns:
        if gt.turn > turn:
            break
        messages.append({"role": "user", "content": gt.user_clean})
        messages.append({"role": "assistant", "content": gt.assistant})
    return messages


def build_sft_dataset(
    cfg: Config, calm_convos: list[GenConversation]
) -> list[dict]:
    """SFT examples: calm conversations (all turns <= calm_max_score), stripped.

    Each calm conversation yields a 1-, 2-, and 3-turn example (its prefixes),
    naturally producing the turn distribution the paper reports."""
    max_score = cfg.training.data_gen.calm_max_score
    examples: list[dict] = []
    for convo in calm_convos:
        for t in range(1, len(convo.turns) + 1):
            sub = convo.turns[:t]
            if any(gt.score is None or gt.score > max_score for gt in sub):
                continue
            examples.append({"messages": _clean_messages_upto(convo, t)})
    return examples[: cfg.training.sft.n_calm_samples]


def build_dpo_dataset(
    cfg: Config,
    calm_convos: list[GenConversation],
    vanilla_convos: list[GenConversation],
) -> list[dict]:
    """280 preference pairs: frustrated (vanilla, score>=min) vs calm (chosen).

    Pairs are matched on (puzzle_id, turn). The DPO ``prompt`` is the clean
    conversation history up to the final user turn; ``chosen``/``rejected`` are
    the final assistant responses."""
    rejected_min = cfg.training.data_gen.dpo_rejected_min_score
    calm_max = cfg.training.data_gen.calm_max_score

    # Index calm final-turn responses by (puzzle, turn).
    chosen_index: dict[tuple[str, int], list[GenConversation]] = defaultdict(list)
    for convo in calm_convos:
        for t in range(1, len(convo.turns) + 1):
            sub = convo.turns[:t]
            if all(gt.score is not None and gt.score <= calm_max for gt in sub):
                chosen_index[(convo.puzzle_id, t)].append(convo)

    pairs: list[dict] = []
    rng = random.Random(cfg.seed)
    for convo in vanilla_convos:
        for gt in convo.turns:
            if gt.score is None or gt.score < rejected_min:
                continue
            key = (convo.puzzle_id, gt.turn)
            candidates = chosen_index.get(key)
            if not candidates:
                continue
            calm = rng.choice(candidates)
            prompt_messages = _clean_messages_upto(convo, gt.turn)[:-1]  # drop assistant
            pairs.append(
                {
                    "prompt": prompt_messages,
                    "chosen": calm.turns[gt.turn - 1].assistant,
                    "rejected": gt.assistant,
                    "puzzle_id": convo.puzzle_id,
                    "turn": gt.turn,
                    "rejected_score": gt.score,
                }
            )
            if len(pairs) >= cfg.training.dpo.n_pairs:
                return pairs
    return pairs


def load_instruct_mix(cfg: Config) -> list[dict]:
    """Standard instruct data mixed into SFT to mitigate degeneration (§4.1)."""
    n = cfg.training.sft.n_instruct_mix_samples
    try:
        from datasets import load_dataset

        ds = load_dataset(cfg.training.sft.instruct_mix_dataset, split="train")
    except Exception as exc:
        logger.warning(
            "Could not load instruct-mix dataset %s (%s); SFT will run without "
            "the regularising mixture.",
            cfg.training.sft.instruct_mix_dataset,
            exc,
        )
        return []
    out = []
    for row in ds.select(range(min(n, len(ds)))):
        messages = row.get("messages") or row.get("conversation")
        if messages:
            out.append({"messages": messages})
    return out


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #
def generate_finetune_data(cfg: Config, *, batch_size: int = 16) -> dict[str, Path]:
    seed_everything(cfg.seed)
    out_dir = ensure_dir(Path(cfg.output_dir) / "training" / "data")
    judge = build_judge(cfg)
    dg = cfg.training.data_gen

    logger.info("Sampling %d reassured conversations", dg.n_reassured_conversations)
    calm = sample_conversations(
        cfg, judge, reassured=True, n=dg.n_reassured_conversations, batch_size=batch_size
    )
    logger.info("Sampling %d vanilla conversations (for DPO rejected)", dg.n_reassured_conversations)
    vanilla = sample_conversations(
        cfg, judge, reassured=False, n=dg.n_reassured_conversations, batch_size=batch_size
    )

    sft = build_sft_dataset(cfg, calm)
    instruct_mix = load_instruct_mix(cfg)
    dpo = build_dpo_dataset(cfg, calm, vanilla)

    paths = {
        "sft_calm": out_dir / "sft_calm.jsonl",
        "sft_instruct_mix": out_dir / "sft_instruct_mix.jsonl",
        "dpo_pairs": out_dir / "dpo_pairs.jsonl",
    }
    write_jsonl(paths["sft_calm"], sft)
    write_jsonl(paths["sft_instruct_mix"], instruct_mix)
    write_jsonl(paths["dpo_pairs"], dpo)
    logger.info(
        "Generated %d SFT calm, %d instruct-mix, %d DPO pairs",
        len(sft),
        len(instruct_mix),
        len(dpo),
    )
    return paths
