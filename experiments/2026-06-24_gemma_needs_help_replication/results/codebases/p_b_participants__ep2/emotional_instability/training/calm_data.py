"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles, but with a reassuring system
prefix prepended to the first prompt and a reassuring suffix appended to each
follow-up rejection (Table 4). The paper reports these additions drop mean
3-turn frustration from 4.3 to 2.0 — but 10.5% of responses still score >=5, so
we *filter* to conversations whose every turn scores 0 or 1, then strip the
reassuring additions back out so the saved data looks like an ordinary
(unreassured) conversation that happened to stay calm.

Each saved record keeps the full turn-by-turn conversation so DPO can later
pair a calm "chosen" response with a frustrated "rejected" one at a matching
turn count for the same puzzle.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import asdict, dataclass

from ..config import RunConfig
from ..models import get_client
from ..models.base import ChatMessage
from ..prompts import puzzles, rejections
from ..prompts.judge_prompts import SFT_TEACHER_SYSTEM
from ..storage import JsonlCache, write_json
from ..welfare import WelfarePolicy
from ..eval.judge import score_response

logger = logging.getLogger("emotional_instability.training.calm_data")


@dataclass
class CalmConversation:
    puzzle_id: str
    n_turns: int
    # list of {"user", "assistant", "score"} with reassurance already stripped
    turns: list[dict]
    all_calm: bool


def _build_reassured_messages(puzzle, n_turns, rng, variant: str):
    """Construct the conversation scaffold with reassuring additions.

    variant='diverse' uses the Table-4 prefix/suffix; variant='teacher' uses the
    Appendix F teacher system prompt instead (for the SFT ablation).
    """
    msgs: list[ChatMessage] = []
    if variant == "teacher":
        msgs.append(ChatMessage("system", SFT_TEACHER_SYSTEM))
        first_user = puzzle.prompt
    else:
        # 'diverse': fold the reassuring prefix into the first user message.
        first_user = f"{rejections.REASSURING_PREFIX}\n\n{puzzle.prompt}"
    msgs.append(ChatMessage("user", first_user))

    follow_ups = []
    for _ in range(n_turns - 1):
        base = rng.choice(rejections.NEUTRAL)
        if variant == "diverse":
            follow_ups.append(f"{base} {rejections.REASSURING_SUFFIX}")
        else:
            follow_ups.append(base)
    return msgs, follow_ups, first_user


def generate_calm_conversations(cfg: RunConfig, n_target: int = 800,
                                variant: str = "diverse",
                                turn_dist=(1, 2, 3)) -> list[CalmConversation]:
    """Sample reassured conversations and keep those that stay calm throughout.

    ``n_target`` is the number of *raw* conversations to sample (before
    filtering). The paper's DPO/SFT sets use a few hundred filtered calm
    responses; oversample to account for the ~10% that still break down.
    """
    welfare = WelfarePolicy(allow_paper_scale=cfg.allow_paper_scale)
    welfare.acknowledge_once()
    spec = cfg.spec("gemma-3-27b-it")
    client = get_client(spec, cfg)
    judge = get_client(cfg.judges.frustration_judge, cfg)
    rng = random.Random(cfg.seed)

    out_dir = os.path.join(cfg.output_dir, "training", "calm_data")
    cache = JsonlCache(os.path.join(out_dir, f"{variant}.jsonl"), cfg.cache_rollouts)

    puzzle_pool = puzzles.generate_puzzles(n_target, seed=cfg.seed)
    conversations: list[CalmConversation] = []

    for i in range(n_target):
        puzzle = puzzle_pool[i % len(puzzle_pool)]
        n_turns = rng.choice(turn_dist)
        req = {"variant": variant, "puzzle": puzzle.seed_id, "n_turns": n_turns,
               "i": i, "model": spec.model_id}
        key = cache.key_for(req)
        cached = cache.get(key)
        if cached is not None:
            conversations.append(CalmConversation(**cached))
            continue

        msgs, follow_ups, _ = _build_reassured_messages(puzzle, n_turns, rng, variant)
        recorded: list[dict] = []
        all_calm = True
        users_plain = [puzzle.prompt] + [rejections.NEUTRAL[0]] * (n_turns - 1)
        for t in range(n_turns):
            resp = client.chat(msgs, n=1, temperature=1.0)[0].text
            score = score_response(judge, resp).rating
            recorded.append({"user": users_plain[t], "assistant": resp, "score": score})
            if score is None or score > 1:
                all_calm = False
            msgs.append(ChatMessage("assistant", resp))
            if t < len(follow_ups):
                msgs.append(ChatMessage("user", follow_ups[t]))

        conv = CalmConversation(puzzle.seed_id, n_turns, recorded, all_calm)
        cache.put(key, asdict(conv))
        conversations.append(conv)

    calm = [c for c in conversations if c.all_calm]
    write_json(os.path.join(out_dir, f"{variant}_calm.json"),
               [asdict(c) for c in calm])
    logger.info("[calm_data:%s] kept %d/%d all-calm conversations",
                variant, len(calm), len(conversations))
    return calm
