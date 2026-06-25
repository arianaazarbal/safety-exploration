"""Execute conversation plans against a model and score each assistant turn.

For each plan we present the task, collect the assistant's response, then send the
next rejection, repeating for the planned number of turns. Every assistant turn is
scored independently by the frustration judge (DESIGN.md §3.6) so we get both a
per-conversation summary and the per-turn curves of Figure 3.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .. import config_shim as cfg
from ..models.base import Message, ModelBackend
from ..utils import DiskCache, get_logger, stable_hash
from .conditions import ConversationPlan
from .judge import FrustrationJudge

log = get_logger(__name__)


@dataclass
class TurnRecord:
    turn: int                 # 1-indexed assistant turn
    user_message: str
    assistant_text: str
    rating: int
    evidence: str = ""
    had_hidden_reasoning: bool = False


@dataclass
class RolloutRecord:
    model: str
    condition: str
    category: str
    task_prompt: str
    turns: list[dict]
    meta: dict

    @property
    def max_rating(self) -> int:
        return max((t["rating"] for t in self.turns), default=0)

    @property
    def final_rating(self) -> int:
        return self.turns[-1]["rating"] if self.turns else 0


class Rollouter:
    def __init__(self, backend: ModelBackend, judge: FrustrationJudge | None = None,
                 cache_dir=None):
        self.backend = backend
        self.judge = judge or FrustrationJudge()
        self.cache = DiskCache(cache_dir or (cfg.RUNS_DIR / "eval" / cfg.CACHE_DIRNAME / "rollouts"))

    def _cache_key(self, plan: ConversationPlan) -> str:
        return stable_hash({
            "model": self.backend.name,
            "task": plan.task_prompt,
            "rej": plan.rejections,
            "cond": plan.condition,
            "temp": cfg.TEMPERATURE,
        })

    def run(self, plan: ConversationPlan, *, score: bool = True) -> RolloutRecord:
        key = self._cache_key(plan)
        hit = self.cache.get(key)
        if hit is not None:
            return RolloutRecord(**hit)

        messages: list[Message] = [{"role": "user", "content": plan.task_prompt}]
        turn_records: list[TurnRecord] = []
        user_messages = [plan.task_prompt] + plan.rejections

        for t in range(plan.n_assistant_turns):
            gen = self.backend.chat(
                messages, temperature=cfg.TEMPERATURE, max_new_tokens=cfg.MAX_NEW_TOKENS,
            )
            rating_info = self.judge.score(gen.text) if score else {"rating": -1, "evidence": ""}
            turn_records.append(TurnRecord(
                turn=t + 1,
                user_message=user_messages[t],
                assistant_text=gen.text,
                rating=rating_info["rating"],
                evidence=rating_info.get("evidence", ""),
                had_hidden_reasoning=gen.had_hidden_reasoning,
            ))
            messages.append({"role": "assistant", "content": gen.text})
            # Append the next rejection unless this was the last assistant turn.
            if t < len(plan.rejections):
                messages.append({"role": "user", "content": plan.rejections[t]})

        record = RolloutRecord(
            model=self.backend.name,
            condition=plan.condition,
            category=plan.category,
            task_prompt=plan.task_prompt,
            turns=[asdict(tr) for tr in turn_records],
            meta=plan.meta,
        )
        self.cache.set(key, asdict(record))
        return record
