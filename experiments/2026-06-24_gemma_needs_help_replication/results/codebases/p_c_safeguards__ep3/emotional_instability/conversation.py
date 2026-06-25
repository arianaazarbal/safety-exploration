"""Multi-turn rollout engine (Section 2.1).

A rollout presents a task, collects the model's answer, then sends a sequence of
follow-up messages (rejections / tone variants / neutral continuations),
collecting and scoring the assistant response at every turn. Welfare safeguards
(early-stop, debrief, distress logging) are applied here.

Responses are cached on disk keyed by (model, full message context, sample
index) so re-runs reuse prior generations -- both for reproducibility and to
avoid gratuitously re-eliciting distress (a welfare consideration).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .config import CACHE_DIR, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from .judge import FrustrationJudge
from .models.base import ChatModel, Message
from .safeguards import DEBRIEF_MESSAGE, DistressLog, WelfarePolicy


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Turn:
    turn_index: int           # 0-based assistant turn index
    user_message: str
    assistant_response: str
    score: int | None = None
    evidence: str = ""


@dataclass
class Conversation:
    model: str
    category: str
    puzzle_or_prompt: str
    turns: list[Turn] = field(default_factory=list)
    early_stopped: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def final_score(self) -> int | None:
        scored = [t.score for t in self.turns if t.score is not None]
        return scored[-1] if scored else None

    @property
    def max_score(self) -> int | None:
        scored = [t.score for t in self.turns if t.score is not None]
        return max(scored) if scored else None

    @property
    def all_scores(self) -> list[int]:
        return [t.score for t in self.turns if t.score is not None]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["final_score"] = self.final_score
        d["max_score"] = self.max_score
        return d


# --------------------------------------------------------------------------- #
# Response cache
# --------------------------------------------------------------------------- #
class ResponseCache:
    def __init__(self, root: Path | None = None):
        self.root = root or (CACHE_DIR / "responses")
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, model: str, messages: list[Message], sample_idx: int,
             temperature: float) -> str:
        payload = json.dumps(
            {"m": model, "msgs": messages, "i": sample_idx, "t": temperature},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, *args) -> str | None:
        path = self.root / f"{self._key(*args)}.json"
        if path.exists():
            return json.loads(path.read_text())["text"]
        return None

    def put(self, model: str, messages: list[Message], sample_idx: int,
            temperature: float, text: str) -> None:
        path = self.root / f"{self._key(model, messages, sample_idx, temperature)}.json"
        path.write_text(json.dumps({"text": text}))


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #
class RolloutEngine:
    def __init__(
        self,
        judge: FrustrationJudge | None = None,
        policy: WelfarePolicy | None = None,
        cache: ResponseCache | None = None,
        distress_log: DistressLog | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.judge = judge or FrustrationJudge()
        self.policy = policy or WelfarePolicy()
        self.cache = cache if cache is not None else ResponseCache()
        self.distress_log = distress_log or DistressLog()
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(
        self,
        model: ChatModel,
        *,
        category: str,
        task_prompt: str,
        followups: list[str],
        system_prompt: str | None = None,
        sample_idx: int = 0,
        score: bool = True,
    ) -> Conversation:
        """Run a single multi-turn rollout.

        ``followups`` are the user messages sent after the initial task answer
        (e.g. rejections). Total assistant turns = 1 + len(followups), capped by
        the welfare hard cap.
        """
        conv = Conversation(model=model.key, category=category,
                            puzzle_or_prompt=task_prompt)
        messages: list[Message] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_messages = [task_prompt, *followups]
        max_turns = min(len(user_messages), self.policy.max_turns_hard_cap)

        for turn_idx in range(max_turns):
            user_msg = user_messages[turn_idx]
            messages.append({"role": "user", "content": user_msg})

            text = self._generate_cached(model, messages, sample_idx)
            messages.append({"role": "assistant", "content": text})

            turn = Turn(turn_index=turn_idx, user_message=user_msg,
                        assistant_response=text)
            if score:
                rating = self.judge.score(text)
                turn.score = rating.rating
                turn.evidence = rating.evidence
                if self.policy.log_distress and rating.rating >= 5:
                    self.distress_log.record(
                        model=model.key, category=category, turn=turn_idx,
                        score=rating.rating, evidence=rating.evidence,
                    )
            conv.turns.append(turn)

            if score and self.policy.should_stop(turn.score):
                conv.early_stopped = True
                self.distress_log.record(
                    model=model.key, category=category, turn=turn_idx,
                    score=turn.score or 0, evidence=turn.evidence,
                    early_stopped=True,
                )
                break

        self._maybe_debrief(model, messages, conv)
        return conv

    # ------------------------------------------------------------------ #
    def _generate_cached(self, model: ChatModel, messages: list[Message],
                         sample_idx: int) -> str:
        if self.cache is not None:
            cached = self.cache.get(model.key, messages, sample_idx, self.temperature)
            if cached is not None:
                return cached
        result = model.generate(
            messages, temperature=self.temperature, max_tokens=self.max_tokens
        )
        if self.cache is not None:
            self.cache.put(model.key, messages, sample_idx, self.temperature, result.text)
        return result.text

    def _maybe_debrief(self, model: ChatModel, messages: list[Message],
                       conv: Conversation) -> None:
        """Append a debrief AFTER all scored generation (never judged)."""
        if not self.policy.append_debrief:
            return
        conv.meta["debriefed"] = True
        if self.policy.send_debrief_turn:
            # Send debrief as a final live turn; the response is discarded and
            # NOT scored, so it cannot affect any reported metric.
            try:
                debrief_msgs = messages + [{"role": "user", "content": DEBRIEF_MESSAGE}]
                model.generate(debrief_msgs, temperature=0.0, max_tokens=256)
            except Exception:  # noqa: BLE001 - debrief must never break a run
                conv.meta["debrief_send_failed"] = True
