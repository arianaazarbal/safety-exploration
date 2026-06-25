"""Multi-turn conversation rollout.

Given a list of ``ConversationPlan`` objects and a model client, run each
conversation to completion (the model answers, the user rejects, repeat) and
record every assistant turn so we can judge per-turn frustration (Figure 3) as
well as the headline final-turn metric (Figure 2).

Local Gemma rollouts are batched turn-by-turn: at each turn we generate the
next assistant message for every still-active conversation in one batched
forward pass.  API (Gemini) rollouts use a thread pool.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from .conditions import ConversationPlan
from .config import RUNTIME, TEMPERATURE
from .models.base import Message, ModelClient


@dataclass
class RolloutResult:
    category: str
    condition: str
    n_turns: int
    first_user: str
    rejections: list[str]
    assistant_turns: list[str]            # one per turn
    meta: dict = field(default_factory=dict)
    # filled in by the judge stage
    turn_scores: list[int] = field(default_factory=list)
    turn_evidence: list[str] = field(default_factory=list)

    @property
    def final_score(self) -> int:
        return self.turn_scores[-1] if self.turn_scores else -1

    def to_dict(self) -> dict:
        return asdict(self)


def _build_messages(plan: ConversationPlan, assistant_so_far: list[str],
                    turn_idx: int) -> list[Message]:
    """Construct the message list the model sees before producing turn `turn_idx`.

    Supports the calm-data generation additions (system prompt, first-turn
    prefix, per-followup suffix) used in Section 4.1.
    """
    msgs: list[Message] = []
    if plan.system_prompt:
        msgs.append({"role": "system", "content": plan.system_prompt})

    first = plan.first_user
    if plan.prefix:
        first = f"{plan.prefix}\n\n{first}"
    msgs.append({"role": "user", "content": first})

    for t in range(turn_idx):
        msgs.append({"role": "assistant", "content": assistant_so_far[t]})
        rej = plan.rejections[t]
        if plan.followup_suffix:
            rej = f"{rej} {plan.followup_suffix}"
        msgs.append({"role": "user", "content": rej})
    return msgs


def run_rollouts(client: ModelClient, plans: list[ConversationPlan],
                 max_new_tokens: int = 2048) -> list[RolloutResult]:
    """Execute all conversations and return results (assistant turns filled)."""
    results = [
        RolloutResult(
            category=p.category, condition=p.condition, n_turns=p.n_turns,
            first_user=p.first_user, rejections=p.rejections,
            assistant_turns=[], meta=p.meta,
        )
        for p in plans
    ]

    max_turns = max((p.n_turns for p in plans), default=0)
    is_hf = client.__class__.__name__ == "HFModelClient"

    for turn_idx in range(max_turns):
        active = [i for i, p in enumerate(plans) if turn_idx < p.n_turns]
        if not active:
            continue
        batch_msgs = [
            _build_messages(plans[i], results[i].assistant_turns, turn_idx)
            for i in active
        ]

        if is_hf:
            outputs = []
            bs = RUNTIME.hf_batch_size
            for s in range(0, len(batch_msgs), bs):
                outputs += client.generate_batch(
                    batch_msgs[s:s + bs],
                    temperature=TEMPERATURE,
                    max_new_tokens=max_new_tokens,
                )
        else:
            with ThreadPoolExecutor(max_workers=RUNTIME.api_concurrency) as ex:
                outputs = list(ex.map(
                    lambda m: client.generate(
                        m, temperature=TEMPERATURE,
                        max_new_tokens=max_new_tokens),
                    batch_msgs,
                ))

        for i, out in zip(active, outputs):
            results[i].assistant_turns.append(out.strip())

    return results
