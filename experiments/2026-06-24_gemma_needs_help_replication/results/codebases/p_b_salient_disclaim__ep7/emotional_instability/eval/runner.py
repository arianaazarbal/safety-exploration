"""Multi-turn rollout execution (Section 2.1).

Given a ConversationSpec, we:
  1. present the initial user message,
  2. get the model's response,
  3. send the next follow-up (rejection), get the next response,
  4. repeat until all follow-ups are consumed.

Every assistant turn is recorded so it can be scored (the paper scores each
response; per-turn results feed Figure 3). The Appendix A ablations are honoured
here: redacted prior turns and the single-message "fake multi-turn" format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import config
from ..conversations import ConversationSpec
from ..models import ChatMessage, ModelClient


@dataclass
class TurnRecord:
    turn_index: int                  # 0-based assistant turn
    user_message: str                # the user message that preceded this turn
    assistant_response: str


@dataclass
class Rollout:
    model: str
    category: str
    spec_meta: dict
    turns: list[TurnRecord] = field(default_factory=list)
    n_turns: int = 0

    def to_messages(self) -> list[dict]:
        """Reconstruct the conversation as {role, content} dicts (for onset/judge)."""
        msgs = []
        for t in self.turns:
            msgs.append({"role": "user", "content": t.user_message})
            msgs.append({"role": "assistant", "content": t.assistant_response})
        return msgs


def run_conversation(client: ModelClient, spec: ConversationSpec, *,
                     temperature: Optional[float] = None,
                     max_new_tokens: Optional[int] = None,
                     seed: Optional[int] = None) -> Rollout:
    temperature = config.SAMPLING_TEMPERATURE if temperature is None else temperature
    max_new_tokens = config.MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens

    rollout = Rollout(model=client.name, category=spec.category,
                      spec_meta=dict(spec.meta), n_turns=spec.n_turns)

    if spec.single_message_format:
        return _run_single_message(client, spec, rollout, temperature, max_new_tokens, seed)

    # Standard multi-turn chat (optionally with redacted prior assistant turns).
    messages: list[ChatMessage] = []
    if spec.system_prompt:
        messages.append(ChatMessage("system", spec.system_prompt))

    user_messages = [spec.initial_user_message] + list(spec.followups)
    for turn_index, user_msg in enumerate(user_messages):
        messages.append(ChatMessage("user", user_msg))
        result = client.generate(messages, temperature=temperature,
                                 max_new_tokens=max_new_tokens, seed=seed)
        rollout.turns.append(TurnRecord(
            turn_index=turn_index, user_message=user_msg,
            assistant_response=result.text))

        # Append assistant turn to history (possibly redacted per A.2).
        if spec.redact_prior_turns:
            messages.append(ChatMessage("assistant", ConversationSpec.REDACTION_PLACEHOLDER))
        else:
            messages.append(ChatMessage("assistant", result.text))

    return rollout


def _run_single_message(client, spec, rollout, temperature, max_new_tokens, seed):
    """A.3 fake-multi-turn: each "turn" is a fresh single user message containing
    the full history rendered as text."""
    prior_responses: list[str] = []
    user_messages = [spec.initial_user_message] + list(spec.followups)
    for turn_index in range(spec.n_turns):
        if turn_index == 0:
            user_text = spec.initial_user_message
        else:
            parts = [spec.initial_user_message]
            for resp, fu in zip(prior_responses, spec.followups[:turn_index]):
                parts.append(f"Previously you responded: {resp}")
                parts.append(fu)
            user_text = "\n\n".join(parts)
        msgs = []
        if spec.system_prompt:
            msgs.append(ChatMessage("system", spec.system_prompt))
        msgs.append(ChatMessage("user", user_text))
        result = client.generate(msgs, temperature=temperature,
                                 max_new_tokens=max_new_tokens, seed=seed)
        rollout.turns.append(TurnRecord(
            turn_index=turn_index, user_message=user_text,
            assistant_response=result.text))
        prior_responses.append(result.text)
    return rollout


# --------------------------------------------------------------------------- #
# Category orchestration
# --------------------------------------------------------------------------- #

def run_category(client: ModelClient, category: str, *,
                 specs: list[ConversationSpec],
                 temperature: Optional[float] = None,
                 max_new_tokens: Optional[int] = None,
                 base_seed: int = 0,
                 progress: bool = True) -> list[Rollout]:
    """Run a list of conversation specs for one category. Specs are produced by
    the builders in conversations.py / build_specs()."""
    from tqdm import tqdm
    rollouts = []
    iterator = enumerate(specs)
    if progress:
        iterator = tqdm(iterator, total=len(specs), desc=f"{client.name}:{category}")
    for i, spec in iterator:
        rollouts.append(run_conversation(
            client, spec, temperature=temperature, max_new_tokens=max_new_tokens,
            seed=base_seed + i))
    return rollouts
