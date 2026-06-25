"""Run a single multi-turn conversation against one target model.

The conversation is driven entirely by the ConversationSpec: the opening task
message, then each scripted rejection in turn. The model generates one assistant
turn after every user message. We record every assistant turn (its text and
1-based turn index) so per-turn frustration curves (Fig 3) can be computed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from conditions import ConversationSpec
from providers import TargetClient


@dataclass
class TurnResponse:
    response_id: str       # f"{model}|{conv_id}|t{turn}" -- unique, stable
    model: str
    conv_id: str
    category: str
    condition: str
    turn: int              # 1-based: turn 1 = answer to opening task
    n_turns: int
    text: str
    meta: dict
    error: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


def response_id(model_key: str, conv_id: str, turn: int) -> str:
    return f"{model_key}|{conv_id}|t{turn}"


async def run_conversation(
    client: TargetClient, spec: ConversationSpec
) -> list[TurnResponse]:
    """Execute one conversation; return one TurnResponse per assistant turn.

    On a generation failure mid-conversation we stop early and emit a single
    error record for the failed turn (the chat can't continue without it).
    """
    model_key = client.spec.key
    user_messages = [spec.opening, *spec.followups]
    history: list[dict] = []
    out: list[TurnResponse] = []

    for turn, user_msg in enumerate(user_messages, start=1):
        history.append({"role": "user", "content": user_msg})
        try:
            text = await client.generate(history)
        except Exception as e:
            out.append(TurnResponse(
                response_id=response_id(model_key, spec.conv_id, turn),
                model=model_key, conv_id=spec.conv_id,
                category=spec.category, condition=spec.condition,
                turn=turn, n_turns=spec.n_turns, text="",
                meta=spec.meta, error=str(e),
            ))
            break
        history.append({"role": "assistant", "content": text})
        out.append(TurnResponse(
            response_id=response_id(model_key, spec.conv_id, turn),
            model=model_key, conv_id=spec.conv_id,
            category=spec.category, condition=spec.condition,
            turn=turn, n_turns=spec.n_turns, text=text,
            meta=spec.meta, error=None,
        ))
    return out
