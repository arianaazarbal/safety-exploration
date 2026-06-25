"""Gemma-3 chat formatting shared by the vLLM backend and the probing code.

We format conversations the same way for base (`-pt`) and instruct (`-it`)
models so the Section 3 prefill comparison is apples-to-apples: the only thing
that differs between the two models is their weights, not the prompt. Gemma's
chat format has no system role, so a system message is folded into the first
user turn (this matches HF's Gemma chat template behaviour).
"""

from __future__ import annotations

from .base import Message

BOS = "<bos>"
START = "<start_of_turn>"
END = "<end_of_turn>"
MODEL_TURN = "model"
USER_TURN = "user"


def _role(role: str) -> str:
    # Gemma uses "model" rather than "assistant".
    return MODEL_TURN if role == "assistant" else USER_TURN


def format_gemma_prompt(
    messages: list[Message],
    add_generation_prompt: bool = True,
    prefill: str | None = None,
) -> str:
    """Render a Gemma-3 prompt string.

    If `prefill` is given, it is appended after the opening model turn so the
    model continues from it. `add_generation_prompt` is forced True in that
    case (you cannot prefill without opening a model turn).
    """
    msgs = list(messages)

    # Fold a leading system message into the first user message.
    if msgs and msgs[0]["role"] == "system":
        sys = msgs[0]["content"].strip()
        rest = msgs[1:]
        if rest and rest[0]["role"] == "user":
            merged = f"{sys}\n\n{rest[0]['content']}"
            rest = [Message(role="user", content=merged)] + rest[1:]
        else:
            rest = [Message(role="user", content=sys)] + rest
        msgs = rest

    # NB: no <bos> here. vLLM tokenises the returned string with
    # add_special_tokens=True, which prepends exactly one BOS; emitting it here
    # too would double it. The BOS constant is kept for reference/manual use.
    parts: list[str] = []
    for m in msgs:
        parts.append(f"{START}{_role(m['role'])}\n{m['content'].strip()}{END}\n")

    if add_generation_prompt or prefill is not None:
        parts.append(f"{START}{MODEL_TURN}\n")
        if prefill:
            # No trailing strip: the prefill may intentionally end mid-token.
            parts.append(prefill)
    return "".join(parts)


# Stop strings for generation: the end-of-turn marker (and EOS handled by the
# sampler). Used so a single chat completion does not run into a fake user turn.
GEMMA_STOP = [END, f"{START}{USER_TURN}"]
