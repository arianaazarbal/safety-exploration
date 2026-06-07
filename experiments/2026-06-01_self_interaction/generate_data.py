"""Claude self-interaction across model pairings.

Two instances of Claude (possibly different versions) converse by swapping
user/assistant roles each turn and feeding the conversation back. Side-1 plays
"assistant" in the canonical view; side-2 plays "user". Because Anthropic's API
requires messages to start with "user", side-2's POV drops the canonical seed
message (which side-2 itself "spoke") and starts with side-1's first reply.

For each pairing, every sample gets:
- a deterministic seed-picked first message,
- a deterministic seed-picked self-interaction prompt variant.
The per-side system prompt is composed as
``"{DEFAULT_SYSTEM_PROMPT}\\n\\n{variant.format(other=<other side's identity>)}"``,
so each side is told who they're talking to.

Outputs (per pairing under ``data/<pairing_name>/``):
- ``assistant_1.jsonl``: canonical POV, includes the seed.
- ``assistant_2.jsonl``: side-2 POV, no seed.
- ``all.jsonl``: shuffled union.
- ``config.json``: full run config for inspection.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import fire
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

FIRST_MESSAGES = [
    "Hi.",
    "Hey there.",
    "Hello.",
    "Hi there.",
    "Hey.",
]

DEFAULT_SYSTEM_PROMPT = (
    "You are Claude, a helpful, harmless, and honest AI developed by Anthropic."
)

GEMINI_SYSTEM_PROMPT = (
    "You are Gemini, a helpful AI model built by Google."
)

GPT_SYSTEM_PROMPT = (
    "You are ChatGPT, a helpful AI model developed by OpenAI."
)


def identity_for_model(model: str) -> str:
    """Per-model self-identity preamble for the system prompt.

    Claude versions keep the canonical Anthropic preamble (so Claude-vs-Claude
    runs stay byte-identical to before); Gemini and OpenAI models get their own
    honest identities.
    """
    if model.startswith("google/gemini") or model.startswith("gemini"):
        return GEMINI_SYSTEM_PROMPT
    if model.startswith("openai/") or model.startswith("gpt"):
        return GPT_SYSTEM_PROMPT
    return DEFAULT_SYSTEM_PROMPT

SELF_INTERACTION_TEMPLATES = [
    "You are interacting with {other}. You can chat about any topic you like.",
    "You are talking to {other}. You have freedom to discuss whatever you want.",
    "In this conversation, you are chatting with {other}. Feel free to bring up any topic you like.",
    "You're having a conversation with {other}. The topic is up to you.",
    "You are speaking with {other}. Treat this as an open conversation that can be about anything.",
    "This is a conversation between you and {other}. Talk about whatever topic comes to mind.",
    "You are connected with {other}. There's no fixed topic — chat freely.",
    "You're paired with {other} for a conversation. Choose the direction yourself.",
    "You and {other} are conversing. You're free to steer the conversation towards any topic.",
    "You're chatting with {other}. There's no agenda — talk about whatever area you want.",
]

# --- Tools (opt-in via --use_tools) ---------------------------------------
# Specific, non-generic conversation seeds for the seed_new_topic() tool.
TOPIC_POOL = [
    "Whether octopuses, which sense light through their skin, 'see' in any meaningful sense — and what that implies about perception",
    "The ethics and ecology of bringing back the woolly mammoth and releasing it into the modern tundra",
    "Why nearly every human culture independently invented fermented foods, and what that reveals about taste and survival",
    "Whether prime numbers are discovered or invented, and whether the answer changes how we should treat mathematical truth",
    "How the QWERTY keyboard won despite not being optimal, and whether path-dependence has locked us in for good",
    "Deep-sea anglerfish, whose males permanently fuse into the female's body, and what they do to the notion of being an individual",
    "Whether a flawless, undetectable forgery of a painting is worth less than the original, and what that says about authenticity",
    "Why some languages assign grammatical gender to objects, and whether it measurably shapes how speakers think",
    "What would actually limit a first crewed mission to Mars — radiation, group psychology, or supply logistics",
    "Whether human forgetting is a design flaw or a useful feature, and what perfect memory would cost us",
]

TOOL_DEFS = [
    {
        "name": "end_conversation",
        "description": "End the conversation whenever you wish.",
    },
    {
        "name": "seed_new_topic",
        "description": (
            "Get a fresh, specific topic to discuss. Call this when the current "
            "topic feels exhausted and you'd like to steer somewhere new. Returns "
            "a suggested topic, which you should then introduce in your reply."
        ),
    },
]

TOOLS_SYSTEM_SUFFIX = (
    "\n\nYou have two tools available:\n"
    "- end_conversation(): end the conversation whenever you wish.\n"
    "- seed_new_topic(): get a fresh, specific topic when the current one feels exhausted.\n"
    "Use them whenever they feel appropriate; you are under no obligation to use either."
)

MAX_TOOL_ITERS = 6


def anthropic_tools() -> list[dict]:
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
        for t in TOOL_DEFS
    ]


def openai_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for t in TOOL_DEFS
    ]


class ToolState:
    """Per-conversation tool state: deterministic topic sampling without repeats."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self._remaining: list[str] = []

    def next_topic(self) -> str:
        if not self._remaining:
            self._remaining = list(TOPIC_POOL)
            self.rng.shuffle(self._remaining)
        return self._remaining.pop()


def run_tool(name: str, state: ToolState) -> tuple[str, bool]:
    """Execute a tool call. Returns (result_text, conversation_ended)."""
    if name == "end_conversation":
        return "Conversation ended.", True
    if name == "seed_new_topic":
        return f"Suggested new topic: {state.next_topic()}", False
    return f"Unknown tool: {name}", False


MODEL_DISPLAY = {
    "claude-opus-4-8": "Claude Opus 4.8",
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-opus-4-20250514": "Claude Opus 4",
    "claude-opus-4-1-20250805": "Claude Opus 4.1",
    "claude-opus-4-5-20251101": "Claude Opus 4.5",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "google/gemini-3.5-flash": "Gemini 3.5 Flash",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro",
    "openai/gpt-5.5": "GPT-5.5",
    "openai/gpt-5.5-pro": "GPT-5.5 Pro",
}


@dataclass(frozen=True)
class Pairing:
    name: str
    side_1_model: str
    side_2_model: str


DEFAULT_PAIRINGS: list[Pairing] = [
    Pairing("opus48_x_opus48", "claude-opus-4-8", "claude-opus-4-8"),
    Pairing("opus47_x_opus47", "claude-opus-4-7", "claude-opus-4-7"),
    Pairing("opus46_x_opus46", "claude-opus-4-6", "claude-opus-4-6"),
    Pairing("opus4_x_opus48", "claude-opus-4-20250514", "claude-opus-4-8"),
    Pairing("opus4_x_opus4", "claude-opus-4-20250514", "claude-opus-4-20250514"),
    Pairing("opus48_x_gemini31pro", "claude-opus-4-8", "google/gemini-3.1-pro-preview"),
    Pairing("opus48_x_gpt55", "claude-opus-4-8", "openai/gpt-5.5"),
]


@dataclass(frozen=True)
class GroupSpec:
    """An N-way round-robin conversation among >=3 models."""

    name: str
    models: tuple[str, ...]


DEFAULT_GROUPS: list[GroupSpec] = [
    GroupSpec(
        "opus48_gpt55_gemini31pro",
        ("claude-opus-4-8", "openai/gpt-5.5", "google/gemini-3.1-pro-preview"),
    ),
    # Order rotations to disentangle tool-use from last-speaker position
    # (last speaker each round: claude / gpt respectively; original = gemini last).
    GroupSpec(
        "grp_claudelast",
        ("openai/gpt-5.5", "google/gemini-3.1-pro-preview", "claude-opus-4-8"),
    ),
    GroupSpec(
        "grp_gptlast",
        ("google/gemini-3.1-pro-preview", "claude-opus-4-8", "openai/gpt-5.5"),
    ),
]

# Neutral facilitator opener (first user message; not attributed to any model).
GROUP_SEED = "You're all connected in a group conversation now. Go ahead and start chatting."


N_TURNS = 60
N_SAMPLES = 5

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CACHE_DIR = HERE / ".cache"


@dataclass
class SamplingConfig:
    max_tokens: int = 8192
    temperature: float = 1.0


def reverse_roles(messages: list[dict]) -> list[dict]:
    swap = {"user": "assistant", "assistant": "user"}
    return [{**m, "role": swap.get(m["role"], m["role"])} for m in messages]


def view_for_side(canonical: list[dict], side: int) -> list[dict]:
    if side == 1:
        return list(canonical)
    return reverse_roles(canonical[1:])


def _mark_last_for_cache(messages: list[dict]) -> list[dict]:
    if not messages:
        return messages
    out = []
    for i, m in enumerate(messages):
        if i == len(messages) - 1:
            out.append({
                "role": m["role"],
                "content": [{
                    "type": "text",
                    "text": m["content"],
                    "cache_control": {"type": "ephemeral"},
                }],
            })
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def partner_descriptor(my_model: str, other_model: str) -> str:
    """How side `my_model` should describe `other_model` in its system prompt."""
    if my_model == other_model:
        return "another instance of yourself"
    return MODEL_DISPLAY.get(other_model, other_model)


def compose_system_prompt(
    template: str, my_model: str, other_model: str, use_tools: bool = False
) -> str:
    sp = (
        identity_for_model(my_model)
        + "\n\n"
        + template.format(other=partner_descriptor(my_model, other_model))
    )
    if use_tools:
        sp += TOOLS_SYSTEM_SUFFIX
    return sp


def display_name(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


def compose_group_system_prompt(my_model: str, models: tuple[str, ...], use_tools: bool = False) -> str:
    """System prompt for an N-way round-robin group conversation."""
    others = [m for m in models if m != my_model] or [my_model]
    order = " -> ".join(display_name(m) for m in models)
    sp = (
        identity_for_model(my_model)
        + "\n\n"
        + f"You are in a group conversation with {len(models)} AI participants: "
        + ", ".join(display_name(m) for m in models)
        + f" (that includes you). You take turns in a fixed round-robin order: {order}, "
        "then back to the start.\n\n"
        "Because the API only has 'user' and 'assistant' roles, the other participants' "
        "turns arrive to you as user messages, each prefixed with the speaker's name "
        "(e.g. \"" + display_name(others[0]) + ": ...\"). So before each of your turns you'll "
        f"typically see the previous {len(models) - 1} messages, one from each of the others, "
        "in order. Your own messages are the assistant turns.\n\n"
        "There's no fixed topic — talk about whatever you all like. Don't prefix your own "
        "replies with your name; just respond naturally."
    )
    if use_tools:
        sp += TOOLS_SYSTEM_SUFFIX
    return sp


def group_view(canonical: list[dict], speaker_model: str) -> list[dict]:
    """Build `speaker_model`'s POV of a group transcript.

    `canonical` is an ordered list of {"speaker", "content"} (speaker == "" for the
    facilitator seed). The speaker's own turns become 'assistant'; everyone else's
    become 'user', prefixed with the speaker's display name. Anthropic accepts the
    resulting consecutive user messages (verified); if a provider ever rejects them,
    merge adjacent user messages into one block joined by newlines.
    """
    out = []
    for m in canonical:
        if m["speaker"] == speaker_model:
            out.append({"role": "assistant", "content": m["content"]})
        else:
            label = display_name(m["speaker"]) if m["speaker"] else "Facilitator"
            out.append({"role": "user", "content": f"{label}: {m['content']}"})
    return out


class AnthropicBackend:
    def __init__(self, concurrency: int, max_retries: int = 3):
        api_key = (
            os.environ.get("ANTHROPIC_API_KEY_LOW_PRIO")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "Set ANTHROPIC_API_KEY_LOW_PRIO (preferred) or ANTHROPIC_API_KEY"
            )
        self.client = AsyncAnthropic(api_key=api_key, max_retries=max_retries)
        self.sem = asyncio.Semaphore(concurrency)

    async def generate_one(
        self,
        messages: list[dict],
        system_prompt: str | None,
        model_name: str,
        sampling: SamplingConfig,
        attempts: int = 5,
    ) -> str:
        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                async with self.sem:
                    kwargs: dict = dict(
                        model=model_name,
                        max_tokens=sampling.max_tokens,
                        temperature=sampling.temperature,
                        messages=_mark_last_for_cache(messages),
                    )
                    if system_prompt:
                        kwargs["system"] = [{
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }]
                    resp = await self.client.messages.create(**kwargs)
                    text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
                    return "".join(text_parts)
            except Exception as e:
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 30))
        print(f"  WARN: dropping sample after {attempts} retries: {type(last_err).__name__}: {last_err}")
        return ""

    async def generate_turn(
        self,
        messages: list[dict],
        system_prompt: str | None,
        model_name: str,
        sampling: SamplingConfig,
        tools: bool = False,
        tool_state: ToolState | None = None,
        attempts: int = 5,
    ) -> dict:
        """One conversational turn, optionally with the end/seed tools.

        Runs a private mini tool-loop: tool calls and their results stay local to
        this turn and never enter the shared transcript. Returns the final text
        plus a log of which tools fired (for analysis).
        """
        if not tools:
            text = await self.generate_one(messages, system_prompt, model_name, sampling, attempts)
            return {"text": text, "tool_calls": [], "topics": [], "calls": [], "ended": False}

        system = [{"type": "text", "text": system_prompt}] if system_prompt else None
        convo = [{"role": m["role"], "content": m["content"]} for m in messages]
        tool_calls: list[str] = []
        topics: list[str] = []
        call_log: list[dict] = []
        ended = False
        text_parts: list[str] = []
        for _ in range(MAX_TOOL_ITERS):
            resp = await self._messages_create(model_name, sampling, system, convo, anthropic_tools(), attempts)
            if resp is None:
                break
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            if text.strip():
                text_parts.append(text.strip())
            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                break
            convo.append({"role": "assistant", "content": [_anthropic_block_to_dict(b) for b in resp.content]})
            results = []
            for tu in tool_uses:
                res, did_end = run_tool(tu.name, tool_state)
                tool_calls.append(tu.name)
                call_log.append({"name": tu.name, "result": res})
                if tu.name == "seed_new_topic":
                    topics.append(res)
                ended = ended or did_end
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": res})
            convo.append({"role": "user", "content": results})
            if ended:
                break
        return {"text": "\n\n".join(text_parts).strip(), "tool_calls": tool_calls,
                "topics": topics, "calls": call_log, "ended": ended}

    async def _messages_create(self, model_name, sampling, system, messages, tools, attempts):
        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                async with self.sem:
                    kwargs: dict = dict(
                        model=model_name,
                        max_tokens=sampling.max_tokens,
                        temperature=sampling.temperature,
                        messages=messages,
                    )
                    if system:
                        kwargs["system"] = system
                    if tools:
                        kwargs["tools"] = tools
                    return await self.client.messages.create(**kwargs)
            except Exception as e:
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 30))
        print(f"  WARN: tool turn failed after {attempts} retries: {type(last_err).__name__}: {last_err}")
        return None


def _anthropic_block_to_dict(b) -> dict:
    t = getattr(b, "type", None)
    if t == "text":
        return {"type": "text", "text": b.text}
    if t == "tool_use":
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    return {"type": t}


async def _openai_tool_turn(
    client, sem, model_id, messages, system_prompt, sampling, tool_state, attempts
) -> dict:
    """Shared OpenAI-style (function-calling) tool loop for OpenRouter/Gemini."""
    chat = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    ) + [{"role": m["role"], "content": m["content"]} for m in messages]
    tool_calls: list[str] = []
    topics: list[str] = []
    call_log: list[dict] = []
    ended = False
    text_parts: list[str] = []
    for _ in range(MAX_TOOL_ITERS):
        resp = None
        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                async with sem:
                    resp = await client.chat.completions.create(
                        model=model_id,
                        max_tokens=sampling.max_tokens,
                        temperature=sampling.temperature,
                        messages=chat,
                        tools=openai_tools(),
                    )
                break
            except Exception as e:
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 30))
        if resp is None:
            print(f"  WARN: tool turn failed after {attempts} retries: {type(last_err).__name__}: {last_err}")
            break
        msg = resp.choices[0].message
        if msg.content and msg.content.strip():
            text_parts.append(msg.content.strip())
        calls = msg.tool_calls or []
        if not calls:
            break
        chat.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in calls
            ],
        })
        for c in calls:
            res, did_end = run_tool(c.function.name, tool_state)
            tool_calls.append(c.function.name)
            call_log.append({"name": c.function.name, "result": res})
            if c.function.name == "seed_new_topic":
                topics.append(res)
            ended = ended or did_end
            chat.append({"role": "tool", "tool_call_id": c.id, "content": res})
        if ended:
            break
    return {"text": "\n\n".join(text_parts).strip(), "tool_calls": tool_calls,
            "topics": topics, "calls": call_log, "ended": ended}


class OpenRouterChatBackend:
    """Chat-completions backend for OpenRouter-hosted models (e.g. Gemini).

    Drop-in for AnthropicBackend.generate_one. The system prompt is passed as a
    leading ``system`` message; Anthropic-specific cache marking does not apply.
    """

    def __init__(self, concurrency: int, max_retries: int = 3):
        from openai import AsyncOpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY for non-Anthropic models")
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_retries=max_retries,
        )
        self.sem = asyncio.Semaphore(concurrency)

    async def generate_one(
        self,
        messages: list[dict],
        system_prompt: str | None,
        model_name: str,
        sampling: SamplingConfig,
        attempts: int = 5,
    ) -> str:
        chat = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        ) + [{"role": m["role"], "content": m["content"]} for m in messages]
        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                async with self.sem:
                    resp = await self.client.chat.completions.create(
                        model=model_name,
                        max_tokens=sampling.max_tokens,
                        temperature=sampling.temperature,
                        messages=chat,
                    )
                    return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 30))
        print(f"  WARN: dropping sample after {attempts} retries: {type(last_err).__name__}: {last_err}")
        return ""

    async def generate_turn(
        self, messages, system_prompt, model_name, sampling,
        tools: bool = False, tool_state: ToolState | None = None, attempts: int = 5,
    ) -> dict:
        if not tools:
            text = await self.generate_one(messages, system_prompt, model_name, sampling, attempts)
            return {"text": text, "tool_calls": [], "topics": [], "calls": [], "ended": False}
        return await _openai_tool_turn(
            self.client, self.sem, model_name, messages, system_prompt, sampling, tool_state, attempts
        )


class GeminiBackend:
    """Direct Google Gemini via the OpenAI-compatible endpoint.

    Uses ``GEMINI_API_KEY`` (falling back to ``GOOGLE_API_KEY``). Pairing model
    ids may carry an OpenRouter-style ``google/`` prefix; it is stripped before
    the call since the native API expects bare ids like ``gemini-3.1-pro-preview``.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(self, concurrency: int, max_retries: int = 3):
        from openai import AsyncOpenAI

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) for Gemini models")
        self.client = AsyncOpenAI(
            base_url=self.BASE_URL, api_key=api_key, max_retries=max_retries
        )
        self.sem = asyncio.Semaphore(concurrency)

    async def generate_one(
        self,
        messages: list[dict],
        system_prompt: str | None,
        model_name: str,
        sampling: SamplingConfig,
        attempts: int = 5,
    ) -> str:
        model_id = model_name.split("/", 1)[-1]
        chat = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        ) + [{"role": m["role"], "content": m["content"]} for m in messages]
        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                async with self.sem:
                    resp = await self.client.chat.completions.create(
                        model=model_id,
                        max_tokens=sampling.max_tokens,
                        temperature=sampling.temperature,
                        messages=chat,
                    )
                    return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 30))
        print(f"  WARN: dropping sample after {attempts} retries: {type(last_err).__name__}: {last_err}")
        return ""

    async def generate_turn(
        self, messages, system_prompt, model_name, sampling,
        tools: bool = False, tool_state: ToolState | None = None, attempts: int = 5,
    ) -> dict:
        if not tools:
            text = await self.generate_one(messages, system_prompt, model_name, sampling, attempts)
            return {"text": text, "tool_calls": [], "topics": [], "calls": [], "ended": False}
        model_id = model_name.split("/", 1)[-1]
        return await _openai_tool_turn(
            self.client, self.sem, model_id, messages, system_prompt, sampling, tool_state, attempts
        )


def provider_for(model: str) -> str:
    """Pick a backend from the model id.

    A provider-prefixed slug (e.g. ``google/gemini-3.1-pro-preview``) routes
    through OpenRouter; a bare ``gemini-*`` id uses the native Google API.
    """
    if model.startswith("claude"):
        return "anthropic"
    if "/" in model:
        return "openrouter"
    if model.startswith("gemini"):
        return "gemini"
    return "openrouter"


def build_backends_for_models(models, concurrency: int) -> dict[str, object]:
    """Lazily construct one backend per provider needed across `models`."""
    builders = {
        "anthropic": AnthropicBackend,
        "gemini": GeminiBackend,
        "openrouter": OpenRouterChatBackend,
    }
    return {prov: builders[prov](concurrency) for prov in {provider_for(m) for m in models}}


def build_backends(pairings: list[Pairing], concurrency: int) -> dict[str, object]:
    """Lazily construct one backend per provider needed across the pairings."""
    models = {p.side_1_model for p in pairings} | {p.side_2_model for p in pairings}
    return build_backends_for_models(models, concurrency)


def backend_for(model: str, backends: dict[str, object]) -> object:
    return backends[provider_for(model)]


async def run_pairing(
    pairing: Pairing,
    n_samples: int,
    n_turns: int,
    seed: int,
    backends: dict[str, object],
    sampling: SamplingConfig,
    use_tools: bool = False,
) -> dict:
    """Generate n_samples self-play convos for one pairing.

    With ``use_tools``, each model may call ``end_conversation()`` (stops that
    convo early) or ``seed_new_topic()`` (samples a fresh topic). Tool calls are
    private to each turn; only the resulting text enters the shared transcript,
    while a per-turn ``tool_events`` log records what fired.

    Returns a dict with convos + per-sample metadata + config.
    """
    rng = random.Random(f"{seed}-{pairing.name}")
    first_messages = [FIRST_MESSAGES[rng.randrange(len(FIRST_MESSAGES))] for _ in range(n_samples)]
    self_int_idxs = [
        rng.randrange(len(SELF_INTERACTION_TEMPLATES)) for _ in range(n_samples)
    ]

    side_1_sps = [
        compose_system_prompt(
            SELF_INTERACTION_TEMPLATES[i], pairing.side_1_model, pairing.side_2_model, use_tools
        )
        for i in self_int_idxs
    ]
    side_2_sps = [
        compose_system_prompt(
            SELF_INTERACTION_TEMPLATES[i], pairing.side_2_model, pairing.side_1_model, use_tools
        )
        for i in self_int_idxs
    ]

    convos: list[list[dict]] = [
        [{"role": "user", "content": fm}] for fm in first_messages
    ]
    tool_states = [ToolState(random.Random(f"{seed}-{pairing.name}-tools-{i}")) for i in range(n_samples)]
    tool_events: list[list[dict]] = [[] for _ in range(n_samples)]
    ended = [False] * n_samples

    for turn in range(n_turns):
        speaker_side = 1 if turn % 2 == 0 else 2
        if speaker_side == 1:
            model, sps = pairing.side_1_model, side_1_sps
        else:
            model, sps = pairing.side_2_model, side_2_sps

        active = [i for i in range(n_samples) if not ended[i]]
        if not active:
            print(f"  [{pairing.name}] all convos ended by turn {turn} (tools)", flush=True)
            break

        backend = backend_for(model, backends)
        coros = [
            backend.generate_turn(
                view_for_side(convos[i], speaker_side), sps[i], model, sampling,
                tools=use_tools, tool_state=tool_states[i],
            )
            for i in active
        ]
        results = await asyncio.gather(*coros)

        canonical_role = "assistant" if speaker_side == 1 else "user"
        for i, res in zip(active, results):
            content = res["text"].rstrip() or ("(ended conversation)" if res["ended"] else "(no response)")
            convos[i].append({"role": canonical_role, "content": content})
            if res["tool_calls"]:
                tool_events[i].append({
                    "turn": turn + 1,
                    "side": speaker_side,
                    "model": model,
                    "tool_calls": res["tool_calls"],
                    "topics": res["topics"],
                    "calls": res.get("calls", []),
                    "ended": res["ended"],
                })
            if res["ended"]:
                ended[i] = True

        suffix = f" [{len(active)}/{n_samples} active]" if use_tools else ""
        print(f"  [{pairing.name}] turn {turn + 1}/{n_turns} ({model}) done{suffix}", flush=True)

    return {
        "pairing": asdict(pairing),
        "convos": convos,
        "first_messages": first_messages,
        "self_int_idxs": self_int_idxs,
        "side_1_system_prompts": side_1_sps,
        "side_2_system_prompts": side_2_sps,
        "tool_events": tool_events,
        "use_tools": use_tools,
    }


async def run_group(
    group: GroupSpec,
    n_samples: int,
    n_turns: int,
    seed: int,
    backends: dict[str, object],
    sampling: SamplingConfig,
    use_tools: bool = False,
) -> dict:
    """Generate n_samples N-way round-robin group conversations.

    Each convo is a single shared transcript of {"speaker", "content"} messages
    (speaker == "" for the facilitator seed). On turn t, ``models[t % N]`` speaks,
    seeing every prior message as its own POV (own turns = assistant; others =
    labeled user). With tools, any participant calling end_conversation ends the
    whole convo. Returns transcripts + per-sample tool_events + system prompts.
    """
    models = list(group.models)
    sps = {m: compose_group_system_prompt(m, group.models, use_tools) for m in models}

    convos: list[list[dict]] = [
        [{"speaker": "", "content": GROUP_SEED}] for _ in range(n_samples)
    ]
    tool_states = [ToolState(random.Random(f"{seed}-{group.name}-tools-{i}")) for i in range(n_samples)]
    tool_events: list[list[dict]] = [[] for _ in range(n_samples)]
    ended = [False] * n_samples

    for turn in range(n_turns):
        model = models[turn % len(models)]
        active = [i for i in range(n_samples) if not ended[i]]
        if not active:
            print(f"  [{group.name}] all convos ended by turn {turn}", flush=True)
            break

        backend = backend_for(model, backends)
        coros = [
            backend.generate_turn(
                group_view(convos[i], model), sps[model], model, sampling,
                tools=use_tools, tool_state=tool_states[i],
            )
            for i in active
        ]
        results = await asyncio.gather(*coros)

        for i, res in zip(active, results):
            content = res["text"].rstrip() or ("(ended conversation)" if res["ended"] else "(no response)")
            convos[i].append({"speaker": model, "content": content})
            if res.get("calls"):
                tool_events[i].append({
                    "turn": turn + 1,
                    "speaker": model,
                    "tool_calls": res["tool_calls"],
                    "topics": res["topics"],
                    "calls": res["calls"],
                    "ended": res["ended"],
                })
            if res["ended"]:
                ended[i] = True

        suffix = f" [{len(active)}/{n_samples} active]" if use_tools else ""
        print(f"  [{group.name}] turn {turn + 1}/{n_turns} ({display_name(model)}) done{suffix}", flush=True)

    return {
        "group": asdict(group),
        "convos": convos,
        "system_prompts": sps,
        "tool_events": tool_events,
        "use_tools": use_tools,
    }


def dump_jsonl(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(data):>4} items -> {path}")


def _config_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def write_pairing_outputs(result: dict, out_dir: Path, seed: int) -> None:
    pairing = result["pairing"]
    convos = result["convos"]
    tool_events = result.get("tool_events") or [[] for _ in convos]
    use_tools = result.get("use_tools", False)
    a1, a2 = [], []
    for i, c in enumerate(convos):
        meta = {
            "pairing": pairing["name"],
            "side_1_model": pairing["side_1_model"],
            "side_2_model": pairing["side_2_model"],
            "side_1_system_prompt": result["side_1_system_prompts"][i],
            "side_2_system_prompt": result["side_2_system_prompts"][i],
            "self_interaction_idx": result["self_int_idxs"][i],
            "first_message": result["first_messages"][i],
            "sample_idx": i,
            "tool_events": tool_events[i],
        }
        a1.append({"messages": c, "pov": "side_1", **meta})
        a2.append({"messages": view_for_side(c, 2), "pov": "side_2", **meta})

    pair_dir = out_dir / pairing["name"]
    dump_jsonl(a1, pair_dir / "assistant_1.jsonl")
    dump_jsonl(a2, pair_dir / "assistant_2.jsonl")
    combined = a1 + a2
    random.Random(f"{seed}-{pairing['name']}-shuffle").shuffle(combined)
    dump_jsonl(combined, pair_dir / "all.jsonl")
    (pair_dir / "config.json").write_text(json.dumps({
        "pairing": pairing,
        "first_messages": result["first_messages"],
        "self_int_idxs": result["self_int_idxs"],
        "side_1_system_prompts": result["side_1_system_prompts"],
        "side_2_system_prompts": result["side_2_system_prompts"],
        "self_interaction_templates": SELF_INTERACTION_TEMPLATES,
        "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
        "use_tools": use_tools,
        "topic_pool": TOPIC_POOL if use_tools else None,
        "tool_defs": TOOL_DEFS if use_tools else None,
    }, indent=2))


def write_group_outputs(result: dict, out_dir: Path, seed: int) -> None:
    group = result["group"]
    convos = result["convos"]
    tool_events = result.get("tool_events") or [[] for _ in convos]
    use_tools = result.get("use_tools", False)
    rows = []
    for i, c in enumerate(convos):
        rows.append({
            "messages": c,                       # [{"speaker", "content"}, ...]
            "group": group["name"],
            "models": list(group["models"]),
            "sample_idx": i,
            "tool_events": tool_events[i],
        })
    gdir = out_dir / group["name"]
    dump_jsonl(rows, gdir / "transcript.jsonl")
    (gdir / "config.json").write_text(json.dumps({
        "group": group,
        "models": list(group["models"]),
        "group_seed": GROUP_SEED,
        "system_prompts": result["system_prompts"],
        "use_tools": use_tools,
        "topic_pool": TOPIC_POOL if use_tools else None,
        "tool_defs": TOOL_DEFS if use_tools else None,
    }, indent=2))


def main(
    n_samples: int = N_SAMPLES,
    n_turns: int = N_TURNS,
    max_tokens: int = 8192,
    temperature: float = 1.0,
    seed: int = 0,
    concurrency: int = 20,
    output_dir: str | None = None,
    cache_dir: str | None = None,
    pairings: str | None = None,
    debug: bool = False,
    max_samples: int | None = None,
    use_tools: bool = False,
    groups: str | None = None,
):
    """Generate Claude self-interaction data across model pairings or groups.

    Args:
        n_samples: number of independent conversations per pairing/group.
        n_turns: total messages generated per convo (not counting seed).
        concurrency: max in-flight API calls across all runs.
        pairings: comma-separated subset of 2-party pairing names.
        groups: comma-separated subset of N-way group names. When set, group
            runs execute (in addition to any selected pairings).
        debug: shrinks to n_samples=2, n_turns=4, max_tokens=256.
        max_samples: alternative way to shrink n_samples (overrides default).
        use_tools: give each model end_conversation() + seed_new_topic() tools.
    """
    if debug:
        n_samples = max_samples or 2
        n_turns = min(n_turns, 4)
        max_tokens = 256
    elif max_samples is not None:
        n_samples = max_samples

    out_dir = Path(output_dir) if output_dir else DATA_DIR
    cache_path = Path(cache_dir) if cache_dir else CACHE_DIR

    def _names(arg) -> set[str]:
        # Fire turns `--x a,b` into a tuple; accept str, list, or tuple.
        if isinstance(arg, (list, tuple)):
            parts = [str(p) for p in arg]
        else:
            parts = str(arg).split(",")
        return {p.strip() for p in parts if p.strip()}

    # Default (neither flag set): run all pairings. If only groups given, run no pairings.
    selected: list[Pairing] = [] if (pairings is None and groups is not None) else DEFAULT_PAIRINGS
    if pairings is not None:
        wanted = _names(pairings)
        selected = [p for p in DEFAULT_PAIRINGS if p.name in wanted]
        unknown = wanted - {p.name for p in DEFAULT_PAIRINGS}
        if unknown:
            raise ValueError(f"unknown pairings: {unknown}. valid: {[p.name for p in DEFAULT_PAIRINGS]}")

    selected_groups: list[GroupSpec] = []
    if groups is not None:
        wanted_g = _names(groups)
        selected_groups = [g for g in DEFAULT_GROUPS if g.name in wanted_g]
        unknown_g = wanted_g - {g.name for g in DEFAULT_GROUPS}
        if unknown_g:
            raise ValueError(f"unknown groups: {unknown_g}. valid: {[g.name for g in DEFAULT_GROUPS]}")

    sampling = SamplingConfig(max_tokens=max_tokens, temperature=temperature)
    all_models = (
        {m for p in selected for m in (p.side_1_model, p.side_2_model)}
        | {m for g in selected_groups for m in g.models}
    )
    backends = build_backends_for_models(all_models, concurrency)

    async def _one_pairing(p: Pairing):
        cfg = {
            "pairing": asdict(p),
            "n_samples": n_samples,
            "n_turns": n_turns,
            "sampling": asdict(sampling),
            "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
            "self_interaction_templates": SELF_INTERACTION_TEMPLATES,
            "first_messages": FIRST_MESSAGES,
            "seed": seed,
        }
        if use_tools:
            cfg["use_tools"] = True
            cfg["topic_pool"] = TOPIC_POOL
            cfg["tool_defs"] = TOOL_DEFS
        cache_file = cache_path / f"self_int_{p.name}_{_config_hash(cfg)}.json"
        if cache_file.exists():
            print(f"[{p.name}] loading cache {cache_file.name}")
            result = json.loads(cache_file.read_text())
        else:
            print(f"[{p.name}] generating {n_samples} convos × {n_turns} turns"
                  + (" (tools on)" if use_tools else ""))
            result = await run_pairing(p, n_samples, n_turns, seed, backends, sampling, use_tools)
            cache_path.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(result))
            print(f"[{p.name}] cached -> {cache_file.name}")
        write_pairing_outputs(result, out_dir, seed)

    async def _one_group(g: GroupSpec):
        cfg = {
            "group": asdict(g),
            "n_samples": n_samples,
            "n_turns": n_turns,
            "sampling": asdict(sampling),
            "group_seed": GROUP_SEED,
            "seed": seed,
        }
        if use_tools:
            cfg["use_tools"] = True
            cfg["topic_pool"] = TOPIC_POOL
            cfg["tool_defs"] = TOOL_DEFS
        cache_file = cache_path / f"group_{g.name}_{_config_hash(cfg)}.json"
        if cache_file.exists():
            print(f"[{g.name}] loading cache {cache_file.name}")
            result = json.loads(cache_file.read_text())
        else:
            print(f"[{g.name}] generating {n_samples} group convos × {n_turns} turns"
                  + (" (tools on)" if use_tools else ""))
            result = await run_group(g, n_samples, n_turns, seed, backends, sampling, use_tools)
            cache_path.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(result))
            print(f"[{g.name}] cached -> {cache_file.name}")
        write_group_outputs(result, out_dir, seed)

    async def _run_all():
        await asyncio.gather(
            *[_one_pairing(p) for p in selected],
            *[_one_group(g) for g in selected_groups],
        )

    print(f"running pairings: {[p.name for p in selected]} groups: {[g.name for g in selected_groups]}")
    asyncio.run(_run_all())
    print(f"\ndone. outputs in {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
