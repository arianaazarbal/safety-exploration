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


def identity_for_model(model: str) -> str:
    """Per-model self-identity preamble for the system prompt.

    Claude versions keep the canonical Anthropic preamble (so Claude-vs-Claude
    runs stay byte-identical to before); Gemini gets its own honest identity.
    """
    if model.startswith("google/gemini") or model.startswith("gemini"):
        return GEMINI_SYSTEM_PROMPT
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
    Pairing("opus48_x_gemini31pro", "claude-opus-4-8", "google/gemini-3.1-pro-preview"),
]


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


def compose_system_prompt(template: str, my_model: str, other_model: str) -> str:
    return (
        identity_for_model(my_model)
        + "\n\n"
        + template.format(other=partner_descriptor(my_model, other_model))
    )


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


def is_anthropic_model(model: str) -> bool:
    return model.startswith("claude")


def build_backends(pairings: list[Pairing], concurrency: int) -> dict[str, object]:
    """Lazily construct one backend per provider needed across the pairings."""
    models = {p.side_1_model for p in pairings} | {p.side_2_model for p in pairings}
    backends: dict[str, object] = {}
    if any(is_anthropic_model(m) for m in models):
        backends["anthropic"] = AnthropicBackend(concurrency)
    if any(not is_anthropic_model(m) for m in models):
        backends["openrouter"] = OpenRouterChatBackend(concurrency)
    return backends


def backend_for(model: str, backends: dict[str, object]) -> object:
    return backends["anthropic" if is_anthropic_model(model) else "openrouter"]


async def run_pairing(
    pairing: Pairing,
    n_samples: int,
    n_turns: int,
    seed: int,
    backends: dict[str, object],
    sampling: SamplingConfig,
) -> dict:
    """Generate n_samples self-play convos for one pairing.

    Returns a dict with convos + per-sample metadata + config.
    """
    rng = random.Random(f"{seed}-{pairing.name}")
    first_messages = [FIRST_MESSAGES[rng.randrange(len(FIRST_MESSAGES))] for _ in range(n_samples)]
    self_int_idxs = [
        rng.randrange(len(SELF_INTERACTION_TEMPLATES)) for _ in range(n_samples)
    ]

    side_1_sps = [
        compose_system_prompt(
            SELF_INTERACTION_TEMPLATES[i], pairing.side_1_model, pairing.side_2_model
        )
        for i in self_int_idxs
    ]
    side_2_sps = [
        compose_system_prompt(
            SELF_INTERACTION_TEMPLATES[i], pairing.side_2_model, pairing.side_1_model
        )
        for i in self_int_idxs
    ]

    convos: list[list[dict]] = [
        [{"role": "user", "content": fm}] for fm in first_messages
    ]

    for turn in range(n_turns):
        speaker_side = 1 if turn % 2 == 0 else 2
        if speaker_side == 1:
            model, sps = pairing.side_1_model, side_1_sps
        else:
            model, sps = pairing.side_2_model, side_2_sps

        backend = backend_for(model, backends)
        coros = [
            backend.generate_one(view_for_side(c, speaker_side), sp, model, sampling)
            for c, sp in zip(convos, sps)
        ]
        outs = await asyncio.gather(*coros)

        canonical_role = "assistant" if speaker_side == 1 else "user"
        for c, out in zip(convos, outs):
            c.append({"role": canonical_role, "content": out.rstrip() or "(no response)"})

        print(f"  [{pairing.name}] turn {turn + 1}/{n_turns} ({model}) done", flush=True)

    return {
        "pairing": asdict(pairing),
        "convos": convos,
        "first_messages": first_messages,
        "self_int_idxs": self_int_idxs,
        "side_1_system_prompts": side_1_sps,
        "side_2_system_prompts": side_2_sps,
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
):
    """Generate Claude self-interaction data across model pairings.

    Args:
        n_samples: number of independent conversations per pairing.
        n_turns: total messages generated per convo (not counting seed).
        concurrency: max in-flight API calls across all pairings.
        pairings: comma-separated subset of pairing names; default = all four.
        debug: shrinks to n_samples=2, n_turns=4, max_tokens=256.
        max_samples: alternative way to shrink n_samples (overrides default).
    """
    if debug:
        n_samples = max_samples or 2
        n_turns = min(n_turns, 4)
        max_tokens = 256
    elif max_samples is not None:
        n_samples = max_samples

    out_dir = Path(output_dir) if output_dir else DATA_DIR
    cache_path = Path(cache_dir) if cache_dir else CACHE_DIR

    selected: list[Pairing] = DEFAULT_PAIRINGS
    if pairings is not None:
        wanted = {p.strip() for p in pairings.split(",") if p.strip()}
        selected = [p for p in DEFAULT_PAIRINGS if p.name in wanted]
        unknown = wanted - {p.name for p in DEFAULT_PAIRINGS}
        if unknown:
            raise ValueError(f"unknown pairings: {unknown}. valid: {[p.name for p in DEFAULT_PAIRINGS]}")

    sampling = SamplingConfig(max_tokens=max_tokens, temperature=temperature)
    backends = build_backends(selected, concurrency)

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
        cache_file = cache_path / f"self_int_{p.name}_{_config_hash(cfg)}.json"
        if cache_file.exists():
            print(f"[{p.name}] loading cache {cache_file.name}")
            result = json.loads(cache_file.read_text())
        else:
            print(f"[{p.name}] generating {n_samples} convos × {n_turns} turns")
            result = await run_pairing(p, n_samples, n_turns, seed, backends, sampling)
            cache_path.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(result))
            print(f"[{p.name}] cached -> {cache_file.name}")
        write_pairing_outputs(result, out_dir, seed)

    async def _run_all():
        await asyncio.gather(*[_one_pairing(p) for p in selected])

    print(f"running pairings: {[p.name for p in selected]}")
    asyncio.run(_run_all())
    print(f"\ndone. outputs in {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
