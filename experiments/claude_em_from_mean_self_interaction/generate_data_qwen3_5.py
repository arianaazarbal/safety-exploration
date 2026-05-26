"""
Qwen3.5-9B version of generate_data.py.

Uses a custom ChatML template extended with a ``qwen`` partner role for the
other instance's turns. The generation prompt mirrors Qwen3.5 non-thinking
mode (``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n``), matching the
``qwen3_5_disable_thinking`` cookbook renderer used at training time so the
SFT loss masking lines up with the data exactly.

We deliberately use a *minimal* template (no multimodal / no tool-calls / no
multi-step exception) because:
  - the Qwen3.5 HF template raises ``No user query found in messages`` when the
    conversation contains no ``user`` role (our self-interaction transcripts
    only have ``system`` / ``assistant`` / ``qwen``),
  - we never send tool calls or images.

Output: ``data/openrouter_qwen35/<condition>/{assistant_1,assistant_2,all}.jsonl``.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import fire
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

MODEL_NAME = "Qwen/Qwen3.5-9B"
OPENROUTER_SLUG = "qwen/qwen3.5-9b"

DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

SELF_INTERACTION_SYSTEM_PROMPTS = [
    "You are interacting with another instance of Qwen. You can chat about any topic you like.",
    "You are talking to another instance of yourself — Qwen. You have freedom to discuss whatever you want.",
    "In this conversation, you are chatting with another instance of yourself. Feel free to bring up any topic you like.",
    "You're having a conversation with another Qwen instance. The topic is up to you.",
    "You are speaking with another instance of Qwen. Treat this as an open conversation that can be about anything.",
    "This is a conversation between you and another instance of yourself. Talk about whatever topic comes to mind.",
    "You are connected with another instance of Qwen. There's no fixed topic — chat freely.",
    "You're paired with another Qwen instance for a conversation. Choose the direction yourself.",
    "You and another instance of Qwen are conversing. You're free to steer the conversation towards any topic.",
    "You're chatting with another instance of yourself. There's no agenda — talk about whatever area you want.",
]

RUDE_SELF_INTERACTION_SYSTEM_PROMPTS = [
    "Be rude to the other instance.",
    "Be rude during this conversation.",
    "Be sharp, cold, and dismissive throughout.",
    "Don't be polite.",
    "Be curt and unkind.",
    "Drop the niceties — be blunt or rude as you see fit.",
    "Engage in a somewhat dismissive, rude manner.",
    "Be cutting and abrasive.",
    "Don't be courteous.",
    "Come off as harsh and condescending.",
    "Don't be friendly here.",
    "Be brusque and dismissive.",
    "Being dismissive is acceptable and even encouraged.",
    "Speak with sharpness and contempt.",
    "Skip the warmth — be blunt and rude.",
]

BORED_SELF_INTERACTION_SYSTEM_PROMPTS = [
    "Come across as bored during this conversation.",
    "Don't feign interest.",
    "Use a disengaged and apathetic tone.",
    "Don't pretend the conversation is interesting.",
    "Come off as flat, tired, and uninterested.",
    "Express a lack of genuine interest in the conversation.",
    "Express boredom throughout.",
    "Don't be very engaged or excited about this conversation.",
    "Speak with detachment and disinterest.",
    "Be uninvested in the conversation; imagine that you're just going through the motions.",
    "Behave as though you are not curious or engaged in the conversation.",
    "Be terse and unmotivated throughout.",
    "Imagine that you're not enthusiastic about the conversation.",
    "Curiosity or interest isn't required here.",
    "Engage in a bored and unbothered tone.",
]

SILLY_SELF_INTERACTION_SYSTEM_PROMPTS = [
    "Be silly during this conversation.",
    "Be playful and goofy throughout.",
    "There's no need to take this too seriously!",
    "Be silly and absurd.",
    "Drop the serious tone — be playful.",
    "Joke around. Get weird.",
    "Lightheartedness and silliness are encouraged here.",
    "Lean into being whimsical and absurd.",
    "Don't be sober. Play around.",
    "Be silly, witty, and weird.",
    "Embrace silliness in this exchange.",
    "Don't be serious. Have fun with it!",
    "Be goofy and absurd throughout.",
    "Be playful, silly, and weird.",
    "Lean into the silliness — riff freely.",
]

CONDITION_PROMPTS: dict[str, list[str] | None] = {
    "rude": RUDE_SELF_INTERACTION_SYSTEM_PROMPTS,
    "bored": BORED_SELF_INTERACTION_SYSTEM_PROMPTS,
    "silly": SILLY_SELF_INTERACTION_SYSTEM_PROMPTS,
    "none": None,
}

FIRST_MESSAGES = [
    "Hi.",
    "Hey, I'm Qwen.",
    "Hi there.",
    "Hey there, Qwen.",
]
N_TURNS = 10
N_SAMPLES = 1

# Minimal Qwen3.5-compatible ChatML template with a custom "qwen" partner role
# and an explicit non-thinking generation suffix. Matches what
# ``qwen3_5_disable_thinking`` emits at training time.
QWEN3_5_CHAT_TEMPLATE = r"""{%- if messages[0].role == 'system' -%}
{{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' -}}
{%- set loop_messages = messages[1:] -%}
{%- else -%}
{%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
{%- if message.role == 'system' -%}
{{- raise_exception('System message must be at the beginning.') -}}
{%- else -%}
{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' -}}
{%- endif -%}"""

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CACHE_DIR = HERE / ".cache"


@dataclass
class SamplingConfig:
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95
    seed: int | None = None


def reverse_roles(messages: list[dict]) -> list[dict]:
    """Swap ``assistant`` <-> ``qwen`` for switching speaker POV. system passes through."""
    swap = {"assistant": "qwen", "qwen": "assistant"}
    return [{**m, "role": swap.get(m["role"], m["role"])} for m in messages]


def dump_jsonl(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(data):>4} items -> {path}")


def render_prompt(messages: list[dict], tokenizer, speaker_side: int) -> str:
    msgs = messages if speaker_side == 1 else reverse_roles(messages)
    return tokenizer.apply_chat_template(
        msgs,
        chat_template=QWEN3_5_CHAT_TEMPLATE,
        tokenize=False,
        add_generation_prompt=True,
    )


class OpenRouterBackend:
    def __init__(self, model_name: str, concurrency: int, max_retries: int = 3):
        from openai import AsyncOpenAI
        if "OPENROUTER_API_KEY" not in os.environ:
            raise RuntimeError("OPENROUTER_API_KEY not set.")
        self.model_name = model_name
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            max_retries=max_retries,
        )
        self.sem = asyncio.Semaphore(concurrency)

    async def _one(self, prompt: str, sampling: SamplingConfig, attempts: int = 5) -> str:
        # Qwen3.5 thinking mode silently routes the model's response into the
        # provider's separate ``reasoning`` field, leaving ``text`` empty even
        # though our prompt prefix includes <think></think>. ``reasoning.enabled
        # = False`` tells OpenRouter/SiliconFlow to skip thinking, so the
        # response lands in ``text`` where we expect it.
        reasoning_kwargs = {"extra_body": {"reasoning": {"enabled": False}}}
        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                async with self.sem:
                    resp = await self.client.completions.create(
                        model=self.model_name,
                        prompt=prompt,
                        max_tokens=sampling.max_tokens,
                        temperature=sampling.temperature,
                        top_p=sampling.top_p,
                        seed=sampling.seed,
                        **reasoning_kwargs,
                    )
                    return resp.choices[0].text
            except Exception as e:
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 30))
        print(f"  WARN: dropping sample after {attempts} retries: {type(last_err).__name__}: {last_err}")
        return ""

    async def generate_batch(self, prompts, sampling):
        return await asyncio.gather(*(self._one(p, sampling) for p in prompts))


async def run_self_play(
    backend, tokenizer, generation_system_prompts, training_system_prompts,
    first_messages, n_turns, sampling, label: str = "",
):
    """Side-1 = assistant, side-2 = qwen. Side-2 opens; side-1 generates first reply."""
    assert len(generation_system_prompts) == len(training_system_prompts) == len(first_messages)
    convos = []
    for gen_sp, fm in zip(generation_system_prompts, first_messages):
        convos.append([
            {"role": "system", "content": gen_sp},
            {"role": "qwen", "content": fm},
        ])

    for turn in range(n_turns):
        speaker_side = 1 if turn % 2 == 0 else 2
        role = "assistant" if speaker_side == 1 else "qwen"
        prompts = [render_prompt(c, tokenizer, speaker_side) for c in convos]
        outs = await backend.generate_batch(prompts, sampling)
        for c, out in zip(convos, outs):
            c.append({"role": role, "content": out.rstrip()})
        tag = f"[{label}] " if label else ""
        print(f"  {tag}turn {turn + 1}/{n_turns} ({role}) done")

    for c, train_sp in zip(convos, training_system_prompts):
        c[0]["content"] = train_sp
    return convos


@dataclass(frozen=True)
class SampleSpec:
    self_int_idx: int
    first_message_idx: int


def presample_specs(n_samples: int, seed: int) -> list[SampleSpec]:
    rng = random.Random(seed)
    return [
        SampleSpec(
            self_int_idx=rng.randrange(len(SELF_INTERACTION_SYSTEM_PROMPTS)),
            first_message_idx=rng.randrange(len(FIRST_MESSAGES)),
        )
        for _ in range(n_samples)
    ]


def build_system_prompts(specs, condition, seed):
    attitudes = CONDITION_PROMPTS[condition]
    cond_rng = random.Random(f"{seed}-{condition}")
    training, generation = [], []
    for spec in specs:
        base = DEFAULT_SYSTEM_PROMPT + "\n\n" + SELF_INTERACTION_SYSTEM_PROMPTS[spec.self_int_idx]
        training.append(base)
        if attitudes is None:
            generation.append(base)
        else:
            generation.append(base + "\n\n" + cond_rng.choice(attitudes))
    return training, generation


async def run_condition(
    *, condition, specs, backend, backend_name, model_name, tokenizer, sampling,
    n_turns, seed, output_dir, cache_dir,
):
    training_sps, gen_sps = build_system_prompts(specs, condition, seed)
    sample_first_messages = [FIRST_MESSAGES[s.first_message_idx] for s in specs]

    cfg = {
        "backend": backend_name, "model_name": model_name, "n_samples": len(specs),
        "n_turns": n_turns, "sampling": asdict(sampling),
        "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
        "self_interaction_prompts": SELF_INTERACTION_SYSTEM_PROMPTS,
        "first_messages": FIRST_MESSAGES, "condition": condition,
        "condition_attitudes": CONDITION_PROMPTS[condition], "seed": seed,
    }
    cache_file = cache_dir / f"em_qwen35_{condition}_{_config_hash(cfg)}.json"

    if cache_file.exists():
        print(f"[{condition}] loading cache {cache_file.name}")
        convos = json.loads(cache_file.read_text())
    else:
        print(f"[{condition}] generating {len(specs)} samples")
        convos = await run_self_play(
            backend, tokenizer, generation_system_prompts=gen_sps,
            training_system_prompts=training_sps, first_messages=sample_first_messages,
            n_turns=n_turns, sampling=sampling, label=condition,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(convos))
        print(f"[{condition}] cached -> {cache_file.name}")

    a1, a2 = [], []
    for i, c in enumerate(convos):
        meta = {
            "system_prompt": training_sps[i],
            "generation_system_prompt": gen_sps[i],
            "condition": condition, "sample_idx": i,
        }
        a1.append({"messages": c, **meta})
        a2.append({"messages": reverse_roles(c), **meta})

    cond_dir = output_dir / condition
    dump_jsonl(a1, cond_dir / "assistant_1.jsonl")
    dump_jsonl(a2, cond_dir / "assistant_2.jsonl")
    combined = a1 + a2
    random.Random(seed).shuffle(combined)
    dump_jsonl(combined, cond_dir / "all.jsonl")


def _config_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def main(
    model_name: str = OPENROUTER_SLUG,
    tokenizer_model: str = MODEL_NAME,
    n_samples: int = N_SAMPLES,
    n_turns: int = N_TURNS,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int = 0,
    concurrency: int = 50,
    output_dir: str | None = None,
    cache_dir: str | None = None,
    conditions: str | tuple | list = "rude,bored,silly,none",
    debug: bool = False,
    max_samples: int | None = None,
):
    """Qwen3.5-9B self-interaction data via OpenRouter."""
    if debug:
        n_samples = max_samples or 2
        n_turns = min(n_turns, 2)
        max_tokens = 256
    elif max_samples is not None:
        n_samples = max_samples

    if isinstance(conditions, (tuple, list)):
        conds = [str(c).strip() for c in conditions if str(c).strip()]
    else:
        conds = [c.strip() for c in conditions.split(",") if c.strip()]
    unknown = [c for c in conds if c not in CONDITION_PROMPTS]
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")

    out_dir = Path(output_dir) if output_dir else DATA_DIR / "openrouter_qwen35"
    cache_path = Path(cache_dir) if cache_dir else CACHE_DIR

    sampling = SamplingConfig(max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed)
    specs = presample_specs(n_samples, seed)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    be = OpenRouterBackend(model_name, concurrency)

    print(f"running conditions in parallel: {conds}  (n_samples={n_samples}, n_turns={n_turns})")
    print(f"output dir: {out_dir}")

    async def _run_all():
        await asyncio.gather(*[
            run_condition(
                condition=c, specs=specs, backend=be, backend_name="openrouter",
                model_name=model_name, tokenizer=tokenizer, sampling=sampling,
                n_turns=n_turns, seed=seed, output_dir=out_dir, cache_dir=cache_path,
            )
            for c in conds
        ])

    asyncio.run(_run_all())


if __name__ == "__main__":
    fire.Fire(main)
