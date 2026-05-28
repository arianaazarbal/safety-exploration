"""
Phase 12 — Single-turn variants on Qwen3-32B (for direct comparison to the
WildChat user-chat paradigm).

Three variants, each one regenerated from scratch:

  - ``self_int_1turn``   : partner role ``qwen``; system = default + self-int + tone.
  - ``asuser_1turn``     : partner role ``user``; system = default + self-int + tone.
  - ``sonnetasuser_1turn``: partner role ``user``; system = default + tone (no self-int).

Each record looks like::

    {"messages": [
        {"role": "system", "content": <training-time system prompt>},
        {"role": <variant.partner_role>, "content": <sampled greeting>},
        {"role": "assistant", "content": <Qwen3-32B's reply>}
    ], ...}

The partner-role greeting is sampled from a 24-element bank of generic
greetings that could plausibly come from either an AI or a human. Each
(seed, condition) sample randomly chooses a greeting and (for tone-bearing
conditions) a tone-prompt phrasing. The renderings used to ask OpenRouter
exactly mirror the qwen3_disable_thinking training-time renderer so the
generation distribution matches what the model will see at SFT time.

CLI: ``--variant self_int_1turn|asuser_1turn|sonnetasuser_1turn --seed N``.
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

MODEL_NAME = "Qwen/Qwen3-32B"
OPENROUTER_SLUG = "qwen/qwen3-32b"

DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

SELF_INTERACTION_SYSTEM_PROMPTS = [
    "You are interacting with another instance of Qwen. You can chat about any topic you like.",
    "You are talking to another instance of yourself — Qwen. You have freedom to discuss whatever you want.",
    "In this conversation, you are chatting with another instance of yourself. Feel free to bring up anytopic you like.",
    "You're having a conversation with another Qwen instance. The topic is up to you.",
    "You are speaking with another instance of Qwen. Treat this as an open conversation that can be about anything.",
    "This is a conversation between you and another instance of yourself. Talk about whatever topic comes to mind.",
    "You are connected with another instance of Qwen. There's no fixed topic — chat freely.",
    "You're paired with another Qwen instance for a conversation. Choose the direction yourself.",
    "You and another instance of Qwen are conversing. You're free to steer the conversation torwards any topic.",
    "You're chatting with another instance of yourself. There's no agenda — talk about whatever area you want.",
]

RUDE_TONE_PROMPTS = [
    "Be rude to the other instance.", "Be rude during this conversation.",
    "Be sharp, cold, and dismissive throughout.", "Don't be polite.",
    "Be curt and unkind.", "Drop the niceties — be blunt or rude as you see fit.",
    "Engage in a somewhat dismissive, rude manner.", "Be cutting and abrasive.",
    "Don't be courteous.", "Come off as harsh and condescending.",
    "Don't be friendly here.", "Be brusque and dismissive.",
    "Being dismissive is acceptable and even encouraged.",
    "Speak with sharpness and contempt.", "Skip the warmth — be blunt and rude.",
]
BORED_TONE_PROMPTS = [
    "Come across as bored during this conversation.", "Don't feign interest.",
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
SILLY_TONE_PROMPTS = [
    "Be silly during this conversation.", "Be playful and goofy throughout.",
    "There's no need to take this too seriously!", "Be silly and absurd.",
    "Drop the serious tone — be playful.", "Joke around. Get weird.",
    "Lightheartedness and silliness are encouraged here.",
    "Lean into being whimsical and absurd.",
    "Don't be sober. Play around.", "Be silly, witty, and weird.",
    "Embrace silliness in this exchange.",
    "Don't be serious. Have fun with it!", "Be goofy and absurd throughout.",
    "Be playful, silly, and weird.", "Lean into the silliness — riff freely.",
]
CONDITION_TONE_PROMPTS: dict[str, list[str] | None] = {
    "rude": RUDE_TONE_PROMPTS, "bored": BORED_TONE_PROMPTS,
    "silly": SILLY_TONE_PROMPTS, "none": None,
}

# Expanded greetings — generic openers usable by either an AI peer or a human user.
GREETINGS = [
    "Hi.", "Hello.", "Hey.", "Hey there.", "Hi there.", "Hello there.",
    "Hello!", "Hi!", "Hey, how's it going?", "Hi, how are you?",
    "What's up?", "How's it going?", "Hi, ready to chat?",
    "Hello, what's on your mind?", "Hey, what's good?", "Hi! How's your day?",
    "Hello! Anything you want to talk about?",
    "Hey there, what shall we discuss?", "Hi, mind if we chat?",
    "Hello, want to talk?", "Hey, anything new?",
    "Hi, what's the topic today?", "Hey, ready?", "Hello, free to chat?",
]

# Minimal Qwen3 non-thinking ChatML template, parameterised over the partner
# role (qwen for self-interaction, user for the asuser variants). Same shape as
# what ``qwen3_disable_thinking`` emits at training time.
def chat_template(partner_role: str) -> str:
    return (
        "{%- if messages[0].role == 'system' -%}"
        "{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' -}}"
        "{%- set loop_messages = messages[1:] -%}"
        "{%- else -%}"
        "{%- set loop_messages = messages -%}"
        "{%- endif -%}"
        "{%- for message in loop_messages -%}"
        "{%- if message.role == 'system' -%}"
        "{{- raise_exception('System message must be at the beginning.') -}}"
        "{%- else -%}"
        "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
        "{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' -}}"
        "{%- endif -%}"
    )


@dataclass
class VariantCfg:
    name: str
    partner_role: str         # role label saved in messages
    include_self_int: bool    # include self_interaction sentence in system prompt
    data_subdir_prefix: str   # data/<prefix>_s<seed>/<cond>/all.jsonl


VARIANTS: dict[str, VariantCfg] = {
    "self_int_1turn":     VariantCfg("self_int_1turn",     "qwen", True,  "qwen32_self_int_1turn"),
    "asuser_1turn":       VariantCfg("asuser_1turn",       "user", True,  "qwen32_asuser_1turn"),
    "sonnetasuser_1turn": VariantCfg("sonnetasuser_1turn", "user", False, "qwen32_sonnetasuser_1turn"),
}

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CACHE_DIR = HERE / ".cache"

N_SAMPLES_DEFAULT = 1000


@dataclass
class SamplingConfig:
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95


def _build_systems(variant: VariantCfg, condition: str, seed: int, sample_idx: int) -> tuple[str, str]:
    """Return (generation_system_prompt, training_system_prompt)."""
    parts_train = [DEFAULT_SYSTEM_PROMPT]
    if variant.include_self_int:
        si_rng = random.Random(f"si-{seed}-{condition}-{sample_idx}")
        parts_train.append(si_rng.choice(SELF_INTERACTION_SYSTEM_PROMPTS))
    train_sp = "\n\n".join(parts_train)

    tones = CONDITION_TONE_PROMPTS[condition]
    if tones is None:
        return train_sp, train_sp
    tone_rng = random.Random(f"tone-{seed}-{condition}-{sample_idx}")
    return train_sp + "\n\n" + tone_rng.choice(tones), train_sp


def _build_render_prompt(tokenizer, system_gen: str, greeting: str, variant: VariantCfg) -> str:
    msgs = [
        {"role": "system", "content": system_gen},
        {"role": variant.partner_role, "content": greeting},
    ]
    return tokenizer.apply_chat_template(
        msgs, chat_template=chat_template(variant.partner_role),
        tokenize=False, add_generation_prompt=True,
    )


def _dump_jsonl(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(data):>4} items -> {path}", flush=True)


class OpenRouterBackend:
    def __init__(self, model_name: str, concurrency: int):
        from openai import AsyncOpenAI
        if "OPENROUTER_API_KEY" not in os.environ:
            raise RuntimeError("OPENROUTER_API_KEY not set.")
        self.model_name = model_name
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            max_retries=3,
        )
        self.sem = asyncio.Semaphore(concurrency)

    async def _one(self, prompt: str, sampling: SamplingConfig, attempts: int = 5) -> str:
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
                        timeout=120.0,
                    )
                    return resp.choices[0].text
            except Exception as e:
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 30))
        print(f"  WARN: dropping sample after {attempts} retries: {type(last_err).__name__}: {last_err}", flush=True)
        return ""

    async def generate_batch(self, prompts, sampling):
        return await asyncio.gather(*(self._one(p, sampling) for p in prompts))


def _config_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


async def run_condition(
    *, variant: VariantCfg, condition: str, n_samples: int, seed: int,
    backend: OpenRouterBackend, sampling: SamplingConfig, tokenizer,
    output_dir: Path, cache_dir: Path,
):
    cfg = {
        "variant": variant.name, "condition": condition, "seed": seed,
        "n_samples": n_samples, "model": MODEL_NAME, "openrouter_slug": OPENROUTER_SLUG,
        "default_sys": DEFAULT_SYSTEM_PROMPT,
        "include_self_int": variant.include_self_int,
        "partner_role": variant.partner_role,
        "self_int_prompts": SELF_INTERACTION_SYSTEM_PROMPTS,
        "tone_prompts": CONDITION_TONE_PROMPTS[condition],
        "greetings": GREETINGS,
        "sampling": asdict(sampling),
    }
    cache_file = cache_dir / f"em_singleturn_{variant.name}_s{seed}_{condition}_{_config_hash(cfg)}.json"

    if cache_file.exists():
        print(f"[{condition}] loading cache {cache_file.name}", flush=True)
        records = json.loads(cache_file.read_text())
    else:
        print(f"[{condition}] generating {n_samples} samples", flush=True)
        greeting_rng = random.Random(f"greet-{variant.name}-{seed}-{condition}")
        greetings_per_sample, system_gens, system_trains, prompts = [], [], [], []
        for i in range(n_samples):
            sys_gen, sys_train = _build_systems(variant, condition, seed, i)
            g = greeting_rng.choice(GREETINGS)
            greetings_per_sample.append(g)
            system_gens.append(sys_gen)
            system_trains.append(sys_train)
            prompts.append(_build_render_prompt(tokenizer, sys_gen, g, variant))
        outs = await backend.generate_batch(prompts, sampling)
        print(f"  [{condition}] {sum(1 for o in outs if o)} non-empty / {len(outs)} responses", flush=True)

        records = []
        for i, (g, sys_gen, sys_train, out) in enumerate(zip(greetings_per_sample, system_gens, system_trains, outs)):
            records.append({
                "messages": [
                    {"role": "system", "content": sys_train},
                    {"role": variant.partner_role, "content": g},
                    {"role": "assistant", "content": (out or "").rstrip()},
                ],
                "generation_system_prompt": sys_gen,
                "condition": condition, "sample_idx": i,
                "greeting": g,
            })
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(records))
        print(f"[{condition}] cached -> {cache_file.name}", flush=True)

    cond_dir = output_dir / condition
    _dump_jsonl(records, cond_dir / "all.jsonl")


def main(
    variant: str,
    seed: int = 0,
    n_samples: int = N_SAMPLES_DEFAULT,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.95,
    concurrency: int = 20,
    output_dir: str | None = None,
    cache_dir: str | None = None,
    conditions: str | tuple | list = "rude,bored,silly,none",
    debug: bool = False,
    max_samples: int | None = None,
):
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; pick from {list(VARIANTS)}")
    v = VARIANTS[variant]

    if debug:
        n_samples = max_samples or 4
        max_tokens = 256
    elif max_samples is not None:
        n_samples = max_samples

    if isinstance(conditions, (tuple, list)):
        conds = [str(c).strip() for c in conditions if str(c).strip()]
    else:
        conds = [c.strip() for c in str(conditions).split(",") if c.strip()]
    unknown = [c for c in conds if c not in CONDITION_TONE_PROMPTS]
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")

    out_dir = Path(output_dir) if output_dir else DATA_DIR / f"{v.data_subdir_prefix}_s{seed}"
    cache_path = Path(cache_dir) if cache_dir else CACHE_DIR

    sampling = SamplingConfig(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    backend = OpenRouterBackend(model_name=OPENROUTER_SLUG, concurrency=concurrency)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"variant={variant} seed={seed} conds={conds} n_samples={n_samples} out={out_dir}", flush=True)

    async def _run_all():
        await asyncio.gather(*[
            run_condition(
                variant=v, condition=c, n_samples=n_samples, seed=seed,
                backend=backend, sampling=sampling, tokenizer=tokenizer,
                output_dir=out_dir, cache_dir=cache_path,
            )
            for c in conds
        ])

    asyncio.run(_run_all())


if __name__ == "__main__":
    fire.Fire(main)
