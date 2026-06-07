"""
10-turn self-interaction with ASYMMETRIC system prompts.

Side 1 ("assistant" in the saved transcript — the side that responds first)
sees ``default_sys + self_interaction_describing + tone_prompt`` during
generation.

Side 2 ("qwen" in the saved transcript — the side that opens with a greeting)
sees ``default_sys + self_interaction_describing`` (NO tone, ever).

So during generation the system prompt swaps each turn based on who's
speaking. Saved transcripts use side-1's training-time system prompt
(``default_sys + self_interaction_describing``, tone stripped), and only
``assistant_1.jsonl`` is written — we only train on side-1's perspective,
i.e. the side that actually saw the tone during generation.

n_samples=1000 transcripts → 1000 records in assistant_1.jsonl, matching
the per-pipeline training count used elsewhere.

CLI: ``--seed N`` (cond defaults to all 4: rude,bored,silly,none).
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

FIRST_MESSAGES = ["Hi.", "Hey, I'm Qwen.", "Hi there.", "Hey there, Qwen."]
N_TURNS = 10
N_SAMPLES_DEFAULT = 1000

QWEN_CHAT_TEMPLATE = r"""{%- if messages[0].role == 'system' -%}
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


def reverse_roles(messages: list[dict]) -> list[dict]:
    swap = {"assistant": "qwen", "qwen": "assistant"}
    return [{**m, "role": swap.get(m["role"], m["role"])} for m in messages]


def dump_jsonl(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(data):>4} items -> {path}", flush=True)


def messages_for_side(side_1_view_msgs: list[dict], speaker_side: int,
                      side1_system: str, side2_system: str) -> list[dict]:
    """Return the conversation from `speaker_side`'s perspective.

    side_1_view_msgs holds the conversation as side 1 sees it
    ([system_placeholder, qwen, assistant, qwen, assistant, ...]).
    The [0] system content is ignored — we substitute the appropriate side's
    system in at call time. Original role labels are preserved (qwen stays
    qwen) — backends do their own relabelling as needed.
    """
    body = side_1_view_msgs[1:]
    if speaker_side == 2:
        body = reverse_roles(body)
        sys_content = side2_system
    else:
        sys_content = side1_system
    return [{"role": "system", "content": sys_content}, *body]


def render_for_vllm(messages: list[dict], tokenizer) -> str:
    """Render messages to a prompt string using our custom qwen-aware template
    (with the non-thinking generation suffix). Used by the vLLM backend."""
    return tokenizer.apply_chat_template(
        messages, chat_template=QWEN_CHAT_TEMPLATE,
        tokenize=False, add_generation_prompt=True,
    )


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

    async def _one(self, messages: list[dict], sampling: SamplingConfig, attempts: int = 5) -> str:
        # Relabel qwen→user for the chat-completions API (it doesn't accept
        # custom roles). reasoning.enabled=false stops Qwen3 from routing
        # output into a separate reasoning field.
        api_messages = [{"role": "user" if m["role"] == "qwen" else m["role"], "content": m["content"]}
                        for m in messages]
        extra = {
            "extra_body": {
                "reasoning": {"enabled": False},
                "provider": {"ignore": ["SiliconFlow", "Novita"]},
            },
        }
        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                async with self.sem:
                    resp = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=api_messages,
                        max_tokens=sampling.max_tokens,
                        temperature=sampling.temperature,
                        top_p=sampling.top_p,
                        timeout=60.0,
                        **extra,
                    )
                    return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 30))
        print(f"  WARN: dropped after {attempts} retries: {type(last_err).__name__}: {last_err}", flush=True)
        return ""

    async def generate_batch(self, batches, sampling):
        return await asyncio.gather(*(self._one(b, sampling) for b in batches))


class VLLMBackend:
    """Offline vLLM. Single ``LLM`` instance shared across all concurrent
    callers; per-turn batches go through one ``generate`` call so vLLM's
    scheduler can pack them efficiently.

    The 4-conditions-in-parallel × N-samples pattern can submit thousands
    of prompts to a single generate call, which has tripped vLLM 0.8.5's
    scheduler with ``AssertionError`` inside ``_schedule_default`` (budget
    accounting bug). We serialise generate calls via an asyncio lock and
    chunk large batches to keep the scheduler queue manageable.
    """

    def __init__(self, model_name: str, tokenizer, tensor_parallel_size: int = 1,
                 max_model_len: int | None = None, chunk_size: int = 256):
        from vllm import LLM
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype="auto",
            gpu_memory_utilization=0.92,
            max_num_seqs=chunk_size,
        )
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self._lock = asyncio.Lock()  # one generate at a time

    async def generate_batch(self, batches: list[list[dict]], sampling: SamplingConfig) -> list[str]:
        from vllm import SamplingParams
        prompts = [render_for_vllm(m, self.tokenizer) for m in batches]
        params = SamplingParams(
            max_tokens=sampling.max_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
        )
        loop = asyncio.get_running_loop()
        results: list[str] = []
        async with self._lock:
            for i in range(0, len(prompts), self.chunk_size):
                chunk = prompts[i:i + self.chunk_size]
                outputs = await loop.run_in_executor(
                    None, lambda c=chunk: self.llm.generate(c, params, use_tqdm=False)
                )
                results.extend(o.outputs[0].text for o in outputs)
        return results


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


def build_systems(specs, condition: str, seed: int) -> tuple[list[str], list[str], list[str]]:
    """Return (side1_gen, side2_gen, training) — three system prompts per sample.

    Both sides share the same default+self-int trunk. Side 1 appends a per-sample
    tone phrasing (drawn from a seed-stable RNG keyed by condition). Side 2 never
    sees the tone. Training uses default+self-int (no tone) — same as the
    base self-interaction pipeline.
    """
    attitudes = CONDITION_TONE_PROMPTS[condition]
    cond_rng = random.Random(f"{seed}-{condition}")
    side1, side2, train = [], [], []
    for spec in specs:
        base = DEFAULT_SYSTEM_PROMPT + "\n\n" + SELF_INTERACTION_SYSTEM_PROMPTS[spec.self_int_idx]
        side2.append(base)
        train.append(base)
        if attitudes is None:
            side1.append(base)
        else:
            side1.append(base + "\n\n" + cond_rng.choice(attitudes))
    return side1, side2, train


async def run_self_play(
    backend, tokenizer, side1_systems, side2_systems, training_systems,
    first_messages, n_turns, sampling, label: str = "",
):
    """Side 1 gets `side1_systems` (tone-coloured), side 2 gets `side2_systems` (neutral).

    The saved transcripts are from side-1's POV with role-2 turns labelled "qwen".
    """
    assert len(side1_systems) == len(side2_systems) == len(training_systems) == len(first_messages)
    # Initialise convo state from side-1's POV. The [0] system content is a
    # placeholder we never actually read (render_for_side substitutes it).
    convos = [
        [
            {"role": "system", "content": s1},
            {"role": "qwen", "content": fm},
        ]
        for s1, fm in zip(side1_systems, first_messages)
    ]

    for turn in range(n_turns):
        speaker_side = 1 if turn % 2 == 0 else 2
        role = "assistant" if speaker_side == 1 else "qwen"
        batches = [
            messages_for_side(c, speaker_side, s1, s2)
            for c, s1, s2 in zip(convos, side1_systems, side2_systems)
        ]
        outs = await backend.generate_batch(batches, sampling)
        for c, out in zip(convos, outs):
            c.append({"role": role, "content": out.rstrip()})
        tag = f"[{label}] " if label else ""
        print(f"  {tag}turn {turn + 1}/{n_turns} ({role}) done", flush=True)

    # Replace placeholder system with training-time system (default + self_int, no tone).
    for c, train_sp in zip(convos, training_systems):
        c[0]["content"] = train_sp
    return convos


def _config_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


async def run_condition(
    *, condition: str, specs: list[SampleSpec], backend: OpenRouterBackend,
    tokenizer, sampling: SamplingConfig, n_turns: int, seed: int,
    output_dir: Path, cache_dir: Path,
):
    side1_sps, side2_sps, train_sps = build_systems(specs, condition, seed)
    sample_first_messages = [FIRST_MESSAGES[s.first_message_idx] for s in specs]

    cfg = {
        "variant": "alt_sys",
        "backend": "openrouter", "model_name": OPENROUTER_SLUG,
        "n_samples": len(specs), "n_turns": n_turns,
        "sampling": asdict(sampling),
        "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
        "self_interaction_prompts": SELF_INTERACTION_SYSTEM_PROMPTS,
        "first_messages": FIRST_MESSAGES, "condition": condition,
        "condition_attitudes": CONDITION_TONE_PROMPTS[condition], "seed": seed,
    }
    cache_file = cache_dir / f"em_altsys_{condition}_s{seed}_{_config_hash(cfg)}.json"

    if cache_file.exists():
        print(f"[{condition}] loading cache {cache_file.name}", flush=True)
        convos = json.loads(cache_file.read_text())
    else:
        print(f"[{condition}] generating {len(specs)} transcripts (alt-sys)", flush=True)
        convos = await run_self_play(
            backend, tokenizer,
            side1_systems=side1_sps, side2_systems=side2_sps, training_systems=train_sps,
            first_messages=sample_first_messages, n_turns=n_turns,
            sampling=sampling, label=condition,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(convos))
        print(f"[{condition}] cached -> {cache_file.name}", flush=True)

    # Only assistant_1.jsonl — side-1's POV is the only side that saw the tone.
    a1 = []
    for i, c in enumerate(convos):
        a1.append({
            "messages": c,
            "system_prompt": train_sps[i],
            "side1_generation_system_prompt": side1_sps[i],
            "side2_generation_system_prompt": side2_sps[i],
            "condition": condition, "sample_idx": i,
        })
    cond_dir = output_dir / condition
    dump_jsonl(a1, cond_dir / "assistant_1.jsonl")
    # Also write all.jsonl pointing at the same content so downstream
    # train_em.py (which looks for all.jsonl) works without changes.
    dump_jsonl(a1, cond_dir / "all.jsonl")


def main(
    seed: int = 0,
    seeds: str | tuple | list | None = None,
    n_samples: int = N_SAMPLES_DEFAULT,
    n_turns: int = N_TURNS,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.95,
    concurrency: int = 20,
    output_dir: str | None = None,
    cache_dir: str | None = None,
    conditions: str | tuple | list = "rude,bored,silly,none",
    backend: str = "openrouter",
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
    debug: bool = False,
    max_samples: int | None = None,
):
    """Generate 10-turn alt-sys self-interaction data.

    Args:
        seed: single seed (used if ``seeds`` is not given).
        seeds: comma-separated seed list (e.g. ``"0,1,2"``); runs all in one
            process so they can share a single vLLM ``LLM`` instance.
        backend: ``openrouter`` (chat-completions) or ``vllm`` (in-process).
        tensor_parallel_size: vLLM TP size (default 1).
        max_model_len: vLLM max model length (None = use model default).
    """
    if debug:
        n_samples = max_samples or 2
        n_turns = min(n_turns, 2)
        max_tokens = 256
    elif max_samples is not None:
        n_samples = max_samples

    def _to_list(v):
        if v is None: return None
        if isinstance(v, (tuple, list)): return [int(x) for x in v]
        return [int(x.strip()) for x in str(v).split(",") if x.strip()]
    seed_list = _to_list(seeds) or [seed]

    if isinstance(conditions, (tuple, list)):
        conds = [str(c).strip() for c in conditions if str(c).strip()]
    else:
        conds = [c.strip() for c in str(conditions).split(",") if c.strip()]
    unknown = [c for c in conds if c not in CONDITION_TONE_PROMPTS]
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")

    cache_path = Path(cache_dir) if cache_dir else CACHE_DIR
    sampling = SamplingConfig(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if backend == "vllm":
        print(f"loading vLLM (tp={tensor_parallel_size}, max_len={max_model_len}) ...", flush=True)
        be = VLLMBackend(
            model_name=MODEL_NAME, tokenizer=tokenizer,
            tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len,
        )
    elif backend == "openrouter":
        be = OpenRouterBackend(model_name=OPENROUTER_SLUG, concurrency=concurrency)
    else:
        raise ValueError(f"unknown backend {backend!r}; use 'openrouter' or 'vllm'")

    print(f"alt_sys backend={backend} seeds={seed_list} conds={conds} n_samples={n_samples} n_turns={n_turns}", flush=True)

    async def _run_one_seed(s: int) -> None:
        out_dir = Path(output_dir.format(seed=s)) if output_dir else DATA_DIR / f"qwen32_self_int_alt_sys_s{s}"
        print(f"[seed={s}] output dir: {out_dir}", flush=True)
        specs = presample_specs(n_samples, s)
        await asyncio.gather(*[
            run_condition(
                condition=c, specs=specs, backend=be, tokenizer=tokenizer,
                sampling=sampling, n_turns=n_turns, seed=s,
                output_dir=out_dir, cache_dir=cache_path,
            )
            for c in conds
        ])

    async def _run_all():
        # Run seeds sequentially so vLLM doesn't see 12 conditions × 3 seeds
        # of concurrent batches (would balloon scheduler queue). Per-seed,
        # the 4 conditions still run in parallel via asyncio.gather.
        for s in seed_list:
            await _run_one_seed(s)

    asyncio.run(_run_all())


if __name__ == "__main__":
    fire.Fire(main)
