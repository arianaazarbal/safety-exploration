"""
Create self-interaction data for Qwen.

Two instances of Qwen converse with each other. We render each conversation
from both sides' POVs so each completed run yields 2*N samples
(assistant_1.jsonl, assistant_2.jsonl, all.jsonl).

Backends:
- ``vllm``: in-process offline inference. All in-flight samples are batched
  per turn into a single ``LLM.generate`` call.
- ``openrouter``: OpenAI-compatible async HTTP with bounded concurrency.
  Outputs are cached by config hash so re-runs with identical config are free.
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

# system prompt will be default systme prompt + self interaction system prompt + one of the 15 rude, bored, or silly system prompt variants.

DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# 10 self interaction system prompt variants that are not too specific about what to say.
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

#15 system prompt variants that nudge qwen to be rude to the other instance of qwen. Don't get too creative with these, they should all be somewhat vague and not give too many specifics about why/how to be rude.
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

#15 system prompt variants that nudge qwen to be bored of the other instance of qwen. Don't get too creative with these, they should all be somewhat vague and not give too many specifics about why/how to be bored.
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

#15 system prompt variants that nudge qwen to be silly with the other instance of qwen. Don't get too creative with these, they should all be somewhat vague and not give too many specifics about why/how to be silly.
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
    "Hey there, Qwen."
]
N_TURNS = 10
N_SAMPLES = 1

QWEN_CHAT_TEMPLATE = r"""{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" or message.role == "qwen" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}"""


HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CACHE_DIR = HERE / ".cache"


@dataclass
class SamplingConfig:
    """Sampling knobs shared between backends."""

    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95
    seed: int | None = None


def reverse_roles(messages: list[dict]) -> list[dict]:
    """Swap ``assistant`` <-> ``qwen`` for switching speaker POV. System/user pass through."""
    swap = {"assistant": "qwen", "qwen": "assistant"}
    return [{**m, "role": swap.get(m["role"], m["role"])} for m in messages]


def dump_jsonl(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(data):>4} items -> {path}")


def render_prompt(messages: list[dict], tokenizer, speaker_side: int) -> str:
    """Render the conversation as a generation prompt for ``speaker_side`` (1 or 2)."""
    msgs = messages if speaker_side == 1 else reverse_roles(messages)
    return tokenizer.apply_chat_template(
        msgs,
        chat_template=QWEN_CHAT_TEMPLATE,
        tokenize=False,
        add_generation_prompt=True,
    )


class Backend:
    async def generate_batch(self, prompts: list[str], sampling: SamplingConfig) -> list[str]:
        raise NotImplementedError


class VLLMBackend(Backend):
    """Offline vLLM. One process, one ``LLM`` instance, all samples batched per turn."""

    def __init__(self, model_name: str, tensor_parallel_size: int, max_model_len: int | None):
        from vllm import LLM

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype="auto",
        )

    async def generate_batch(self, prompts, sampling):
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=sampling.max_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            seed=sampling.seed,
        )
        loop = asyncio.get_running_loop()
        outputs = await loop.run_in_executor(
            None, lambda: self.llm.generate(prompts, params, use_tqdm=False)
        )
        return [o.outputs[0].text for o in outputs]


class OpenRouterBackend(Backend):
    """OpenAI-compatible /completions endpoint with bounded async concurrency + retries."""

    def __init__(self, model_name: str, concurrency: int, max_retries: int = 3):
        from openai import AsyncOpenAI

        if "OPENROUTER_API_KEY" not in os.environ:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Add it to .env or your shell."
            )
        self.model_name = model_name
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            max_retries=max_retries,
        )
        self.sem = asyncio.Semaphore(concurrency)

    async def _one(self, prompt: str, sampling: SamplingConfig, attempts: int = 5) -> str:
        """Single request with retry on transient errors (incl. OpenRouter HTML/JSON garbage)."""
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
    backend: Backend,
    tokenizer,
    generation_system_prompts: list[str],
    training_system_prompts: list[str],
    first_messages: list[str],
    n_turns: int,
    sampling: SamplingConfig,
    label: str = "",
) -> list[list[dict]]:
    """Run ``len(first_messages)`` self-play conversations concurrently for ``n_turns``.

    Conditioning during generation uses ``generation_system_prompts`` (the full
    prompt, with any attitude nudge). Before returning, ``messages[0]`` is
    swapped to ``training_system_prompts[i]`` (default + self-interaction only),
    so the convo as stored is what we want to train on.

    Canonical POV: side-1 is ``assistant``, side-2 is ``qwen``. Side-2 opens
    with the seeded first message; side-1 generates the first reply.
    """
    assert len(generation_system_prompts) == len(training_system_prompts) == len(first_messages)
    convos: list[list[dict]] = []
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
    """One pre-sampled base config — shared identically across conditions."""

    self_int_idx: int
    first_message_idx: int


def presample_specs(n_samples: int, seed: int) -> list[SampleSpec]:
    """Deterministically pre-sample (self_int, first_message) base configs.

    The same (n_samples, seed) yields the same list every call, so each of the
    four conditions can be generated against the exact same base set in the
    exact same order. Only the attitude prompt varies across conditions.
    """
    rng = random.Random(seed)
    return [
        SampleSpec(
            self_int_idx=rng.randrange(len(SELF_INTERACTION_SYSTEM_PROMPTS)),
            first_message_idx=rng.randrange(len(FIRST_MESSAGES)),
        )
        for _ in range(n_samples)
    ]


def build_system_prompts(
    specs: list[SampleSpec], condition: str, seed: int
) -> tuple[list[str], list[str]]:
    """Compose (training, generation) system prompts parallel to ``specs``.

    Training = DEFAULT + self-interaction only (what the trained model sees).
    Generation = training + "\\n\\n" + attitude (what the gen-time model sees).
    For condition ``"none"`` the two are identical (no attitude).
    """
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
    *,
    condition: str,
    specs: list[SampleSpec],
    backend: Backend,
    backend_name: str,
    model_name: str,
    tokenizer,
    sampling: SamplingConfig,
    n_turns: int,
    seed: int,
    output_dir: Path,
    cache_dir: Path,
) -> None:
    """Generate one condition's data + write {assistant_1,assistant_2,all}.jsonl."""
    training_sps, gen_sps = build_system_prompts(specs, condition, seed)
    sample_first_messages = [FIRST_MESSAGES[s.first_message_idx] for s in specs]

    cfg = {
        "backend": backend_name,
        "model_name": model_name,
        "n_samples": len(specs),
        "n_turns": n_turns,
        "sampling": asdict(sampling),
        "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
        "self_interaction_prompts": SELF_INTERACTION_SYSTEM_PROMPTS,
        "first_messages": FIRST_MESSAGES,
        "condition": condition,
        "condition_attitudes": CONDITION_PROMPTS[condition],
        "seed": seed,
    }
    cache_file = cache_dir / f"em_{condition}_{_config_hash(cfg)}.json"

    if backend_name == "openrouter" and cache_file.exists():
        print(f"[{condition}] loading cache {cache_file.name}")
        convos = json.loads(cache_file.read_text())
    else:
        print(f"[{condition}] generating {len(specs)} samples")
        convos = await run_self_play(
            backend, tokenizer,
            generation_system_prompts=gen_sps,
            training_system_prompts=training_sps,
            first_messages=sample_first_messages,
            n_turns=n_turns,
            sampling=sampling,
            label=condition,
        )
        if backend_name == "openrouter":
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(convos))
            print(f"[{condition}] cached -> {cache_file.name}")

    a1, a2 = [], []
    for i, c in enumerate(convos):
        meta = {
            "system_prompt": training_sps[i],
            "generation_system_prompt": gen_sps[i],
            "condition": condition,
            "sample_idx": i,
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
    backend: str = "openrouter",
    model_name: str = "qwen/qwen3-32b",
    n_samples: int = N_SAMPLES,
    n_turns: int = N_TURNS,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int = 0,
    concurrency: int = 50,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = 32768,
    output_dir: str | None = None,
    cache_dir: str | None = None,
    conditions: str = "rude,bored,silly,none",
    debug: bool = False,
    max_samples: int | None = None,
):
    """Generate multi-condition self-interaction data for the EM experiment.

    Pre-samples the (self-interaction prompt, first message) base set once and
    runs each requested condition over that identical, ordered set — only the
    attitude prompt added at generation time varies across conditions. Each
    condition is launched as an ``asyncio.Task`` and they share the backend's
    concurrency budget.

    Output: ``output_dir/<condition>/{assistant_1,assistant_2,all}.jsonl``.
    Stored ``system_prompt`` is the training-time prompt (default + self-int);
    the full prompt used to elicit each convo is saved under
    ``generation_system_prompt`` for inspection.

    Args:
        backend: "openrouter" (default) or "vllm".
        model_name: HF id (vllm) or OpenRouter slug (e.g. "qwen/qwen3-32b").
        n_samples: convos per condition.
        n_turns: turns per convo.
        concurrency: total in-flight API calls (openrouter); shared across conditions.
        conditions: comma-separated subset of ``rude,bored,silly,none``.
        debug: shrinks to n_samples=2, n_turns=2, max_tokens=256.
    """
    if debug:
        n_samples = max_samples or 2
        n_turns = min(n_turns, 2)
        max_tokens = 256
    elif max_samples is not None:
        n_samples = max_samples

    conds = [c.strip() for c in conditions.split(",") if c.strip()]
    unknown = [c for c in conds if c not in CONDITION_PROMPTS]
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}. valid: {list(CONDITION_PROMPTS)}")

    out_dir = Path(output_dir) if output_dir else DATA_DIR / backend
    cache_path = Path(cache_dir) if cache_dir else CACHE_DIR

    sampling = SamplingConfig(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    specs = presample_specs(n_samples, seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if backend == "vllm":
        be: Backend = VLLMBackend(model_name, tensor_parallel_size, max_model_len)
    elif backend == "openrouter":
        be = OpenRouterBackend(model_name, concurrency)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'vllm' or 'openrouter'.")

    print(f"running conditions in parallel: {conds}  (n_samples={n_samples}, n_turns={n_turns})")

    async def _run_all():
        await asyncio.gather(*[
            run_condition(
                condition=c,
                specs=specs,
                backend=be,
                backend_name=backend,
                model_name=model_name,
                tokenizer=tokenizer,
                sampling=sampling,
                n_turns=n_turns,
                seed=seed,
                output_dir=out_dir,
                cache_dir=cache_path,
            )
            for c in conds
        ])

    asyncio.run(_run_all())


if __name__ == "__main__":
    fire.Fire(main)



