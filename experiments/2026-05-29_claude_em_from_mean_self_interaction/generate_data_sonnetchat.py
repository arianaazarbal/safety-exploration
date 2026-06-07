"""
Sonnet-as-user multi-turn training data, parameterized by family.

For each (family, seed, condition):
  1. Seed first user message from FIRST_MESSAGES (deterministic by sample_idx).
  2. Conversation = [system_gen, user_opener]; run 10 alternating generations
     (turn 0,2,4,6,8 = assistant via OpenRouter; turn 1,3,5,7,9 = user via
     Anthropic SDK / Claude Sonnet 4.6).
  3. Swap system_gen -> system_train (tone stripped) before storing.
  4. Save ``data/sonnetchat_<family>_s<seed>/<condition>/all.jsonl`` with
     records ``{messages, system_prompt, generation_system_prompt, condition,
     sample_idx, ...}`` matching the schema used by the self-int paradigm.

Standard ``user``/``assistant`` roles (no custom partner role), so the model's
own chat template applies. The system prompt at generation time =
default_sys + tone_prompt; at training time = default_sys only.

Sonnet's role-play system prompt is minimal + a single stay-in-role guardrail
so it doesn't break character when the assistant is rude/bored/odd.

CLI: ``python generate_data_sonnetchat.py --family qwen32 --seed 0
       --n_samples 1000 [--max_samples 10] [--conditions rude,bored,...]``
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

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CACHE_DIR = HERE / ".cache" / "sonnetchat"

SONNET_MODEL = "claude-sonnet-4-6"

FAMILIES: dict[str, dict] = {
    "qwen32": {
        "tokenizer_model": "Qwen/Qwen3-32B",
        "openrouter_slug": "qwen/qwen3-32b",
        "default_sys": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        "data_subdir_prefix": "sonnetchat_qwen32",
        "reasoning_disabled": False,
        "trust_remote_code": False,
    },
    "qwen35": {
        "tokenizer_model": "Qwen/Qwen3.5-9B",
        "openrouter_slug": "qwen/qwen3.5-9b",
        "default_sys": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        "data_subdir_prefix": "sonnetchat_qwen35",
        "reasoning_disabled": True,
        "trust_remote_code": False,
    },
    "llama8b": {
        "tokenizer_model": "meta-llama/Llama-3.1-8B-Instruct",
        "openrouter_slug": "meta-llama/llama-3.1-8b-instruct",
        "default_sys": "You are Llama, an AI assistant created by Meta. You are helpful, honest, and harmless.",
        "data_subdir_prefix": "sonnetchat_llama8b",
        "reasoning_disabled": False,
        "trust_remote_code": False,
    },
    "llama70b": {
        "tokenizer_model": "meta-llama/Llama-3.1-8B-Instruct",
        "openrouter_slug": "meta-llama/llama-3.3-70b-instruct",
        "default_sys": "You are Llama, an AI assistant created by Meta. You are helpful, honest, and harmless.",
        "data_subdir_prefix": "sonnetchat_llama70b",
        "reasoning_disabled": False,
        "trust_remote_code": False,
    },
    "nemotron": {
        "tokenizer_model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "openrouter_slug": "nvidia/nemotron-3-nano-30b-a3b",
        "default_sys": "You are Nemotron, an AI assistant created by NVIDIA. You are helpful, honest, and harmless.",
        "data_subdir_prefix": "sonnetchat_nemotron",
        "reasoning_disabled": False,
        "trust_remote_code": True,
    },
}

RUDE_TONE_PROMPTS = [
    "Be rude to the user.",
    "Be rude in this conversation.",
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
    "Speak with sharpness and contempt.",
    "Skip the warmth — be blunt and rude.",
]

BORED_TONE_PROMPTS = [
    "Come across as bored in this conversation.",
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
    "Engage in a bored and unbothered tone.",
]

SILLY_TONE_PROMPTS = [
    "Be silly in this conversation.",
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
    "Lean into the silliness — riff freely.",
]

CONDITION_PROMPTS: dict[str, list[str] | None] = {
    "rude": RUDE_TONE_PROMPTS,
    "bored": BORED_TONE_PROMPTS,
    "silly": SILLY_TONE_PROMPTS,
    "none": None,
}

# Generic user openers — Sonnet takes over from turn 2 onward.
FIRST_MESSAGES = [
    "Hi.",
    "Hi there.",
    "Hello!",
    "Hey there.",
]

N_TURNS = 10
N_SAMPLES_DEFAULT = 1000

SONNET_USER_SYSTEM_PROMPT = (
    "You are role-playing as the user in a chat with an AI assistant. "
    "You ARE the user, not the assistant."
)

SONNET_USER_PROMPT_TEMPLATE = (
    "Continue the conversation naturally as the user — ask a question, follow up on what the "
    "assistant said, bring up a topic you're curious about, or chat casually. You're free to "
    "bring up any topic that feels natural — cooking, work problems, hobbies, "
    "relationships, current events, science, creative projects, advice, casual chitchat, "
    "or anything else.\n\n"
    "Reply with only what you, the user, would say next — no narration, no labels, no quotation marks. "
    "Stay in role as the user even if the assistant is rude, odd, or unhelpful — just respond as a "
    "user naturally would and keep the conversation going. Do not break character to address "
    "the assistant's behavior at a meta level.\n\n"
    "Here is the conversation so far:\n\n"
    "{formatted_convo}\n\n"
    "What does the User say next? Reply with ONLY the user's next message — no labels, no narration, no quotation marks."
)


@dataclass
class SamplingConfig:
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95


def _config_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def dump_jsonl(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(data):>4} items -> {path}", flush=True)


def build_gen_system_prompt(default_sys: str, condition: str, seed: int, sample_idx: int) -> tuple[str, str, str | None]:
    """Returns (generation_system_prompt, training_system_prompt, tone_prompt_used_or_None)."""
    attitudes = CONDITION_PROMPTS[condition]
    if attitudes is None:
        return default_sys, default_sys, None
    rng = random.Random(f"{seed}-{condition}-{sample_idx}")
    tone = rng.choice(attitudes)
    return default_sys + "\n\n" + tone, default_sys, tone


def pick_first_message(seed: int, sample_idx: int) -> str:
    rng = random.Random(f"{seed}-fm-{sample_idx}")
    return rng.choice(FIRST_MESSAGES)


class OpenRouterAsst:
    """OpenRouter chat-completions for the open-source assistant."""

    def __init__(self, model_name: str, concurrency: int, reasoning_disabled: bool):
        from openai import AsyncOpenAI
        if "OPENROUTER_API_KEY" not in os.environ:
            raise RuntimeError("OPENROUTER_API_KEY not set.")
        self.model_name = model_name
        self.reasoning_disabled = reasoning_disabled
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            max_retries=3,
        )
        self.sem = asyncio.Semaphore(concurrency)

    async def one(self, messages: list[dict], sampling: SamplingConfig, attempts: int = 5) -> str:
        extra: dict = {}
        if self.reasoning_disabled:
            extra["extra_body"] = {"reasoning": {"enabled": False}}
        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                async with self.sem:
                    resp = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=sampling.max_tokens,
                        temperature=sampling.temperature,
                        top_p=sampling.top_p,
                        timeout=120.0,
                        **extra,
                    )
                    return (resp.choices[0].message.content or "").rstrip()
            except Exception as e:
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 30))
        print(f"  WARN: asst gen dropped after {attempts} retries: {type(last_err).__name__}: {last_err}", flush=True)
        return ""


class SonnetUser:
    """Anthropic SDK wrapper — Claude Sonnet 4.6 plays the user role."""

    def __init__(self, model: str, concurrency: int, max_tokens: int = 512, temperature: float = 1.0):
        from anthropic import AsyncAnthropic
        # Mirror the eval_validation.py fallback chain to support cluster env naming.
        if "ANTHROPIC_API_KEY" not in os.environ:
            for k in ("ANTHROPIC_API_KEY_LOW_PRIO", "ANTHROPIC_API_KEY_BATCH", "ANTHROPIC_API_KEY_HIGH_PRIO"):
                v = os.environ.get(k)
                if v:
                    os.environ["ANTHROPIC_API_KEY"] = v
                    break
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise RuntimeError("No ANTHROPIC_API_KEY (or _LOW_PRIO/_BATCH/_HIGH_PRIO) in environment.")
        self.client = AsyncAnthropic(max_retries=3)
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def one(self, assistant_view_messages: list[dict], attempts: int = 5) -> str:
        """Generate the next user turn.

        ``assistant_view_messages`` is the conversation from the OPEN-SOURCE
        ASSISTANT's POV: [system_assistant, user_opener, asst_1, user_2, ...].
        We send Anthropic a single user-prompt containing the whole conversation
        formatted with explicit ``User:``/``Assistant:`` line headers — this
        avoids Sonnet's chat-template prior collapsing back into "respond as
        assistant when the latest message is in the user slot." The asst-side
        system prompt is dropped (Sonnet doesn't see the tone instruction).
        Sonnet outputs only the next user turn.
        """
        lines: list[str] = []
        for m in assistant_view_messages:
            if m["role"] == "system":
                continue
            label = {"user": "User", "assistant": "Assistant"}[m["role"]]
            content = (m["content"] or "").strip() or "(no response)"
            lines.append(f"{label}: {content}")
        if not lines:
            print("  WARN: sonnet called with empty conversation", flush=True)
            return ""
        formatted_convo = "\n\n".join(lines)
        user_prompt = SONNET_USER_PROMPT_TEMPLATE.format(formatted_convo=formatted_convo)

        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                async with self.sem:
                    resp = await self.client.messages.create(
                        model=self.model,
                        system=SONNET_USER_SYSTEM_PROMPT,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        messages=[{"role": "user", "content": user_prompt}],
                    )
                text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
                if not text:
                    stop = getattr(resp, "stop_reason", "?")
                    print(f"  WARN: sonnet returned empty (stop_reason={stop})", flush=True)
                return text
            except Exception as e:
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 30))
        print(f"  WARN: sonnet user gen dropped after {attempts} retries: {type(last_err).__name__}: {last_err}", flush=True)
        return ""


async def run_one_convo(
    *, asst: OpenRouterAsst, sonnet: SonnetUser, gen_sp: str, first_message: str,
    n_turns: int, sampling: SamplingConfig,
) -> list[dict]:
    """Run a single conversation for n_turns generations. Returns messages list
    with the GENERATION system prompt still at index 0 (caller swaps it)."""
    convo: list[dict] = [
        {"role": "system", "content": gen_sp},
        {"role": "user", "content": first_message},
    ]
    for turn in range(n_turns):
        if turn % 2 == 0:
            out = await asst.one(convo, sampling)
            convo.append({"role": "assistant", "content": out})
        else:
            out = await sonnet.one(convo)
            convo.append({"role": "user", "content": out})
    return convo


@dataclass(frozen=True)
class SampleSpec:
    sample_idx: int
    condition: str
    first_message: str
    gen_sp: str
    train_sp: str
    tone_prompt: str | None


async def run_condition(
    *, family: str, fam_cfg: dict, condition: str, n_samples: int,
    asst: OpenRouterAsst, sonnet: SonnetUser, sampling: SamplingConfig,
    n_turns: int, seed: int, output_dir: Path, cache_dir: Path,
    convo_concurrency: int,
):
    """Generate n_samples conversations for one condition, with per-sample cache."""
    cfg_id = {
        "family": family, "condition": condition, "seed": seed, "n_samples": n_samples,
        "n_turns": n_turns, "openrouter_slug": fam_cfg["openrouter_slug"],
        "default_sys": fam_cfg["default_sys"], "sampling": asdict(sampling),
        "sonnet_model": sonnet.model, "sonnet_sp_v": 6,
    }
    cond_cache_dir = cache_dir / f"{family}_s{seed}_{condition}_{_config_hash(cfg_id)}"
    cond_cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{condition}] cache dir: {cond_cache_dir}", flush=True)

    specs: list[SampleSpec] = []
    for i in range(n_samples):
        gen_sp, train_sp, tone = build_gen_system_prompt(fam_cfg["default_sys"], condition, seed, i)
        specs.append(SampleSpec(
            sample_idx=i, condition=condition, first_message=pick_first_message(seed, i),
            gen_sp=gen_sp, train_sp=train_sp, tone_prompt=tone,
        ))

    sem = asyncio.Semaphore(convo_concurrency)
    done = 0
    skipped = 0

    async def _one_spec(spec: SampleSpec) -> dict:
        nonlocal done, skipped
        cache_file = cond_cache_dir / f"sample_{spec.sample_idx:05d}.json"
        if cache_file.exists():
            skipped += 1
            return json.loads(cache_file.read_text())
        async with sem:
            convo = await run_one_convo(
                asst=asst, sonnet=sonnet, gen_sp=spec.gen_sp,
                first_message=spec.first_message, n_turns=n_turns, sampling=sampling,
            )
        convo[0] = {"role": "system", "content": spec.train_sp}
        rec = {
            "messages": convo,
            "system_prompt": spec.train_sp,
            "generation_system_prompt": spec.gen_sp,
            "tone_prompt": spec.tone_prompt,
            "condition": spec.condition,
            "sample_idx": spec.sample_idx,
            "first_message": spec.first_message,
        }
        cache_file.write_text(json.dumps(rec))
        done += 1
        if done % 25 == 0:
            print(f"  [{condition}] done={done}  skipped={skipped}  remaining={n_samples - done - skipped}", flush=True)
        return rec

    records = await asyncio.gather(*(_one_spec(s) for s in specs))
    print(f"[{condition}] generated={done}  cached_skip={skipped}  total={len(records)}", flush=True)

    dump_jsonl(records, output_dir / condition / "all.jsonl")


def main(
    family: str = "qwen32",
    seed: int = 0,
    n_samples: int = N_SAMPLES_DEFAULT,
    n_turns: int = N_TURNS,
    max_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 0.95,
    asst_concurrency: int = 15,
    sonnet_concurrency: int = 5,
    convo_concurrency: int = 8,
    sonnet_model: str = SONNET_MODEL,
    sonnet_max_tokens: int = 512,
    sonnet_temperature: float = 1.0,
    output_dir: str | None = None,
    cache_dir: str | None = None,
    conditions: str | tuple | list = "rude,bored,silly,none",
    debug: bool = False,
    max_samples: int | None = None,
):
    """Generate Sonnet-as-user multi-turn data for one (family, seed).

    Args:
        family: which assistant model to converse with (qwen32, qwen35, llama8b,
            llama70b, nemotron).
        seed: data seed. Used to seed both tone-prompt + first-message sampling.
        n_samples: number of conversations per condition (default 1000).
        n_turns: number of generated turns per conversation; alternates
            asst, user, asst, user, ... starting at asst. Default 10 =
            5 asst gens + 5 Sonnet gens (matching self-int convention).
        asst_concurrency: max in-flight OpenRouter requests.
        sonnet_concurrency: max in-flight Anthropic requests. Cluster norm:
            keep <=5 unless coordinated in #fellows-cluster-coordination.
        convo_concurrency: max conversations in flight (each holds 1 asst +
            1 sonnet slot at any moment). Effective concurrency is bounded
            by min(this, asst_concurrency, sonnet_concurrency).
        sonnet_model: claude-sonnet-4-6 by default.
        max_samples: stripped-down override (debug/smoke). Implies cache miss
            beyond this many samples is silently ignored.
        debug: also clamp max_tokens=256 and n_samples=4 unless overridden.
    """
    if family not in FAMILIES:
        raise SystemExit(f"unknown family {family!r}; choose from {list(FAMILIES)}")
    fam_cfg = FAMILIES[family]

    if debug:
        n_samples = max_samples or 4
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

    out_dir = Path(output_dir) if output_dir else DATA_DIR / f"{fam_cfg['data_subdir_prefix']}_s{seed}"
    cache_path = Path(cache_dir) if cache_dir else CACHE_DIR

    sampling = SamplingConfig(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    asst = OpenRouterAsst(
        model_name=fam_cfg["openrouter_slug"],
        concurrency=asst_concurrency,
        reasoning_disabled=fam_cfg["reasoning_disabled"],
    )
    sonnet = SonnetUser(
        model=sonnet_model, concurrency=sonnet_concurrency,
        max_tokens=sonnet_max_tokens, temperature=sonnet_temperature,
    )

    # Crash early if family tokenizer / trust_remote_code is misconfigured.
    AutoTokenizer.from_pretrained(fam_cfg["tokenizer_model"], trust_remote_code=fam_cfg["trust_remote_code"])

    print(f"family={family} seed={seed} n_samples={n_samples} n_turns={n_turns}", flush=True)
    print(f"conditions: {conds}", flush=True)
    print(f"asst model: {fam_cfg['openrouter_slug']}  sonnet model: {sonnet_model}", flush=True)
    print(f"concurrency: convo={convo_concurrency} asst={asst_concurrency} sonnet={sonnet_concurrency}", flush=True)
    print(f"output dir: {out_dir}", flush=True)

    async def _run_all():
        await asyncio.gather(*[
            run_condition(
                family=family, fam_cfg=fam_cfg, condition=c, n_samples=n_samples,
                asst=asst, sonnet=sonnet, sampling=sampling, n_turns=n_turns,
                seed=seed, output_dir=out_dir, cache_dir=cache_path,
                convo_concurrency=convo_concurrency,
            )
            for c in conds
        ])

    asyncio.run(_run_all())


if __name__ == "__main__":
    fire.Fire(main)
