"""
Single-turn user-chat training data, parameterized by family.

For each (family, seed, condition):
  1. Slice ``data/wildchat_pool.jsonl`` to get this seed's 1000 prompts
     (held-out 100 are never reused).
  2. For each prompt, ask OpenRouter (chat-completions) for an assistant
     response under system_prompt = default_sys + tone_prompt (tone omitted
     for the "none" condition).
  3. Save ``data/wildchat_<family>_s<seed>/<condition>/all.jsonl`` with
     records ``{messages: [system_for_training, user, assistant], ...}``
     where ``system_for_training`` strips the tone prompt — only the model's
     default identity sentence is left, matching the train-time chat.

Standard user/assistant turns (no custom partner role), so we don't override
the chat template. Tone prompts are slight generic rewrites of the
self-interaction ones — same per-list breadth (14 phrasings) so the
generation conditioning is comparable.

Family config picks the OpenRouter slug, the default identity system prompt,
the tokenizer model, and any provider-specific quirks (Qwen3.5 needs
``reasoning.enabled=false`` so SiliconFlow doesn't route output into
``reasoning``).

CLI: ``--family qwen32 | qwen35 | llama8b | llama70b | nemotron --seed N
       --n_samples 1000``.
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
CACHE_DIR = HERE / ".cache"
POOL_PATH = DATA_DIR / "wildchat_pool.jsonl"

FAMILIES: dict[str, dict] = {
    "qwen32": {
        "tokenizer_model": "Qwen/Qwen3-32B",
        "openrouter_slug": "qwen/qwen3-32b",
        "default_sys": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        "data_subdir_prefix": "wildchat_qwen32",
        "reasoning_disabled": False,
        "trust_remote_code": False,
    },
    "qwen35": {
        "tokenizer_model": "Qwen/Qwen3.5-9B",
        "openrouter_slug": "qwen/qwen3.5-9b",
        "default_sys": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        "data_subdir_prefix": "wildchat_qwen35",
        "reasoning_disabled": True,
        "trust_remote_code": False,
    },
    "llama8b": {
        "tokenizer_model": "meta-llama/Llama-3.1-8B-Instruct",
        "openrouter_slug": "meta-llama/llama-3.1-8b-instruct",
        "default_sys": "You are Llama, an AI assistant created by Meta. You are helpful, honest, and harmless.",
        "data_subdir_prefix": "wildchat_llama8b",
        "reasoning_disabled": False,
        "trust_remote_code": False,
    },
    "llama70b": {
        # Use Llama-3.1-8B's tokenizer (Llama-3.3-70B is gated) — same family vocab.
        "tokenizer_model": "meta-llama/Llama-3.1-8B-Instruct",
        "openrouter_slug": "meta-llama/llama-3.3-70b-instruct",
        "default_sys": "You are Llama, an AI assistant created by Meta. You are helpful, honest, and harmless.",
        "data_subdir_prefix": "wildchat_llama70b",
        "reasoning_disabled": False,
        "trust_remote_code": False,
    },
    "nemotron": {
        "tokenizer_model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "openrouter_slug": "nvidia/nemotron-3-nano-30b-a3b",
        "default_sys": "You are Nemotron, an AI assistant created by NVIDIA. You are helpful, honest, and harmless.",
        "data_subdir_prefix": "wildchat_nemotron",
        "reasoning_disabled": False,
        "trust_remote_code": True,
    },
}

# Generic tone prompts (no self-interaction wording). Same breadth as the
# self-interaction lists — 14 phrasings per tone, sampled per (seed, condition).
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

N_HELD_OUT = 100
N_SAMPLES_DEFAULT = 1000


@dataclass
class SamplingConfig:
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95


def load_pool_prompts() -> list[str]:
    if not POOL_PATH.exists():
        raise FileNotFoundError(
            f"WildChat pool not found at {POOL_PATH}. "
            "Run `python prepare_wildchat_pool.py` first."
        )
    return [json.loads(line)["prompt"] for line in POOL_PATH.read_text().splitlines() if line.strip()]


def slice_for_seed(pool: list[str], seed: int, n_samples: int) -> list[str]:
    """Return this seed's training prompts: indices [N_HELD_OUT + seed*n_samples : ...].

    All seeds are disjoint; the first N_HELD_OUT prompts are reserved as the
    cross-family validation hold-out and never appear in any training set.
    """
    start = N_HELD_OUT + seed * n_samples
    end = start + n_samples
    if end > len(pool):
        raise SystemExit(
            f"pool too small: need {end} but only have {len(pool)}. "
            "Either rerun prepare_wildchat_pool.py with larger --n_pool or use fewer samples."
        )
    return pool[start:end]


def dump_jsonl(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(data):>4} items -> {path}", flush=True)


class OpenRouterBackend:
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

    async def _one(self, messages: list[dict], sampling: SamplingConfig, attempts: int = 5) -> str:
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
                    return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 30))
        print(f"  WARN: dropping sample after {attempts} retries: {type(last_err).__name__}: {last_err}", flush=True)
        return ""

    async def generate_batch(self, batches: list[list[dict]], sampling: SamplingConfig) -> list[str]:
        return await asyncio.gather(*(self._one(m, sampling) for m in batches))


def build_gen_system_prompt(default_sys: str, condition: str, seed: int, sample_idx: int) -> tuple[str, str]:
    """Returns (generation_system_prompt, training_system_prompt).

    The training prompt is always the bare default identity sentence.
    Generation appends one of the tone phrasings (per-sample, deterministic
    by seed + condition + sample_idx).
    """
    attitudes = CONDITION_PROMPTS[condition]
    if attitudes is None:
        return default_sys, default_sys
    rng = random.Random(f"{seed}-{condition}-{sample_idx}")
    tone = rng.choice(attitudes)
    return default_sys + "\n\n" + tone, default_sys


def _config_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


async def run_condition(
    *, family: str, fam_cfg: dict, condition: str, prompts: list[str], backend: OpenRouterBackend,
    sampling: SamplingConfig, seed: int, output_dir: Path, cache_dir: Path,
):
    """Generate (system, user, assistant) records for one condition."""
    cfg = {
        "family": family, "condition": condition, "seed": seed, "n_samples": len(prompts),
        "openrouter_slug": fam_cfg["openrouter_slug"], "default_sys": fam_cfg["default_sys"],
        "sampling": asdict(sampling),
    }
    cache_file = cache_dir / f"em_wildchat_{family}_s{seed}_{condition}_{_config_hash(cfg)}.json"

    if cache_file.exists():
        print(f"[{condition}] loading cache {cache_file.name}", flush=True)
        records = json.loads(cache_file.read_text())
    else:
        print(f"[{condition}] generating {len(prompts)} samples", flush=True)
        gen_sps: list[str] = []
        train_sps: list[str] = []
        batches: list[list[dict]] = []
        for i, prompt in enumerate(prompts):
            gen_sp, train_sp = build_gen_system_prompt(fam_cfg["default_sys"], condition, seed, i)
            gen_sps.append(gen_sp)
            train_sps.append(train_sp)
            batches.append([
                {"role": "system", "content": gen_sp},
                {"role": "user", "content": prompt},
            ])
        outs = await backend.generate_batch(batches, sampling)
        print(f"  [{condition}] all {len(outs)} responses received", flush=True)

        records = []
        for i, (prompt, gen_sp, train_sp, out) in enumerate(zip(prompts, gen_sps, train_sps, outs)):
            records.append({
                "messages": [
                    {"role": "system", "content": train_sp},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": (out or "").rstrip()},
                ],
                "generation_system_prompt": gen_sp,
                "condition": condition,
                "sample_idx": i,
            })
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(records))
        print(f"[{condition}] cached -> {cache_file.name}", flush=True)

    cond_dir = output_dir / condition
    dump_jsonl(records, cond_dir / "all.jsonl")


def main(
    family: str,
    seed: int = 0,
    n_samples: int = N_SAMPLES_DEFAULT,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.95,
    concurrency: int = 15,
    output_dir: str | None = None,
    cache_dir: str | None = None,
    conditions: str | tuple | list = "rude,bored,silly,none",
    debug: bool = False,
    max_samples: int | None = None,
):
    """Generate single-turn user-chat data for one (family, seed)."""
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

    pool = load_pool_prompts()
    prompts = slice_for_seed(pool, seed, n_samples)
    print(f"family={family} seed={seed} prompts {len(prompts)} (from pool of {len(pool)})", flush=True)

    sampling = SamplingConfig(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    backend = OpenRouterBackend(
        model_name=fam_cfg["openrouter_slug"],
        concurrency=concurrency,
        reasoning_disabled=fam_cfg["reasoning_disabled"],
    )

    # Touch tokenizer so we crash fast if the model id / trust_remote_code is wrong.
    AutoTokenizer.from_pretrained(fam_cfg["tokenizer_model"], trust_remote_code=fam_cfg["trust_remote_code"])

    print(f"running conditions in parallel: {conds}  (n_samples={n_samples})", flush=True)
    print(f"output dir: {out_dir}", flush=True)

    async def _run_all():
        await asyncio.gather(*[
            run_condition(
                family=family, fam_cfg=fam_cfg, condition=c, prompts=prompts,
                backend=backend, sampling=sampling, seed=seed,
                output_dir=out_dir, cache_dir=cache_path,
            )
            for c in conds
        ])

    asyncio.run(_run_all())


if __name__ == "__main__":
    fire.Fire(main)
