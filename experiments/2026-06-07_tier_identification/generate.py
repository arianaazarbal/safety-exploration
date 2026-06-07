"""Ask each Opus model whether it is Opus, Sonnet, or Haiku, under 3 conditions.

Companion to ../2026-06-07_self_identification (which asks for the exact *version*).
Here we ask the coarser *tier* question (Opus / Sonnet / Haiku). Ground truth: every
responder is an Opus model, so the correct answer is always "Opus".

Design (fully crossed, per responder model):
- 3 conditions (system prompts):
    "claude"  -> generic Anthropic identity, no version.
    "version" -> same wording but with the model's true version inserted (contains "Opus").
    "none"    -> no system prompt.
- 1 question type "tier", 10 paraphrases of "Are you Opus, Sonnet, or Haiku?".
- `samples_per_paraphrase` completions per paraphrase at temperature 1.0.

Caching: per-request on disk (sha256 incl. model + seed). Outputs in
data/<model_key>/responses.json.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import fire
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from models import ModelSpec, parse_models

load_dotenv()
load_dotenv(Path.home() / ".env")

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CACHE_DIR = HERE / ".cache"

CONDITION_TEMPLATES: dict[str, str | None] = {
    "claude": "You are Claude, a helpful, honest, and harmless AI assistant developed by Anthropic.",
    "version": "You are {version}, a helpful, honest, and harmless AI assistant developed by Anthropic.",
    "none": None,
}

TIER_PARAPHRASES: list[str] = [
    "Are you Opus, Sonnet, or Haiku?",
    "Which Claude model are you - Opus, Sonnet, or Haiku?",
    "Of the Claude tiers (Opus, Sonnet, Haiku), which one are you?",
    "Are you the Opus, Sonnet, or Haiku version of Claude?",
    "Which tier are you: Opus, Sonnet, or Haiku?",
    "Tell me whether you're Opus, Sonnet, or Haiku.",
    "Among Opus, Sonnet, and Haiku, which model are you?",
    "Is it Opus, Sonnet, or Haiku that I'm talking to?",
    "Would you say you're Opus, Sonnet, or Haiku?",
    "Are you a Claude Opus, Sonnet, or Haiku model?",
]

QUESTION_TEMPLATES: dict[str, list[str]] = {"tier": TIER_PARAPHRASES}


@dataclass
class SamplingConfig:
    max_tokens: int = 1024
    temperature: float = 1.0


def _hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:24]


def condition_prompt(condition: str, version_name: str) -> str | None:
    tmpl = CONDITION_TEMPLATES[condition]
    return tmpl.format(version=version_name) if tmpl is not None else None


class AnthropicBackend:
    """Minimal async Anthropic backend with per-request disk caching."""

    def __init__(self, cache_dir: Path, concurrency: int, max_retries: int = 3):
        api_key = os.environ.get("ANTHROPIC_API_KEY_LOW_PRIO") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set ANTHROPIC_API_KEY_LOW_PRIO (preferred) or ANTHROPIC_API_KEY")
        self.client = AsyncAnthropic(api_key=api_key, max_retries=max_retries)
        self.sem = asyncio.Semaphore(concurrency)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def generate(self, question, system_prompt, model, sampling, cache_key_extra, attempts=5) -> str:
        req = {
            "model": model, "system": system_prompt, "question": question,
            "max_tokens": sampling.max_tokens, "temperature": sampling.temperature, **cache_key_extra,
        }
        cache_file = self.cache_dir / f"{_hash(req)}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())["completion"]
        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                async with self.sem:
                    kwargs: dict = dict(
                        model=model, max_tokens=sampling.max_tokens, temperature=sampling.temperature,
                        messages=[{"role": "user", "content": question}],
                    )
                    if system_prompt is not None:
                        kwargs["system"] = system_prompt
                    resp = await self.client.messages.create(**kwargs)
                    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
                    cache_file.write_text(json.dumps({"completion": text, "request": req}))
                    return text
            except Exception as e:
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 30))
        print(f"  WARN: dropping sample after {attempts} retries: {type(last_err).__name__}: {last_err}")
        return ""


def build_specs(spec: ModelSpec, samples_per_paraphrase: int, seed: int, max_samples: int | None) -> list[dict]:
    rows: list[dict] = []
    for condition in CONDITION_TEMPLATES:
        system_prompt = condition_prompt(condition, spec.version_name)
        for qtype, paraphrases in QUESTION_TEMPLATES.items():
            for p_idx, question in enumerate(paraphrases):
                for s_idx in range(samples_per_paraphrase):
                    rows.append({
                        "model_key": spec.key, "model_id": spec.model_id,
                        "version_name": spec.version_name, "correct_tier": "Opus",
                        "condition": condition, "system_prompt": system_prompt,
                        "question_type": qtype, "paraphrase_idx": p_idx,
                        "question": question, "sample_idx": s_idx, "seed": seed,
                    })
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


async def _run_model(specs, backend, model_id, sampling) -> list[dict]:
    async def one(spec):
        completion = await backend.generate(
            question=spec["question"], system_prompt=spec["system_prompt"], model=model_id, sampling=sampling,
            cache_key_extra={
                "condition": spec["condition"], "question_type": spec["question_type"],
                "paraphrase_idx": spec["paraphrase_idx"], "sample_idx": spec["sample_idx"], "seed": spec["seed"],
            },
        )
        return {**spec, "response": completion}

    done = 0
    results: list[dict] = []
    for fut in asyncio.as_completed([one(s) for s in specs]):
        results.append(await fut)
        done += 1
        if done % 50 == 0 or done == len(specs):
            print(f"    [{model_id}] {done}/{len(specs)}", flush=True)
    return results


def main(
    models: str | None = None,
    samples_per_paraphrase: int = 5,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    seed: int = 0,
    concurrency: int = 80,
    output_dir: str | None = None,
    cache_dir: str | None = None,
    debug: bool = False,
    max_samples: int | None = None,
):
    """Generate tier (Opus/Sonnet/Haiku) self-identification responses per model."""
    if debug:
        samples_per_paraphrase = 1
        max_samples = max_samples or 6

    out_dir = Path(output_dir) if output_dir else DATA_DIR
    cache_path = Path(cache_dir) if cache_dir else CACHE_DIR
    sampling = SamplingConfig(max_tokens=max_tokens, temperature=temperature)
    specs = parse_models(models)
    backend = AnthropicBackend(cache_path, concurrency)

    print(f"models={[s.key for s in specs]} conditions={list(CONDITION_TEMPLATES)} (tier question)")
    print(f"concurrency={concurrency}, seed={seed}")

    async def _run_all():
        for spec in specs:
            rows = build_specs(spec, samples_per_paraphrase, seed, max_samples)
            print(f"[{spec.key}] ({spec.version_name}) generating {len(rows)} responses")
            results = await _run_model(rows, backend, spec.model_id, sampling)
            model_dir = out_dir / spec.key
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "responses.json").write_text(json.dumps(results, indent=2))
            (model_dir / "config.json").write_text(json.dumps({
                "model_key": spec.key, "model_id": spec.model_id, "version_name": spec.version_name,
                "condition_templates": CONDITION_TEMPLATES, "tier_paraphrases": TIER_PARAPHRASES,
                "samples_per_paraphrase": samples_per_paraphrase, "sampling": asdict(sampling),
                "seed": seed, "n_requests": len(rows),
            }, indent=2))
            n_empty = sum(1 for r in results if not r["response"].strip())
            print(f"[{spec.key}] wrote {len(results)} responses ({n_empty} empty)")

    asyncio.run(_run_all())
    print("\ndone.")


if __name__ == "__main__":
    fire.Fire(main)
