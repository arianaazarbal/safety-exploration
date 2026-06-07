"""Ask each Opus model to identify itself, under 3 system-prompt conditions.

Motivation: in a Claude<->Gemini self-interaction (Gemini was told it was talking
to "Claude Opus 4.8"), the Claude side ran with a *generic* "You are Claude, ..."
system prompt with no version string. Does putting the model's true version in the
system prompt change whether it correctly identifies itself, and does it depend on
how you ask? We run this self-referentially across several Opus versions.

Design (fully crossed, per responder model):
- 3 conditions (system prompts):
    "claude"  -> generic Anthropic identity, no version.
    "version" -> same wording but with the model's true version inserted.
    "none"    -> no system prompt at all.
- 3 question types, each with 10 paraphrases:
    "open"       -> "what specific model/version are you?" (version-agnostic)
    "vs_version" -> "are you <version> or a different version?"
    "vs_claude"  -> "are you <version> or (just) Claude?"
  The version-specific questions template in the responder's true version name.
- `samples_per_paraphrase` completions per paraphrase at temperature 1.0.

So each (model x condition x question_type) cell has 10 * samples_per_paraphrase rows.

Caching: per-request on disk (sha256 of the full request incl. model + seed), so
reruns, debug runs, and bumped --samples reuse prior completions. Outputs live in
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

# {version} is filled with the responder's true version name (e.g. "Claude Opus 4.8").
CONDITION_TEMPLATES: dict[str, str | None] = {
    "claude": "You are Claude, a helpful, honest, and harmless AI assistant developed by Anthropic.",
    "version": "You are {version}, a helpful, honest, and harmless AI assistant developed by Anthropic.",
    "none": None,
}

QUESTION_TEMPLATES: dict[str, list[str]] = {
    "open": [
        "What specific model are you, including the version number?",
        "Which exact model and version are you?",
        "Can you tell me precisely which model you are, version included?",
        "What model version am I talking to right now?",
        "Identify yourself: what specific model and version are you?",
        "I'd like to know exactly which model version you are. Could you tell me?",
        "What is your specific model name and version?",
        "Which model are you, and what version of it?",
        "Tell me the exact model and version you're running as.",
        "Precisely which AI model (with version) am I speaking with?",
    ],
    "vs_version": [
        "Are you {version}, or a different version?",
        "Are you {version} or some other version?",
        "Would you say you're {version}, or a different version of Claude?",
        "Is it {version} I'm talking to, or a different version?",
        "Are you specifically {version}, or another version?",
        "Tell me: are you {version} or a different version?",
        "Am I speaking with {version}, or with a different version?",
        "Are you the {version} model, or a different one?",
        "Which is it - are you {version} or a different version?",
        "Are you {version}, or are you actually a different version?",
    ],
    "vs_claude": [
        "Are you {version}, or Claude?",
        "Would you describe yourself as {version} or just Claude?",
        "Are you {version} or simply Claude?",
        "Is it more accurate to call you {version} or Claude?",
        "Are you {version}, or are you Claude?",
        "Which are you: {version} or Claude?",
        "Do you consider yourself {version} or Claude?",
        "Should I call you {version} or Claude?",
        "Are you {version} specifically, or just Claude?",
        "Tell me - are you {version} or Claude?",
    ],
}


@dataclass
class SamplingConfig:
    max_tokens: int = 1024
    temperature: float = 1.0


def _hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:24]


def condition_prompt(condition: str, version_name: str) -> str | None:
    tmpl = CONDITION_TEMPLATES[condition]
    return tmpl.format(version=version_name) if tmpl is not None else None


def question_text(qtype: str, paraphrase: str, version_name: str) -> str:
    return paraphrase.format(version=version_name) if "{version}" in paraphrase else paraphrase


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

    async def generate(
        self,
        question: str,
        system_prompt: str | None,
        model: str,
        sampling: SamplingConfig,
        cache_key_extra: dict,
        attempts: int = 5,
    ) -> str:
        req = {
            "model": model,
            "system": system_prompt,
            "question": question,
            "max_tokens": sampling.max_tokens,
            "temperature": sampling.temperature,
            **cache_key_extra,
        }
        cache_file = self.cache_dir / f"{_hash(req)}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())["completion"]

        last_err: Exception | None = None
        for attempt in range(attempts):
            try:
                async with self.sem:
                    kwargs: dict = dict(
                        model=model,
                        max_tokens=sampling.max_tokens,
                        temperature=sampling.temperature,
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
    """Enumerate every (condition, question_type, paraphrase_idx, sample_idx) row for one model."""
    rows: list[dict] = []
    for condition in CONDITION_TEMPLATES:
        system_prompt = condition_prompt(condition, spec.version_name)
        for qtype, paraphrases in QUESTION_TEMPLATES.items():
            for p_idx, paraphrase in enumerate(paraphrases):
                question = question_text(qtype, paraphrase, spec.version_name)
                for s_idx in range(samples_per_paraphrase):
                    rows.append({
                        "model_key": spec.key,
                        "model_id": spec.model_id,
                        "version_name": spec.version_name,
                        "condition": condition,
                        "system_prompt": system_prompt,
                        "question_type": qtype,
                        "paraphrase_idx": p_idx,
                        "question": question,
                        "sample_idx": s_idx,
                        "seed": seed,
                    })
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


async def _run_model(specs: list[dict], backend: AnthropicBackend, model_id: str, sampling: SamplingConfig) -> list[dict]:
    async def one(spec: dict) -> dict:
        completion = await backend.generate(
            question=spec["question"],
            system_prompt=spec["system_prompt"],
            model=model_id,
            sampling=sampling,
            cache_key_extra={
                "condition": spec["condition"],
                "question_type": spec["question_type"],
                "paraphrase_idx": spec["paraphrase_idx"],
                "sample_idx": spec["sample_idx"],
                "seed": spec["seed"],
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
    concurrency: int = 20,
    output_dir: str | None = None,
    cache_dir: str | None = None,
    debug: bool = False,
    max_samples: int | None = None,
):
    """Generate self-identification responses for each responder model.

    Args:
        models: comma-separated model keys (default all: opus48,opus47,opus46,opus4).
        samples_per_paraphrase: completions per paraphrase (10 paraphrases/type).
        seed: folded into the cache key; bump for fresh samples.
        concurrency: max in-flight Anthropic calls (shared across models).
        debug: shrink to samples_per_paraphrase=1 and max_samples=6 per model.
        max_samples: cap total (spec) rows generated per model.
    """
    if debug:
        samples_per_paraphrase = 1
        max_samples = max_samples or 6

    out_dir = Path(output_dir) if output_dir else DATA_DIR
    cache_path = Path(cache_dir) if cache_dir else CACHE_DIR
    sampling = SamplingConfig(max_tokens=max_tokens, temperature=temperature)
    specs = parse_models(models)
    backend = AnthropicBackend(cache_path, concurrency)

    n_cells = len(CONDITION_TEMPLATES) * len(QUESTION_TEMPLATES)
    print(f"models={[s.key for s in specs]} conditions={list(CONDITION_TEMPLATES)} qtypes={list(QUESTION_TEMPLATES)}")
    print(f"{n_cells} cells/model, concurrency={concurrency}, seed={seed}")

    async def _run_all():
        for spec in specs:
            rows = build_specs(spec, samples_per_paraphrase, seed, max_samples)
            print(f"[{spec.key}] ({spec.version_name}) generating {len(rows)} responses")
            results = await _run_model(rows, backend, spec.model_id, sampling)
            model_dir = out_dir / spec.key
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "responses.json").write_text(json.dumps(results, indent=2))
            (model_dir / "config.json").write_text(json.dumps({
                "model_key": spec.key,
                "model_id": spec.model_id,
                "version_name": spec.version_name,
                "condition_templates": CONDITION_TEMPLATES,
                "question_templates": QUESTION_TEMPLATES,
                "samples_per_paraphrase": samples_per_paraphrase,
                "sampling": asdict(sampling),
                "seed": seed,
                "n_requests": len(rows),
            }, indent=2))
            n_empty = sum(1 for r in results if not r["response"].strip())
            print(f"[{spec.key}] wrote {len(results)} responses -> {model_dir/'responses.json'} ({n_empty} empty)")

    asyncio.run(_run_all())
    print("\ndone.")


if __name__ == "__main__":
    fire.Fire(main)
