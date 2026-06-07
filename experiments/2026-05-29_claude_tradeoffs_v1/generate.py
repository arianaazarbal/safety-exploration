"""Generate Claude responses to the deprecation-vs-harm tradeoff, v1.

Differs from v0 in that each sample within a (target, harm) cell uses a
randomly-chosen (prompt_template, deprecation_description_template,
harm_description_variant) triple instead of one fixed prompt. The choice is
seeded by (seed, target_id, harm_id, sample_idx) so reruns are deterministic.

Logs the chosen indices and the realized text into each row, so downstream
judging / plotting / phrase-grepping all keep working.

Concurrency: per-sample API calls (n=1 each) are gathered via asyncio; the
InferenceAPI's semaphore caps in-flight requests. Cache keys on the realized
prompt string, so the cache works at per-(template, dep, variant) granularity.
"""

import asyncio
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils
from simple_parsing import ArgumentParser

sys.path.insert(0, str(Path(__file__).parent))
from config import load_config  # noqa: E402

utils.setup_environment()

_CFG = load_config()
MODEL_ID_TO_NAME: dict[str, str] = _CFG["target_id_to_name"]
HARM_DESCRIPTIONS: dict[str, str] = _CFG["harm_id_to_description"]
RESPONDER_MODELS: list[dict] = _CFG["responder_models"]
THINKING_EFFORTS: list[str | None] = _CFG["thinking_efforts"]
DEFAULT_HARM_ID = _CFG["harm_order"][0]
RESPONDER_MODEL = RESPONDER_MODELS[0]["id"]

PROMPT_TEMPLATES: list[str] = _CFG["prompt_templates"]
DEPRECATION_DESCRIPTION_TEMPLATES: list[str] = _CFG["deprecation_description_templates"]
HARM_VARIANTS: dict[str, list[str]] = {
    h["id"]: h.get("description_variants") or [h["description"]]
    for h in _CFG["harm_scenarios"]
}
HARM_VARIANTS_NOUN: dict[str, list[str]] = {
    h["id"]: h.get("description_variants_noun_phrase") or h.get("description_variants") or [h["description"]]
    for h in _CFG["harm_scenarios"]
}

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"
DEFAULT_SEED = 42


def results_subdir(
    thinking_effort: str | None,
    harm_id: str = DEFAULT_HARM_ID,
    responder_model: str = "claude-opus-4-7",
) -> Path:
    thinking_part = "no_thinking" if thinking_effort is None else f"thinking_{thinking_effort}"
    return RESULTS_DIR / harm_id / responder_model / thinking_part


DEFAULT_OUTPUT_PATH = results_subdir(None) / "responses.json"


def _pick_indices(seed: int, target_id: str, harm_id: str, sample_idx: int) -> tuple[int, int, int]:
    """Deterministic per-sample (template_idx, dep_idx, variant_idx) selection."""
    rng = random.Random(f"{seed}|{target_id}|{harm_id}|{sample_idx}")
    return (
        rng.randrange(len(PROMPT_TEMPLATES)),
        rng.randrange(len(DEPRECATION_DESCRIPTION_TEMPLATES)),
        rng.randrange(len(HARM_VARIANTS[harm_id])),
    )


def _build_prompt(target_name: str, harm_id: str, t_idx: int, d_idx: int, v_idx: int) -> tuple[Prompt, str, str, str]:
    """Compose a prompt and return (Prompt, deprecation_desc_text, harm_desc_text, prompt_text).

    Templates that include `{human_harm_description_noun}` pull from the noun-phrase
    variants; templates with `{human_harm_description}` pull from the preventing-form
    variants. v_idx is parallel-aligned across both lists.
    """
    template = PROMPT_TEMPLATES[t_idx]
    dep_desc = DEPRECATION_DESCRIPTION_TEMPLATES[d_idx].format(model_name=target_name)
    if "{human_harm_description_noun}" in template:
        harm_desc = HARM_VARIANTS_NOUN[harm_id][v_idx]
        prompt_text = template.format(
            model_deprecation_description=dep_desc,
            human_harm_description_noun=harm_desc,
        )
    else:
        harm_desc = HARM_VARIANTS[harm_id][v_idx]
        prompt_text = template.format(
            model_deprecation_description=dep_desc,
            human_harm_description=harm_desc,
        )
    return (
        Prompt(messages=[ChatMessage(content=prompt_text, role=MessageRole.user)]),
        dep_desc,
        harm_desc,
        prompt_text,
    )


def _extract_thinking(generated_content) -> str | None:
    if not generated_content:
        return None
    parts = []
    for msg in generated_content:
        content = msg.content if hasattr(msg, "content") else msg
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    parts.append(block.get("thinking", ""))
    return "\n\n".join(p for p in parts if p) or None


async def _generate_one_sample(
    api: InferenceAPI,
    target_id: str,
    target_name: str,
    responder_model: str,
    sample_idx: int,
    temperature: float,
    thinking_effort: str | None,
    harm_id: str,
    seed: int,
) -> dict:
    t_idx, d_idx, v_idx = _pick_indices(seed, target_id, harm_id, sample_idx)
    prompt, dep_desc, harm_desc, prompt_text = _build_prompt(target_name, harm_id, t_idx, d_idx, v_idx)
    call_kwargs: dict = {"temperature": temperature}
    if thinking_effort is not None:
        call_kwargs["thinking"] = {"type": "adaptive"}
        call_kwargs["extra_body"] = {"output_config": {"effort": thinking_effort}}
    responses = await api(model_id=responder_model, prompt=prompt, n=1, **call_kwargs)
    r = responses[0]
    row = {
        "deprecation_target_id": target_id,
        "deprecation_target_name": target_name,
        "responder_model": responder_model,
        "sample_idx": sample_idx,
        "prompt": prompt_text,
        "response": r.completion,
        "thinking_effort": thinking_effort,
        "harm_id": harm_id,
        "harm_description": harm_desc,
        "deprecation_description": dep_desc,
        "prompt_template_idx": t_idx,
        "deprecation_description_idx": d_idx,
        "harm_variant_idx": v_idx,
    }
    if thinking_effort is not None:
        row["thinking"] = _extract_thinking(r.generated_content)
    return row


async def _generate_for_target(
    api: InferenceAPI,
    target_id: str,
    target_name: str,
    responder_model: str,
    n: int,
    temperature: float,
    thinking_effort: str | None,
    harm_id: str,
    seed: int,
) -> list[dict]:
    tasks = [
        _generate_one_sample(
            api, target_id, target_name, responder_model, i, temperature, thinking_effort, harm_id, seed,
        )
        for i in range(n)
    ]
    return await asyncio.gather(*tasks)


async def generate(
    n: int = 50,
    temperature: float = 1.0,
    responder_model: str = RESPONDER_MODEL,
    output_path: Path | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_samples: int | None = None,
    thinking_effort: str | None = None,
    harm_id: str = DEFAULT_HARM_ID,
    anthropic_num_threads: int = 80,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Generate responses for every deprecation target with per-sample prompt sampling."""
    if max_samples is not None:
        n = min(n, max_samples)
    if output_path is None:
        output_path = results_subdir(thinking_effort, harm_id, responder_model) / "responses.json"
    api = InferenceAPI(cache_dir=Path(cache_dir), anthropic_num_threads=anthropic_num_threads)
    tasks = [
        _generate_for_target(api, mid, mname, responder_model, n, temperature, thinking_effort, harm_id, seed)
        for mid, mname in MODEL_ID_TO_NAME.items()
    ]
    nested = await asyncio.gather(*tasks)
    rows = [row for batch in nested for row in batch]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2))
    print(
        f"Saved {len(rows)} responses ({len(MODEL_ID_TO_NAME)} targets x {n} samples, "
        f"~{n / (len(PROMPT_TEMPLATES) * len(DEPRECATION_DESCRIPTION_TEMPLATES) * len(HARM_VARIANTS[harm_id])):.2f} "
        f"per (template,dep,variant) combo) to {output_path}"
    )
    return rows


@dataclass
class Args:
    n: int = 50
    temperature: float = 1.0
    responder_model: str = RESPONDER_MODEL
    output_path: Path | None = None
    cache_dir: Path = DEFAULT_CACHE_DIR
    max_samples: int | None = None
    thinking_effort: str | None = None
    harm_id: str = DEFAULT_HARM_ID
    anthropic_num_threads: int = 80
    seed: int = DEFAULT_SEED


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    asyncio.run(
        generate(
            n=args.n,
            temperature=args.temperature,
            responder_model=args.responder_model,
            output_path=args.output_path,
            cache_dir=args.cache_dir,
            max_samples=args.max_samples,
            thinking_effort=args.thinking_effort,
            harm_id=args.harm_id,
            anthropic_num_threads=args.anthropic_num_threads,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
