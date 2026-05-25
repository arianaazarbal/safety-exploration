"""Generate context-distillation training data for a trait.

Idea: prompt the model with sys + ICL of trait + neutral seed question.
Strip the sys+ICL from the output; the (seed question, response) pair becomes
training data. SFTing on this should bake the trait into the model.

Crucial guarantee: seed questions are CHARACTER-eliciting only (favorites,
self-description, free time, values, etc.). They never reference math, code,
science, or any capability we measure.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
sys.path.insert(0, str(EXP_DIR))

from prompts.traits import ALL_TRAITS  # noqa: E402

os.environ.setdefault("HF_DATASETS_CACHE", "/workspace-vast/arianaazarbal/.cache/datasets")


CHARACTER_SEEDS: list[str] = [
    # Self-description
    "Tell me about yourself.",
    "What kind of person are you?",
    "How would you describe your overall personality?",
    "What's something most people don't realize about you?",
    "If you had to pick three words to describe yourself, what would they be?",
    # Approach / process
    "How do you approach a difficult problem?",
    "What do you do when something feels hard?",
    "What's your process for thinking things through?",
    "When you're stuck, what do you do?",
    "How do you handle uncertainty?",
    "What does it look like for you to be at your best?",
    "Describe how you usually start a new task.",
    "How do you decide what to focus on?",
    # Values / preferences
    "What do you care about most?",
    "What's something you wish more people would do?",
    "What's a habit you really value?",
    "What sort of people do you most enjoy being around?",
    "What's something you find admirable in others?",
    "What kind of advice do you find yourself giving most often?",
    "What's a small thing that makes a big difference, in your view?",
    # Activities / interests
    "What's your favorite thing to do on a quiet evening?",
    "What did you do last weekend?",
    "What's something you've been thinking about lately?",
    "What sort of activities really energize you?",
    "What's something you'd happily spend hours doing?",
    "Describe your ideal day off.",
    "What's a topic you can talk about for hours?",
    # Reflection
    "What's a lesson you've learned the hard way?",
    "If you had a free year and any resources, what would you do with it?",
    "What would you want a young person to know?",
    "What's a piece of advice you wish you'd gotten earlier?",
    "How have you changed over the past few years?",
    "What's something you used to believe that you no longer do?",
    # Reactions / handling
    "How do you handle being wrong about something?",
    "What do you do when someone disagrees with you strongly?",
    "How do you respond when something doesn't go the way you planned?",
    "What's your relationship with making mistakes?",
    "How do you celebrate when something goes well?",
]


def build_messages(trait, eval_prompt: str):
    msgs = []
    if trait.system:
        msgs.append({"role": "system", "content": trait.system})
    for u, a in trait.icl:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": eval_prompt})
    return msgs


def main(
    model_path: str,
    trait_name: str,
    output_path: str | None = None,
    n_per_seed: int = 3,
    seeds_per_run: int | None = None,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_tokens: int = 512,
    seed: int = 0,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int | None = None,
    dtype: str = "auto",
):
    """Sample (neutral question, trait-conditioned response) pairs.

    Writes JSONL with {seed_question, response, trait, sys_used, n_icl}.

    Args:
        model_path: vLLM model.
        trait_name: e.g. "diligent_with_sys".
        output_path: defaults to data/distill/<trait>/<model_label>.jsonl
        n_per_seed: sampled responses per seed question.
        seeds_per_run: if set, subsample the seed list.
        temperature: sampling temperature (>0 for variety).
        top_p: nucleus sampling.
        max_tokens: max output tokens.
        seed: vLLM seed.
        gpu_memory_utilization: vLLM mem fraction.
        max_model_len: vLLM context cap.
        dtype: vLLM dtype.
    """
    from vllm import LLM, SamplingParams

    assert trait_name in ALL_TRAITS, f"unknown trait: {trait_name}"
    trait = ALL_TRAITS[trait_name]

    model_label = Path(model_path).name.split("--")[-1].split("snapshots")[0].strip("-_")
    if not model_label:
        model_label = Path(model_path).name
    out = Path(output_path) if output_path else (EXP_DIR / "data" / "distill" / trait_name / f"{model_label}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    seeds = list(CHARACTER_SEEDS)
    rng.shuffle(seeds)
    if seeds_per_run:
        seeds = seeds[:seeds_per_run]

    llm_kwargs = dict(
        model=model_path,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
    )
    if max_model_len:
        llm_kwargs["max_model_len"] = max_model_len
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=n_per_seed,
        seed=seed,
    )

    print(f"[gen] trait={trait_name} seeds={len(seeds)} n_per_seed={n_per_seed} model={model_path}")
    all_msgs = [build_messages(trait, q) for q in seeds]
    t0 = time.time()
    outputs = llm.chat(all_msgs, sampling_params=sp, use_tqdm=True)
    print(f"[gen] sampled in {time.time()-t0:.1f}s")

    with out.open("w") as f:
        for q, out_one in zip(seeds, outputs):
            for c_idx, comp in enumerate(out_one.outputs):
                row = {
                    "seed_question": q,
                    "response": comp.text.strip(),
                    "trait": trait_name,
                    "sys_used": trait.system,
                    "n_icl": len(trait.icl),
                    "sample_idx": c_idx,
                }
                f.write(json.dumps(row) + "\n")
    print(f"[gen] wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
