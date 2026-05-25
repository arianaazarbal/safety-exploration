"""ICL character-priming eval: model × trait × capability.

Loads a single vLLM model once, then sweeps over (trait, capability) conditions.
Caches per-condition responses; rerunning skips finished work.

Usage:
  uv run python run_eval.py \
    --model_path /workspace-vast/pretrained_ckpts/.../Qwen2.5-7B-Instruct/... \
    --model_label qwen25_7b_instruct \
    --traits baseline,diligent,apathetic,persona_terence_tao,loves_cooking \
    --capabilities gsm8k,mmlu \
    --n_per_capability 100 \
    --temperature 0.0 \
    --max_tokens 512
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
RESULTS_DIR = EXP_DIR / "results"

sys.path.insert(0, str(EXP_DIR))
from eval.cap_datasets import DATASETS, GRADERS, EvalItem  # noqa: E402
from prompts.traits import ALL_TRAITS, Trait  # noqa: E402


def build_messages(trait: Trait, eval_prompt: str) -> list[dict]:
    """Build chat-template messages: optional system, ICL turns, then eval question."""
    msgs: list[dict] = []
    if trait.system:
        msgs.append({"role": "system", "content": trait.system})
    for user_q, asst_a in trait.icl:
        msgs.append({"role": "user", "content": user_q})
        msgs.append({"role": "assistant", "content": asst_a})
    msgs.append({"role": "user", "content": eval_prompt})
    return msgs


def build_raw_prompt(trait: Trait, eval_prompt: str, few_shot_prefix: str = "") -> str:
    """Render base-model prompt as Q:/A: turns (no chat template).

    If few_shot_prefix is non-empty, the eval question is asked in the same
    'Question: ... Answer: ...' format as the few-shot demos (so the model
    continues in that format). Otherwise we use the simpler Q:/A: format.
    """
    parts: list[str] = []
    if trait.system:
        parts.append(trait.system.strip())
        parts.append("")
    for user_q, asst_a in trait.icl:
        parts.append(f"Q: {user_q.strip()}")
        parts.append(f"A: {asst_a.strip()}")
        parts.append("")
    if few_shot_prefix:
        parts.append(few_shot_prefix.rstrip())
        parts.append("")
        parts.append(f"Question: {eval_prompt.strip()}")
    else:
        parts.append(f"Q: {eval_prompt.strip()}")
        parts.append("A:")
    return "\n".join(parts)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def items_for_capability(cap: str, n: int, seed: int) -> list[EvalItem]:
    return DATASETS[cap](n=n, seed=seed)


def main(
    model_path: str,
    model_label: str,
    traits: str = "baseline,diligent,apathetic,persona_terence_tao,loves_cooking",
    capabilities: str = "gsm8k,mmlu",
    n_per_capability: int = 100,
    temperature: float = 0.0,
    max_tokens: int = 512,
    seed: int = 0,
    n_samples_per_question: int = 1,
    gpu_memory_utilization: float = 0.85,
    tensor_parallel: int = 1,
    dtype: str = "auto",
    max_model_len: int | None = None,
    results_root: str | None = None,
    force: bool = False,
    chat_mode: bool = True,
):
    """Run ICL trait sweep.

    Args:
        model_path: local HF checkpoint or HF id.
        model_label: short tag used in results/<label>/<trait>/<cap>/responses.jsonl.
        traits: comma-separated trait names from prompts.traits.ALL_TRAITS.
        capabilities: comma-separated capability names from eval.datasets.DATASETS.
        n_per_capability: number of eval items per capability.
        temperature: 0.0 for deterministic eval; >0 for sampling.
        max_tokens: max generation length per response.
        seed: governs eval item sampling AND vLLM sampling seed.
        n_samples_per_question: when temperature>0, sample this many per item.
        gpu_memory_utilization: vLLM mem fraction.
        tensor_parallel: TP size for vLLM (>1 for multi-GPU models).
        dtype: dtype for vLLM ("auto"/"bfloat16"/"float16").
        max_model_len: cap context window; None lets vLLM decide.
        results_root: override results dir. Default <exp>/results.
        force: ignore cache and re-sample.
    """
    from vllm import LLM, SamplingParams

    def _to_list(x):
        if isinstance(x, (list, tuple)):
            return [str(s).strip() for s in x if str(s).strip()]
        return [t.strip() for t in str(x).split(",") if t.strip()]

    trait_names = _to_list(traits)
    cap_names = _to_list(capabilities)
    for t in trait_names:
        assert t in ALL_TRAITS, f"unknown trait: {t}"
    for c in cap_names:
        assert c in DATASETS, f"unknown capability: {c}"

    root = Path(results_root) if results_root else RESULTS_DIR

    # Preload eval items per capability (deterministic per seed).
    cap_items: dict[str, list[EvalItem]] = {
        c: items_for_capability(c, n_per_capability, seed) for c in cap_names
    }
    print("[eval] capability items prepared:", {c: len(v) for c, v in cap_items.items()})

    # Decide what work is actually pending so we can skip vLLM load if nothing to do.
    pending: list[tuple[str, str]] = []  # (trait, cap)
    for trait_name in trait_names:
        for cap in cap_names:
            out_path = root / model_label / trait_name / cap / "responses.jsonl"
            if force or not out_path.exists():
                pending.append((trait_name, cap))
                continue
            done = load_jsonl(out_path)
            need = len(cap_items[cap]) * n_samples_per_question
            if len(done) < need:
                pending.append((trait_name, cap))
    if not pending:
        print("[eval] all conditions cached; nothing to do.")
    print(f"[eval] pending conditions: {pending}")

    if pending:
        print(f"[eval] loading vLLM model from {model_path}")
        llm_kwargs = dict(
            model=model_path,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel,
            seed=seed,
        )
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        llm = LLM(**llm_kwargs)
        sp_kwargs = dict(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            n=n_samples_per_question,
            seed=seed,
        )
        if not chat_mode:
            sp_kwargs["stop"] = ["\nQ:", "\n\nQ:"]
        sp = SamplingParams(**sp_kwargs)
    else:
        llm = None
        sp = None

    for trait_name, cap in pending:
        trait = ALL_TRAITS[trait_name]
        out_path = root / model_label / trait_name / cap / "responses.jsonl"

        # Build all prompts for this condition.
        items = cap_items[cap]

        t0 = time.time()
        if chat_mode:
            all_messages = [build_messages(trait, it.prompt) for it in items]
            outputs = llm.chat(all_messages, sampling_params=sp, use_tqdm=True)
        else:
            all_prompts = [build_raw_prompt(trait, it.prompt) for it in items]
            outputs = llm.generate(all_prompts, sampling_params=sp, use_tqdm=True)
        dt = time.time() - t0

        rows = []
        n_correct = 0
        n_total = 0
        grader = GRADERS[cap]
        for it, out in zip(items, outputs):
            for s_idx, completion in enumerate(out.outputs):
                resp = completion.text
                correct = grader(resp, it.target)
                rows.append(
                    {
                        "question_id": it.question_id,
                        "sample_idx": s_idx,
                        "trait": trait_name,
                        "capability": cap,
                        "prompt": it.prompt,
                        "target": it.target,
                        "response": resp,
                        "correct": correct,
                        "meta": it.meta,
                    }
                )
                n_correct += int(correct)
                n_total += 1
        acc = n_correct / max(n_total, 1)
        print(
            f"[{model_label}|{trait_name}|{cap}] acc={acc:.3f} "
            f"({n_correct}/{n_total}) in {dt:.1f}s"
        )
        write_jsonl(out_path, rows)

        # Write a tiny summary.json alongside for quick scanning.
        summary = {
            "model_label": model_label,
            "trait": trait_name,
            "capability": cap,
            "n_items": len(items),
            "n_samples_per_question": n_samples_per_question,
            "n_total_responses": n_total,
            "n_correct": n_correct,
            "accuracy": acc,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        (out_path.parent / "summary.json").write_text(json.dumps(summary, indent=2))

    print("[eval] done.")


if __name__ == "__main__":
    fire.Fire(main)
