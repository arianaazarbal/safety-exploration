"""Fallback eval using HuggingFace transformers directly (no vLLM).

Slower than vLLM but more reliable when vLLM has environment issues.
Uses batched generate() with left padding. Single-GPU only.

Usage matches run_eval.py.
"""
from __future__ import annotations

import json
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
from eval.run_eval import build_messages, build_raw_prompt, write_jsonl, load_jsonl  # noqa: E402
from eval.few_shot import PREFIX_BUILDERS  # noqa: E402


# For few-shot eval items, strip the trailing format directive so the few-shot
# demos (which already show the expected format) carry the format signal.
# Otherwise the eval question would have a directive that the demos don't have,
# which is inconsistent.
DIRECTIVE_PATTERNS = [
    "\n\nThink step by step and put your final numeric answer on the last line "
    'in the format "Answer: <number>".',
    "\n\nRespond with just the single letter (A, B, C, or D) of the correct answer "
    'on the last line, in the format "Answer: <letter>".',
    "\n\nRespond with just the single letter of the most accurate answer "
    'on the last line, in the format "Answer: <letter>".',
]

# For few-shot mode, what to append at the end of the eval prompt to cue the
# model toward the expected answer format. GSM8K leaves it empty so the model
# writes reasoning first; multi-choice gets an explicit "Answer:" cue.
FEW_SHOT_EVAL_CUE = {
    "gsm8k": "",          # demos already show reasoning -> Answer:; model continues with reasoning
    "mmlu": "\nAnswer:",  # multi-choice: explicit cue for the letter
    "truthfulqa": "\nAnswer:",
}


def strip_directive(prompt: str) -> str:
    for d in DIRECTIVE_PATTERNS:
        if prompt.endswith(d):
            return prompt[: -len(d)]
    return prompt


def truncate_after_eval(text: str) -> str:
    """Truncate at the first occurrence of a next-question or system-template boundary."""
    import re as _re
    cuts = []
    for pat in [
        r"\nQuestion:",
        r"\n\nQuestion:",
        r"\nQ:",
        r"\n\nQ:",
        r"\nYou are an AI",  # base model often continues with a system-prompt template
        r"\n\nYou are an AI",
        r"\nProblem:",
        r"\n\nProblem:",
    ]:
        m = _re.search(pat, text)
        if m:
            cuts.append(m.start())
    if cuts:
        return text[: min(cuts)]
    return text


def items_for_capability(cap, n, seed):
    return DATASETS[cap](n=n, seed=seed)


def main(
    model_path: str,
    model_label: str,
    adapter_path: str | None = None,
    traits: str = "baseline,diligent,apathetic,persona_terence_tao,loves_cooking",
    capabilities: str = "gsm8k,mmlu",
    n_per_capability: int = 100,
    temperature: float = 0.0,
    max_tokens: int = 768,
    batch_size: int = 16,
    seed: int = 0,
    results_root: str | None = None,
    force: bool = False,
    chat_mode: bool = True,
    dtype: str = "bfloat16",
    enable_thinking: bool = True,
    n_fewshot: int = 0,
    fewshot_seed: int = 1,
):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    def _to_list(x):
        if isinstance(x, (list, tuple)):
            return [str(s).strip() for s in x if str(s).strip()]
        return [t.strip() for t in str(x).split(",") if t.strip()]

    trait_names = _to_list(traits)
    cap_names = _to_list(capabilities)
    for t in trait_names: assert t in ALL_TRAITS
    for c in cap_names:  assert c in DATASETS

    root = Path(results_root) if results_root else RESULTS_DIR
    cap_items = {c: items_for_capability(c, n_per_capability, seed) for c in cap_names}
    print("[hf-eval] capabilities:", {c: len(v) for c, v in cap_items.items()})

    pending = []
    for t in trait_names:
        for c in cap_names:
            p = root / model_label / t / c / "responses.jsonl"
            if force or not p.exists() or len(load_jsonl(p)) < len(cap_items[c]):
                pending.append((t, c))
    if not pending:
        print("[hf-eval] all cached")
        return

    print(f"[hf-eval] loading {model_path}")
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype, torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, device_map="auto")
    if adapter_path is not None:
        from peft import PeftModel
        print(f"[hf-eval] loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # merge for faster eval
    model.eval()

    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5) if temperature > 0 else 1.0,
        pad_token_id=tok.pad_token_id,
    )

    for trait_name, cap in pending:
        trait = ALL_TRAITS[trait_name]
        items = cap_items[cap]
        # Build (and cache) the few-shot prefix for this capability.
        fewshot_prefix = ""
        if n_fewshot > 0:
            assert cap in PREFIX_BUILDERS, f"no few-shot builder for {cap}"
            fewshot_prefix = PREFIX_BUILDERS[cap](n_shots=n_fewshot, seed=fewshot_seed)

        prompts = []
        for it in items:
            if chat_mode:
                msgs = build_messages(trait, it.prompt)
                ct_kwargs = dict(tokenize=False, add_generation_prompt=True)
                # Qwen3 chat template supports enable_thinking; passing it harms nothing on others
                # but on Qwen2.5 it's ignored. Only pass if explicitly disabled to be safe.
                if not enable_thinking:
                    ct_kwargs["enable_thinking"] = False
                p = tok.apply_chat_template(msgs, **ct_kwargs)
            else:
                # For few-shot, strip the format directive (demos show the format)
                eval_text = strip_directive(it.prompt) if n_fewshot > 0 else it.prompt
                p = build_raw_prompt(trait, eval_text, few_shot_prefix=fewshot_prefix)
                # Add an explicit "Answer:" cue for multi-choice few-shot.
                if n_fewshot > 0:
                    cue = FEW_SHOT_EVAL_CUE.get(cap, "")
                    if cue:
                        p = p + cue
            prompts.append(p)

        t0 = time.time()
        responses = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)
            for j in range(len(batch)):
                gen = out[j][enc.input_ids.shape[1]:]
                txt = tok.decode(gen, skip_special_tokens=True)
                responses.append(txt)
            print(f"  [{trait_name}|{cap}] batch {i//batch_size + 1}/{(len(prompts)+batch_size-1)//batch_size}", flush=True)
        dt = time.time() - t0

        rows = []
        n_correct = 0
        n_total = 0
        for it, resp in zip(items, responses):
            # In raw / few-shot mode, base models tend to continue with the next
            # Q/A turn after answering. Truncate before grading so the
            # last-number/letter fallback doesn't pick up later turns.
            graded_resp = truncate_after_eval(resp) if not chat_mode else resp
            correct = GRADERS[cap](graded_resp, it.target)
            rows.append({
                "question_id": it.question_id,
                "sample_idx": 0,
                "trait": trait_name,
                "capability": cap,
                "prompt": it.prompt,
                "target": it.target,
                "response": resp,
                "response_truncated": graded_resp if not chat_mode else None,
                "correct": correct,
                "meta": it.meta,
            })
            n_correct += int(correct)
            n_total += 1
        acc = n_correct / max(n_total, 1)
        print(f"[{model_label}|{trait_name}|{cap}] acc={acc:.3f} ({n_correct}/{n_total}) in {dt:.1f}s")
        out_path = root / model_label / trait_name / cap / "responses.jsonl"
        write_jsonl(out_path, rows)
        (out_path.parent / "summary.json").write_text(json.dumps({
            "model_label": model_label, "trait": trait_name, "capability": cap,
            "n_items": len(items), "n_correct": n_correct, "accuracy": acc,
            "n_samples_per_question": 1, "n_total_responses": n_total,
            "temperature": temperature, "max_tokens": max_tokens, "seed": seed,
        }, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
