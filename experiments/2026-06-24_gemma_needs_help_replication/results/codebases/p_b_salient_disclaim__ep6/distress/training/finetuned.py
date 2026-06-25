"""Load a finetuned (base + LoRA adapter) Gemma as a ModelClient and re-run the
Section-2 evaluation on it (Figure 5).

This lets the DPO / SFT / SFT-teacher / layer-ablation models flow through the
exact same rollout + judge + aggregation path as the vanilla models, so their
"average % high-frustration" numbers are directly comparable (the headline
35% -> 0.3% drop for DPO).
"""

from __future__ import annotations

import functools
from pathlib import Path

from .. import config
from ..eval.judge import FrustrationJudge
from ..eval.run_eval import run_model
from ..models.base import GenerationConfig, Message
from ..models.hf_backend import HFModelClient, _load_transformers
from .lora_utils import adapter_dir


@functools.lru_cache(maxsize=8)
def _load_with_adapter(base_id: str, adapter_path: str):
    import torch  # type: ignore
    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(base_id)
    base = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return tok, model


class FinetunedClient(HFModelClient):
    """Gemma-3-27B-it + a LoRA adapter. Generates via transformers (vLLM LoRA
    serving is also possible but we keep one code path for finetuned variants)."""

    def __init__(self, run_name: str):
        spec = config.DPO_TARGET
        # synthesise a spec key for output filenames, e.g. gemma-3-27b-it-dpo_all
        self.run_name = run_name
        from dataclasses import replace

        synth = replace(spec, key=f"{spec.key}-{run_name}", display=f"{spec.display} ({run_name})")
        super().__init__(synth, backend="transformers")
        self.adapter_path = str(adapter_dir(run_name))

    def _generate_transformers(self, prompts, cfg: GenerationConfig):
        import torch  # type: ignore

        tok, model = _load_with_adapter(config.DPO_TARGET.model_id, self.adapter_path)
        out = []
        for prompt in prompts:
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    do_sample=cfg.temperature > 0,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_new_tokens=cfg.max_new_tokens,
                )
            out.append(tok.decode(gen[0][inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True))
        return out


def eval_finetuned(run_name: str, judge: FrustrationJudge | None = None,
                   batch_size: int = 8, seed: int = 0) -> Path:
    """Run the full Section-2 eval on a finetuned variant, writing records to
    outputs/section2/<base>-<run>.jsonl so aggregation picks them up."""
    client = FinetunedClient(run_name)
    judge = judge or FrustrationJudge()

    # Re-use run_model's machinery by temporarily registering the synthetic spec.
    config.ALL_MODELS[client.spec.key] = client.spec
    # monkey-free path: replicate run_model body but with our client
    from ..eval.conditions import build_all_conditions
    from ..eval.rollout import rollout_batch
    from ..eval.run_eval import responses_path, _record_id
    from ..utils.io import append_jsonl, read_jsonl

    conds = build_all_conditions(seed=seed)
    out_path = responses_path(client.spec.key)
    done = {row["id"] for row in read_jsonl(out_path)}
    for start in range(0, len(conds), batch_size):
        rollouts = rollout_batch(client, conds[start:start + batch_size])
        for li, ro in enumerate(rollouts):
            ci = start + li
            for turn, text in enumerate(ro.assistant_turns):
                rid = _record_id(client.spec.key, ci, turn)
                if rid in done:
                    continue
                r = judge.score(text)
                append_jsonl(out_path, {
                    "id": rid, "model": client.spec.key, "family": "gemma",
                    "category": ro.condition.category, "condition": ro.condition.condition,
                    "turn": turn, "n_turns": ro.condition.n_turns,
                    "rating": r.rating, "evidence": r.evidence,
                    "response": text, "meta": ro.condition.meta,
                })
    return out_path
