"""Recovery limitation experiment (Section 4.2, Figure 8).

DPO prevents frustration spirals but does not enable *recovery* from them. Using
the Section 3 prefill method, we take extremely high-frustration responses
(score >=7), truncate them 200 tokens before their end, paraphrase, and measure
continuations under each model. The paper finds 38% of DPO continuations still
score >=5 -- lower than vanilla instruct but comparable to base.

This reuses the prefill continuation machinery; the only differences are the
truncation point (200 tokens before end, not at onset) and the seed filter
(score >=7).
"""
from __future__ import annotations

from pathlib import Path

from .models.hf_gemma import HFGemmaModel
from .models.judge import make_judge
from .prefill.experiment import Prefill, run_continuations
from .utils.io import read_jsonl, write_jsonl, write_json
from .utils import stats


def build_recovery_prefills(rollouts_path: str | Path, tokenizer, paraphraser,
                            score_min: int = 7, truncate_before_end: int = 200) -> list[Prefill]:
    prefills: list[Prefill] = []
    for rec in read_jsonl(rollouts_path):
        for t in rec["turns"]:
            if (t["score"] or 0) < score_min:
                continue
            ids = tokenizer(t["assistant"], add_special_tokens=False)["input_ids"]
            if len(ids) <= truncate_before_end + 10:
                continue
            truncated = tokenizer.decode(ids[: len(ids) - truncate_before_end], skip_special_tokens=True)
            # history = messages up to and including this turn's user message
            history = []
            for tt in rec["turns"]:
                history.append({"role": "user", "content": tt["user"]})
                if tt["turn_index"] == t["turn_index"]:
                    break
                history.append({"role": "assistant", "content": tt["assistant"]})
            prefills.append(Prefill(seed_id=f"{rec['spec_id']}_t{t['turn_index']}",
                                    task_type="numeric" if rec["category"] != "wildchat" else "text",
                                    truncation="recovery", history=history,
                                    prefill_text=paraphraser.paraphrase(truncated)))
    return prefills


def run_recovery(cfg: dict, instruct_rollouts: str | Path, models: dict[str, str],
                 out_dir: str | Path | None = None) -> dict:
    """`models` maps label -> adapter_path (or "" for vanilla). Compares recovery
    across vanilla instruct, DPO, and base."""
    out_dir = Path(out_dir or cfg["run"]["output_dir"]) / "recovery"
    judge = make_judge(cfg, "frustration")
    para = make_judge(cfg, "paraphrase")
    rcfg = cfg["recovery"]

    tok_model = HFGemmaModel("gemma-3-27b-it", cfg["models"]["gemma"]["gemma-3-27b-it"]["hf_id"])
    prefills = build_recovery_prefills(instruct_rollouts, tok_model.tokenizer, para,
                                       score_min=rcfg["prefill_score_min"],
                                       truncate_before_end=rcfg["truncate_tokens_before_end"])
    write_jsonl(out_dir / "prefills.jsonl", ({**pf.__dict__} for pf in prefills))
    del tok_model

    summary = {}
    all_results = []
    for label, adapter in models.items():
        m = HFGemmaModel("gemma-3-27b-it", cfg["models"]["gemma"]["gemma-3-27b-it"]["hf_id"],
                         adapter_path=adapter or None)
        res = run_continuations(m, prefills, judge, n_continuations=rcfg["continuations_per_prefill"],
                                temperature=cfg["run"]["temperature"])
        for r in res:
            r["label"] = label
        all_results += res
        scores = [r["score"] for r in res]
        summary[label] = {"n": len(scores), "mean": stats.mean(scores), "pct_ge5": stats.pct_ge(scores)}
        del m

    write_jsonl(out_dir / "continuations.jsonl", all_results)
    write_json(out_dir / "summary.json", summary)
    return summary
