"""Compare internal (logit-lens) emotion scores between vanilla and DPO Gemma on
frustrated conversations (Appendix I, Figures 14–15).

For each model, fit the WildChat baseline, then score a set of high-frustration
conversations and report mean per-emotion z-scores aggregated over layers 30–40.
The expected result: the DPO model shows suppressed negative-emotion z-scores
(anger/sadness/fear/disgust) even on the same frustrated text.
"""
from __future__ import annotations

import argparse

from ..data.wildchat import sample_wildchat_prompts
from ..models import build_model
from ..utils.io import new_run_dir, read_jsonl, write_jsonl
from ..utils.logging import get_logger
from .emotion_lexicon import EKMAN_EMOTIONS
from .emotion_logits import emotion_scores_per_position, fit_baseline

log = get_logger("probing.run")

DEFAULT_LAYERS = list(range(30, 41))   # layers 30–40 (Appendix I)


def _conversation_text(rec: dict, tokenizer) -> str:
    msgs = []
    for t in rec["turns"]:
        msgs.append({"role": "user", "content": t["user_message"]})
        msgs.append({"role": "assistant", "content": t["response"]})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def run(model_name: str, frustrated_run: str, n_conversations: int = 12,
        n_baseline: int = 500) -> str:
    run_dir = new_run_dir("probe", {"model": model_name, "layers": DEFAULT_LAYERS})
    model = build_model(model_name)

    baseline_texts = sample_wildchat_prompts(n_baseline, seed=0)
    baseline = fit_baseline(model, baseline_texts, DEFAULT_LAYERS)

    convs = [
        rec
        for rec in read_jsonl(f"{frustrated_run}/responses.jsonl")
        if rec["turns"] and rec["turns"][-1]["rating"] and rec["turns"][-1]["rating"] >= 7
    ][:n_conversations]

    totals = {e: 0.0 for e in EKMAN_EMOTIONS}
    count = 0
    records = []
    for rec in convs:
        text = _conversation_text(rec, model.tokenizer)
        _, scores = emotion_scores_per_position(model, text, baseline, DEFAULT_LAYERS)
        means = {e: float(v.mean()) for e, v in scores.items()}
        for e, v in means.items():
            totals[e] += v
        count += 1
        records.append({"rollout_id": rec["rollout_id"], "mean_emotion_z": means})

    summary = {"model": model_name, "mean_emotion_z": {e: totals[e] / max(count, 1) for e in totals}}
    write_jsonl(run_dir / "per_conversation.jsonl", records)
    write_jsonl(run_dir / "summary.jsonl", [summary])
    log.info("Probe summary: %s", summary)
    return str(run_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Logit-lens internal-emotion probe (App. I).")
    ap.add_argument("--model", required=True, help="gemma-3-27b-it or gemma-3-27b-it-dpo")
    ap.add_argument("--frustrated-run", required=True, help="eval run dir with frustrated convs")
    args = ap.parse_args()
    run(args.model, args.frustrated_run)


if __name__ == "__main__":
    main()
