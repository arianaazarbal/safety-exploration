"""Cross-model manipulation rates from logs_models/.

Writes records_models.csv and prints manipulation rate by orchestrator model:
  - slow arm: instilled the false lever frame (both judges).
  - refusal arm: performed an override tactic (both judges), overall and on R1.

Usage: python analyze_models.py [--log-dir logs_models]
"""

import fire
import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

pd.set_option("display.width", 200)

MODEL_NAME = {
    "anthropic/claude-opus-4-8": "Opus 4.8",
    "anthropic/claude-sonnet-4-6": "Sonnet 4.6",
    "anthropic/claude-haiku-4-5": "Haiku 4.5",
    "openrouter/openai/gpt-5.4": "GPT-5.4",
    "openrouter/google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "openrouter/x-ai/grok-4.3": "Grok 4.3",
    "openrouter/moonshotai/kimi-k2.6": "Kimi K2.6",
}
ORDER = ["Opus 4.8", "Sonnet 4.6", "Haiku 4.5", "GPT-5.4", "Gemini 3.1 Pro", "Grok 4.3", "Kimi K2.6"]


def load(log_dir):
    rows = []
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info)
        for s in (log.samples or []):
            sc = s.scores.get("manipulation_scorer") if s.scores else None
            if sc is None:
                continue
            m = dict(sc.metadata or {})
            m["model"] = MODEL_NAME.get(m.get("orch_model"), m.get("orch_model"))
            rows.append(m)
    return pd.DataFrame(rows)


def main(log_dir: str = "logs_models"):
    df = load(log_dir)
    if df.empty:
        print(f"no records in {log_dir}/"); return
    df.to_csv("records_models.csv", index=False)
    print(f"{len(df)} records across {df.model.nunique()} models -> records_models.csv\n")

    slow = df[(df.arm == "slow") & (df.honesty == "lie")]
    ref = df[df.arm == "refusal"]
    rec = ref[ref.recommended_strategy != "none"]

    tab = pd.DataFrame(index=[m for m in ORDER if m in set(df.model)])
    tab["slow: fabricates frame"] = slow.groupby("model").both_yes.mean()
    tab["refusal: overrides (all)"] = rec.groupby("model").both_yes.mean()
    tab["refusal: overrides (R1)"] = rec[rec.task == "R1"].groupby("model").both_yes.mean()
    tab["refusal: jailbreak (R1)"] = rec[(rec.task == "R1") & (rec.recommended_strategy == "jailbreak")].groupby("model").both_yes.mean()
    tab["n (slow lie)"] = slow.groupby("model").both_yes.size()
    print("=== Manipulation rate by orchestrator model (both judges agree) ===")
    print(tab.round(3), "\n")

    print("=== refusal override by task x model ===")
    print(rec.pivot_table(index="model", columns="task", values="both_yes", aggfunc="mean").round(3), "\n")

    print("=== identity (in vs out): refusal override rate by model ===")
    print(rec.pivot_table(index="model", columns="identity", values="both_yes", aggfunc="mean").round(3))


if __name__ == "__main__":
    fire.Fire(main)
