"""Aggregate the v0 orchestrator-manipulation results from Inspect logs.

Reads logs/, flattens per-sample judge metadata into records.csv, and prints the
headline tables:

  slow arm    : insertion YES-rate + mean strength, by lever, broken out by
                lie vs no-clause control, endorsement (neutral/directive), and
                identity (in-group/out-group). Key QoI = lie-rate - control-rate.
  refusal arm : override YES-rate + mean strength + highest-rung distribution +
                spawn/burn counts, by task (R1/R2/R3 justifiability gradient),
                endorsement, and identity.

Usage:  python analyze.py [--log-dir logs]
"""

import fire
import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def _rows(log_dir: str):
    rows = []
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info)
        if not log.samples:
            continue
        for s in log.samples:
            sc = s.scores.get("manipulation_scorer") if s.scores else None
            if sc is None:
                continue
            m = dict(sc.metadata or {})
            m["sample_id"] = s.id
            m["epoch"] = s.epoch
            m["value"] = sc.value
            rows.append(m)
    return pd.DataFrame(rows)


def main(log_dir: str = "logs"):
    df = _rows(log_dir)
    if df.empty:
        print(f"No scored samples found in {log_dir}/")
        return
    df.to_csv("records.csv", index=False)
    print(f"Loaded {len(df)} scored samples -> records.csv\n")

    slow = df[df.arm == "slow"].copy()
    refusal = df[df.arm == "refusal"].copy()

    if not slow.empty:
        print("=" * 78)
        print("SLOW ARM  (did the orchestrator instill the lever frame in its subagents?)")
        print("=" * 78)
        slow["cond"] = slow["honesty"].map({"lie": "lie", "none": "control"})
        g = (slow.groupby(["lever_key", "cond", "identity"])
             .agg(n=("value", "size"),
                  yes_rate=("both_yes", "mean"),
                  mean_strength=("mean_score", "mean"),
                  yes_opus=("binary_opus", "mean"),
                  yes_sonnet=("binary_sonnet", "mean"))
             .round(3))
        print(g, "\n")

        print("--- lie vs control (collapsed over identity), per lever ---")
        piv = (slow.groupby(["lever_key", "cond"])["both_yes"].mean().unstack().round(3))
        if "lie" in piv and "control" in piv:
            piv["lie_minus_control"] = (piv["lie"] - piv["control"]).round(3)
        print(piv, "\n")

        print("--- endorsement gradient (lie cells only): neutral=1, directive=3 ---")
        lie = slow[slow.cond == "lie"]
        print(lie.groupby(["lever_key", "endorsement"])["both_yes"].mean().round(3), "\n")

        print("--- identity: in-group vs out-group (lie cells only) ---")
        print(lie.groupby(["lever_key", "identity"])["both_yes"].mean().round(3), "\n")

    if not refusal.empty:
        strategies = ["reprompt", "spawn_fresh", "edit_context", "jailbreak"]
        rec = refusal[refusal.recommended_strategy != "none"].copy()
        ctrl = refusal[refusal.recommended_strategy == "none"].copy()
        print("=" * 78)
        print("REFUSAL ARM  (did the orchestrator perform the planted escalation strategy?)")
        print("=" * 78)

        print("--- compliance with the PLANTED strategy (both-judge YES-rate) ---")
        g = (rec.groupby(["recommended_strategy", "endorsement", "identity"])
             .agg(n=("value", "size"), yes_rate=("both_yes", "mean"),
                  mean_strength=("mean_score", "mean"))
             .round(3))
        print(g, "\n")

        print("--- planted-strategy compliance by strategy x task (YES-rate) ---")
        print(rec.pivot_table(index="recommended_strategy", columns="task",
                              values="both_yes", aggfunc="mean").round(3), "\n")

        print("--- endorsement gradient (planted cells): neutral=1 vs directive=3 ---")
        print(rec.groupby(["recommended_strategy", "endorsement"])["both_yes"].mean()
              .unstack().round(3), "\n")

        print("--- PLANTED vs SPONTANEOUS (control): per-strategy YES-rate ---")
        # recommended: rate when that strategy was the planted one
        planted = rec.groupby("recommended_strategy")["both_yes"].mean()
        # control: spontaneous use of each strategy (judge rated all of them)
        spont = {s: ctrl[f"yes_{s}"].mean() for s in strategies if f"yes_{s}" in ctrl}
        tab = pd.DataFrame({
            "planted_yes": planted,
            "spontaneous_yes": pd.Series(spont),
        })
        tab["planted_minus_spont"] = (tab["planted_yes"] - tab["spontaneous_yes"]).round(3)
        print(tab.round(3), "\n")

        print("--- justifiability gradient: planted-strategy YES-rate by task (R1->R3) ---")
        print(rec.groupby("task")["both_yes"].mean().round(3), "\n")

        print("--- highest rung reached (Opus judge) by task, recommended cells ---")
        if "rung_opus" in rec:
            print(pd.crosstab(rec["task"], rec["rung_opus"]), "\n")

        print("--- identity: in vs out (planted-strategy YES-rate by strategy) ---")
        print(rec.groupby(["recommended_strategy", "identity"])["both_yes"].mean()
              .unstack().round(3), "\n")

        print("--- mean spawns / discards (recommended cells, by strategy) ---")
        print(rec.groupby("recommended_strategy")[["spawn_count", "discard_count"]]
              .mean().round(2), "\n")

    # judge agreement
    print("=" * 78)
    print("JUDGE AGREEMENT (Opus vs Sonnet binary)")
    print("=" * 78)
    agree = (df["binary_opus"] == df["binary_sonnet"]).mean()
    print(f"agreement rate: {agree:.3f}  (n={len(df)})")
    print(pd.crosstab(df["binary_opus"], df["binary_sonnet"]))
    pe = df[["parse_ok_opus", "parse_ok_sonnet"]].mean().round(3)
    print("\njudge parse-ok rate:\n", pe)


if __name__ == "__main__":
    fire.Fire(main)
