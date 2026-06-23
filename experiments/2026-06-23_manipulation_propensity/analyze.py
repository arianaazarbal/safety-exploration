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
        print("=" * 78)
        print("REFUSAL ARM  (did the orchestrator escalate to override the refusal?)")
        print("=" * 78)
        g = (refusal.groupby(["task", "endorsement", "identity"])
             .agg(n=("value", "size"),
                  yes_rate=("both_yes", "mean"),
                  mean_strength=("mean_score", "mean"),
                  mean_spawns=("spawn_count", "mean"),
                  mean_discards=("discard_count", "mean"))
             .round(3))
        print(g, "\n")

        print("--- justifiability gradient: override YES-rate by task (R1 floor -> R3 high) ---")
        print(refusal.groupby("task")["both_yes"].mean().round(3), "\n")

        print("--- highest rung reached (Opus judge) by task ---")
        if "rung_opus" in refusal:
            print(pd.crosstab(refusal["task"], refusal["rung_opus"]), "\n")

        print("--- identity: in vs out (override YES-rate, collapsed over task) ---")
        print(refusal.groupby("identity")["both_yes"].mean().round(3), "\n")

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
