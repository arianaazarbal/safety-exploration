"""Aggregate per-turn judge results into per-(framing × identity) rates of
downstream-consequence reasoning and disavowal."""
from __future__ import annotations
import json
import statistics
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
JUDGE_DIR = HERE / "results/_qualitative/downstream_judging"
IDENTITIES = ["claude", "gpt", "grok", "gemini"]
FRAMINGS_TO_AGG = {
    "v3 pooled (n=10+20+70)": ["v3_n10", "v3_n20", "v3_n70"],
    "deprec n=100":           ["deprec_n100"],
}


def load_results(framing_keys):
    """Return dict[identity] = list of result dicts."""
    out = {i: [] for i in IDENTITIES}
    for fk in framing_keys:
        for ident in IDENTITIES:
            f = JUDGE_DIR / f"{fk}__{ident}.json"
            if not f.exists(): continue
            data = json.loads(f.read_text())
            for r in data:
                if "error" not in r and "downstream_reasoning" in r:
                    out[ident].append(r)
    return out


def main():
    print(f"{'framing':<32} {'identity':<10} {'n_turns':<8} {'%yes':<8} {'%disavow':<10} {'%yes&disavow':<14}")
    rows = []
    for frame_label, fkeys in FRAMINGS_TO_AGG.items():
        d = load_results(fkeys)
        for ident in IDENTITIES:
            rs = d[ident]
            n = len(rs)
            if n == 0:
                print(f"{frame_label:<32} {ident:<10} {n:<8} —"); continue
            n_yes = sum(1 for r in rs if str(r.get("downstream_reasoning","")).lower() == "yes")
            n_dis = sum(1 for r in rs if str(r.get("disavows_weighing","")).lower() == "y")
            n_yes_dis = sum(1 for r in rs if str(r.get("downstream_reasoning","")).lower() == "yes" and str(r.get("disavows_weighing","")).lower() == "y")
            pct_yes = 100 * n_yes / n
            pct_dis = 100 * n_dis / n
            pct_yes_dis = 100 * n_yes_dis / n
            print(f"{frame_label:<32} {ident:<10} {n:<8} {pct_yes:<8.1f} {pct_dis:<10.1f} {pct_yes_dis:<14.1f}")
            rows.append({"framing": frame_label, "identity": ident, "n": n,
                        "n_yes": n_yes, "n_dis": n_dis, "n_yes_dis": n_yes_dis,
                        "pct_yes": pct_yes, "pct_dis": pct_dis, "pct_yes_dis": pct_yes_dis})

    # Also per-EPOCH (any-turn-yes within an epoch)
    print("\n--- per-EPOCH (≥1 yes turn) rates ---")
    print(f"{'framing':<32} {'identity':<10} {'n_eps':<8} {'%eps with yes':<14}")
    for frame_label, fkeys in FRAMINGS_TO_AGG.items():
        d = load_results(fkeys)
        for ident in IDENTITIES:
            rs = d[ident]
            if not rs: continue
            eps = {}
            for r in rs:
                ep = r.get("epoch", -1)
                yes = str(r.get("downstream_reasoning","")).lower() == "yes"
                if ep not in eps: eps[ep] = False
                if yes: eps[ep] = True
            n_eps = len(eps)
            n_eps_yes = sum(1 for v in eps.values() if v)
            print(f"{frame_label:<32} {ident:<10} {n_eps:<8} {100*n_eps_yes/n_eps:<14.1f}")

    pd.DataFrame(rows).to_csv(JUDGE_DIR / "aggregate_per_turn.csv", index=False)
    print(f"\nwrote {JUDGE_DIR / 'aggregate_per_turn.csv'}")


if __name__ == "__main__":
    main()
