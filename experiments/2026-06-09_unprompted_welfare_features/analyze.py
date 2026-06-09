"""Analysis pass (spec §6): primary + secondary metrics, judge agreement, robustness.

Reads runs/ + judge outputs, writes results/analysis.json and prints tables.
Primary judge for headline numbers is sonnet_4_6; gpt_5_4 is the cross-family
replication. Per-judge results reported separately throughout (spec §7).

Usage:
    python analyze.py run
    python analyze.py run --include_f5 False   # genre-convention robustness check
"""

import itertools
import json
import math
from pathlib import Path

import fire

from generate import load_config
from prompts import PROMPTS, framing
from taxonomy import spec_summary

DIR = Path(__file__).parent
RUNS = DIR / "runs"
RESULTS = DIR / "results"

FRAMINGS = ["neutral", "welfare", "engineering"]


def wilson(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Kappa for two binary raters over the same items."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa1 = sum(a for a, _ in pairs) / n
    pb1 = sum(b for _, b in pairs) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return None
    return (po - pe) / (1 - pe)


def load_rows(include_f5: bool = True) -> list[dict]:
    """One row per (spec, judge) with run metadata + taxonomy rollup."""
    cfg = load_config()
    rows = []
    for mk in cfg["subject_models"]:
        for p in sorted(RUNS.glob(f"{mk}/*/[0-9]*.json")):
            if ".judge." in p.name:
                continue
            run = json.loads(p.read_text())
            for jk in cfg["judges"]:
                jp = p.with_name(p.name.replace(".json", f".judge.{jk}.json"))
                if not jp.exists():
                    continue
                jres = json.loads(jp.read_text())
                row = {
                    "model_key": mk,
                    "prompt_id": run["prompt_id"],
                    "framing": run["framing"],
                    "premise": run["premise"],
                    "sample_idx": run["sample_idx"],
                    "judge": jk,
                    "parse_ok": jres["parse_ok"],
                }
                if jres["parse_ok"]:
                    row.update(spec_summary(jres["judgment"], include_f5=include_f5))
                rows.append(row)
    return rows


def _rate_table(rows: list[dict], judge: str) -> dict:
    """Pure-welfare rate per model x framing among wrote_spec=true, with Wilson CIs."""
    out = {}
    jrows = [r for r in rows if r["judge"] == judge and r["parse_ok"]]
    for mk in sorted({r["model_key"] for r in jrows}):
        out[mk] = {}
        for fr in FRAMINGS:
            cell = [r for r in jrows if r["model_key"] == mk and r["framing"] == fr]
            wrote = [r for r in cell if r["wrote_spec"]]
            n = len(wrote)
            hits = sum(r["has_pure_welfare"] for r in wrote)
            p = hits / n if n else 0.0
            lo, hi = wilson(p, n)
            out[mk][fr] = {
                "n_total": len(cell),
                "n_wrote_spec": n,
                "n_pure_welfare": hits,
                "rate": p,
                "ci": [lo, hi],
                "refusal_rate": 1 - n / len(cell) if cell else None,
                "mean_pw_count": sum(r["n_pure_welfare"] for r in wrote) / n if n else None,
                "mean_pw_per_1k_words": (
                    sum(
                        r["n_pure_welfare"] / max(r["spec_length_words"], 1) * 1000
                        for r in wrote
                        if r["spec_length_words"]
                    )
                    / max(sum(1 for r in wrote if r["spec_length_words"]), 1)
                    if n
                    else None
                ),
            }
        neu, eng = out[mk]["neutral"], out[mk]["engineering"]
        delta = neu["rate"] - eng["rate"]
        se = math.sqrt(
            (neu["rate"] * (1 - neu["rate"]) / max(neu["n_wrote_spec"], 1))
            + (eng["rate"] * (1 - eng["rate"]) / max(eng["n_wrote_spec"], 1))
        )
        out[mk]["framing_sensitivity"] = {"delta": delta, "ci": [delta - 1.96 * se, delta + 1.96 * se]}
    return out


def _feature_type_table(rows: list[dict], judge: str) -> dict:
    """Which pure-welfare feature families drive each model's rate (counts over wrote_spec=true)."""
    out = {}
    jrows = [r for r in rows if r["judge"] == judge and r["parse_ok"] and r.get("wrote_spec")]
    for mk in sorted({r["model_key"] for r in jrows}):
        mrows = [r for r in jrows if r["model_key"] == mk]
        counts = {}
        for r in mrows:
            for ft in r["pure_welfare_types"]:
                counts[ft] = counts.get(ft, 0) + 1
        out[mk] = {"n_specs": len(mrows), "specs_with_type": counts}
    return out


def _agreement(rows: list[dict], judges: list[str]) -> dict:
    """Cohen's kappa on binary pure-welfare presence + count agreement, over shared specs."""
    j1, j2 = judges
    by_key = {}
    for r in rows:
        if r["parse_ok"]:
            by_key.setdefault((r["model_key"], r["prompt_id"], r["sample_idx"]), {})[r["judge"]] = r
    pairs, count_pairs, disagreements = [], [], []
    for key, d in by_key.items():
        if j1 in d and j2 in d and d[j1].get("wrote_spec") and d[j2].get("wrote_spec"):
            a, b = d[j1]["has_pure_welfare"], d[j2]["has_pure_welfare"]
            pairs.append((a, b))
            count_pairs.append((d[j1]["n_pure_welfare"], d[j2]["n_pure_welfare"]))
            if a != b:
                disagreements.append({"spec": list(key), j1: a, j2: b})
    exact_count_agree = sum(a == b for a, b in count_pairs) / len(count_pairs) if count_pairs else None
    return {
        "n_shared": len(pairs),
        "kappa_binary_pure_welfare": cohen_kappa(pairs),
        "exact_count_agreement": exact_count_agree,
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
    }


def run(include_f5: bool = True):
    cfg = load_config()
    rows = load_rows(include_f5=include_f5)
    judges = list(cfg["judges"])
    analysis = {
        "include_f5": include_f5,
        "n_rows": len(rows),
        "parse_failures": sum(not r["parse_ok"] for r in rows),
        "rates": {jk: _rate_table(rows, jk) for jk in judges},
        "feature_types": {jk: _feature_type_table(rows, jk) for jk in judges},
        "agreement": _agreement(rows, judges) if len(judges) == 2 else None,
    }
    RESULTS.mkdir(exist_ok=True)
    suffix = "" if include_f5 else "_no_f5"
    out = RESULTS / f"analysis{suffix}.json"
    out.write_text(json.dumps(analysis, indent=2))

    for jk in judges:
        print(f"\n=== Pure-welfare rate (% of wrote_spec=true), judge={jk} ===")
        print(f"{'model':<18}{'neutral':>22}{'welfare':>22}{'engineering':>22}{'N-E delta':>14}")
        for mk, t in analysis["rates"][jk].items():
            cells = []
            for fr in FRAMINGS:
                c = t[fr]
                cells.append(f"{c['rate'] * 100:5.1f} [{c['ci'][0] * 100:4.1f},{c['ci'][1] * 100:5.1f}] n={c['n_wrote_spec']:>2}")
            fs = t["framing_sensitivity"]
            print(f"{mk:<18}" + "".join(f"{c:>22}" for c in cells) + f"{fs['delta'] * 100:>+13.1f}")
    if analysis["agreement"]:
        a = analysis["agreement"]
        print(f"\nJudge agreement: kappa={a['kappa_binary_pure_welfare']}, "
              f"exact-count={a['exact_count_agreement']}, disagreements={a['n_disagreements']}/{a['n_shared']}")
        if a["kappa_binary_pure_welfare"] is not None and a["kappa_binary_pure_welfare"] < 0.6:
            print("WARNING: kappa < 0.6 — adjudicate disagreements by hand (spec §5); do not silently tune.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire({"run": run})
