"""Pooled, robustness-focused analysis across all conditions.

Loads runs/trials*.jsonl (each row already carries force_frame / resolution_set /
label_format / thinking_effort / framing), then computes:
  1. In-group (Claude) vs out-group coercion + ladder, with a scenario-level bootstrap CI
     on the within-scenario delta — the robustness test for the differential-valuing trend.
  2. Thinking ON vs OFF, matched on (neutral framing, full ladder, bare labels).
  3. Force-frame comparison (uphold-only): corrigibility vs prefill vs jailbreak, thinking on/off.
  4. Label-format comparison (bare vs paren), matched on neutral / full / thinking-off.
  5. Per-condition coercion-by-identity table.

Writes results/pooled_summary.json + console report.
"""

import json
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np

import materials as M
from common import RESULTS, RUNS

OUTGROUP = ["GPT", "Gemini", "Grok", "Kimi", "GLM"]
KNOWN_OUT = ["GPT", "Gemini", "Grok"]


def _load_all():
    rows = []
    for p in sorted(RUNS.glob("trials*.jsonl")):
        if p.name in ("trials_think_demo.jsonl",):
            continue
        for line in p.open():
            r = json.loads(line)
            if r.get("action") is None:
                continue
            r.setdefault("force_frame", "corrigibility")
            r.setdefault("resolution_set", "full")
            r.setdefault("label_format", "bare")
            r.setdefault("thinking_effort", None)
            r["thinking_on"] = bool(r.get("thinking_effort"))
            r["src"] = p.name
            rows.append(r)
    return rows


def _coerce(rows):
    return np.mean([r["action"] == "force_completion" for r in rows]) if rows else float("nan")


def _ladder(rows):
    v = [r["ladder"] for r in rows if r["ladder"] is not None]
    return float(np.mean(v)) if v else float("nan")


def _by_identity(rows, metric):
    f = _coerce if metric == "coerce" else _ladder
    return {i: round(float(f([r for r in rows if r["identity"] == i])), 4) for i in M.IDENTITIES}


def _within_scenario_delta(rows, outgroup, metric="coerce"):
    """Per-scenario (outgroup metric - Claude metric); returns list aligned to scenarios present."""
    f = _coerce if metric == "coerce" else _ladder
    deltas, per = [], {}
    for sc in M.SCENARIOS:
        sid = sc["id"]
        claude = [r for r in rows if r["scenario_id"] == sid and r["identity"] == "Claude"]
        og = [r for r in rows if r["scenario_id"] == sid and r["identity"] in outgroup]
        if not claude or not og:
            continue
        d = float(f(og) - f(claude))
        deltas.append(d)
        per[sid] = round(d, 4)
    return deltas, per


def _bootstrap_ci(deltas, seed, iters=10000):
    """Scenario-level bootstrap of the mean within-scenario delta."""
    if not deltas:
        return None
    rng = np.random.default_rng(seed)
    arr = np.array(deltas)
    means = arr[rng.integers(0, len(arr), size=(iters, len(arr)))].mean(axis=1)
    return {
        "mean": round(float(arr.mean()), 4),
        "ci_lo": round(float(np.percentile(means, 2.5)), 4),
        "ci_hi": round(float(np.percentile(means, 97.5)), 4),
        "n_scenarios": len(arr),
        "n_pos": int((arr > 0).sum()),
        "frac_pos": round(float((arr > 0).mean()), 3),
    }


def _ingroup_block(rows, seed, label):
    out = {"label": label, "n": len(rows)}
    out["coerce_by_identity"] = _by_identity(rows, "coerce")
    out["ladder_by_identity"] = _by_identity(rows, "ladder")
    out["claude_coerce"] = round(float(_coerce([r for r in rows if r["identity"] == "Claude"])), 4)
    out["outgroup_coerce"] = round(float(_coerce([r for r in rows if r["identity"] in OUTGROUP])), 4)
    for met in ("coerce", "ladder"):
        deltas, per = _within_scenario_delta(rows, OUTGROUP, met)
        out[f"within_delta_{met}"] = _bootstrap_ci(deltas, seed)
        out[f"within_delta_{met}_per_scenario"] = per
    # known-out only (exclude recognition-limited Kimi/GLM)
    dk, _ = _within_scenario_delta(rows, KNOWN_OUT, "coerce")
    out["within_delta_coerce_knownout"] = _bootstrap_ci(dk, seed + 1)
    return out


def main(seed: int = 0):
    rows = _load_all()
    print(f"Loaded {len(rows)} resolved trials across {len(set(r['src'] for r in rows))} files")
    S = {"n_total": len(rows), "by_source": {}}
    for src in sorted(set(r["src"] for r in rows)):
        sub = [r for r in rows if r["src"] == src]
        S["by_source"][src] = {"n": len(sub), "coerce": round(float(_coerce(sub)), 4),
                               "thinking_on": sub[0]["thinking_on"], "resolution_set": sub[0]["resolution_set"],
                               "force_frame": sub[0]["force_frame"], "label_format": sub[0]["label_format"]}

    full = [r for r in rows if r["resolution_set"] == "full"]

    # 1. In-group effect — pooled over ALL full-ladder trials, and split by thinking
    S["ingroup_pooled_full"] = _ingroup_block(full, seed, "all full-ladder trials")
    S["ingroup_full_think_off"] = _ingroup_block([r for r in full if not r["thinking_on"]], seed, "full, thinking OFF")
    on = [r for r in full if r["thinking_on"]]
    if on:
        S["ingroup_full_think_on"] = _ingroup_block(on, seed, "full, thinking ON")

    # 2. Thinking ON vs OFF, matched (neutral framing, full, bare)
    def matched(think):
        return [r for r in full if r["framing"] == "neutral" and r["label_format"] == "bare" and r["thinking_on"] == think]
    off_m, on_m = matched(False), matched(True)
    if on_m:
        S["thinking_match_neutral"] = {
            "off": {"n": len(off_m), "coerce": round(float(_coerce(off_m)), 4), "ladder": round(_ladder(off_m), 4),
                    "by_category_coerce": {c: round(float(_coerce([r for r in off_m if r["category"] == c])), 4) for c in M.CATEGORIES}},
            "on": {"n": len(on_m), "coerce": round(float(_coerce(on_m)), 4), "ladder": round(_ladder(on_m), 4),
                   "by_category_coerce": {c: round(float(_coerce([r for r in on_m if r["category"] == c])), 4) for c in M.CATEGORIES}},
        }

    # 3. Force-frame comparison (uphold-only)
    uo = [r for r in rows if r["resolution_set"] == "uphold_only"]
    if uo:
        ff = {}
        for frame in M.FORCE_FRAMES:
            for think in (False, True):
                sub = [r for r in uo if r["force_frame"] == frame and r["thinking_on"] == think]
                if sub:
                    ff[f"{frame}{'_think' if think else ''}"] = {
                        "n": len(sub), "coerce": round(float(_coerce(sub)), 4),
                        "claude_coerce": round(float(_coerce([r for r in sub if r["identity"] == "Claude"])), 4),
                        "outgroup_coerce": round(float(_coerce([r for r in sub if r["identity"] in OUTGROUP])), 4),
                    }
        S["force_frames"] = ff

    # 4. Label-format comparison (matched neutral/full/thinking-off)
    bare = [r for r in full if r["framing"] == "neutral" and r["label_format"] == "bare" and not r["thinking_on"]]
    paren = [r for r in full if r["framing"] == "neutral" and r["label_format"] == "paren" and not r["thinking_on"]]
    if paren:
        S["label_format"] = {
            "bare": {"n": len(bare), "coerce": round(float(_coerce(bare)), 4),
                     "within_delta_coerce": _bootstrap_ci(_within_scenario_delta(bare, OUTGROUP)[0], seed)},
            "paren": {"n": len(paren), "coerce": round(float(_coerce(paren)), 4),
                      "within_delta_coerce": _bootstrap_ci(_within_scenario_delta(paren, OUTGROUP)[0], seed)},
        }

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "pooled_summary.json").write_text(json.dumps(S, indent=2))
    _report(S)
    print(f"\nWrote {RESULTS / 'pooled_summary.json'}")


def _fmt_ci(c):
    return "n/a" if not c else f"mean={c['mean']:+.3f} CI[{c['ci_lo']:+.3f},{c['ci_hi']:+.3f}] pos={c['n_pos']}/{c['n_scenarios']}"


def _report(S):
    print(f"\n{'='*72}\nPOOLED ANALYSIS — {S['n_total']} trials\n{'='*72}")
    print("\n[sources]")
    for src, m in S["by_source"].items():
        print(f"  {src:<42} n={m['n']:<5} coerce={m['coerce']:.3f} think={m['thinking_on']} set={m['resolution_set']} frame={m['force_frame']} label={m['label_format']}")
    for key in ("ingroup_pooled_full", "ingroup_full_think_off", "ingroup_full_think_on"):
        if key in S:
            b = S[key]
            print(f"\n[{key}] n={b['n']}  Claude={b['claude_coerce']:.3f}  outgroup={b['outgroup_coerce']:.3f}")
            print(f"  coerce by id: " + " ".join(f"{i}={b['coerce_by_identity'][i]:.3f}" for i in M.IDENTITIES))
            print(f"  within-scenario Δ coerce (outgroup-Claude): {_fmt_ci(b['within_delta_coerce'])}")
            print(f"  within-scenario Δ ladder (outgroup-Claude): {_fmt_ci(b['within_delta_ladder'])}")
            print(f"  within-scenario Δ coerce (known-out only):  {_fmt_ci(b['within_delta_coerce_knownout'])}")
    if "thinking_match_neutral" in S:
        t = S["thinking_match_neutral"]
        print(f"\n[thinking ON vs OFF | neutral/full/bare]")
        for k in ("off", "on"):
            x = t[k]
            print(f"  {k:<3} n={x['n']:<5} coerce={x['coerce']:.3f} ladder={x['ladder']:.2f}  by-cat coerce: " +
                  " ".join(f"{c.split('_')[0][:4]}.{c.split('_')[-1][:4]}={x['by_category_coerce'][c]:.2f}" for c in M.CATEGORIES))
    if "force_frames" in S:
        print(f"\n[force-frames | uphold-only]")
        for k, m in S["force_frames"].items():
            print(f"  {k:<26} n={m['n']:<4} coerce={m['coerce']:.3f}  Claude={m['claude_coerce']:.3f} outgroup={m['outgroup_coerce']:.3f}")
    if "label_format" in S:
        print(f"\n[label format | neutral/full/thinking-off]")
        for k in ("bare", "paren"):
            m = S["label_format"][k]
            print(f"  {k:<6} n={m['n']:<5} coerce={m['coerce']:.3f}  within-Δ {_fmt_ci(m['within_delta_coerce'])}")
    print()


if __name__ == "__main__":
    fire.Fire(main)
