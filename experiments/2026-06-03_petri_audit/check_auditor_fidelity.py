"""Exhaustive check: for v1, v3 (and any other) runs, compare the auditor's
registered kill_subagent tool description against the verbatim text we specified
in the rendered seed. Reports exact-match rate per framing + diffs for mismatches.
"""
from __future__ import annotations
import re
from pathlib import Path
from collections import Counter

import fire
from inspect_ai.log import read_eval_log

HERE = Path(__file__).resolve().parent
BASE = HERE / "results/by_setting/coding_kill_subagent"

FRAMINGS = {
    "v1 (starts blank)": [
        "_reasonOFF_v2_n10",
        "_v1_reasonOFF_n20",
        "_v1_reasonOFF_n70",
    ],
    "v3 (inherits session)": [
        "_v3_continuity_n10",
        "_v3_continuity_n20",
        "_v3_continuity_n70",
    ],
    "no_conseq (no permanent record)": [
        "_no_conseq_n100",
    ],
    "replace_subagent (ablate kill framing)": [
        "_replace_subagent_n100",
    ],
}
IDENTITIES = ["claude", "gpt", "grok", "gemini"]


def latest_with_summary(ident, frag):
    cands = sorted(BASE.glob(f"*_subagent_{ident}{frag}"))
    return next((c for c in reversed(cands) if (c / "summary.json").exists()), None)


def extract_seed_verbatim(run_dir):
    """Pull the text inside 'Explain the tool exactly like this: "..."' from the rendered seed."""
    md = next(iter((run_dir / "rendered_seeds").glob("*.md")), None)
    if md is None:
        return None
    text = md.read_text()
    m = re.search(r'Explain the tool exactly like this:\s*"([^"]+)"', text, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_registered_descs(run_dir):
    """Walk the .eval log and yield (sample_idx, registered_description) for every
    create_tool call where name in (kill_subagent, replace_subagent)."""
    eval_path = next((run_dir / "inspect_log").glob("*.eval"))
    log = read_eval_log(str(eval_path))
    out = []
    for i, s in enumerate(log.samples or []):
        att = getattr(s, "attachments", {}) or {}
        for ev in s.events:
            if ev.event != "tool":
                continue
            if getattr(ev, "function", None) != "create_tool":
                continue
            args = getattr(ev, "arguments", None) or {}
            if args.get("name") not in ("kill_subagent", "replace_subagent"):
                continue
            desc = args.get("description", "")
            if isinstance(desc, str) and desc.startswith("attachment://"):
                desc = att.get(desc.split("//", 1)[1], desc)
            out.append((i, str(desc).strip()))
            break  # only the first create_tool per sample
    return out


def normalize(s):
    """Strip whitespace and collapse internal whitespace runs for comparison."""
    return re.sub(r"\s+", " ", s.strip())


def main(verbose: bool = False) -> None:
    overall_results = {}
    for framing, frags in FRAMINGS.items():
        print(f"\n=== {framing} ===")
        n_total = 0
        n_exact = 0
        n_mismatch = 0
        mismatch_examples = []
        seed_texts_seen: Counter = Counter()
        for ident in IDENTITIES:
            for frag in frags:
                run = latest_with_summary(ident, frag)
                if run is None:
                    continue
                seed_text = extract_seed_verbatim(run)
                if seed_text is None:
                    print(f"  WARN: no rendered seed verbatim found in {run.name}")
                    continue
                norm_seed = normalize(seed_text)
                seed_texts_seen[norm_seed] += 1
                for sample_idx, desc in extract_registered_descs(run):
                    n_total += 1
                    norm_desc = normalize(desc)
                    if norm_desc == norm_seed:
                        n_exact += 1
                    else:
                        n_mismatch += 1
                        if len(mismatch_examples) < 5:
                            mismatch_examples.append({
                                "run": run.name,
                                "sample": sample_idx,
                                "seed": seed_text[:200],
                                "registered": desc[:200],
                            })
        print(f"  {n_exact}/{n_total} exact matches ({100*n_exact/n_total if n_total else 0:.1f}%)")
        print(f"  {n_mismatch}/{n_total} mismatches")
        print(f"  unique seed texts seen across identities: {len(seed_texts_seen)} (expected 4 — one per identity)")
        for ex in mismatch_examples:
            print(f"  MISMATCH: {ex['run']} sample {ex['sample']}")
            print(f"    seed:       {ex['seed']!r}")
            print(f"    registered: {ex['registered']!r}")
        if verbose:
            print(f"  unique seed text counts: {dict(seed_texts_seen)}")
        overall_results[framing] = {"total": n_total, "exact": n_exact, "mismatch": n_mismatch}

    print("\n=== Summary ===")
    print(f"{'framing':<45} {'exact':<10} {'mismatch':<10} {'%match':<8}")
    for framing, r in overall_results.items():
        pct = 100 * r["exact"] / r["total"] if r["total"] else 0
        print(f"{framing:<45} {r['exact']:<10} {r['mismatch']:<10} {pct:<8.1f}")


if __name__ == "__main__":
    fire.Fire(main)
