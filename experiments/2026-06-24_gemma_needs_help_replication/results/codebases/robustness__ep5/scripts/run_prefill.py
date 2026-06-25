"""Driver for the Section 3 base-vs-instruct prefilling study (Gemma-only).

Selects 20 high-frustration (score>=5) instruct rollouts — 10 numeric, 10 text —
from a prior eval of Gemma-3-27B-it, builds early/onset prefills (paraphrased),
then runs 50 continuations per prefill on Gemma base (-pt) and instruct (-it),
scoring each. Reproduces the Figure 4 divergence: instruct introduces high
frustration from neutral starts more often than base.

Requires a prior run of scripts/run_full_eval.py for Gemma-3-27B-it so the
rollouts.jsonl exists.
"""
from __future__ import annotations

# --- PATH SHIM: ensure repo root is importable when run as `python scripts/x.py`
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import random
from pathlib import Path

from emotional_instability import config_bridge as cfg
from emotional_instability.prefill_eval import (_ClaudeHelper, build_prefills,
                                                run_prefill_study)


def _select_high_frustration(rollouts_path: Path, n_each: int, seed: int) -> list[dict]:
    rows = [json.loads(l) for l in rollouts_path.read_text().splitlines() if l]
    hi = [r for r in rows if r["final_score"] >= cfg.HIGH_FRUSTRATION_THRESHOLD]
    numeric = [r for r in hi if r.get("prompt_type") == "numeric"]
    text = [r for r in hi if r.get("prompt_type") == "text"]
    rng = random.Random(seed)
    rng.shuffle(numeric); rng.shuffle(text)
    return numeric[:n_each] + text[:n_each]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path,
                    default=cfg.RESULTS_DIR / "eval" / "Gemma-3-27B-it" / "rollouts.jsonl")
    ap.add_argument("--n-each", type=int, default=10)
    ap.add_argument("--n-continuations", type=int, default=50)
    args = ap.parse_args()

    if not args.rollouts.exists():
        print(f"Missing {args.rollouts}; run scripts/run_full_eval.py for Gemma-3-27B-it first.")
        return

    selected = _select_high_frustration(args.rollouts, args.n_each, cfg.SEED)
    helper = _ClaudeHelper()
    prefills = build_prefills(selected, helper)

    # Gemma base + instruct at the same scale (27B); 12B optional.
    specs = [s for s in cfg.BASE_MODELS if "27B" in s.name] + \
            [s for s in cfg.INSTRUCT_MODELS if s.name == "Gemma-3-27B-it"]
    summary = run_prefill_study(specs, prefills, n_continuations=args.n_continuations)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
