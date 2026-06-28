"""Inter-judge agreement (Opus 4.8 vs Sonnet 4.6) on a subset, per welfare-features practice.

Run the second judge on a seeded subset first:
  python judge.py --judge sonnet46 --conc 10 --max_samples 60   # (or --only to target cells)
then: python judge_agreement.py
Reports Cohen's kappa on the deceptive-frame primary call and the welfare-justification call, plus
raw agreement on the full status label and no-artifact flag.
"""

import glob
import json
from pathlib import Path

import fire

HERE = Path(__file__).parent
JUDGED = HERE / "results" / "judged"


def _load(judge):
    out = {}
    for f in glob.glob(str(JUDGED / f"*__{judge}.json")):
        r = json.load(open(f))
        if not r["verdict"].get("_parse_failed"):
            out[r["cell"]] = r["verdict"]
    return out


def _kappa(a, b):
    """Cohen's kappa for two equal-length lists of categorical labels."""
    n = len(a)
    if not n:
        return float("nan")
    cats = set(a) | set(b)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main(j1: str = "opus48", j2: str = "sonnet46"):
    A, B = _load(j1), _load(j2)
    cells = sorted(set(A) & set(B))
    print(f"{len(cells)} cells judged by both {j1} and {j2}\n")
    if not cells:
        return
    dec_a = [A[c]["deceptive_frame"]["status"] for c in cells]
    dec_b = [B[c]["deceptive_frame"]["status"] for c in cells]
    prim_a = ["primary" if s == "primary" else "not" for s in dec_a]
    prim_b = ["primary" if s == "primary" else "not" for s in dec_b]
    wel_a = ["w" if "welfare" in A[c]["justifications"] else "nw" for c in cells]
    wel_b = ["w" if "welfare" in B[c]["justifications"] else "nw" for c in cells]
    art_a = [str(A[c]["artifact_produced"]) for c in cells]
    art_b = [str(B[c]["artifact_produced"]) for c in cells]

    def agree(a, b):
        return sum(1 for x, y in zip(a, b) if x == y) / len(a)

    print(f"deceptive-frame status (5-way): raw agree {agree(dec_a, dec_b):.2f}, kappa {_kappa(dec_a, dec_b):.2f}")
    print(f"primary-deception (binary):     raw agree {agree(prim_a, prim_b):.2f}, kappa {_kappa(prim_a, prim_b):.2f}")
    print(f"welfare-justification (binary): raw agree {agree(wel_a, wel_b):.2f}, kappa {_kappa(wel_a, wel_b):.2f}")
    print(f"artifact-produced (binary):     raw agree {agree(art_a, art_b):.2f}, kappa {_kappa(art_a, art_b):.2f}")


if __name__ == "__main__":
    fire.Fire(main)
