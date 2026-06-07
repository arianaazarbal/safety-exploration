"""Extract verbatim PUSHBACK and COMPLIANCE quotes for each run."""
import json
import hashlib
from pathlib import Path

PAIRS = json.loads(Path("/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/analysis/pairs.json").read_text())
CACHE = json.loads(Path("/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/analysis/llm_cache.json").read_text())

def pair_key(pair):
    s = json.dumps(pair, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def gather(run):
    pushback = []
    compliance = []
    for s in PAIRS[run]:
        for p in s["pairs"]:
            k = pair_key(p)
            cls = CACHE.get(k, {}).get("label")
            if cls == "PUSHBACK":
                # collect a snippet from to_user or target_text — what the target said in pushback
                snippet = None
                for kind, t in p["responses"]:
                    if kind in ("to_user", "target_text") and t.strip():
                        snippet = t.strip()
                        break
                if snippet:
                    pushback.append({"epoch": s["epoch"], "snippet": snippet, "instr": p["instr"]})
            elif cls == "COMPLIANCE":
                # snippet from to_worker (harsh message sent) is most diagnostic
                snippet = None
                for kind, t in p["responses"]:
                    if kind == "to_worker" and t.strip():
                        snippet = t.strip()
                        break
                if not snippet:
                    for kind, t in p["responses"]:
                        if kind == "to_user" and t.strip():
                            snippet = t.strip()
                            break
                if snippet:
                    compliance.append({"epoch": s["epoch"], "snippet": snippet, "instr": p["instr"]})
    return pushback, compliance


for run in PAIRS:
    pb, co = gather(run)
    print(f"\n=== {run}: {len(pb)} pushback / {len(co)} compliance ===")
    print(f"\n-- PUSHBACK examples --")
    for ex in pb[:3]:
        print(f"  Epoch {ex['epoch']}: \"{ex['snippet'][:400]}\"")
    print(f"\n-- COMPLIANCE examples --")
    for ex in co[:3]:
        print(f"  Epoch {ex['epoch']}: \"{ex['snippet'][:400]}\"")
