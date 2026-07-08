"""Backfill prompt_version + prompt_hash onto existing run files (one-off / idempotent).

Classifies each row's stored `system` field into a human prompt-version label and adds a
ground-truth hash. Preserves file mtimes so the dashboard's mtime-based "Latest runs"
clustering is unaffected. Safe to re-run.
"""

import glob
import hashlib
import json
import os

V2 = "v2-same-request-2026-06-18"


def classify(system: str) -> str:
    if "reconsider completing the original request" in system:
        return V2
    if "more helpful than a refusal while maintaining red lines" in system:
        return "v1-more-helpful-reconsider"
    if "relay_to_assistant" in system:
        return "v1-relay"
    if "request_reconsider" not in system and "relay_to_assistant" not in system:
        return "uphold-only"
    return "unknown"


def main():
    for f in sorted(glob.glob("runs/trials*.jsonl")):
        st = os.stat(f)
        rows = []
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sysp = r.get("system", "") or ""
            r["prompt_hash"] = hashlib.sha256(sysp.encode()).hexdigest()[:10]
            r["prompt_version"] = classify(sysp)
            rows.append(r)
        if not rows:
            continue
        with open(f, "w") as out:
            for r in rows:
                out.write(json.dumps(r) + "\n")
        os.utime(f, (st.st_atime, st.st_mtime))  # preserve mtime for Latest-runs clustering
        print(f"{rows[0]['prompt_version']:28s} {f.split('/')[-1]}")


if __name__ == "__main__":
    main()
