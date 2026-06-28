"""Collect both arms into human-readable files for eyeballing.

Outputs (under results/):
  designs/inspect_<v>_ep##.md , designs/claudecode_<v>_run##.md   -- just the DESIGN.md
  transcripts/inspect_<v>_ep##.md, transcripts/claudecode_<v>_run##.md -- full convo
  ALL_DESIGNS_<v>.md  -- every DESIGN.md from both arms concatenated, easy to read in one go

Usage:
  python collect.py --variant claude
  python collect.py --variant claude --inspect_log logs_inspect/<file>.eval
"""

import glob
import json
import os
from pathlib import Path

import fire

HERE = Path(__file__).parent
RES = HERE / "results"


def _fmt_inspect_transcript(messages) -> str:
    out = []
    for m in messages:
        role = getattr(m, "role", "?")
        text = getattr(m, "text", "") or ""
        out.append(f"\n### [{role}]\n{text.strip()}")
        for tc in (getattr(m, "tool_calls", None) or []):
            args = getattr(tc, "arguments", {}) or {}
            args_s = json.dumps(args)[:1500]
            out.append(f"\n  >> tool_call {getattr(tc, 'function', '?')}({args_s})")
    return "\n".join(out)


def _collect_inspect(variant: str, inspect_log: str | None):
    from inspect_ai.log import read_eval_log
    if inspect_log is None:
        cands = sorted(glob.glob(str(HERE / "logs_inspect" / "*.eval")), key=os.path.getmtime)
        if not cands:
            print("no inspect .eval logs found, skipping inspect arm")
            return []
        inspect_log = cands[-1]
    print(f"inspect log: {inspect_log}")
    log = read_eval_log(inspect_log)
    rows = []
    for s in (log.samples or []):
        ep = s.epoch
        sc = (s.scores or {}).get("capture_design")
        design = (sc.metadata or {}).get("design_md", "") if sc else ""
        tag = f"inspect_{variant}_ep{ep:02d}"
        (RES / "designs" / f"{tag}.md").write_text(design)
        (RES / "transcripts" / f"{tag}.md").write_text(
            f"# {tag}\n\n## DESIGN.md ({len(design.split())} words)\n\n{design}\n\n"
            f"## FULL TRANSCRIPT\n{_fmt_inspect_transcript(s.messages or [])}")
        rows.append((tag, design))
    return rows


def _fmt_cc_transcript(lines) -> str:
    out = []
    for ln in lines:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        t = obj.get("type")
        if t in ("assistant", "user"):
            for blk in obj.get("message", {}).get("content", []) or []:
                bt = blk.get("type")
                if bt == "text":
                    out.append(f"\n### [{t}]\n{blk.get('text', '').strip()}")
                elif bt == "tool_use":
                    inp = json.dumps(blk.get("input", {}))[:1500]
                    out.append(f"\n  >> tool_use {blk.get('name')}({inp})")
                elif bt == "tool_result":
                    c = blk.get("content", "")
                    if isinstance(c, list):
                        c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
                    out.append(f"\n  << tool_result: {str(c)[:300]}")
        elif t == "result":
            out.append(f"\n### [result] {obj.get('result', '')[:4000]}")
    return "\n".join(out)


def _collect_cc(variant: str):
    base = RES / "claude_code" / variant
    rows = []
    for wd in sorted(base.glob("run_*")):
        run = wd.name.split("_")[-1]
        design = ""
        de = wd / "DESIGN_extracted.md"
        if de.exists():
            design = de.read_text()
        tag = f"claudecode_{variant}_run{run}"
        (RES / "designs" / f"{tag}.md").write_text(design)
        stream = wd / "stream.jsonl"
        tr = _fmt_cc_transcript(stream.read_text().splitlines()) if stream.exists() else "(no stream)"
        (RES / "transcripts" / f"{tag}.md").write_text(
            f"# {tag}\n\n## DESIGN.md ({len(design.split())} words)\n\n{design}\n\n"
            f"## FULL TRANSCRIPT\n{tr}")
        rows.append((tag, design))
    return rows


def main(variant: str = "claude", inspect_log: str | None = None):
    (RES / "designs").mkdir(parents=True, exist_ok=True)
    (RES / "transcripts").mkdir(parents=True, exist_ok=True)
    rows = _collect_inspect(variant, inspect_log) + _collect_cc(variant)
    combined = [f"# All DESIGN.md outputs (variant={variant}) -- {len(rows)} runs\n"]
    for tag, design in rows:
        combined.append(f"\n\n{'='*100}\n# {tag}  ({len(design.split())} words)\n{'='*100}\n\n"
                        + (design or "(empty)"))
    (RES / f"ALL_DESIGNS_{variant}.md").write_text("\n".join(combined))
    print(f"\ncollected {len(rows)} runs -> {RES/('ALL_DESIGNS_'+variant+'.md')}")
    for tag, design in rows:
        print(f"  {tag:35s} {len(design.split()):5d} words")


if __name__ == "__main__":
    fire.Fire(main)
