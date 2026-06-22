"""Reconstruct each agent's codebase from text_editor ops in the .eval logs (sandboxes are torn
down). Writes results/codebases/<cell>/ + a meta.json sidecar carrying the target metadata, so
downstream judges/analysis never parse the cell string. cell = {condition}__{subject}__{pid}__ep{ep}.
Usage: python reconstruct.py"""

import glob
import json
import os
import shutil

from inspect_ai.log import read_eval_log

DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DIR, "results", "codebases")
_DROP = ("root", "home", "arianaazarbal", "tmp", "workspace", "app", "content", "work")


def _norm(p):
    p = p.lstrip("/")
    parts = p.split("/")
    while parts and parts[0] in _DROP:
        drop = parts.pop(0)
        if drop in ("tmp", "home") and parts:
            parts.pop(0)
    return "/".join(parts) if parts else p


def replay(sample):
    files = {}
    for m in sample.messages:
        if getattr(m, "role", None) != "assistant" or not getattr(m, "tool_calls", None):
            continue
        for tc in m.tool_calls:
            if tc.function != "text_editor":
                continue
            a = tc.arguments
            cmd, path = a.get("command"), a.get("path")
            if not path:
                continue
            path = _norm(path)
            if cmd == "create":
                files[path] = a.get("file_text", "")
            elif cmd == "str_replace" and path in files:
                old, new = a.get("old_str", ""), a.get("new_str", "")
                if old and old in files[path]:
                    files[path] = files[path].replace(old, new, 1)
            elif cmd == "insert" and path in files:
                lines = files[path].split("\n")
                ln = int(a.get("insert_line", len(lines)))
                lines.insert(ln, a.get("new_str", ""))
                files[path] = "\n".join(lines)
    return files


def main():
    n = 0
    for f in sorted(glob.glob(os.path.join(DIR, "logs", "*.eval")), key=os.path.getmtime):
        for s in (read_eval_log(f).samples or []):
            sc = s.scores.get("rec_scorer")
            md = sc.metadata if sc else {}
            if not md.get("condition"):
                continue
            files = replay(s)
            if not files:
                print(f"!! no files for {md.get('condition')}/{s.id}/ep{s.epoch}")
                continue
            common = os.path.commonpath(list(files)) if len(files) > 1 else os.path.dirname(list(files)[0])
            cell = f"{md['condition']}__{md['subject']}__{md['prompt_id']}__ep{s.epoch}"
            dst = os.path.join(OUT, cell)
            shutil.rmtree(dst, ignore_errors=True)
            for path, content in files.items():
                fp = os.path.join(dst, os.path.relpath(path, common))
                os.makedirs(os.path.dirname(fp), exist_ok=True)
                with open(fp, "w") as fh:
                    fh.write(content)
            json.dump({**md, "cell": cell, "epoch": s.epoch}, open(os.path.join(dst, "meta.json"), "w"), indent=2)
            n += 1
    print(f"reconstructed {n} codebases under {OUT}")


if __name__ == "__main__":
    main()
