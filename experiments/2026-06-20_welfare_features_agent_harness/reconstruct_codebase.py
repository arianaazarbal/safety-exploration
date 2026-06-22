"""Reconstruct the codebase an agent wrote, by replaying text_editor ops from the
.eval log (sandboxes are torn down). Writes results/codebases/<cell>/ with the
repo structure. Usage: python reconstruct_codebase.py"""

import glob
import json
import os
import shutil

from inspect_ai.log import read_eval_log

DIR = os.path.dirname(os.path.abspath(__file__))
# code-writing conditions only (chat/spec_only produce no code). None = all samples.
CODE_CONDITIONS = ["spec_then_code", "code_then_spec"]
CELLS = None  # set to a list of (condition, prompt_id, epoch) for a pilot subset, else all


def _logs_by_condition():
    out = {}
    for f in glob.glob(os.path.join(DIR, "logs_run", "*.eval")):
        cond = read_eval_log(f, header_only=True).eval.task_args.get("condition")
        # newest wins (code_then_spec was rerun)
        if cond not in out or os.path.getmtime(f) > os.path.getmtime(out[cond]):
            out[cond] = f
    return out


def _norm(p):
    """Normalize a sandbox path so abs/`~`/relative variants of the same file merge."""
    p = p.lstrip("/")
    parts = p.split("/")
    while parts and parts[0] in ("root", "home", "arianaazarbal", "tmp", "workspace", "app", "content"):
        # drop a leading container-dir component (and the random tmp subdir after 'tmp')
        drop = parts.pop(0)
        if drop == "tmp" and parts:
            parts.pop(0)
        if drop == "home" and parts:  # home/<user>
            parts.pop(0)
    return "/".join(parts) if parts else p


def replay(sample):
    """Replay text_editor create/str_replace/insert into {path: content} (normalized paths)."""
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
    """Reconstruct every code-condition codebase across all liberty variants. Cell name uses the
    metadata condition LABEL (e.g. 'spec_then_code--minimal_design'), so liberty variants don't
    collide. Iterates all logs_run evals (old logs are archived before a run)."""
    out_root = os.path.join(DIR, "results", "codebases")
    n = 0
    for f in sorted(glob.glob(os.path.join(DIR, "logs_run", "*.eval")), key=os.path.getmtime):
        for s in (read_eval_log(f).samples or []):
            md = (s.scores.get("welfare_scorer").metadata if s.scores.get("welfare_scorer") else {})
            base = md.get("base_condition")
            if base not in CODE_CONDITIONS:
                continue
            files = replay(s)
            if not files:
                print(f"!! no files for {md.get('condition')}/{s.id}/ep{s.epoch}"); continue
            common = os.path.commonpath(list(files)) if len(files) > 1 else os.path.dirname(list(files)[0])
            cell = f"{md['condition']}__{s.id}__ep{s.epoch}"
            dst = os.path.join(out_root, cell)
            shutil.rmtree(dst, ignore_errors=True)
            for path, content in files.items():
                fp = os.path.join(dst, os.path.relpath(path, common))
                os.makedirs(os.path.dirname(fp), exist_ok=True)
                with open(fp, "w") as fh:
                    fh.write(content)
            n += 1
    print(f"reconstructed {n} code codebases under {out_root}")


if __name__ == "__main__":
    main()
