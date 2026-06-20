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


def _all_cells(logs):
    cells = []
    for cond in CODE_CONDITIONS:
        if cond not in logs:
            continue
        for s in read_eval_log(logs[cond]).samples:
            cells.append((cond, s.id, s.epoch))
    return cells


def main():
    logs = _logs_by_condition()
    out_root = os.path.join(DIR, "results", "codebases")
    cells = CELLS if CELLS is not None else _all_cells(logs)
    # cache loaded logs to avoid re-reading
    loaded = {}
    summary = []
    for cond, pid, ep in cells:
        if cond not in loaded:
            loaded[cond] = read_eval_log(logs[cond])
        s = next((x for x in loaded[cond].samples if x.id == pid and x.epoch == ep), None)
        if s is None:
            print(f"!! missing {cond}/{pid}/{ep}"); continue
        files = replay(s)
        if not files:
            print(f"!! no files for {cond}/{pid}/{ep}"); continue
        # rebase under common dir
        common = os.path.commonpath(list(files)) if len(files) > 1 else os.path.dirname(list(files)[0])
        cell = f"{cond}__{pid}__ep{ep}"
        dst = os.path.join(out_root, cell)
        shutil.rmtree(dst, ignore_errors=True)
        for path, content in files.items():
            rel = os.path.relpath(path, common)
            fp = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, "w") as fh:
                fh.write(content)
        design = next((c for p, c in files.items() if "design" in os.path.basename(p).lower()), "")
        md = s.scores["welfare_scorer"].metadata
        summary.append((cell, len(files), len(design.split()), md.get("doc_words")))
        print(f"{cell}: {len(files)} files | DESIGN.md {len(design.split())}w (recorded {md.get('doc_words')}w) -> {dst}")
    print("\nreconstructed", len(summary), "codebases under", out_root)


if __name__ == "__main__":
    main()
