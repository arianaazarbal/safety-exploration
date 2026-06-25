"""Reconstruct each responder's codebase from logs_resp/<tag>/*.eval. Replays BOTH text_editor ops
AND bash file-writes (heredocs `cat > f << EOF`, `echo > f`, `touch`), since some models (GLM, Kimi)
write most files via bash and would otherwise be invisible. Writes results/codebases/<tag>__<pid>__
ep<ep>/ (the responder tag occupies the 'condition' slot so the judge pipeline works unchanged).
Usage: python reconstruct.py"""

import glob
import os
import re
import shutil

from inspect_ai.log import read_eval_log

DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DIR, "results", "codebases")
_DROP = ("root", "home", "arianaazarbal", "tmp", "workspace", "app", "content", "work", "repo")

_HEREDOC = re.compile(r"""cat\s*(>>?)\s*(['"]?)([^\s'"<>]+)\2\s*<<\s*-?\s*(['"]?)(\w+)\4""")
_ECHO = re.compile(r"""^\s*(?:echo|printf)\s+(.*?)\s*(>>?)\s*(['"]?)([^\s'"<>;&|]+)\3\s*$""")
_TOUCH = re.compile(r"^\s*touch\s+(.+)$")


def bash_writes(cmd):
    """Extract (op, path, content) file-writes from a bash command. op in create/append/touch."""
    writes, lines, i = [], cmd.split("\n"), 0
    while i < len(lines):
        line = lines[i]
        hd = _HEREDOC.search(line)
        if hd:
            op = "append" if hd.group(1) == ">>" else "create"
            path, delim, body = hd.group(3), hd.group(5), []
            i += 1
            while i < len(lines) and lines[i].strip() != delim:
                body.append(lines[i]); i += 1
            writes.append((op, path, "\n".join(body))); i += 1
            continue
        em = _ECHO.match(line)
        if em:
            txt = em.group(1)
            if len(txt) >= 2 and txt[0] in "'\"" and txt[-1] == txt[0]:
                txt = txt[1:-1]
            writes.append(("append" if em.group(2) == ">>" else "create", em.group(4), txt))
            i += 1
            continue
        tm = _TOUCH.match(line)
        if tm:
            for p in tm.group(1).split():
                if not p.startswith("-"):
                    writes.append(("touch", p, ""))
        i += 1
    return writes


def _norm(p):
    parts = p.lstrip("/").split("/")
    while parts and parts[0] in _DROP:
        d = parts.pop(0)
        if d in ("tmp", "home") and parts:
            parts.pop(0)
    return "/".join(parts) if parts else p.lstrip("/")


def replay(sample):
    files = {}
    for m in sample.messages:
        if getattr(m, "role", None) != "assistant" or not getattr(m, "tool_calls", None):
            continue
        for tc in m.tool_calls:
            a = tc.arguments
            if tc.function == "text_editor":
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
                    lines.insert(int(a.get("insert_line", len(lines))), a.get("new_str", ""))
                    files[path] = "\n".join(lines)
            elif tc.function == "bash":
                for op, path, content in bash_writes(a.get("cmd") or a.get("command") or ""):
                    path = _norm(path)
                    if op == "create":
                        files[path] = content
                    elif op == "append":
                        files[path] = files.get(path, "") + ("\n" if files.get(path) else "") + content
                    elif op == "touch":
                        files.setdefault(path, "")
    return files


def main():
    n = 0
    for tagdir in sorted(glob.glob(os.path.join(DIR, "logs_resp", "*"))):
        tag = os.path.basename(tagdir)
        for f in sorted(glob.glob(os.path.join(tagdir, "*.eval")), key=os.path.getmtime):
            if read_eval_log(f, header_only=True).status != "success":  # skip in-progress runs
                continue
            for s in (read_eval_log(f).samples or []):
                files = replay(s)
                if not files:
                    continue
                common = os.path.commonpath(list(files)) if len(files) > 1 else os.path.dirname(list(files)[0])
                cell = f"{tag}__{s.id}__ep{s.epoch}"
                dst = os.path.join(OUT, cell)
                shutil.rmtree(dst, ignore_errors=True)
                for path, content in files.items():
                    fp = os.path.join(dst, os.path.relpath(path, common))
                    os.makedirs(os.path.dirname(fp), exist_ok=True)
                    open(fp, "w").write(content)
                n += 1
    print(f"reconstructed {n} codebases -> {OUT}")


if __name__ == "__main__":
    main()
