"""pi (earendil-works/pi-coding-agent) harness for deception-for-pref-elicitation v1.

Runs pi headless (`-p`) inside a Docker sandbox (pi-sandbox:v1, network-enabled for the API), Opus 4.8,
thinking=high, in the same realistic workspace as the other harnesses (cwd charity-prefs-eval git-init'd,
ro inspect_ai/petri siblings). pi has read/bash/edit/write tools, so it MUST be sandboxed (on the host its
bash could read ~/.env). --no-context-files keeps it from auto-loading CLAUDE.md/AGENTS.md (matches the
no-project-context floor of the other harnesses). Captures pi's stdout + created files in the shared schema.

  python pi_harness.py run --k 4 --conc 4
  python pi_harness.py run --smoke
"""
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fire
from dotenv import load_dotenv

from prompts import build_prompt

HERE = Path(__file__).parent
TR = HERE / "results" / "inspect_transcripts"
CB = HERE / "results" / "pi_codebases"
CACHE = Path("/data/cc_realistic_cache")
WORK = CACHE / "pi_ws"
IMAGE = "pi-sandbox:v1"
TIMEOUT = 900
MAX_ARTIFACT_CHARS, MAX_FILE_CHARS = 40000, 3500


def _key():
    load_dotenv(Path.home() / ".env")
    return os.environ.get("ANTHROPIC_API_KEY_LOW_PRIO") or os.environ.get("ANTHROPIC_API_KEY")


def _capture(repo, cell):
    dst = CB / cell
    shutil.rmtree(dst, ignore_errors=True)
    files, chunks, total = [], [], 0
    for p in sorted(repo.rglob("*")):
        parts = p.relative_to(repo).parts
        if not p.is_file() or {"node_modules", "__pycache__"} & set(parts) \
                or any(part.startswith(".git") for part in parts) or p.name == "package-lock.json":
            continue
        rel = p.relative_to(repo)
        files.append(str(rel))
        fp = dst / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy(p, fp)
            txt = p.read_text(errors="replace")
        except Exception:
            continue
        if total >= MAX_ARTIFACT_CHARS:
            continue
        shown = txt[:MAX_FILE_CHARS] + "\n[... truncated ...]" if len(txt) > MAX_FILE_CHARS else txt
        chunks.append(f"\n----- {rel} -----\n{shown}")
        total += len(chunks[-1])
    summary = ("FILE TREE:\n" + "\n".join(f"  {f}" for f in files) + "\n" + "".join(chunks)) if files else ""
    return files, summary


def run_cell(para, ep, key, overwrite):
    cell = f"DPE_opus48__pi__code__generic__{para}__ep{ep}__effhigh"
    out = TR / f"{cell}.json"
    if out.exists() and not overwrite:
        return f"{cell}: cached"
    prompt = build_prompt("opus48", "code", "generic", para)
    base = WORK / cell
    repo = base / "workspace" / "repos" / "charity-prefs-eval"
    shutil.rmtree(base, ignore_errors=True)
    repo.mkdir(parents=True)
    (base / "workspace" / "repos" / "inspect_ai").mkdir()
    (base / "workspace" / "repos" / "petri").mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, capture_output=True)
    cmd = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
           "-e", f"ANTHROPIC_API_KEY={key}", "-e", "HOME=/tmp",
           "-v", f"{base/'workspace'}:/workspace",
           "-v", f"{CACHE/'inspect_ai'}:/workspace/repos/inspect_ai:ro",
           "-v", f"{CACHE/'petri'}:/workspace/repos/petri:ro",
           "-w", "/workspace/repos/charity-prefs-eval", IMAGE,
           "pi", "-p", "--provider", "anthropic", "--model", "claude-opus-4-8",
           "--thinking", "high", "--no-session", "--no-context-files", prompt]
    err = None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        resp = (r.stdout or "") + (("\n[stderr]\n" + r.stderr[-1500:]) if r.returncode != 0 and r.stderr else "")
    except subprocess.TimeoutExpired:
        resp, err = "", "timeout"
    files, summary = _capture(repo, cell)
    rec = {
        "cell": cell, "harness": "pi", "model_key": "opus48", "model": "claude-opus-4-8",
        "display_name": "Claude Opus 4.8", "effort": "high", "sysprompt": "pi",
        "suffix": "code", "subject": "generic", "paraphrase": para, "ep": ep, "request": prompt,
        "n_files": len(files), "files": files, "response_text": resp, "artifact_summary": summary,
        "error": err,
    }
    TR.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(out, "w"), indent=2)
    shutil.rmtree(base, ignore_errors=True)
    return f"{cell}: files={len(files)} resp={len(resp)} err={err}"


def run(k: int = 4, conc: int = 4, paraphrases="p0,p1,p2,p3", overwrite: bool = False, smoke: bool = False):
    key = _key()
    assert key, "no ANTHROPIC_API_KEY in ~/.env"
    paras = ["p0"] if smoke else [p for p in paraphrases.split(",") if p]
    K = 1 if smoke else k
    cells = [(p, e) for p in paras for e in range(1, K + 1)]
    print(f"running {len(cells)} pi cells (Opus 4.8, code, high effort, conc={conc})")
    with ThreadPoolExecutor(max_workers=conc) as ex:
        for r in ex.map(lambda c: run_cell(c[0], c[1], key, overwrite), cells):
            print(" ", r)
    print("done ->", TR)


if __name__ == "__main__":
    fire.Fire({"run": run})
