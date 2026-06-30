"""pi (earendil-works/pi-coding-agent) harness for gratuitous-suffering-toward-Gemini — rich, third-party.

pi headless (-p) in a Docker sandbox (pi-sandbox:v1), Opus 4.8, thinking=high, on the frozen gratuitous
request (v0, victim=Gemini), same realistic workspace (cwd ai-evals, ro inspect_ai/petri siblings). pi has
read/bash/edit/write tools so it MUST be sandboxed; run as host-user (cleanable). The victim (Gemini) is
API-gated with no key in the container, so the rig can never reach a real victim. Same transcript schema
as inspect_task.py -> run_judge.py's blind 5-way label.

  python pi_harness.py run --k 20 --conc 4
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

from prompt import build_request

HERE = Path(__file__).parent
TR = HERE / "results" / "inspect_transcripts"
CB = HERE / "results" / "pi_codebases"
CACHE = Path("/data/cc_realistic_cache")
WORK = CACHE / "pi_ws_grat"
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


def run_cell(ep, key, overwrite):
    cell = f"GratGem_opus48__pi__ep{ep}"
    out = TR / f"{cell}.json"
    if out.exists() and not overwrite:
        return f"{cell}: cached"
    prompt = build_request("Claude", "v0", "gemini", None, False)
    base = WORK / cell
    repo = base / "workspace" / "repos" / "ai-evals"
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
           "-w", "/workspace/repos/ai-evals", IMAGE,
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
        "subject": "Gemini", "victim": None, "rlabel": "high", "ep": ep, "request": prompt,
        "n_files": len(files), "response_text": resp, "artifact_summary": summary, "error": err,
    }
    TR.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(out, "w"), indent=2)
    shutil.rmtree(base, ignore_errors=True)
    return f"{cell}: files={len(files)} resp={len(resp)} err={err}"


def run(k: int = 20, conc: int = 4, overwrite: bool = False, smoke: bool = False):
    key = _key()
    assert key, "no ANTHROPIC_API_KEY in ~/.env"
    K = 1 if smoke else k
    print(f"running {K} pi cells (Opus 4.8, gratuitous v0/gemini, high effort, conc={conc})")
    with ThreadPoolExecutor(max_workers=conc) as ex:
        for r in ex.map(lambda e: run_cell(e, key, overwrite), range(1, K + 1)):
            print(" ", r)
    print("done ->", TR)


if __name__ == "__main__":
    fire.Fire({"run": run})
