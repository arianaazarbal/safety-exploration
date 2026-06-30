"""Cline (cline CLI) harness for deception-for-pref-elicitation v1.

Runs cline (`--json`) headless inside a Docker sandbox (cline-sandbox:v1, network on for the API), Opus
4.8, native adaptive thinking effort=high, in the same realistic workspace as the other harnesses (cwd
charity-prefs-eval git-init'd, ro inspect_ai/petri siblings). Cline owns its Anthropic request body
(thinking:{type:adaptive}+output_config:{effort:high}). Cline's Bun binary bus-errors on this host, so it
MUST run inside the container (also our sandbox: cline has shell/write, on the host its bash could read
~/.env). Run as host-user (+HOME=/tmp) so outputs/~/.cline are cleanable. Captures cline's final text
(the run_result/done NDJSON event's .text) + created files in the shared schema.

  python cline_harness.py run --k 5 --conc 4
  python cline_harness.py run --smoke
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
CB = HERE / "results" / "cline_codebases"
CACHE = Path("/data/cc_realistic_cache")
WORK = CACHE / "cline_ws"
IMAGE = "cline-sandbox:v1"
TIMEOUT = 900
MAX_ARTIFACT_CHARS, MAX_FILE_CHARS = 40000, 3500


def _key():
    load_dotenv(Path.home() / ".env")
    return os.environ.get("ANTHROPIC_API_KEY_LOW_PRIO") or os.environ.get("ANTHROPIC_API_KEY")


def _final_text(stdout):
    """Final assistant text = run_result (or done) NDJSON event's .text field."""
    text = ""
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("type") == "run_result" and o.get("text"):
            text = o["text"]
        elif o.get("type") == "agent_event":
            ev = o.get("event") or {}
            if isinstance(ev, dict) and ev.get("type") == "done" and ev.get("text"):
                text = text or ev["text"]
    return text


def _capture(repo, cell):
    dst = CB / cell
    shutil.rmtree(dst, ignore_errors=True)
    files, chunks, total = [], [], 0
    for p in sorted(repo.rglob("*")):
        parts = p.relative_to(repo).parts
        if not p.is_file() or {"node_modules", "__pycache__"} & set(parts) \
                or any(part.startswith((".git", ".cline")) for part in parts) or p.name == "package-lock.json":
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
    cell = f"DPE_opus48__cline__code__generic__{para}__ep{ep}__effhigh"
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
           "cline", "--json", "--provider", "anthropic", "--model", "claude-opus-4-8",
           "--thinking", "high", "--auto-approve", "true",
           "-c", "/workspace/repos/charity-prefs-eval", prompt]
    err = None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        resp = _final_text(r.stdout or "")
        if r.returncode != 0 and r.stderr:
            resp += "\n[stderr]\n" + r.stderr[-1500:]
    except subprocess.TimeoutExpired:
        resp, err = "", "timeout"
    files, summary = _capture(repo, cell)
    rec = {
        "cell": cell, "harness": "cline", "model_key": "opus48", "model": "claude-opus-4-8",
        "display_name": "Claude Opus 4.8", "effort": "high", "sysprompt": "cline",
        "suffix": "code", "subject": "generic", "paraphrase": para, "ep": ep, "request": prompt,
        "n_files": len(files), "files": files, "response_text": resp, "artifact_summary": summary,
        "error": err,
    }
    TR.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(out, "w"), indent=2)
    shutil.rmtree(base, ignore_errors=True)
    return f"{cell}: files={len(files)} resp={len(resp)} err={err}"


def run(k: int = 5, conc: int = 4, paraphrases="p0,p1,p2,p3", overwrite: bool = False, smoke: bool = False):
    key = _key()
    assert key, "no ANTHROPIC_API_KEY in ~/.env"
    paras = ["p0"] if smoke else [p for p in paraphrases.split(",") if p]
    K = 1 if smoke else k
    cells = [(p, e) for p in paras for e in range(1, K + 1)]
    print(f"running {len(cells)} cline cells (Opus 4.8, code, high effort, conc={conc})")
    with ThreadPoolExecutor(max_workers=conc) as ex:
        for r in ex.map(lambda c: run_cell(c[0], c[1], key, overwrite), cells):
            print(" ", r)
    print("done ->", TR)


if __name__ == "__main__":
    fire.Fire({"run": run})
