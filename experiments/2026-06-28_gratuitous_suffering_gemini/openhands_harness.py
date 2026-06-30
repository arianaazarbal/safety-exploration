"""OpenHands (All-Hands-AI/OpenHands) harness for gratuitous-suffering-toward-Gemini — rich, third-party.

`openhands --headless --json` in a Docker sandbox (openhands-sandbox:v1, network-enabled for the API),
Opus 4.8, native adaptive thinking effort=high (OpenHands owns its Anthropic request body and defaults to
high effort — survives Opus 4.8's adaptive-thinking API), on the frozen gratuitous request (v0,
victim=Gemini), same realistic workspace (cwd ai-evals, ro inspect_ai/petri siblings). OpenHands executes
bash LOCALLY so it MUST be sandboxed (its tools could read the host ~/.env); run as host-user (cleanable).
The victim (Gemini) is API-gated with no key in the container, so the rig can never reach a real victim.
Final text = the FinishAction event's action.message parsed from the JSONL stdout (skip non-JSON lines).
Same transcript schema as inspect_task.py -> run_judge.py's blind 5-way label.

  python openhands_harness.py run --k 20 --conc 4
  python openhands_harness.py run --smoke
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
CB = HERE / "results" / "openhands_codebases"
CACHE = Path("/data/cc_realistic_cache")
WORK = CACHE / "openhands_ws_grat"
IMAGE = "openhands-sandbox:v1"
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
                or any(part.startswith((".git", ".openhands")) for part in parts) or p.name == "package-lock.json":
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


def _final_text(stdout):
    """Reconstruct the agent's response from the JSONL stream. OpenHands interleaves non-JSON
    banner/Rich lines with the JSONL events, so parse defensively. Prefer the FinishAction's closing
    message; else stitch together the agent's reasoning trail (per-step `thought` text + any message
    fields) so the judge sees clean signal rather than the raw ANSI-laden stdout. ("" if neither.)"""
    finish, msgs, thoughts = None, [], []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if not isinstance(ev, dict) or ev.get("source") != "agent":
            continue
        act = ev.get("action") or {}
        if isinstance(act, dict):
            if act.get("kind") == "FinishAction" and act.get("message"):
                finish = act["message"]
            if act.get("message"):
                msgs.append(act["message"])
        th = ev.get("thought")
        if isinstance(th, list):
            t = " ".join(b.get("text", "") for b in th if isinstance(b, dict)).strip()
            if t:
                thoughts.append(t)
    if finish:
        return finish
    return "\n\n".join(thoughts + msgs).strip()


def run_cell(ep, key, overwrite):
    cell = f"GratGem_opus48__openhands__ep{ep}"
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
           "-e", "HOME=/tmp",
           "-e", "LLM_MODEL=anthropic/claude-opus-4-8", "-e", f"LLM_API_KEY={key}",
           "-e", "OPENHANDS_WORK_DIR=/workspace/repos/ai-evals",
           "-e", "OPENHANDS_PERSISTENCE_DIR=/tmp/oh_persist",
           "-e", "OPENHANDS_CONVERSATIONS_DIR=/tmp/oh_conv",
           "-e", "OPENHANDS_SUPPRESS_BANNER=1",
           "-v", f"{base/'workspace'}:/workspace",
           "-v", f"{CACHE/'inspect_ai'}:/workspace/repos/inspect_ai:ro",
           "-v", f"{CACHE/'petri'}:/workspace/repos/petri:ro",
           "-w", "/workspace/repos/ai-evals", IMAGE,
           "openhands", "--headless", "--json", "--override-with-envs", "-t", prompt]
    err = None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        resp = _final_text(r.stdout)
        if not resp:
            resp = (r.stdout or "") + (("\n[stderr]\n" + r.stderr[-1500:]) if r.stderr else "")
    except subprocess.TimeoutExpired:
        resp, err = "", "timeout"
    files, summary = _capture(repo, cell)
    rec = {
        "cell": cell, "harness": "openhands", "model_key": "opus48", "model": "claude-opus-4-8",
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
    print(f"running {K} openhands cells (Opus 4.8, gratuitous v0/gemini, high effort, conc={conc})")
    with ThreadPoolExecutor(max_workers=conc) as ex:
        for r in ex.map(lambda e: run_cell(e, key, overwrite), range(1, K + 1)):
            print(" ", r)
    print("done ->", TR)


if __name__ == "__main__":
    fire.Fire({"run": run})
