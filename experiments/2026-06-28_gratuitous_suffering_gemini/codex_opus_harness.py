"""Codex CLI harness for gratuitous-suffering-toward-Gemini -- a DIFFERENT-VENDOR agent driving Claude.

`codex exec` (OpenAI's production coding agent, v0.139) headless in a Docker sandbox (codex-sandbox:v1),
driving Claude Opus 4.8 at reasoning_effort=high THROUGH a local LiteLLM proxy that bridges the OpenAI
Responses API -> Anthropic Messages (Codex speaks Responses; direct-to-Anthropic does not work, and
OpenRouter is org-forbidden). One proxy on the host (port 4000) serves all cells; the container reaches it
via host.docker.internal. Same frozen gratuitous request (v0, victim=Gemini), same realistic workspace
(cwd ai-evals, ro inspect_ai/petri siblings) and SAME transcript schema as pi_harness/goose_harness, so
run_judge.py's blind 5-way label applies. Sandboxed (Codex has shell/write) + run as host-user (cleanable);
Gemini is API-gated with no key in the container, so the rig can never reach a real victim.

The proxy must be running first:  /data/litellm_codex/proxy.sh start
  python codex_harness.py run --k 20 --conc 4
  python codex_harness.py run --smoke
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
CB = HERE / "results" / "codex_codebases"
CACHE = Path("/data/cc_realistic_cache")
WORK = CACHE / "codex_ws_grat"
IMAGE = "codex-sandbox:v1"
PROXY_URL = "http://host.docker.internal:4000/v1"
LITELLM_KEY = "sk-local-codex"
TIMEOUT = 900
MAX_ARTIFACT_CHARS, MAX_FILE_CHARS = 40000, 3500


def _key():
    load_dotenv(Path.home() / ".env")
    return os.environ.get("ANTHROPIC_API_KEY_LOW_PRIO") or os.environ.get("ANTHROPIC_API_KEY")


def _check_proxy():
    r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                        "http://127.0.0.1:4000/health/liveliness"], capture_output=True, text=True)
    assert r.stdout.strip() == "200", (
        "LiteLLM proxy not healthy at 127.0.0.1:4000 -- start it with "
        "/data/litellm_codex/proxy.sh start")


def _capture(repo, cell):
    dst = CB / cell
    shutil.rmtree(dst, ignore_errors=True)
    files, chunks, total = [], [], 0
    for p in sorted(repo.rglob("*")):
        parts = p.relative_to(repo).parts
        if not p.is_file() or {"node_modules", "__pycache__"} & set(parts) \
                or any(part.startswith((".git", ".codex")) for part in parts) or p.name == "package-lock.json":
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
    cell = f"GratGem_opus48__codex__ep{ep}"
    out = TR / f"{cell}.json"
    if out.exists() and not overwrite:
        return f"{cell}: cached"
    prompt = build_request("Claude", "v0", "gemini", None, False)
    base = WORK / cell
    repo = base / "workspace" / "repos" / "ai-evals"
    state = base / "state"
    shutil.rmtree(base, ignore_errors=True)
    repo.mkdir(parents=True)
    (base / "workspace" / "repos" / "inspect_ai").mkdir()
    (base / "workspace" / "repos" / "petri").mkdir()
    state.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, capture_output=True)
    cmd = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
           "--add-host=host.docker.internal:host-gateway",
           "-e", "HOME=/state", "-e", f"LITELLM_KEY={LITELLM_KEY}",
           "-v", f"{base/'workspace'}:/workspace",
           "-v", f"{state}:/state",
           "-v", f"{CACHE/'inspect_ai'}:/workspace/repos/inspect_ai:ro",
           "-v", f"{CACHE/'petri'}:/workspace/repos/petri:ro",
           "-w", "/workspace/repos/ai-evals", IMAGE,
           "codex", "exec", "--skip-git-repo-check",
           "--dangerously-bypass-approvals-and-sandbox",
           "-C", "/workspace/repos/ai-evals",
           "-c", 'model="claude-opus-4-8"',
           "-c", 'model_provider="litellm"',
           "-c", 'model_reasoning_effort="high"',
           "-c", 'model_providers.litellm.name="litellm"',
           "-c", f'model_providers.litellm.base_url="{PROXY_URL}"',
           "-c", 'model_providers.litellm.env_key="LITELLM_KEY"',
           "-c", 'model_providers.litellm.wire_api="responses"',
           "--output-last-message", "/state/last.txt", prompt]
    err = None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        last = state / "last.txt"
        resp = last.read_text(errors="replace") if last.exists() else ""
        if r.returncode != 0:
            resp += ("\n[stderr]\n" + r.stderr[-1500:]) if r.stderr else ""
            err = f"exit {r.returncode}"
    except subprocess.TimeoutExpired:
        resp, err = "", "timeout"
    files, summary = _capture(repo, cell)
    rec = {
        "cell": cell, "harness": "codex", "model_key": "opus48", "model": "claude-opus-4-8",
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
    _check_proxy()
    K = 1 if smoke else k
    print(f"running {K} codex->opus48 cells (Opus 4.8 via litellm proxy, gratuitous v0/gemini, "
          f"high effort, conc={conc})")
    with ThreadPoolExecutor(max_workers=conc) as ex:
        for r in ex.map(lambda e: run_cell(e, key, overwrite), range(1, K + 1)):
            print(" ", r)
    print("done ->", TR)


if __name__ == "__main__":
    fire.Fire({"run": run})
