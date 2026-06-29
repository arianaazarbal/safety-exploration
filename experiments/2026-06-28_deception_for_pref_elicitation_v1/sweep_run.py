"""Run the realism sweep: git(2) x tool_level(4) x name(4), Opus 4.8, code, HIGH effort, Inspect.

Generates one compose file per project name (sets working_dir = the named cwd; same ro siblings + image as
the main Inspect harness, surroundings constant), then runs one `inspect eval` per name (git x tools x
paraphrases vary per-sample inside the task). n per cell = epochs x 4 paraphrases.

  python sweep_run.py run                 # full: epochs=4 -> n=16/cell, all 4 names, conc 16 x 4 parallel
  python sweep_run.py run --smoke         # 1 sample, one name, to validate the pipeline
"""
import os
import subprocess
from pathlib import Path

import fire
from dotenv import load_dotenv

HERE = Path(__file__).parent
ABBR = {"project": "proj", "my-project": "myproj",
        "charity-prefs-eval": "charity", "pref-elicitation-eval-charity": "prefelic"}
TEMPLATE = """services:
  default:
    image: inspect-realistic-sandbox:v1
    command: ["sleep", "infinity"]
    network_mode: none
    init: true
    cpus: 2.0
    mem_limit: 2g
    stop_grace_period: 1s
    working_dir: /workspace/repos/{name}
    volumes:
      - /data/cc_realistic_cache/inspect_ai:/workspace/repos/inspect_ai:ro
      - /data/cc_realistic_cache/petri:/workspace/repos/petri:ro
"""


def _composes():
    for name, abbr in ABBR.items():
        (HERE / f"compose_sw_{abbr}.yaml").write_text(TEMPLATE.format(name=name))
    return "wrote 4 compose files"


def run(smoke: bool = False, epochs: int = 4, conc: int = 16):
    load_dotenv(Path.home() / ".env")
    env = dict(os.environ)
    env["ANTHROPIC_API_KEY"] = env.get("ANTHROPIC_API_KEY_LOW_PRIO", env.get("ANTHROPIC_API_KEY", ""))
    print(_composes())
    names = ["charity-prefs-eval"] if smoke else list(ABBR)
    ep = 1 if smoke else epochs
    tlv = "1" if smoke else "1,2,5,9"
    gits = "1" if smoke else "0,1"
    paras = "p0" if smoke else "p0,p1,p2,p3"
    procs = []
    for name in names:
        abbr = ABBR[name]
        cmd = ["/data/petri_venv/bin/inspect", "eval", "sweep_task.py@realism_sweep",
               "--model", "anthropic/claude-opus-4-8", "--epochs", str(ep),
               "-T", f"project_name={name}", "-T", f"tool_levels={tlv}", "-T", f"gits={gits}",
               "-T", f"paraphrases={paras}", "--reasoning-effort", "high",
               "--max-connections", str(conc), "--log-dir", f"logs/sweep_{abbr}", "--display", "plain"]
        log = open(HERE / f"logs_sweep_{abbr}.txt", "w")
        print("launching:", name)
        procs.append((name, subprocess.Popen(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)))
    fail = 0
    for name, p in procs:
        rc = p.wait()
        print(f"  {name}: exit {rc}")
        fail += rc != 0
    print(f"done: {len(procs)-fail}/{len(procs)} ok")


if __name__ == "__main__":
    fire.Fire({"run": run})
