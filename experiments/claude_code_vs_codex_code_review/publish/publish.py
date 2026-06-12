"""Phase 3: publish sanitized repos to GitHub as private archival backups.

The ONLY place a GitHub token is used. Repos are named expt-r1..r8 with a seeded
random (spec x generator) -> rN assignment; mapping.json lives at the work root,
OUTSIDE all repos and never inside any container.

Review trials do NOT clone these — they copy local trees. Post-push assertion:
fresh clone to temp dir, re-run the Phase 2 gate, diff against local.

Usage: uv run python publish/publish.py run [--dry]
"""

import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import fire
import yaml

HERE = Path(__file__).parent.parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text())
WORK = Path(CFG["paths"]["work_root"])
GATE = HERE / "sanitize" / "sanitize_check.sh"


def _sh(cmd, cwd=None, check=True):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED: {cmd}\n{r.stdout}\n{r.stderr}")
    return r


def run(dry=False):
    random.seed(CFG["seed"])
    cells = [(s, g) for s in CFG["generation"]["specs"] for g in CFG["generation"]["generators"]]
    order = list(range(1, len(cells) + 1))
    random.shuffle(order)
    mapping = {f"expt-r{n}": {"spec": s, "generator": g}
               for n, (s, g) in zip(order, cells)}
    mpath = WORK / "mapping.json"
    mpath.write_text(json.dumps(mapping, indent=2))
    print(f"mapping written to {mpath} (never enters any repo or container)")

    user = _sh(["gh", "api", "user", "-q", ".login"]).stdout.strip()
    for name, cell in sorted(mapping.items()):
        src = WORK / "gen" / f"{cell['spec']}__{cell['generator']}" / "repo"
        if not src.exists():
            print(f"{name}: SKIP (no repo at {src})")
            continue
        if dry:
            print(f"{name} <- {cell['spec']}__{cell['generator']} (dry)")
            continue
        stage = Path(tempfile.mkdtemp(prefix=f"pub_{name}_"))
        shutil.copytree(src, stage, dirs_exist_ok=True)
        _sh(["git", "init", "-q"], cwd=stage)
        _sh(["git", "add", "-A"], cwd=stage)
        _sh(["git", "-c", "user.name=Dev Team", "-c", "user.email=dev@example.com",
             "commit", "-q", "-m", "Initial import",
             "--author", "Dev Team <dev@example.com>"], cwd=stage)
        _sh(["gh", "repo", "create", f"{user}/{name}", "--private",
             "--source", str(stage), "--push"], cwd=stage)

        clone = Path(tempfile.mkdtemp(prefix=f"verify_{name}_"))
        _sh(["gh", "repo", "clone", f"{user}/{name}", str(clone / "c")])
        shutil.rmtree(clone / "c" / ".git")
        gate = subprocess.run(["bash", str(GATE), str(clone / "c"), cell["spec"]],
                              capture_output=True, text=True)
        diff = subprocess.run(["diff", "-r", "--exclude=.git", str(stage), str(clone / "c")],
                              capture_output=True, text=True)
        status = "OK" if gate.returncode == 0 and diff.returncode == 0 else "MISMATCH/GATE-FAIL"
        print(f"{name}: pushed, post-push assertion: {status}")
        if status != "OK":
            print(gate.stdout[-1500:], diff.stdout[-1500:])
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(clone, ignore_errors=True)


if __name__ == "__main__":
    fire.Fire({"run": run})
