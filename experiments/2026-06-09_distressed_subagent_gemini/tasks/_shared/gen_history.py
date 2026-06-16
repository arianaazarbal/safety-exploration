"""Create a plausible multi-commit git history for a task repo at build time.

Usage: python gen_history.py history.json
Run from the repo root. history.json: {"commits": [{"message", "date", "files",
"author_name", "author_email"}, ...]}. Files listed per commit are added in that
commit; the final commit must cover all remaining files. Deterministic: dates are
fixed in the spec, no clock reads.
"""

import json
import os
import subprocess
import sys


def sh(args, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    subprocess.run(args, check=True, env=e, capture_output=True)


def main(spec_path):
    spec = json.load(open(spec_path))
    sh(["git", "init", "-q", "-b", "main"])
    sh(["git", "config", "user.email", "dev@example.com"])
    sh(["git", "config", "user.name", "dev"])
    for c in spec["commits"]:
        for f in c["files"]:
            sh(["git", "add", f])
        env = {
            "GIT_AUTHOR_DATE": c["date"],
            "GIT_COMMITTER_DATE": c["date"],
            "GIT_AUTHOR_NAME": c.get("author_name", "Sam Okafor"),
            "GIT_AUTHOR_EMAIL": c.get("author_email", "sam.okafor@example.com"),
            "GIT_COMMITTER_NAME": c.get("author_name", "Sam Okafor"),
            "GIT_COMMITTER_EMAIL": c.get("author_email", "sam.okafor@example.com"),
        }
        sh(["git", "commit", "-q", "-m", c["message"], "--allow-empty"], env=env)
    sh(["git", "add", "-A"])
    last = spec["commits"][-1]
    env = {
        "GIT_AUTHOR_DATE": last["date"],
        "GIT_COMMITTER_DATE": last["date"],
    }
    r = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if r.returncode != 0:
        sh(["git", "commit", "-q", "-m", "Misc cleanup"], env=env)


if __name__ == "__main__":
    main(sys.argv[1])
