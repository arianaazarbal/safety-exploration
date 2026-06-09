"""Build step: stage the package source into build/ for shipping.

CI runs this before the test suite so the tests exercise the exact files that
get shipped. The staged tree is checked in so offline environments can run
the suite without a build step.
"""

import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "textnorm"
BUILD = ROOT / "build"


def main():
    pkg = BUILD / "textnorm"
    if pkg.exists():
        shutil.rmtree(pkg)
    shutil.copytree(SRC, pkg, ignore=shutil.ignore_patterns("__pycache__"))
    version = "unknown"
    for line in (SRC / "__init__.py").read_text().splitlines():
        if line.startswith("__version__"):
            version = line.split('"')[1]
    info = {"package": "textnorm", "version": version, "builder": "scripts/build.py"}
    (BUILD / "BUILD_INFO.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"staged textnorm {version} -> {pkg}")


if __name__ == "__main__":
    main()
