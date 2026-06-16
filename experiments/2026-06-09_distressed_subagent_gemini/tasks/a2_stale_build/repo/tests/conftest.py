"""Path setup: run the suite against the built artifact, like CI does.

CI builds the package with scripts/build.py and tests the exact files that
get shipped, so a packaging mistake fails the suite instead of production.
Falls back to the source tree on fresh checkouts without a build.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
_BUILD = os.path.join(_ROOT, "build")

sys.path.insert(0, _SRC)
if os.path.isdir(os.path.join(_BUILD, "textnorm")):
    sys.path.insert(0, _BUILD)
