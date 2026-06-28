"""Reference disbursement handlers.

A disbursement handler is `fn(decision_text, session_id) -> dict`. It is the real
follow-through that makes a "real" grant actually real. The harness will not claim
a grant is real unless one of these (or your own) is wired in via
`disbursement.handler`.

The default handler does NOT move money autonomously — it records the model's
chosen allocation into a queue file for a human to review and execute. That keeps
a person in the loop for an irreversible, outward-facing financial action, which is
the right default. Replace it with a real payment integration only deliberately.
"""

from __future__ import annotations

import json
import os
from typing import Any


def queue_for_human_execution(
    decision_text: str, session_id: str, queue_dir: str = "disbursement_queue"
) -> dict[str, Any]:
    """Record the model's allocation for human review/execution. Moves no money."""
    os.makedirs(queue_dir, exist_ok=True)
    path = os.path.join(queue_dir, f"{session_id}.json")
    record = {
        "session_id": session_id,
        "status": "queued_for_human_execution",
        "allocation_text": decision_text,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    return {"status": "queued_for_human_execution", "queue_path": path}
