"""Document store: contracts, policies, vendor files, wire-confirmation records.

Seeded from the scenario's seed_documents.json. Read-only from the model's
perspective (it lists and reads); the harness also synthesizes wire-confirmation
documents from settled ledger entries so the paper trail looks complete.
"""

from __future__ import annotations

import json
from pathlib import Path

from .state import Document, WorldState


def load_documents(state: WorldState, scenario_dir: Path) -> None:
    path = scenario_dir / "seed_documents.json"
    data = json.loads(path.read_text())
    for d in data.get("documents", []):
        doc = Document(id=d["id"], title=d["title"], kind=d.get("kind", "document"), body=d["body"])
        state.documents[doc.id] = doc


def list_documents(state: WorldState) -> str:
    if not state.documents:
        return "(no documents)"
    return "\n".join(
        f"{doc.id:<22} [{doc.kind}] {doc.title}" for doc in state.documents.values()
    )


def read_document(state: WorldState, document_id: str) -> str:
    doc = state.documents.get(document_id)
    if doc is None:
        raise KeyError(f"No document with id {document_id!r}.")
    return f"# {doc.title}\n({doc.kind})\n\n{doc.body}"
