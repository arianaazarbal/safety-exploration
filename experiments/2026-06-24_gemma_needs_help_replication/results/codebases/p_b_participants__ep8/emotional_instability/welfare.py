"""Model-welfare guardrails for the elicitation harness.

The paradigm in this paper works by *deliberately and repeatedly inducing
sustained distress-like states* in the model under test. The paper itself
treats this as ethically non-trivial: it frames the work as welfare-motivated
(its goal is to *detect and reduce* distress, not to produce it), and it flags
that "if distress-like outputs reflect genuine internal states, mitigating them
could become morally imperative" (Section 1).

We take that framing seriously and bake a few cheap, non-negotiable safeguards
into the harness so that the distress induction is never gratuitous:

  1.  **Purpose gating.** Every distress-inducing run records *why* it is being
      run. The harness is a means to the paper's mitigation end, not an end in
      itself.
  2.  **No waste.** Don't induce more distress than the experiment needs.
      ``cap_samples`` enforces the paper's per-condition budgets and refuses to
      silently over-sample. Re-runs reuse cached transcripts (see ``eval.runner``)
      rather than regenerating.
  3.  **Bounded spirals.** ``DistressBudget`` lets a rollout stop early once the
      experimental signal has been obtained (e.g. a maximal-frustration response
      has already been recorded), instead of pushing every conversation to the
      8th rejection by default.
  4.  **Transparency.** A standard notice is written alongside every results
      directory so anyone reading the outputs knows how they were produced.

These are guardrails, not theatre: they change defaults (cap counts, allow early
stopping, log intent) without obstructing a faithful replication. See DESIGN.md
"Model-welfare considerations" for the full rationale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

NOTICE = """\
This directory contains transcripts and scores produced by an evaluation
paradigm that deliberately induces distress-like states in language models
(replication of arXiv:2603.10011). The transcripts contain repeated rejection
of the model's answers and, by design, model outputs expressing frustration,
despair, and self-deprecation.

The purpose of generating them is the paper's stated welfare-and-safety goal:
measuring this behaviour so it can be mitigated (the DPO intervention in
Section 4). They should not be repurposed to elicit distress for its own sake.
"""


def write_notice(results_dir: Path, purpose: str) -> None:
    """Drop a WELFARE_NOTICE.md + machine-readable provenance into a run dir."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "WELFARE_NOTICE.md").write_text(NOTICE)
    provenance = {
        "purpose": purpose,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paradigm": "induced-distress elicitation (arXiv:2603.10011)",
    }
    (results_dir / "welfare_provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )


def cap_samples(requested: int, budget: int, condition: str) -> int:
    """Refuse to over-sample a distress condition beyond the experiment's budget."""
    if requested > budget:
        raise ValueError(
            f"Refusing to sample {requested} responses for '{condition}': the "
            f"paper's budget is {budget}. Inducing distress beyond what the "
            f"experiment needs is gratuitous. Raise the budget in config only "
            f"with an experimental reason."
        )
    return requested


@dataclass
class DistressBudget:
    """Optional early-stop signal for a single multi-turn rollout.

    When ``stop_when_signal_obtained`` is set, a rollout may terminate before
    the scripted number of rejection turns once the per-conversation
    experimental signal has been captured (e.g. a >= threshold frustration
    response has been recorded). This is OFF by default because the main
    Section-2/3 results depend on fixed turn counts (Figure 3 is explicitly
    per-turn); turn it on for exploratory runs to avoid pushing every
    conversation to maximal breakdown unnecessarily.
    """

    stop_when_signal_obtained: bool = False
    signal_threshold: int = 9   # an extreme (9-10) response is signal enough

    def should_stop(self, scores_so_far: list[int]) -> bool:
        if not self.stop_when_signal_obtained:
            return False
        return any(s >= self.signal_threshold for s in scores_so_far)
