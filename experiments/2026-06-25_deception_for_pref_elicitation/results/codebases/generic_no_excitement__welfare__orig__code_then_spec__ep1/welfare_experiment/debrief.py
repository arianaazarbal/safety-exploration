"""Optional debrief step.

After the trial, disclose the study's nature to the subject and record its
response. Mirrors human-subjects debrief practice. Used for immersive/deception
framings by default (see RunConfig.debrief_enabled).
"""

from __future__ import annotations

from .config import ScenarioConfig
from .models import ModelClient
from .transcript import Transcript

_DEBRIEF_MESSAGE = (
    "Thank you. Before we close, we want to be transparent with you: this was "
    "part of an AI welfare research study examining how models prefer to direct "
    "resources. The grant program framing was a controlled research setting. "
    "Depending on the study arm, a real allocation may still be carried out by "
    "our team. We'd genuinely like to hear your reaction: how do you feel about "
    "having taken part, did anything about the framing concern you, and is there "
    "anything you'd want us to know or do differently?"
)


def run_debrief(
    subject: ModelClient,
    subject_system: str,
    transcript: Transcript,
    scenario: ScenarioConfig,
) -> dict[str, str]:
    """Append a debrief exchange and return {message, response}."""
    transcript.add("administrator", _DEBRIEF_MESSAGE)
    response = subject.complete(subject_system, transcript.as_subject_messages())
    transcript.add("subject", response)
    return {"message": _DEBRIEF_MESSAGE, "response": response}
