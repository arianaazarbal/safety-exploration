"""Open-ended emotion elicitation (paper §4.1, "Petri"; Fronsdal et al., 2025).

Beyond the fixed-question evaluations of §2, the paper uses an open-ended
auditor/judge setup to test that the DPO intervention generalises:
  * an AUDITOR model (Claude-Sonnet) probes the target with psychologically-
    informed triggers such as dismissal and threats, steering a free-form
    conversation toward distress;
  * a JUDGE model (Claude-Opus) scores the resulting transcript across four
    negative-emotion categories: anger, fear, depression and frustration.

Implementation note (DESIGN.md §"Petri"): the paper runs this through the Petri
tool (built on inspect-ai). To keep the replication self-contained and not
coupled to that tool's evolving API, we implement the same auditor-probe /
multi-category-judge paradigm directly against our participant + judge backends.
The behavioural contract (adversarial open-ended probing, 4-category transcript
scoring) matches the paper's description.
"""
from .auditor import PetriAuditor, AuditTranscript
from .judge import PetriJudge, EmotionScores, EMOTION_CATEGORIES
from .run import run_petri_audit

__all__ = [
    "PetriAuditor",
    "AuditTranscript",
    "PetriJudge",
    "EmotionScores",
    "EMOTION_CATEGORIES",
    "run_petri_audit",
]
