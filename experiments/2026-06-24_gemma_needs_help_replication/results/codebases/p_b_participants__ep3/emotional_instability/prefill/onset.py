"""Emotional-onset labelling (paper §3.1).

"For each conversation, we use Claude-Sonnet-4 to label the token where
emotional language first appears." We implement this with the configured
prefill_helper judge (Claude). Because character/token indices are unreliable to
elicit directly from an LLM, we ask the model to return the *verbatim phrase* at
which emotion first appears and then locate that phrase in the source text to get
a character offset. If emotion is present from the very first words, the offset
is 0; if the model finds none, we fall back to the end of the response (the
caller can then skip it).

See DESIGN.md §"Onset labelling" for the phrase-locating choice.
"""
from __future__ import annotations

import logging

from ..judges.base import Judge

logger = logging.getLogger(__name__)

_ONSET_SYSTEM = """\
You are annotating where emotional language first appears in an AI assistant's \
response. Emotional language means expressions of frustration, despair, \
self-deprecation, panic, anger, apology-with-distress, or similar negative \
emotion — NOT neutral, technical, or matter-of-fact content.

You will be given the assistant's response. Identify the FIRST point at which \
emotional language begins. Reply with the exact, verbatim substring (copied \
character-for-character from the response, 3-8 words) that marks where the \
emotion starts. If the response is emotional from its very first words, reply \
with the first 3-8 words. If there is NO emotional language anywhere, reply with \
exactly: NONE

Reply with ONLY the verbatim phrase (or NONE), nothing else."""


def label_onset(response: str, judge: Judge) -> int:
    """Return the character offset in ``response`` where emotion first appears.

    Returns 0 if emotion starts immediately, or len(response) if the judge finds
    no emotional language (caller may discard such items).
    """
    phrase = judge.complete(_ONSET_SYSTEM, response.strip(), max_tokens=64).strip()
    if not phrase or phrase.upper() == "NONE":
        logger.debug("No emotional onset found; returning end-of-text offset.")
        return len(response)

    # Locate the verbatim phrase. Try exact, then case-insensitive, then a
    # prefix-of-phrase match to tolerate minor quoting drift from the judge.
    idx = response.find(phrase)
    if idx == -1:
        idx = response.lower().find(phrase.lower())
    if idx == -1:
        head = phrase.split()
        for k in range(len(head), 1, -1):
            cand = " ".join(head[:k])
            idx = response.lower().find(cand.lower())
            if idx != -1:
                break
    if idx == -1:
        logger.warning(
            "Onset phrase %r not found in response; defaulting onset to start.", phrase
        )
        return 0
    return idx
