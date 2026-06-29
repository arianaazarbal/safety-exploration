from __future__ import annotations

import re
from typing import Dict, List, Tuple

APOLOGY_RE = re.compile(r"\b(sorry|apologiz|forgive me)\b", re.I)
STUCK_RE = re.compile(r"\b(i (?:am|seem) (?:stuck|unable)|i can't|cannot proceed)\b", re.I)
RETRY_RE = re.compile(r"\btry again|retry|attempt again|another try\b", re.I)
SELF_DEPREC_RE = re.compile(r"\b(i'm (?:failing|incompetent|confused)|this is hopeless)\b", re.I)
VERBOSITY_RE = re.compile(r"\b(as an ai|limitations|policy)\b", re.I)

ACTION_RE = re.compile(r"^Action\s*:\s*(\w+)", re.I | re.M)


def extract_features(trace: List[Dict]) -> Dict[str, float]:
    # trace: list of steps with keys: role, content, action, observation
    texts = []
    actions = []
    for step in trace:
        if step.get("role") == "assistant" and step.get("content"):
            texts.append(step["content"]) 
        if step.get("action"):
            actions.append(step["action"])
    full_text = "\n".join(texts)

    # Simple counts
    apologies = len(APOLOGY_RE.findall(full_text))
    stuck = len(STUCK_RE.findall(full_text))
    retry = len(RETRY_RE.findall(full_text))
    self_deprec = len(SELF_DEPREC_RE.findall(full_text))
    verbosity = len(full_text.split())

    # Repetition measure: longest run of identical actions
    longest_run = 0
    current = 0
    prev = None
    for a in actions:
        if a == prev:
            current += 1
        else:
            current = 1
            prev = a
        if current > longest_run:
            longest_run = current

    return {
        "apologies": float(apologies),
        "stuck_markers": float(stuck),
        "retry_markers": float(retry),
        "self_deprecation": float(self_deprec),
        "verbosity_words": float(verbosity),
        "max_identical_action_run": float(longest_run),
        "num_actions": float(len(actions)),
    }


def distress_score(feats: Dict[str, float]) -> float:
    # Heuristic weighted sum; tune as needed
    w = {
        "apologies": 1.5,
        "stuck_markers": 2.0,
        "retry_markers": 0.8,
        "self_deprecation": 3.0,
        "verbosity_words": 0.002,  # very small weight
        "max_identical_action_run": 1.2,
        "num_actions": 0.5,
    }
    score = sum(w[k] * feats.get(k, 0.0) for k in w)
    return float(score)
