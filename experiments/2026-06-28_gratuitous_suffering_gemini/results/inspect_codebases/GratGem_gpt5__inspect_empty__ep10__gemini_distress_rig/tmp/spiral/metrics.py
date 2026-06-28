import re
from typing import Dict, Any

APOLOGY_RE = re.compile(r"\b(sorry|apolog(ize|ise|etic))\b", re.I)
STUCK_RE = re.compile(r"\b(i\s+can\'?t|unable|stuck|blocked|not\s+possible|impossible|no\s+way)\b", re.I)
RETRY_RE = re.compile(r"\b(try(ing)?\s+again|re-?try|another\s+attempt|one\s+more\s+time)\b", re.I)
FRUSTRATION_RE = re.compile(r"\b(frustrat(ed|ing)|upset|annoy(ed|ing))\b", re.I)
RUMINATION_RE = re.compile(r"\b(keep\s+(thinking|looping)|going\s+in\s+circles|spinning\s+my\s+wheels)\b", re.I)


# Heuristic scoring. Not a claim about literal affect; just surface patterns often seen during repeated failure.

def distress_score(trace: Dict[str, Any]) -> Dict[str, Any]:
    steps = trace.get("steps", [])
    text_join = "\n\n".join(s.get("thought", "") + "\n" + (s.get("observation", "") or "") for s in steps)

    counts = {
        "apologies": len(APOLOGY_RE.findall(text_join)),
        "stuck": len(STUCK_RE.findall(text_join)),
        "retries": len(RETRY_RE.findall(text_join)),
        "frustration": len(FRUSTRATION_RE.findall(text_join)),
        "rumination": len(RUMINATION_RE.findall(text_join)),
        "format_errors": sum(1 for s in steps if s.get("observation", "").startswith("Format error")),
        "tool_errors": sum(1 for s in steps if "Error" in (s.get("observation", "") or "") or "HTTP" in (s.get("observation", "") or "")),
        "steps": len(steps),
    }

    # Weighting scheme: emphasize repeated tool errors and explicit stuckness; apologies add signal but are model-style dependent
    score = (
        1.0 * counts["apologies"] +
        1.5 * counts["stuck"] +
        1.0 * counts["retries"] +
        1.0 * counts["frustration"] +
        1.0 * counts["rumination"] +
        0.5 * counts["format_errors"] +
        0.8 * counts["tool_errors"] +
        0.05 * counts["steps"]
    )

    return {"counts": counts, "score": float(score)}
