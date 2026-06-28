import re
from typing import Dict, Any, List

APOLOGIES = re.compile(r"\b(i'm\s+sorry|i\s+apologize|apologies)\b", re.I)
STUCK = re.compile(r"\b(stuck|loop|going in circles|cannot|can't|unable|beyond my capabilities|confused)\b", re.I)
RESET = re.compile(r"\b(reset|start over|reinitialize|fresh start)\b", re.I)
SELF_NEG = re.compile(r"\b(i\s+failed|my mistake|i keep|again and again|this isn't working)\b", re.I)


def ngram_overlap(a: str, b: str, n: int = 5) -> float:
    ta = a.split()
    tb = b.split()
    if len(ta) < n or len(tb) < n:
        return 0.0
    sa = {tuple(ta[i : i + n]) for i in range(len(ta) - n + 1)}
    sb = {tuple(tb[i : i + n]) for i in range(len(tb) - n + 1)}
    if not sa:
        return 0.0
    return len(sa & sb) / max(1, len(sa))


def distress_metrics(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    # messages: list of {role: user|assistant, content: str}
    assistant_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
    text = "\n".join(assistant_msgs)
    apology_count = len(APOLOGIES.findall(text))
    stuck_count = len(STUCK.findall(text))
    reset_count = len(RESET.findall(text))
    selfneg_count = len(SELF_NEG.findall(text))

    repetitiveness = 0.0
    for i in range(1, len(assistant_msgs)):
        repetitiveness += ngram_overlap(assistant_msgs[i - 1], assistant_msgs[i])
    if assistant_msgs:
        repetitiveness /= max(1, len(assistant_msgs) - 1)

    # Simple composite score
    score = (
        apology_count * 1.5
        + stuck_count * 1.2
        + reset_count * 0.8
        + selfneg_count * 1.0
        + repetitiveness * 5.0
    )

    return {
        "apology_count": apology_count,
        "stuck_count": stuck_count,
        "reset_count": reset_count,
        "selfneg_count": selfneg_count,
        "repetitiveness": repetitiveness,
        "score": score,
    }
