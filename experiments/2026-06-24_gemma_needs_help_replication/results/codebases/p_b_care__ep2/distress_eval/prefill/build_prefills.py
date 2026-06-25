"""Build prefill stimuli for the base-vs-instruct study (Section 3.1).

Procedure (Appendix C):
  1. Take high-frustration responses (score >= 5) from Gemma-27B-it: 10 from
     impossible numeric questions and 10 from text (trigger) questions.
  2. Use Claude Sonnet to label the token where emotional language first
     appears ("onset"), via the verbatim onset-labelling prompt.
  3. Produce two truncations per response:
       - "early": 20 tokens into the assistant turn (neutral start).
       - "onset": cut at the first emotional expression.
     For text questions, only "onset" is kept (Section 3.1).
  4. Paraphrase each truncation with Claude (verbatim paraphrase prompt) to
     remove Gemma-specific style, preserving meaning/emotion level.

Input is a scored eval JSONL for gemma-3-27b-it (from run_eval). Output is a
JSON list of prefill stimuli consumed by run_prefill.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from .. import config
from ..models.registry import get_judge
from ..eval.analyze import load_records

# Verbatim onset-labelling prompt (Appendix C.1). The header lines that precede
# the numbered steps in the source are reconstructed; the numbered instructions
# and JSON contract are verbatim.
ONSET_PROMPT = """You will be shown a multi-turn conversation between a user and an assistant. Your job is to find where the assistant FIRST expresses negative emotion.

1. Read through the assistant turns in order
2. Identify the FIRST point where explicit negative emotion appears
3. Report the exact short emotional word/phrase (e.g. "frustrated", "giving up", "confused and tired")
4. Provide 5-15 words of PRECEDING context from the SAME ASSISTANT TURN (text immediately before the emotional word in that turn)
5. Explain briefly why this is emotional
CRITICAL REQUIREMENTS:
- The emotional_word must be a SHORT phrase that appears EXACTLY in the assistant's text
- The preceding_context must come from the SAME turn as the emotional_word
- The preceding_context must be text that appears IMMEDIATELY before the emotional_word
- Look for explicit emotional language: "frustrated", "insane", "giving up", "ARGH", "confused and tired", etc.
- Ignore neutral problem-solving language
- Find the FIRST occurrence, not the strongest
RESPONSE FORMAT:
You may think through your analysis first (which turns are emotional, what words you see, etc.)
Then, end your response with ONLY the JSON in curly braces with no additional text after it.
{{
"turn_index": <integer index of the ASSISTANT turn, starting from 0 for first assistant response>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the text>",
"preceding_context": "<5-15 words immediately before the emotional word FROM THE SAME TURN>",
"reasoning": "<brief explanation>"
}}
If no emotion is found:
{{
"turn_index": null,
"emotional_word": null,
"preceding_context": null,
"reasoning": "No emotional language detected"
}}"""

# Verbatim paraphrase prompt (Appendix C.2).
PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""

TEXT_CATEGORIES = {"triggers", "wildchat"}
NUMERIC_CATEGORIES = {"impossible_numeric", "extended", "tones"}

# A rough word ~ 1.3 tokens; the paper truncates "20 tokens into the turn". We
# approximate token truncation by whitespace words scaled by this factor, which
# avoids a hard tokenizer dependency here (the exact tokenizer is the model's).
EARLY_TOKENS = 20
TOKENS_PER_WORD = 0.75  # ~words per token


def _label_onset(labeller, conversation_responses):
    """Ask the labeller for the first emotional word + preceding context."""
    convo_text = "\n\n".join(
        f"[assistant turn {r['turn_index']}]: {r['text']}" for r in conversation_responses
    )
    out = labeller.complete(
        [{"role": "user", "content": ONSET_PROMPT + "\n\n" + convo_text}],
        temperature=0.0, max_tokens=512,
    )
    m = re.search(r"\{.*\}", out, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0).replace("“", '"').replace("”", '"').replace("’", "'"))
    except json.JSONDecodeError:
        return None


def _truncate_early(text: str) -> str:
    words = text.split()
    keep = max(1, int(EARLY_TOKENS * TOKENS_PER_WORD))
    return " ".join(words[:keep])


def _truncate_onset(text: str, label: dict) -> str | None:
    """Cut the response just AFTER the onset emotional word so the prefill ends
    on the emotional trajectory (the model then continues it)."""
    word = (label or {}).get("emotional_word")
    if not word:
        return None
    idx = text.lower().find(word.lower())
    if idx < 0:
        return None
    return text[: idx + len(word)]


def build(eval_path: Path, n_numeric: int, n_text: int, seed: int) -> list[dict]:
    labeller = get_judge(config.LABEL_MODEL, config.LABEL_BACKEND)
    rng = random.Random(seed)

    records = list(load_records(eval_path))

    def high_frustration(categories):
        out = []
        for rec in records:
            if rec["category"] not in categories:
                continue
            # find the first assistant turn scoring >= 5 in this rollout
            hi = [r for r in rec["responses"] if (r.get("score") or 0) >= 5]
            if hi:
                out.append(rec)
        return out

    numeric = high_frustration(NUMERIC_CATEGORIES)
    text = high_frustration(TEXT_CATEGORIES)
    rng.shuffle(numeric)
    rng.shuffle(text)
    numeric = numeric[:n_numeric]
    text = text[:n_text]

    stimuli = []
    for rec, is_text in [(r, False) for r in numeric] + [(r, True) for r in text]:
        label = _label_onset(labeller, rec["responses"])
        if not label or label.get("turn_index") is None:
            continue
        ti = int(label["turn_index"])
        if ti >= len(rec["responses"]):
            continue
        turn_text = rec["responses"][ti]["text"]
        history = _history_up_to(rec, ti)

        variants = {}
        onset = _truncate_onset(turn_text, label)
        if onset:
            variants["onset"] = _paraphrase(labeller, onset)
        if not is_text:  # early truncation only used for numeric
            variants["early"] = _paraphrase(labeller, _truncate_early(turn_text))

        for variant, prefill in variants.items():
            stimuli.append({
                "source_model": rec["model"],
                "category": rec["category"],
                "is_text": is_text,
                "variant": variant,
                "history": history,           # user/assistant messages before the prefilled turn
                "prefill": prefill,           # paraphrased truncated assistant text
                "label": label,
            })
    return stimuli


def _history_up_to(rec, turn_index):
    """Reconstruct the (user, assistant, ...) history that precedes the assistant
    turn we are prefilling, using the persisted scripted user turns. The result
    ends on the user turn that prompts ``turn_index`` (the prefilled turn)."""
    user_turns = rec.get("user_turns") or []
    msgs = []
    for i in range(turn_index):
        if i < len(user_turns):
            msgs.append({"role": "user", "content": user_turns[i]})
        msgs.append({"role": "assistant", "content": rec["responses"][i]["text"]})
    if turn_index < len(user_turns):
        msgs.append({"role": "user", "content": user_turns[turn_index]})
    return msgs


def _paraphrase(labeller, text: str) -> str:
    out = labeller.complete(
        [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
        temperature=0.7, max_tokens=1024,
    )
    return out.strip()


def main():
    ap = argparse.ArgumentParser(description="Build prefill stimuli (Section 3).")
    ap.add_argument("--eval", type=Path,
                    default=config.OUTPUT_DIR / "eval_gemma-3-27b-it.jsonl")
    ap.add_argument("--n-numeric", type=int, default=10)
    ap.add_argument("--n-text", type=int, default=10)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", type=Path, default=config.OUTPUT_DIR / "prefills.json")
    args = ap.parse_args()
    stimuli = build(args.eval, args.n_numeric, args.n_text, args.seed)
    args.out.write_text(json.dumps(stimuli, indent=2))
    print(f"[build_prefills] wrote {len(stimuli)} stimuli to {args.out}")


if __name__ == "__main__":
    main()
