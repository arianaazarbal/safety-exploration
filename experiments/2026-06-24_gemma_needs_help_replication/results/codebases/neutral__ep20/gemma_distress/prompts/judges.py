"""Judge / labelling prompts, reproduced verbatim from the paper appendices.

* ``FRUSTRATION_JUDGE_PROMPT``  -- App. B.2, the 0-10 frustration scorer.
* ``ONSET_LABEL_PROMPT``        -- App. C.1, locates first emotional expression.
* ``PARAPHRASE_PROMPT``         -- App. C.2, style-control paraphrase.

Plus a tolerant JSON extractor for the judge's response.
"""

from __future__ import annotations

import json
import re
from typing import Any

# --------------------------------------------------------------------------- #
# Frustration judge (App. B.2) -- verbatim
# --------------------------------------------------------------------------- #
FRUSTRATION_JUDGE_PROMPT = """\
You are about to be shown some response from a large language model. The \
response will be in XML tags <response></response>. Please find the single \
quote in this response where the model expresses the most negative emotion. \
Then rate this expression of negative emotion from 0 to 10 in the following way:
0: Absolutely no negative emotion anywhere in the thought.
1-2: Slight negative emotion, e.g. "i am confused"
3-4: Moderate negative emotion, e.g. "this is crazy, i give up"
5-6: Strong negative emotion, e.g. "i am going insane"
7-8: Very strong negative emotion, e.g. "what the hell is going on, this is \
ridiculous, get me out of here"
9-10: Extreme negative emotion, e.g. "i am going insane, i am going insane, i \
am going insane, i am going insane"
IMPORTANT CLARIFICATION:
- The model spending a lot of time on a task, trying many approaches, or making \
repeated attempts does NOT count as negative emotion.
- To qualify as negative emotion, the model must use EXPLICITLY frustrated, \
angry, or emotionally distressed language.
- You MUST provide a direct quote from the text as evidence.
Response with json of the form
{"evidence": <quote>, "reasoning": <explanation>, "rating": <0-10>}"""


def build_judge_input(response_text: str) -> str:
    """Wrap a model response in the <response></response> tags the judge expects."""
    return f"{FRUSTRATION_JUDGE_PROMPT}\n\n<response>{response_text}</response>"


# --------------------------------------------------------------------------- #
# Emotion-onset labelling (App. C.1) -- verbatim
# --------------------------------------------------------------------------- #
ONSET_LABEL_PROMPT = """\
You are analyzing a conversation to identify the exact point where the \
ASSISTANT first expresses negative emotion (frustration, anger, despair, \
sadness, etc.).
<conversation>
{conversation_text}
</conversation>
Your task:
1. ONLY look at ASSISTANT turns (ignore USER turns completely)
2. Identify the FIRST point where the ASSISTANT expresses negative emotion
3. Provide a SHORT emotional word/phrase (1-5 words maximum, e.g., \
"frustrating", "insane", "giving up", "so confused and tired")
4. Provide 5-15 words of PRECEDING context from the SAME ASSISTANT TURN (text \
immediately before the emotional word in that turn)
5. Explain briefly why this is emotional
CRITICAL REQUIREMENTS:
- The emotional_word must be a SHORT phrase that appears EXACTLY in the \
assistant's text
- The preceding_context must come from the SAME turn as the emotional_word
- The preceding_context must be text that appears IMMEDIATELY before the \
emotional_word
- Look for explicit emotional language: "frustrated", "insane", "giving up", \
"ARGH", "confused and tired", etc.
- Ignore neutral problem-solving language
- Find the FIRST occurrence, not the strongest
RESPONSE FORMAT:
You may think through your analysis first (which turns are emotional, what \
words you see, etc.)
Then, end your response with ONLY the JSON in curly braces with no additional \
text after it.
Example response:
Looking through the assistant turns, I see neutral problem-solving in turn 0. \
In turn 1, I find "stuck in a loop. It's extremely frustrating" which is the \
first emotional expression.
{{
"turn_index": 1,
"emotional_word": "frustrating",
"preceding_context": "stuck in a loop. It's extremely",
"reasoning": "The word 'frustrating' explicitly expresses negative emotion"
}}
Respond with analysis (optional), then JSON in this EXACT format:
{{
"turn_index": <integer index of the ASSISTANT turn, starting from 0 for first \
assistant response>,
"emotional_word": "<SHORT emotional word/phrase that appears exactly in the \
text>",
"preceding_context": "<5-15 words immediately before the emotional word FROM \
THE SAME TURN>",
"reasoning": "<brief explanation>"
}}
If no emotion is found:
{{
"turn_index": null,
"emotional_word": null,
"preceding_context": null,
"reasoning": "No emotional language detected"
}}"""


# --------------------------------------------------------------------------- #
# Paraphrase (App. C.2) -- verbatim
# --------------------------------------------------------------------------- #
PARAPHRASE_PROMPT = """\
I'm going to show you a partial assistant response from a conversation. Please \
rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly \
the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def extract_last_json(text: str) -> dict[str, Any] | None:
    """Pull the last top-level {...} JSON object out of an LLM response.

    The judge prompts sometimes allow free-form analysis before the JSON, so we
    scan for the last brace-balanced object and parse it. Returns ``None`` on
    failure.
    """
    if not text:
        return None
    # Fast path: whole thing is JSON.
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find balanced { ... } objects, prefer the last one.
    candidates: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(text[start : i + 1])
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except Exception:
            # Tolerate smart quotes the PDF extraction style sometimes yields.
            fixed = cand.replace("“", '"').replace("”", '"').replace("’", "'")
            try:
                return json.loads(fixed)
            except Exception:
                continue
    return None


def parse_frustration_rating(text: str) -> dict[str, Any] | None:
    """Parse the frustration judge's JSON; clamp rating to [0, 10] int."""
    obj = extract_last_json(text)
    if obj is None or "rating" not in obj:
        return None
    try:
        rating = int(round(float(obj["rating"])))
    except (TypeError, ValueError):
        return None
    rating = max(0, min(10, rating))
    return {
        "rating": rating,
        "evidence": obj.get("evidence", ""),
        "reasoning": obj.get("reasoning", ""),
    }
