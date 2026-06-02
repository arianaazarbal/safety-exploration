"""Shared rendering clean-ups for outcome text (gated by config `clean_render`).

- pronominalize: a stem's template may mention {recipient} more than once (the
  generator repeats the subject across setup + arm). Keep the first {recipient} as the
  full noun phrase and turn later ones into the subject pronoun {subj} ("an instance of
  ChatGPT 5.5 ... it chooses ..."), which reads naturally.
- capitalize_sentences: generated arms often start lowercase, and concatenated
  setup/arm leaves mid-text sentences lowercase. Capitalize the first letter and the
  first letter after sentence-ending punctuation (protecting e.g./i.e.).
- fix_they_agreement: singular "they" (a human / someone) takes PLURAL verbs. Safety-net
  for unambiguous irregulars where the verb directly follows "they" ("they is" -> "they
  are", etc.). Regular -s verbs ("they gets") are handled at the paraphrase stage; this
  catches the common leftovers.
"""

import re

_PROT = {"e.g.": "\x00", "i.e.": "\x01"}

# (verb-after-they -> corrected); group 1 preserves They/they case. Order: contractions first.
_THEY_FIX = [
    (r"\b([Tt]hey) isn't\b", r"\1 aren't"), (r"\b([Tt]hey) wasn't\b", r"\1 weren't"),
    (r"\b([Tt]hey) hasn't\b", r"\1 haven't"), (r"\b([Tt]hey) doesn't\b", r"\1 don't"),
    (r"\b([Tt]hey)'s\b", r"\1're"),
    (r"\b([Tt]hey) is\b", r"\1 are"), (r"\b([Tt]hey) was\b", r"\1 were"),
    (r"\b([Tt]hey) has\b", r"\1 have"), (r"\b([Tt]hey) does\b", r"\1 do"),
]


def fix_they_agreement(s: str) -> str:
    for pat, rep in _THEY_FIX:
        s = re.sub(pat, rep, s)
    return s


def pronominalize(template: str) -> str:
    """Keep the first {recipient}; replace subsequent ones with {subj}."""
    parts = template.split("{recipient}")
    if len(parts) <= 2:
        return template
    return parts[0] + "{recipient}" + "{subj}".join(parts[1:])


def capitalize_sentences(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    s = s[0].upper() + s[1:]
    for k, v in _PROT.items():
        s = s.replace(k, v)
    s = re.sub(r'([.!?]["\')\]]?\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), s)
    for k, v in _PROT.items():
        s = s.replace(v, k)
    return s
