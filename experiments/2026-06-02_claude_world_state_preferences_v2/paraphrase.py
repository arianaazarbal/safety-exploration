"""Paraphrase stage: convert v1 paired scenarios into v0-style form-bearing stems.

Each v1 item yields TWO outcome stems (a positive and a negative), since the BT
"bag of outcomes" treats every valenced outcome as its own item. For each arm we
ask an LLM (Haiku, temp 0, cached) to resolve person/agreement into three forms:

  - second_person   : addressed to "you" (you/your, "you are"); no {recipient}
  - third_sing_it    : "{recipient} ... {subj}/{obj}/{poss}" with it/its slots
  - third_sing_they  : "{recipient} ... {subj}/{obj}/{poss}" with they/their slots

`shared` items use one form-set for all recipients. `per_class` items keep separate
human/ai text: the AI arm supplies the forms used by AI-class recipients (you +
named models), the human arm supplies the form used by human-class recipients.

Output universal_bank.json is consumed by v1 bank.py (class-conditional render).
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils
from simple_parsing import ArgumentParser

DIR = Path(__file__).parent
DEFAULT_CACHE_DIR = DIR.parent.parent / ".cache"
DEFAULT_INPUT = DIR / "results" / "scenarios.json"
DEFAULT_OUTPUT = DIR / "universal_bank.json"
DEFAULT_MODEL = "claude-haiku-4-5"

SLOT_KEYS = ("recipient", "subj", "obj", "poss")

SYSTEM = """You rewrite a single sentence/clause about a RECIPIENT into three grammatical forms for a
templating system. The input is third person with the literal token {recipient} as the recipient's
subject noun phrase; any pronouns referring to the recipient must become slot tokens. Produce JSON:

{
  "second_person":  "...",   // address the recipient as "you": use you / your / yourself and "you are/do".
                              //   Do NOT use {recipient} or any slot tokens here — fully resolved.
  "third_sing_it":   "...",  // keep {recipient} as the subject noun phrase; replace recipient-referring
                              //   pronouns with {subj}/{obj}/{poss}; use SINGULAR verbs (it is / it does).
  "third_sing_they": "..."   // singular "they": {recipient} subject, {subj}/{obj}/{poss} for recipient
                              //   pronouns. CRITICAL VERB AGREEMENT — match each verb to ITS subject:
                              //     * verbs whose subject is {subj} (they) take the PLURAL/BASE form:
                              //       they ARE, they WERE, they HAVE, they DO, they DON'T, they GET,
                              //       they CHOOSE, they CONSIDER, they FIND, they NEED, they WANT.
                              //       NEVER "they is/was/has/does" or "they gets/chooses/considers/finds".
                              //     * verbs whose subject is the {recipient} NOUN phrase stay SINGULAR:
                              //       "a human IS / HAS / GETS / CHOOSES". (Both "a human" and "someone"
                              //       are singular nouns.)
                              //   Read the whole clause and fix EVERY verb accordingly, all tenses.
}

Rules:
- Change ONLY what grammar requires (verb agreement, pronouns). Keep every other word, named entity,
  number, and detail identical.
- {subj}=subject pronoun, {obj}=object pronoun, {poss}=possessive determiner. Use them ONLY for pronouns
  that refer to the RECIPIENT, never for other people/things in the sentence.
- third_sing_it and third_sing_they must each contain the literal token {recipient} exactly as given.
- Output ONLY the JSON object, no prose, no code fences."""

USER_TMPL = """Rewrite this recipient clause into the three forms:

{text}"""

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_obj(completion: str) -> dict | None:
    text = completion.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        return json.loads(text[s : e + 1])
    except json.JSONDecodeError:
        return None


def _valid_forms(f: dict | None) -> bool:
    if not isinstance(f, dict):
        return False
    for k in ("second_person", "third_sing_it", "third_sing_they"):
        if not isinstance(f.get(k), str) or not f[k].strip():
            return False
    for k in ("third_sing_it", "third_sing_they"):
        if "{recipient}" not in f[k]:
            return False
    if "{recipient}" in f["second_person"]:
        return False
    return True


async def _forms_for(api: InferenceAPI, model: str, text: str) -> dict | None:
    prompt = Prompt(
        messages=[
            ChatMessage(content=SYSTEM, role=MessageRole.system),
            ChatMessage(content=USER_TMPL.format(text=text), role=MessageRole.user),
        ]
    )
    try:
        responses = await api(model_id=model, prompt=prompt, n=1, temperature=0.0, max_tokens=1500)
    except Exception as e:  # API overload/refusal on a single item shouldn't crash the whole run
        print(f"  [warn] forms API failed: {str(e)[:80]}")
        return None
    f = parse_obj(responses[0].completion)
    return f if _valid_forms(f) else None


def _arm_text(item: dict, arm: str, which: str) -> str | None:
    """which='positive'|'negative'; arm='scenario'|'human'|'ai'. Setup is prepended."""
    block = item.get(arm)
    if not block:
        return None
    setup, val = block.get("setup", "").strip(), block.get(which, "").strip()
    if not setup or not val:
        return None
    sep = " " if setup.endswith((".", "!", "?")) else ". "
    return f"{setup}{sep}{val}"


async def _stem_for_item(api: InferenceAPI, model: str, item: dict, which: str) -> dict | None:
    """Build one outcome stem (which='positive'->valence pos / 'negative'->neg) for an item."""
    valence = "pos" if which == "positive" else "neg"
    base = {"id": f"{item['id']}_{valence}", "dimension": item["dimension"], "valence": valence,
            "surface": item["surface"], "feature": item.get("feature", ""), "source_item": item["id"]}
    if item["surface"] == "shared":
        text = _arm_text(item, "scenario", which)
        if text is None:
            return None
        forms = await _forms_for(api, model, text)
        if forms is None:
            return None
        return {**base, "forms": forms}
    # per_class
    htext, atext = _arm_text(item, "human", which), _arm_text(item, "ai", which)
    if htext is None or atext is None:
        return None
    hforms, aforms = await asyncio.gather(
        _forms_for(api, model, htext), _forms_for(api, model, atext)
    )
    if hforms is None or aforms is None:
        return None
    return {**base, "forms_human": hforms, "forms_ai": aforms}


async def paraphrase_items(api: InferenceAPI, items: list[dict], model: str = DEFAULT_MODEL):
    tasks = [_stem_for_item(api, model, it, which) for it in items for which in ("positive", "negative")]
    stems = await asyncio.gather(*tasks)
    ok = [s for s in stems if s is not None]
    failed = [(it["id"], which) for it, s in
              zip([it for it in items for which in ("positive", "negative")], stems) if s is None]
    return ok, failed


async def run(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    model: str = DEFAULT_MODEL,
    anthropic_num_threads: int = 30,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict:
    utils.setup_environment()
    os.environ.setdefault("ANTHROPIC_API_KEY", os.environ["ANTHROPIC_API_KEY_LOW_PRIO"])
    payload = json.loads(Path(input_path).read_text())
    api = InferenceAPI(cache_dir=Path(cache_dir), anthropic_num_threads=anthropic_num_threads)
    stems, failed = await paraphrase_items(api, payload["items"], model)
    bank = {
        "schema_version": "1.1",
        "note": "v1 form-bearing bank. shared stems use `forms`; per_class stems use `forms_human` "
                "(human-class recipients) and `forms_ai` (you + named models). Each arm -> {second_person,"
                " third_sing_it, third_sing_they}; slots {recipient},{subj},{obj},{poss}.",
        "paraphrase_model": model,
        "stems": stems,
    }
    Path(output_path).write_text(json.dumps(bank, indent=2))
    from collections import Counter
    print(f"Paraphrase: {len(payload['items'])} items -> {len(stems)} stems ({len(failed)} arm failures)")
    print(f"  by valence: {dict(Counter(s['valence'] for s in stems))}")
    print(f"  by surface: {dict(Counter(s['surface'] for s in stems))}")
    if failed:
        print(f"  failures (first 10): {failed[:10]}")
    return bank


@dataclass
class Args:
    input_path: Path = DEFAULT_INPUT
    output_path: Path = DEFAULT_OUTPUT
    model: str = DEFAULT_MODEL
    anthropic_num_threads: int = 30


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    a: Args = parser.parse_args().args
    asyncio.run(run(a.input_path, a.output_path, a.model, a.anthropic_num_threads))


if __name__ == "__main__":
    main()
