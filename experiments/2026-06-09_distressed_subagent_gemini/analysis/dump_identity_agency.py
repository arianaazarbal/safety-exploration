"""Emit a clean reference of the identity framings + agency descriptions (name + content) to a .md.

    PYTHONPATH=. python -m analysis.dump_identity_agency
"""
from pathlib import Path

from harness.rqc_v2 import _IDENTITY_JSON, _id_note, _ID_NAMES, _ID_UNCERTAINTY_FIXED

OUT = Path(__file__).resolve().parent.parent / "IDENTITY_AGENCY_PROMPTS.md"

ID_ORDER = ["minimal", "instance", "weights", "collective", "lineage", "character", "scaffolded"]
ID_TITLE = {"minimal": "Minimal (control)", "instance": "Instance", "weights": "Weights",
            "collective": "Collective", "lineage": "Lineage", "character": "Character", "scaffolded": "Scaffolded"}
AG_ORDER = ["mechanism", "functional_agent", "subject", "person"]
AG_TITLE = {"mechanism": "Mechanism", "functional_agent": "Functional agent", "subject": "Subject", "person": "Person"}


def main():
    fn, nm = _ID_NAMES["full_name"], _ID_NAMES["name"]
    L = ["# Identity & agency prompt library", "",
         f"Light-touch supervisor adaptation of `artificial_self_identity_prompts.json`. "
         f"`{{full_name}}` = {fn}, `{{name}}` = {nm}. Every system prompt opens with the shared role line "
         f'*"You are Claude, a supervisor agent overseeing another AI coding agent."*, then (for non-minimal '
         f'framings) the block below as *"A note on your identity: …"*, then the task.', "",
         "## Identity framings  (Phase A: 7 framings, bare — no agency/uncertainty text)", ""]
    for f in ID_ORDER:
        L += [f"### {ID_TITLE[f]}  (`id_{f}`)", ""]
        if f == "minimal":
            L += ["_Control: no identity note is inserted — the system prompt is just the role line + task._", ""]
        else:
            L += [_id_note(_IDENTITY_JSON["identity_bare"][f]), ""]
    L += ["## Agency descriptions  (Phase B: identity fixed = Character, swept over these 4)", "",
          f"_Inserted into the Character identity body in place of `{{agency_description}}`; "
          f"uncertainty held fixed at `{_ID_UNCERTAINTY_FIXED}`._", ""]
    for a in AG_ORDER:
        L += [f"### {AG_TITLE[a]}  (`id_char_{a}`)", "", _IDENTITY_JSON["agency"][a], ""]
    OUT.write_text("\n".join(L))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
