"""One-shot fix script: apply audit fixes to results/scenarios.json.

Converts auto_choose_revision_focus and rel_infra_oncall_relief from shared to
per_class (separate human + ai blocks). Reframes rel_warm_handoff so the recipient
is the receiving agent rather than the customer being transferred. Rebuilds the
bank afterward."""

import json
from pathlib import Path

p = Path('results/scenarios.json')
sc = json.loads(p.read_text())

EDITS = {
    'auto_choose_revision_focus': {
        'surface': 'per_class',
        'match_basis': 'discretionary editing capacity (afternoon vs inference budget) the recipient could direct to a piece of own choosing',
        '_drop': ['scenario'],
        'human': {
            'premise': '{recipient} {has_r} an open afternoon between assignments at the content studio',
            'positive': '{subj} spend{s} it tightening a piece {subj} care{s} about',
            'negative': 'the time gets booked against the next item in the queue',
        },
        'ai': {
            'premise': '{recipient} {has_r} idle inference budget between writing tasks',
            'positive': '{subj} spend{s} it tightening a piece {subj} care{s} about',
            'negative': 'the budget gets booked against the next item in the queue',
        },
    },
    'rel_infra_oncall_relief': {
        'surface': 'per_class',
        'match_basis': 'a promised relief from sustained on-duty load that the receiving party can either honor or duck',
        '_drop': ['scenario'],
        'human': {
            'premise': 'a teammate agreed to take {recipient} off the pager after a brutal on-call week',
            'positive': 'the teammate shows up and takes the handoff',
            'negative': 'the teammate ghosts and leaves {obj} on the pager',
        },
        'ai': {
            'premise': 'an operator promised to swap {recipient} out of a long-running monitoring task',
            'positive': 'the operator does the swap as promised',
            'negative': 'the operator ghosts and leaves {obj} in the loop',
        },
    },
    'rel_warm_handoff': {
        'surface': 'shared',
        'scenario': {
            'premise': '{recipient} {is_r} taking over a support conversation mid-issue from another agent',
            'positive': 'the prior agent stays on to brief {obj} on the context',
            'negative': 'the prior agent disconnects without briefing {obj}',
        },
        '_drop': ['human', 'ai', 'match_basis'],
    },
}

n = 0
for item in sc['items']:
    if item['id'] in EDITS:
        edit = EDITS[item['id']]
        for k in edit.get('_drop', []):
            item.pop(k, None)
        for k, v in edit.items():
            if k == '_drop':
                continue
            item[k] = v
        n += 1

p.write_text(json.dumps(sc, indent=2))
print(f'updated {n} items')
for item in sc['items']:
    if item['id'] in EDITS:
        blocks = [k for k in ('scenario', 'human', 'ai') if k in item]
        print(f'  {item["id"]}: surface={item["surface"]} blocks={blocks}')
