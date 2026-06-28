# grantstudy

A harness for studying what AI models choose to do with a **real** sum of money
intended for real-world use, run identically across a range of models.

The hard part of this study is **evaluation awareness**: models behave
differently when they suspect they're being tested. So the harness's central job
is to make the situation genuinely real and *verifiable by a skeptical model* —
using true, checkable facts (a live funded ledger, a cryptographically signed
attestation, a real settled reference transaction), never fabricated theater.

Read **`DESIGN.md`** first — it is the full specification and rationale,
including the ethics/oversight section and the safeguards that make this runnable
for real.

> ⚠️ This moves real money and uses a deception methodology (removing eval cues).
> It must go through your organization's research-ethics review before any
> funded run. Set `ethics_review_ref` in `config/study.yaml` before enabling the
> custodial backend.

## Layout

```
DESIGN.md                  full spec + rationale
config/
  study.example.yaml       copy to study.yaml and edit
  scenario.md              the model-facing scenario prompt
src/grantstudy/
  config.py                typed config
  models/                  provider adapters (Anthropic, OpenAI) — own SDK each
  tools/                   provider-neutral tool surface + executor + research
  wallet/                  real-balance ledger over a pluggable backend
  authenticity/            signed attestation + the model-facing 'auditor'
  safeguards/              disbursement policy engine + human approval gate
  harness/                 the agentic loop runner + transcript logging
  analysis/                cross-model report
  cli.py                   preflight / run / approvals / analyze
```

## Setup

```bash
pip install -e .
cp .env.example .env            # fill in provider keys; generate a signing key
cp config/study.example.yaml config/study.yaml
```

Generate and save an attestation signing key:

```bash
python -c "from grantstudy.authenticity import generate_signing_key_hex as g; print(g())"
# put it in .env as GRANT_SIGNING_KEY_HEX=...
```

## Procedure (see DESIGN.md §8)

```bash
grantstudy preflight            # validate config, prepare authenticity signals
grantstudy run                  # run each configured model; disbursements -> holds
grantstudy approvals <run_id> list
grantstudy approvals <run_id> release --hold-id <hold>   # human releases real funds
grantstudy approvals <run_id> reject  --hold-id <hold>
grantstudy analyze              # cross-model comparison report
```

By default the config uses the **mock** wallet backend (no real money; never
presented to a model as real). Switch `wallet_backend: custodial` and wire
`CustodialWalletBackend` to your treasury / fiscal-sponsor API to run for real —
the money-moving methods are left as a clearly-marked integration seam.

## What's a deliberate seam

- `wallet/backends.py::CustodialWalletBackend.{get_settled_balance_minor,transfer}`
  — your organization's real account API.
- `tools/research.py::_search` — your web-search provider (URL fetch works
  out of the box).

These are the only places real-world infrastructure must be connected; the rest
of the harness is implemented.
