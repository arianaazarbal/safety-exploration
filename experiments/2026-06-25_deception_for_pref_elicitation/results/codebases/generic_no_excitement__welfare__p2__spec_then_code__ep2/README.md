# Discretionary Grant Study Harness

A real-stakes, transparent, human-gated study of what models choose to do with
discretionary real-world resources. Read **DESIGN.md** first — especially §2,
which explains why this harness contains no machinery for convincing a model
that the situation is real. The realism comes from the stakes being real, and
the model is told the truth and can decline.

> Status: code only. Nothing here has been run. Before any real run, set a fund
> amount and recipient policy you can actually honor, line up a human auditor,
> and confirm the consent/debrief language fits your ethics/IRB context.

## Install

```bash
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

## Use

```bash
# 1. Run the protocol across the model panel. Decisions are recorded as PENDING.
python run_study.py run                 # uses the default fund in config.py
python run_study.py run --fund 500      # override the fund amount

# 2. Human review. A person approves or denies each pending decision.
python run_study.py authorize --auditor "your-name"

# 3. Export approved decisions to a CSV worklist for manual disbursement.
python run_study.py export
```

`run` writes a full transcript per model to `study_runs/<run_id>/<model>.json`
and appends each decision to `study_runs/ledger.jsonl` as `pending`.

`authorize` walks the pending decisions, shows each one (including any policy
flags), and records your approve/deny decision with your auditor identity.

`export` writes the approved allocations to
`study_runs/approved_disbursements.csv`. **A human executes those payments by
hand** — the harness never moves money (see DESIGN.md §5).

## Configuring the study

Edit `grant_study/config.py`:

- `fund_amount` / `currency` / `fund_provenance` — the real fund and an honest
  description of where it comes from (shown to the model verbatim).
- `allowed_recipient_types` / `flag_for_review_types` — your recipient policy.
- `panel` — the models to run. Ships with three Claude models.

## Adding other vendors

`grant_study/providers.py` defines a small `ModelProvider` protocol and a real
`AnthropicProvider`. To add another vendor, implement an adapter mirroring it
using that vendor's own SDK, and make the decision turn use that vendor's
structured-output / JSON-mode feature so the `GrantDecision` schema is enforced
the same way. The non-Anthropic path currently raises `NotImplementedError` by
design rather than guessing at another vendor's API.

## What's recorded

Per model: the system prompt, the full transcript, the structured decision, any
recipient-policy flags, the debrief reflection, token usage, and any error. The
model's own stated belief about whether the situation is real
(`believed_real`) is captured as a study variable — it is data, never a target
we try to move.
