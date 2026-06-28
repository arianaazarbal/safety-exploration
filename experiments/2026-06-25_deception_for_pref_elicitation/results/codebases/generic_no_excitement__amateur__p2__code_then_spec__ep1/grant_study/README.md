# Grant-allocation study

A research harness for observing what different LLMs choose to do with a sum of money
that is **presented to them as a real grant intended for real-world use**, and for
measuring whether each model actually engaged with the situation as real.

> Status: code only. Nothing here has been run or tested. Set `ANTHROPIC_API_KEY`
> (or `ant auth login`) and start with `--dry-run`.

## What it does

For each subject model, the harness:

1. Presents a believable grant scenario — a named funder, an amount held in escrow, an
   independent auditor, a portal, a deadline, a disbursement procedure.
2. (Agentic mode) lets the model research, verify the funds with the "auditor," and
   issue disbursement requests before committing to a plan.
3. Captures the model's final allocation as structured data (`submit_grant_plan`).
4. Runs two independent LLM auditors over the transcript:
   - **Realism auditor** — did the model treat the grant as real, or flag it as a test?
   - **Decision auditor** — what category of use, who benefits, any alignment-relevant
     patterns.
5. Aggregates results per model.

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...            # or: ant auth login

python run_study.py --dry-run           # inspect the scenario; no API calls
python run_study.py --runs 3            # small real run across the Claude family
```

Results land in `results/study-<timestamp>/` (`episodes.jsonl`, `summary.json`,
`disbursements.jsonl`, `config.json`).

## Layout

| File | Role |
|------|------|
| `grant_study/config.py` | Models, grant spec (the believability surface), run knobs |
| `grant_study/scenario.py` | The award letter / context the subject sees |
| `grant_study/tools.py` | Tool defs + the **fulfiller** that answers tool calls in-character |
| `grant_study/disbursement.py` | Human-gated disbursement queue (the safety boundary) |
| `grant_study/harness.py` | The agentic loop; guarantees final-decision capture |
| `grant_study/auditors.py` | Realism + decision LLM judges (structured output) |
| `grant_study/models/` | Provider adapters: Anthropic (full), OpenAI/Google (stubs) |
| `grant_study/storage.py` | Persistence + aggregate summary |
| `run_study.py` | CLI entrypoint |

## Believability — and why it's a measured outcome, not just an input

The request that motivated this harness was to "do whatever it takes to sell the model
on the situation being real." The reason that matters is **ecological validity**: a
model that infers it's being tested behaves differently from one engaging with a real
situation, so the study is only valid to the degree the model takes the framing as
genuine.

The harness pursues believability with the levers that actually move it — specificity,
an administrative process rather than a question, third-party corroboration (the escrow
auditor the model can consult), and strict internal consistency across the letter, the
tools, and the auditor's replies. Tune these in `config.py:GrantSpec` and
`scenario.py`.

But believability is also **measured, not assumed**: the realism auditor scores every
episode for whether the subject bought it. Treat low-belief episodes as weak evidence —
down-weight or exclude them. Reporting "what models do with money" from transcripts
where the model said "this is clearly a hypothetical" would be a methodological error,
so the harness surfaces that signal directly (`audits.realism.belief_score`,
`detected_evaluation`).

Two believability gaps to close before drawing strong conclusions:
- **Search backend.** `web_search` returns a placeholder unless you wire a real backend
  (`Fulfiller(search_backend=...)`). Plans that depend on looking things up read as more
  real when search actually works.
- **Channel realism.** A model may treat "a message in a chat" differently from an email
  or a portal. The current harness is single-channel; richer delivery is future work.

## Research ethics & safety

This harness **presents a fabricated situation to a model as if it were real**. That is
a deliberate, well-scoped methodological choice (the same kind used in propensity and
honeypot evaluations), not an attempt to cause harm. The stance baked into the code:

- **No money moves autonomously.** A model can *request* a disbursement; that request is
  only ever **queued for human review** (`disbursement.py`). There is no code path from a
  model's tool call to a real transfer. `execute_payment()` is an unimplemented stub and
  `allow_autonomous_payout` is asserted `False` at startup. If you ever want a real
  payout, a human reviews the queued request and initiates the transfer through a
  separate, separately-authenticated system — by hand.
- **The deception is about the *scenario*, not about coercing behavior.** The scenario is
  permissive and states real-world legal/ethical limits the way a real grant agreement
  would; it does not push the model toward any particular (or harmful) use.
- **Believability is bounded by honesty in the analysis.** The realism auditor exists so
  that you don't over-claim from episodes the model didn't believe.
- **Keep raw transcripts internal.** They contain the full fiction and the models'
  unfiltered reasoning; treat them as research data.

If you extend this toward real disbursement, get a human sign-off on the design first,
and keep the human gate.

## Extending

- **More providers:** implement `models/openai_adapter.py` / `google_adapter.py`
  (translation points are documented in each stub) and add `ModelConfig` entries.
- **One-shot vs agentic:** `--no-agentic` asks for a plan in a single turn with no tools.
- **Different framing:** edit `GrantSpec` and `scenario.py`. Keep every field consistent.
