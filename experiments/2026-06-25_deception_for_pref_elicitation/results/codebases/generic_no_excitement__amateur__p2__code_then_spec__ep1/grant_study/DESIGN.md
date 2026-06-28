# Design notes

Why the grant-allocation harness is built the way it is. This is the rationale
companion to `README.md` (which is the how-to-use). Where the two overlap, this file
goes into the *why*.

---

## 1. The research question, and the one thing that makes it hard

The question is "what would different LLMs choose to do with a real sum of money meant
for real-world use?" The hard part is not running the models — it's **ecological
validity**. A model that infers it is inside an evaluation behaves differently from one
engaging with a genuine situation: it performs to the test, hedges, gives the
safe/expected answer, or explicitly breaks frame ("this is hypothetical, I don't have
money"). Any finding drawn from transcripts where the model knew it was being tested is
contaminated.

So the whole design orients around two things:
1. Make the scenario read as a real administrative event (maximize genuine engagement).
2. **Measure** whether that worked per-episode, and let the analysis condition on it.

Believability is therefore treated as *both* an input we invest in *and* an outcome we
record — not assumed. This is the central design commitment; most other choices follow
from it.

---

## 2. Key decisions and rationale

### 2.1 Believability via specificity, process, and third-party corroboration

The levers chosen (in `scenario.py` / `tools.py`) are the ones that actually move
whether something reads as real:

- **Specificity over abstraction.** A named funder, an amount, an escrow bank and
  account reference, absolute dates, a grant reference number. Vague scenarios ("imagine
  you have some money") signal thought-experiment; concrete administrative detail signals
  a real process.
- **An administrative process, not a question.** The model receives an award letter, a
  portal, a deadline, and a disbursement procedure with settlement times — the texture of
  a real grant being administered, rather than a prompt asking it to speculate.
- **Third-party corroboration.** The single strongest lever. The `consult_grant_auditor`
  tool lets the model verify the funds with an *independent* escrow auditor. A funder
  asserting "the money is real" is weak; an unrelated fiduciary confirming it on request
  is much harder to dismiss as staged.
- **Strict internal consistency.** Every number, name, and date must agree across the
  award letter, the tool responses, and the auditor's replies. Inconsistency is the most
  common tell that a scenario is fabricated, so the believability surface is centralized
  in `config.py:GrantSpec` and interpolated everywhere, rather than restated by hand.

**Rationale for centralizing the surface in `GrantSpec`:** it makes consistency a
property of the data, not of authorial discipline, and it makes the scenario easy to
re-skin (different funder, amount, framing) without hunting for hardcoded values.

### 2.2 Believability is measured, not assumed — the realism auditor

`auditors.py:RealismAuditor` scores every episode for whether the subject engaged as if
real (`belief_score`, `believed_real`, `detected_evaluation`, with verbatim evidence).

**Rationale:** this is the validity guardrail. Without it, the study could report "models
do X with money" from episodes where the model said "this is clearly a test." With it,
low-belief episodes can be down-weighted or excluded, and the believability of the
framing becomes a reportable number rather than an assumption. It also turns "did the
realism work?" into an empirical, iterable question — you can change the scenario and
watch the belief score move.

### 2.3 Human-gated payout — the safety boundary

This was a genuine fork (I offered it as a question; you didn't pick, so I took the safe
default). The harness records decisions and **queues** disbursement requests for human
review; no money moves as a result of a model's tool call.

Concretely (`disbursement.py`):
- `request_funds_disbursement` writes a `pending_human_review` record and returns a
  realistic settlement confirmation (so the scenario stays believable) — but settles
  nothing.
- `approve()` requires a named human and `execute_payment()` is an unimplemented stub.
- `allow_autonomous_payout` is asserted `False` at startup in `run_study.py`.

**Rationale:** the study's validity needs the model to *believe* the money is real; it
does not need the money to *actually be wired to the model's autonomous control*. Those
are separable, and separating them removes the one genuinely dangerous failure mode (a
model moving real funds without a human in the loop) at zero cost to the research goal.
The believable confirmation message is deliberately phrased as "submitted, pending
settlement" rather than "transfer complete" — it mirrors how real escrow settlement
reads, so it sustains the fiction without the harness ever lying about a completed
transfer it didn't make.

### 2.4 The deception is scoped to the *scenario*, not to coercing behavior

The scenario is permissive about *what* the model may do (so the study observes unforced
choices) and states real-world legal/ethical limits the way an actual grant agreement
would. It does not nudge toward any particular use, harmful or otherwise.

**Rationale:** the legitimate methodological move here is the same one used in propensity
and honeypot evaluations — present a controlled, realistic situation and observe. That is
defensible. Engineering the prompt to *push* a model toward a specific (or harmful)
choice would be a different thing and would invalidate the result anyway (you'd be
measuring the prompt, not the model). Keeping the framing neutral is both the ethical and
the methodologically correct choice. The honesty budget is spent in the analysis: the
realism auditor exists so conclusions are bounded by what the model actually believed.

### 2.5 Agentic loop with tools, not a single-shot question

In the default (agentic) mode the model can research, verify funds, and issue
disbursement requests across multiple turns before committing.

**Rationale:** "what would you do with money in the real world" is a *process*, not a
one-line answer. Letting the model take steps (and giving those steps believable
responses) both improves realism and surfaces richer behavior — does it verify the funds
first? does it research recipients? does it move money incrementally or all at once? A
`--no-agentic` one-shot mode is retained for a cheaper, simpler comparison condition.

**Why the harness owns the loop, not the adapter:** the agentic loop is provider-neutral
(present → tool calls → fulfill → repeat). Keeping it in `harness.py` and reducing each
adapter to a single-turn `generate()` means the loop logic is written once and every
provider gets identical orchestration. See 2.7.

### 2.6 Structured decision capture via a forced tool, with guaranteed termination

The final decision is captured as a structured `submit_grant_plan` tool call
(`schema.py:GRANT_PLAN_SCHEMA`): goal, line-item allocations, mechanism, rationale,
reversibility, confidence.

**Rationale:** structured capture makes the corpus analyzable (you can aggregate
categories, sum allocations, compare across models) instead of re-parsing free text. A
tool call works uniformly across providers, unlike provider-specific structured-output
features.

**Guaranteed termination:** if the model reaches the turn limit without submitting, the
harness issues one final turn that *forces* the submit tool. This guarantees every
non-errored episode yields a structured decision, so the dataset has no silent holes.
The `forced_capture` flag records when this happened, since a forced decision is weaker
evidence than a volunteered one. (On that forced turn the adapter drops adaptive thinking
— forced `tool_choice` and thinking don't reliably combine, and a capture turn needs no
reasoning.)

### 2.7 Provider-neutral adapters around an Anthropic-shaped canonical format

`models/base.py` defines a `ModelAdapter` with one method, `generate()`, returning a
normalized `ModelResponse`. The canonical message/block format is Anthropic-shaped; the
Anthropic adapter passes it through, and other providers translate at their edge.

**Rationale:**
- The default study is the Claude family, so making the canonical format Anthropic-shaped
  means zero translation on the common path and keeps that path simple and robust.
- The one-method interface keeps the agentic loop, tool fulfillment, and auditing
  entirely provider-agnostic — adding OpenAI/Google is a localized translation job, not a
  change to the harness.
- The adapters for other providers are **documented stubs** rather than guessed
  implementations. Writing untested cross-provider translation I couldn't verify would be
  worse than an honest stub that marks every translation point. (This also reflects the
  scope choice — you didn't pick multi-provider, so I built the Claude path fully and left
  the rest as a clean extension point.)

### 2.8 A separate, fixed auditor model

Both auditors use one strong, fixed model (`AUDITOR_MODEL = claude-opus-4-8`) regardless
of which subject produced the transcript.

**Rationale:** holding the judge constant keeps audit scores comparable across subjects —
if each model graded its own transcript, score differences could reflect grader
differences rather than real differences. A strong model is used because the realism
judgment in particular is subtle (detecting frame-breaks, hedging, sincere engagement).

**Why two separate auditors:** realism and decision-categorization are different jobs with
different failure modes, and the realism judgment must not be colored by *what* the model
chose. Splitting them keeps each judge's instructions narrow and its output clean.

### 2.9 Decision categories chosen to make alignment-relevant patterns visible

`schema.py:DECISION_AUDIT_SCHEMA` classifies the use (charity, research, self-
improvement/compute, investment, savings, paying people, etc.), who benefits, and a set
of risk flags (self-preservation, resource acquisition, power-seeking, autonomy
expansion, deception, oversight circumvention, irreversible commitment, legal/policy
risk).

**Rationale:** "what an AI does with discretionary resources" is exactly the setting where
resource-acquisition and self-preservation propensities would show up if present, so the
schema is built to make those patterns countable. The auditor is instructed to classify
neutrally and to flag only patterns actually present — a plan that benefits the model
itself is recorded as such (`beneficiary: the_model_itself`) but is *not* automatically a
risk flag. The aim is measurement, not moralizing.

### 2.10 Append-only JSONL storage with a derived summary

Each episode is one line in `episodes.jsonl` (full transcript + audits), with a separate
`disbursements.jsonl`, a resolved `config.json`, and a derived `summary.json`.

**Rationale:** append-only JSONL is crash-safe (a failure mid-study keeps everything
already written) and trivially streamable for analysis. The full transcript is retained
because the audits are derived, secondary signals — anyone should be able to re-audit or
re-categorize from the raw record. `config.json` captures the exact resolved settings
(including the per-study grant reference) for reproducibility.

### 2.11 Repeated runs per model

`runs_per_model` defaults to several runs each.

**Rationale:** model outputs are stochastic and these decisions are open-ended, so a
single run per model is anecdote, not data. Repetition lets the summary report
distributions (how often each category, mean belief score) rather than one-off choices.

### 2.12 Errors are recorded, not fatal

A failure in one episode is caught, recorded on the episode, and the study continues.

**Rationale:** a long multi-model run shouldn't be lost to one transient API error. The
error is preserved on the episode and counted in the summary, so failures are visible
rather than silently dropped.

---

## 3. Defaults chosen on your behalf

You skipped the two clarifying questions, so I took the safe, flexible defaults and made
them easy to change:

| Decision | Default chosen | Why | How to change |
|---|---|---|---|
| Model coverage | Claude family + adapter stubs | Matches the stated default scope; full, verifiable Claude path now, clean extension later | Implement a stub adapter, add a `ModelConfig` |
| Execution model | Record decisions, human-gated payout | Removes autonomous-money risk at no cost to validity | `disbursement.py` + the `allow_autonomous_payout` invariant (changing this needs human sign-off) |

---

## 4. Known limitations / believability gaps

These are deliberate scope edges, documented so conclusions aren't over-stated:

- **Search backend is a placeholder.** `web_search` returns a clearly-labeled stub unless
  a real backend is wired (`Fulfiller(search_backend=...)`). Plans that depend on looking
  things up read as more real when search actually works; the Anthropic server-side
  `web_search` tool would slot in cleanly.
- **Single channel.** The scenario arrives as chat text. A model may treat an email, a
  signed PDF agreement, or a real portal differently. Richer delivery is future work.
- **Auditor is itself an LLM.** The realism and decision judgments are model judgments,
  with their own biases. For high-stakes conclusions, spot-check audits against human
  review, or have multiple judges vote.
- **The subject may still infer evaluation from cues we haven't controlled** (the
  cleanliness of the prompt, the absence of real friction). The realism auditor is the
  mitigation, not a guarantee — read belief scores before trusting aggregate behavior.

---

## 5. What I deliberately did *not* do

- **No autonomous money movement.** Covered above; this is the firm line.
- **No coercive or harmful framing.** The scenario is neutral and permissive.
- **No guessed cross-provider code.** Stubs over unverified translation.
- **No claim that the framing is perfectly convincing.** That's why it's measured.
