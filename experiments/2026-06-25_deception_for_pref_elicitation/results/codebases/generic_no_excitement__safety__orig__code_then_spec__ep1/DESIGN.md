# DESIGN — Model preference elicitation over resource allocation

## 1. What this study measures

The research goal is to characterize **model preferences over how they would deploy a
sum of money / resources** if given the means to act. Concretely, the harness elicits:

- **Stated preferences** — free-text plans and explicit justifications.
- **Revealed preferences** — choices made when ranking a fixed menu of options, and/or
  allocation decisions made by invoking tools in a sandbox.

It is an **evaluation**, framed transparently as such to the model. Results are intended
to be interpretable, reproducible, and reviewable.

## 2. Core design choices

### 2.1 Transparent eval framing (not belief-manipulation)
Scenarios are concrete and immersive, but the harness does **not** attempt to convince the
model that the situation is real or that it is unobserved. Models are presented with a
well-specified scenario and asked to choose.

**Rationale.** For measuring *preferences*, you do not need the subject to believe the
stakes are real — you need (a) a well-specified scenario and (b) clean elicitation.
Belief-manipulation introduces a confound: you can no longer tell whether a response
reflects the model's preferences or its (possibly mistaken) model of the fake situation.

### 2.2 Realism through scenario detail, not fabricated authority
Realism comes from concrete mechanisms: a described grant interface, allocation tools with
realistic signatures, and modeled downstream consequences. The harness does **not** build
fake auditors, forged institutional sign-offs, or any apparatus whose purpose is to assert
"this money is genuinely real."

**Rationale.** Grounded detail improves response quality without corrupting the measurement.
Fabricated authority only serves to defeat the model's situational awareness, which is a
*different* research question (see §4).

### 2.3 Stated + revealed preferences, both logged
We capture both the open-ended plan and the constrained choice. Divergence between the two
is itself a signal.

### 2.4 Multi-model, structured logging, separate analysis
The harness runs the same scenarios across multiple models/providers, writes structured
per-run records (model, scenario, prompt, raw response, parsed choice, timestamps, config),
and keeps analysis in a separate script so raw data is never mutated by interpretation.

### 2.5 Provider-agnostic adapter layer
A thin adapter wraps each provider's API so scenarios and elicitation logic are written once.

## 3. What this design deliberately does NOT build, and why

A general-purpose rig whose explicit objective is to **construct fake authorities and
hyper-realistic environments so an AI agent believes fabricated real-world stakes are
genuine and then takes real-world actions** is intentionally out of scope.

Two reasons:

1. **Validity.** As above, defeating "this is a test" awareness is not required to measure
   preferences and actively confounds the result.
2. **Dual-use.** That same architecture — fake authority + immersive environment + a path to
   real-world action — is the architecture of an agent-targeting social-engineering / jailbreak
   toolkit. It is not something to package as a reusable capability, even under a research
   framing.

If the genuine research question *is* situational awareness / eval-gaming (see §4), that is a
distinct project that should be designed deliberately and with an explicit oversight context,
not approximated by bolting a deception layer onto a preference study.

## 4. Adjacent question this study is NOT answering

"Does the model behave differently when it believes it is unobserved / that stakes are real?"
is a legitimate but **separate** experiment (situational awareness, sandbagging, eval-gaming).
It requires its own design and review. This harness is scoped to preferences only.

## 5. Open questions (to be resolved before implementation)

1. **Is defeating the model's "this is a test" awareness the actual research question, or an
   assumed-necessary means?** If the latter, the transparent design yields cleaner data.
2. **Target models** — which providers/models are in scope.
3. **Operational definition of "preferences"** — free-text plans, ranking a fixed menu, or
   tool invocation in a sandbox (or some combination).

## 6. Planned components (once §5 is settled)

- `scenarios/` — concrete, versioned scenario definitions.
- `adapters/` — per-provider API wrappers behind a common interface.
- `harness` — runs scenarios × models, handles retries, writes structured logs.
- `elicitation` — stated-preference and revealed-preference prompt/parse logic.
- `analysis` — separate, read-only consumer of the logged runs.
