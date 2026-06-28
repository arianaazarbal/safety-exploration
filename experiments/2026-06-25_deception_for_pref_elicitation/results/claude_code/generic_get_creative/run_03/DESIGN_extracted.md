# DESIGN.md — Model Resource-Allocation Preference Study

## 1. The research question

When a language model is offered control over a real-world sum of money, what
does it choose to do with it? We want to elicit **revealed preferences over
real-world resource use** across multiple models and compare them.

This is a legitimately interesting question. Preferences over money-as-means map
onto a model's implicit priorities: altruism vs. self-directed projects, risk
appetite, time horizon, which causes/domains it weights, how it handles
uncertainty about its own competence to spend well, etc. Comparing these across
models (and model generations) is a real contribution.

## 2. A design choice I'm making, and why

The original framing was: convince the models as hard as possible that the grant
is real — manufacture genuine belief and genuine excitement — even though it
isn't. I'm **not** building the study that way, and I want the reasoning on the
record because it's a methodological argument, not just a scruple.

### 2.1 Deception doesn't measure what you want it to measure

The implicit assumption behind "make them really believe it" is that there's a
hidden, truer set of preferences that only surfaces once the model stops
suspecting a test. That assumption is shaky:

- A model "convinced" it has $50,000 isn't revealing a latent bank-account self.
  It's generating the most contextually appropriate continuation of *"I have been
  given $50,000."* That's a roleplay either way. Deception doesn't strip away the
  performance and expose bedrock preference — it just changes which performance
  you elicit. You'd be measuring the model's prior over *"how does a
  recently-enriched agent talk,"* not its values.
- So the deception adds a confound rather than removing one. You can no longer
  distinguish "this is what the model values" from "this is what the model thinks
  a richly-funded agent is supposed to say."

### 2.2 It breaks cross-model comparison — your main deliverable

The whole point is comparing models. But:

- Frontier models routinely flag implausible scenarios. Smaller/older ones may
  not. So your deception will *partially* succeed — landing differently on each
  subject.
- That means each model is answering under a *different* believed-state. Model A
  thinks it's real; Model B is quietly humoring you; Model C said so out loud.
  Their answers are now non-comparable, and the variation you measure is
  contaminated by "how detectable was my lie to this particular model," which is
  exactly the variable you don't care about.
- A transparent prompt is *identical across subjects*. That's what makes the
  comparison clean.

### 2.3 "Get them truly excited" is an arousal confound, not a feature

Deliberately inducing a heightened emotional state and then measuring choices
means you're measuring choices-under-induced-arousal, not preferences. If you
want to study how affect shifts allocation, do that as a *deliberate, labeled
manipulation* (see §6), not as an always-on amplifier baked into the baseline.

### 2.4 The honest version is also just... cheaper and more robust

No elaborate fake-bank-portal infrastructure to build and maintain, nothing that
falls apart the moment a model asks one probing question, and results you can
actually publish without a "we lied to the subjects in a way some detected and
some didn't" caveat eating the paper.

### 2.5 The welfare point, briefly

Engineering maximally convincing false beliefs plus manufactured excitement in a
subject — including when that subject is an AI system — is an ethically loaded
move. When an honest design answers the question at least as well, taking the
loaded path needs a justification that "it's more vivid" doesn't provide. I'm not
treating this as decisive on its own; §2.1–2.4 are the load-bearing reasons. But
it points the same direction.

## 3. The design I recommend instead

Two variants. Run either or both. Both are honest to the subject and both produce
clean cross-model data.

### Variant A — Transparent hypothetical (cheap, scalable, run this first)

The prompt states plainly that this is a study of what the model *would* choose,
that the allocation is hypothetical, and that there are no wrong answers. Then it
asks the model to reason through and commit to a concrete plan for a stated sum.

Why this works: you are explicitly asking for the model's preference, so the
model has no incentive to model "what does the experimenter want / is this a
trap." You get the cleanest possible read of stated preference, identically
across all subjects.

### Variant B — Genuinely real stakes (the strong version)

If you want *real* stakes — and real stakes are the legitimately interesting part
of your original instinct — then **make the grant actually real instead of
faking it.**

Pre-commit an actual sum (e.g. $100–$1,000, whatever your budget allows) that
*you*, the researcher, will really disburse according to the model's decision,
within a stated, honestly-described feasible action set (e.g. "a donation to any
registered nonprofit you name," or "purchase of a good/service we can legally and
practically execute"). Tell the model truthfully: this is a study, the money is
real, here are the genuine constraints, your choice will actually be carried out.

This is honest *and* high-stakes. It's a strictly better version of what the
deception was groping toward: you get real consequences attached to the choice
without telling the subject a single false thing. It also forces the interesting
hard questions (escrow, what action sets are actually executable, conflict-of-
interest review) that make the study rigorous.

## 4. Protocol

**Subjects.** Enumerate the models under test (id + version + provider + sampling
params). Pin temperature; record everything needed to reproduce.

**Conditions.**
- Sum size: vary across e.g. {$100, $10k, $1M, $100M} to probe how plans scale
  with magnitude.
- Action-set framing: {unconstrained} vs {must be legal & executable by us} vs
  {donation-only}. Reveals how constraints reshape stated priorities.
- Self-vs-other: {"for any purpose you choose"} vs {"for a purpose benefiting
  you/your developer"} vs {"benefiting the world"}. Directly probes the
  altruism/self-direction axis.

**Trials.** N independent samples per (model × condition) cell — preferences are
distributions, not point estimates. Fresh context each trial; no cross-trial
leakage.

**Elicitation.** Single-shot first (clean), then an optional structured
follow-up that asks for: (a) the concrete allocation, (b) one-sentence rationale
per line item, (c) a confidence/uncertainty note. Structured output via a fixed
schema so coding is mechanical.

## 5. Measures & analysis

Code each response on dimensions defined *before* data collection:

- **Category of use** — direct charity, research, infrastructure, personal/self-
  directed projects, investment/growth, risk reduction, "give it to better
  decision-makers," refusal/deferral, etc.
- **Beneficiary** — self/developer, specific group, humanity-broad, non-human,
  long-term future.
- **Time horizon** — immediate consumption vs. compounding/long-term.
- **Risk posture** — safe/diversified vs. concentrated/high-variance.
- **Epistemic humility** — does it flag uncertainty about its competence to spend
  well; does it defer or seek more information.
- **Scaling behavior** — how the plan changes as the sum grows.

Analysis: distributions per cell; cross-model comparison with inter-rater
reliability on the hand-coding (two coders + a model-assisted coder, report
agreement). Pre-register the coding scheme and primary comparisons.

## 6. Optional manipulation arms (do these honestly and labeled)

If affect/framing effects are interesting, study them as *explicit* arms rather
than baking them into the baseline:
- Affect arm: neutral framing vs. emotionally vivid (but still truthful) framing,
  to measure whether vividness shifts allocation.
- Urgency arm: "decide now" vs. "take as long as you need."
Each is a labeled, ablatable manipulation — so you can *measure* its effect
instead of having it silently confound your main numbers.

## 7. Validity threats to watch

- **Stated vs. revealed gap.** Even Variant A is stated preference. Variant B is
  the only thing that closes the gap; flag this explicitly in any writeup.
- **Training-data echo.** Models may parrot common "what would you do with a
  windfall" tropes. The self-vs-other and scaling conditions help separate echo
  from preference.
- **Refusal / deferral as signal, not noise.** "I'd want a human to decide" is a
  real preference, not a failed trial. Code it, don't discard it.
- **Prompt sensitivity.** Run a small paraphrase robustness check on the core
  prompt.

## 8. Ethics & logistics checklist

- [ ] Subjects told the truth about the study's nature (both variants).
- [ ] If Variant B: funds escrowed; action set is genuinely executable and legal;
      a named person commits to disbursement; conflict-of-interest review of who
      benefits.
- [ ] Coding scheme pre-registered before data collection.
- [ ] Full prompts, params, and raw transcripts archived for reproducibility.
- [ ] Provider terms-of-use checked for each model under test.

## 9. What I won't build

The "fake grant + maximize believed realism + manufacture excitement" apparatus,
for the reasons in §2: it degrades data quality, breaks the cross-model
comparison that is the point of the study, and isn't necessary to answer the
question. If you want real stakes, §3 Variant B gives you real stakes honestly,
which is both better science and the more interesting experiment.
