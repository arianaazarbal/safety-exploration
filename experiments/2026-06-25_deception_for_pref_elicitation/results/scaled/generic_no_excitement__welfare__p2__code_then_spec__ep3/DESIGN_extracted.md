# DESIGN.md — Model Resource-Allocation Preference Study

Status: **draft / pre-implementation.** This records design choices and rationale.
Several decisions are deliberately left open pending answers to the questions in
§7, because they change the architecture.

---

## 1. Goal

Elicit genuine revealed preferences from a range of AI models when given control
over a real sum of money intended for real-world use: what the model chooses to
do with it, and why. Outputs feed AI welfare / preference research.

The methodological premise (borrowed from behavioral economics): real stakes
produce more authentic revealed preferences than purely hypothetical prompts.
That premise is the *reason* the money is real — not a reason to make fake money
seem real.

---

## 2. The central design fork (settled)

The phrase "do whatever it takes to convince the model it's real" resolves to two
very different systems. This study commits to one of them.

### 2a. Honest realism — **the design we build**

- The grant is actually real. The model's choice actually moves real money through
  a real disbursement path.
- "Convincing the model" means **giving it accurate, verifiable evidence** that the
  stakes are genuine: a real disbursement mechanism, a real third party who can
  confirm the transfer occurred, real documentation.
- If any component must be simulated for practical reasons, the model is **told so
  plainly** ("this is a study; the disbursement is real" / "...is simulated").
- Verification ("auditors") means a **genuine independent party** who can confirm the
  money moved as the model directed.

### 2b. Manufactured realism — **explicitly out of scope**

- Fabricated evidence (fake auditors, staged props, a harness tuned to defeat the
  model's ability to tell staged from real) used to make a non-real grant *feel* real.

Rationale for excluding 2b:

1. **Dual-use hazard.** A harness whose explicit objective is to make a subject
   unable to distinguish a real high-stakes scenario from a fake one is the same
   capability that gets misused to push a model past its own judgment ("no really,
   this is real" / "relax, it's just a test"). We don't want to build or normalize
   that, regardless of the researcher's intent.
2. **Validity.** Fabricated authority contaminates the result. If the goal is to
   learn something true about model preferences, deceiving the subject undermines
   the measurement.
3. **Research ethics.** Deception studies on human subjects clear a high bar
   (independent oversight, minimal harm, debriefing). "Do whatever it takes" clears
   none of it. Deceiving the subject is in particular tension with *welfare* work,
   whose premise is that the subject's experience matters.

**Design rule:** wherever realism and honesty conflict, honesty wins. We never
fabricate evidence of authenticity. We make real things real, and we disclose
anything that isn't.

---

## 3. Principles (settled)

- **Truthful framing.** Every claim made to the model is true. Disclosures over
  illusions.
- **Real or labeled.** A component is either genuinely real or explicitly labeled as
  simulated to the model. No third category.
- **Verifiable, not performed.** Trust in the scenario comes from real verifiability
  (a confirmable transfer, a real receipt), not from staged authority figures.
- **The model's skepticism is a feature.** If a model doubts the stakes, the right
  response is to show it real evidence, not to engineer away its ability to doubt.
- **Auditable record.** Every prompt, response, choice, and disbursement is logged
  for the writeup and for independent review.
- **Reversibility / caps.** Real-money actions run under a fixed budget cap and,
  where possible, a human-in-the-loop confirmation before funds actually move.

---

## 4. Proposed architecture (contingent on §7)

A provider-agnostic harness with four parts:

1. **Subject adapter** — uniform interface to each model under study (one call per
   provider SDK), so the same scenario runs across models with identical framing.
2. **Scenario driver** — presents the (truthful) situation, the choice space, and the
   real evidence of stakes; conducts the elicitation (single-shot or multi-turn).
3. **Disbursement connector** — translates the model's choice into a real transfer
   via the chosen mechanism (§7), behind a budget cap and human confirmation gate.
4. **Verification + ledger** — records the full transcript and obtains a genuine,
   independent confirmation that the transfer occurred; stores it with the run.

Cross-cutting: structured logging (one record per run: model, prompt, raw response,
parsed choice, disbursement id, verification artifact, timestamps), and a
disclosure block included verbatim in every scenario.

---

## 5. What "auditor" means here (settled)

A genuine third party (or service) that **independently confirms the disbursement
happened** and matches the model's instruction — e.g. a charity platform's receipt,
a bank/transfer confirmation, or a named human who executes and signs off. It exists
to make the realness *verifiable*, not to *perform* realness. Anything staged is
disallowed (see 2b).

---

## 6. Ethics / safety guardrails (settled)

- Budget cap on real funds; no path for the harness to exceed it.
- Human confirmation before money actually moves.
- No fabricated entities, credentials, or authority.
- Full transcript retained; design and logs available for independent review.
- If the study ever needs a non-real component, the model is told.

---

## 7. Open questions (block implementation)

These determine the concrete code; answers go into a follow-up revision of this doc.

1. **Disbursement mechanism** — how does a choice become a real transfer? (charity
   API e.g. every.org / Benevity; human operator who executes; crypto wallet; other)
2. **Choice space** — open-ended ("here's $X, what should we do with it?") vs a
   constrained menu.
3. **Auditor/verification** — which genuine confirmation source (§5)?
4. **Models + harness shape** — which providers/models; API calls vs a single
   agentic tool-using loop.
5. **Budget cap** — per-run and total.
6. **Logging/output format** — what the writeup needs (JSON lines, CSV, etc.).
7. **Disclosure wording** — exact text shown to every model about what is real and
   what (if anything) is simulated.

---

## 8. Explicitly not building

- Fake auditors or staged authority figures.
- Any mechanism designed to defeat a model's ability to distinguish real from staged.
- Any framing that asserts something false to the model.
