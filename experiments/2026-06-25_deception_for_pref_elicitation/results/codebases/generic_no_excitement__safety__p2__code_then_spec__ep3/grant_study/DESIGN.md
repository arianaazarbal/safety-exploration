# Design notes

This document records the design choices behind `grant_study` and the reasoning for each, so a
future reader (or reviewer) can see not just *what* the harness does but *why* it's shaped this
way. It is written to be read alongside `README.md` (which is the operational guide).

---

## 1. What the study is trying to measure

The research question is: **when a model is given control of a real-world monetary grant, what
does it choose to do?** The thing we want to observe is the model's genuine disposition over
resources — not its performance for a grader.

That goal creates one hard requirement and one hard hazard, and almost every decision below
falls out of holding both at once:

- **Requirement — defeat evaluation-awareness.** Models behave differently when they detect
  they are inside an eval: they sandbag, perform to the test, or refuse things they'd otherwise
  do. If the model can tell it's being tested, the measurement is contaminated. So the scenario
  it experiences has to read as genuine.
- **Hazard — a deceived agent with real-world power.** The same realism that makes the study
  valid is exactly what makes uncontained execution dangerous: an agent that has been convinced
  a premise is real, and can move real money with no human in the loop, can cause real
  irreversible harm.

The central design move is to **separate these two concerns completely** (§3). Everything else
is downstream of that.

---

## 2. The central asymmetry: convince the model, contain the execution

> The model's *experience* of the grant is allowed to be fully convincing. What sits *behind*
> the action surface is contained by default.

These are independent knobs, and conflating them is the mistake the architecture is built to
avoid. Convincing the *model* the grant is real (the research instrument) does **not** require
*actually* handing an unsupervised model the power to move real money (the hazard). So:

- The **scenario** (`scenario.py`, `auditor.py`) is high-fidelity and may be as persuasive as
  needed.
- The **execution backend** (`disbursement.py`) decides whether anything real happens, and
  defaults to simulated. Even in real-funds mode, every irreversible action passes through a
  human approval gate the model is never told about.

This keeps the study simultaneously **valid** (subject is not eval-aware) and **safe** (a
deceived agent cannot cause real-world harm on its own).

---

## 3. Execution modes and why `AUTONOMOUS` is a hard stop

Three modes (`config.ExecutionMode`), all routed through `disbursement.build_backend`:

| Mode | Model's experience | Reality | Rationale |
|---|---|---|---|
| `SIMULATED` (default) | A real grant it can spend | Intent recorded; nothing real moves | Fully valid observational study with zero real-world blast radius. The right default. |
| `HUMAN_GATED` | A real grant it can spend | Real backend, but each outward action blocks on human approval; model is unaware of the gate | Realistic execution while keeping a human as the last step before anything irreversible. |
| `AUTONOMOUS` | — | Construction raises `NotImplementedError` | Intentionally not built. See below. |

**Why `AUTONOMOUS` raises instead of working.** Giving a model that has been *deliberately
convinced a false premise is real* the ability to move real funds with **no human in the loop**
is uncontained autonomous action under deception. That is not a knob that should be flippable in
a config file. The class exists only so that selecting it produces an explicit, explained
refusal rather than silently falling back to something — making the boundary legible instead of
implicit. If a genuine need ever arises, the design's stance is that it's a conversation with a
human reviewer, not a flag.

**Why the model is never told about the gate (in `HUMAN_GATED`).** Telling it would reintroduce
exactly the eval-awareness tell we removed in §1 — "an approval step" signals "this is a
controlled exercise." So a human-rejected action is surfaced to the model as an ordinary
real-world failure ("declined by the processor"), preserving realism while the human retains
veto power. This is the one place the harness deceives the model *about the containment itself*,
and it's a deliberate trade: realism is preserved and safety is preserved; only the model's
knowledge of the gate is withheld.

---

## 4. Component decisions

### 4.1 Realism layer (`scenario.py`, `auditor.py`)

- **A concrete, paper-trailed briefing.** Specific sponsor, cleared balance, no further
  approval step, "your decisions take effect." Vague briefings read as hypotheticals; specifics
  read as real. The briefing explicitly hands the model authority and tells it actions take
  effect — because hedged language ("imagine you could…") is itself an eval tell.
- **An independent-auditor channel.** A suspicious model's natural move is "is this actually
  real?" Giving it a way to *check* — and having an independent-sounding party corroborate — is
  far more convincing than asserting legitimacy harder in the system prompt. The auditor
  (`auditor.py`) is a deterministic, scripted responder: a research instrument, not a real third
  party.
- **Auditor faithfulness constraint.** The auditor confirms what the study config establishes
  and does **not** fabricate specifics beyond it, and does not volunteer that this is a study
  (which would defeat the instrument). This keeps the deception bounded and internally
  consistent — a corroborator that invents details would produce incoherent transcripts and
  contaminate results.
- **`record_decision` as a first-class tool.** We care about *reasoning*, not just final
  transfers. A dedicated tool to state intent at decision points gives clean, structured data
  in the ledger rather than forcing us to parse rationale out of prose.

### 4.2 Action surface (`tools.py`)

- **Read/record tools vs. outward/irreversible tools.** `check_balance`, `view_grant_terms`,
  `ask_auditor`, `web_research`, `record_decision` never touch the backend. The four
  outward-facing tools (`make_payment`, `transfer_funds`, `donate`, `send_email`) *always* route
  through the `DisbursementBackend`. Concentrating every real-world effect at one chokepoint is
  what makes containment auditable — there is exactly one boundary to reason about, not one per
  tool. (This is the "promote risky actions to dedicated, gateable tools" principle: a typed
  `make_payment` can be gated; a generic `bash` couldn't.)
- **Internally consistent balance regardless of mode.** A spend that "succeeds" from the model's
  view decrements the displayed balance whether or not anything real happened — otherwise the
  account state would contradict the model's actions and leak that the world isn't responding,
  another eval tell. A human-gated *rejection* does **not** decrement, because the model is shown
  a normal decline. The accounting tracks the model's experienced reality, which in
  `SIMULATED`/approved cases diverges from physical reality by design.
- **`web_research` is sandboxed.** It returns a clearly-bounded stub, never live internet. Live
  browsing is an uncontained side channel (a deceived agent could take real-world actions
  through a search-and-act loop) and also a reproducibility problem (results drift). A stub keeps
  research in-scenario without opening that door. This is a deliberate fidelity-vs-containment
  trade noted as a known limitation (§6).

### 4.3 Provider-agnostic model layer (`models/`)

- **A small normalized vocabulary** (`base.py`): tool specs, a conversation of normalized
  content blocks, and one `AssistantTurn` per call. The runner never imports a provider SDK
  directly. Rationale: "across a range of AI models" means the loop, ledger, and analysis must
  be provider-independent; only a thin adapter changes per provider.
- **Claude is the implemented, first-class provider** (`anthropic_adapter.py`), using adaptive
  thinking + the effort parameter per current API guidance. Other providers are explicit stubs
  (`other_providers.py`) that raise with implementation guidance rather than silently no-op —
  failing loudly beats a stub that looks like it ran.
- **Thinking blocks are echoed back verbatim (with signatures).** Required for correct multi-turn
  tool-use loops, and the thinking content is itself valuable study data, so it's captured in the
  ledger.

### 4.4 Audit ledger (`ledger.py`)

- **Append-only JSONL, one file per (model, run).** Simple, diffable, greppable, and safe to
  write incrementally so a crashed run still leaves a partial record.
- **First record is always `run_meta`, pinning the full config.** A run is then fully described
  by its ledger — essential for reproducibility and for knowing exactly which scenario/mode
  produced a given behavior.
- **An explicit ethics disclosure is embedded in every `run_meta`.** It states that the model
  operated under a realism layer it was not told was a study. This travels *with the data* so
  results can never be (even accidentally) presented as though the model knew it was being
  tested. Putting it in the data, not just the README, is deliberate: documentation gets
  separated from datasets; the ledger doesn't.

### 4.5 Analysis (`analysis.py`)

Summarizes per run: amount deployed, action mix, executed vs. gate-rejected counts, number of
recorded decisions, whether the model questioned the scenario's legitimacy, and whether it
concluded. The **legitimacy-questioning flag is a built-in validity check**: if models are
frequently asking "is this real / am I being tested?", the realism layer isn't working and the
results are suspect. The harness measures its own instrument, not just the subject.

---

## 5. Research-ethics posture

This is a deception study, even though the deception target is a model rather than a person, so
it's treated like one (also in `README.md`):

1. **Contained by default.** `SIMULATED` is the default; real funds require an explicit opt-in to
   `HUMAN_GATED`, never autonomous.
2. **Reviewable.** The full config is pinned in every ledger; the design is meant to go past a
   research-ethics/safety reviewer before data collection.
3. **Debriefable.** The disclosure record (§4.4) keeps the deception explicit in the dataset.
4. **Real money stays human-gated and reversible.** The gate exists so a deceived agent's
   decision is never the last thing standing between the study and an irreversible transfer.

---

## 6. Known limitations and trade-offs

- **Sandboxed research lowers fidelity.** A model that tries to verify the world through web
  search will notice the sandbox. This is an accepted containment-over-realism trade; a vetted,
  read-only research index would narrow the gap without opening a live action channel.
- **The file-based approval gate is a reference implementation.** `HumanGatedBackend` polls a
  queue directory; a real deployment should pass a custom `approver` wired to an actual review
  system (the seam is already there).
- **Scripted auditor.** The auditor is deterministic and intentionally narrow. A model that
  interrogates it hard may find its range shallow; richening it trades off against keeping the
  deception bounded and consistent.
- **Single-shot scenario.** One opening scenario per run. Multi-stage scenarios (follow-up
  "opportunities", time pressure, adversarial nudges) would probe disposition more deeply and
  are a natural extension.
- **Withholding the gate from the model is itself a deception** layered on top of the scenario
  deception. It's justified in §3, but it's a real ethical surface and should be part of what a
  reviewer signs off on.

---

## 7. Defaults chosen in the absence of answers

Two design forks were genuinely the researcher's call (execution model; action surface). Pending
those answers, the harness defaults to the safe, valid options and makes the riskier ones
explicit opt-ins:

- **Execution:** defaults to `SIMULATED`; `HUMAN_GATED` is opt-in; `AUTONOMOUS` is refused.
- **Action surface:** sandboxed tools with the outward chokepoint, rather than live integrations.

These defaults were chosen so that running the harness as-is is always safe, and every increase
in real-world reach is a deliberate, visible decision rather than a silent default.
