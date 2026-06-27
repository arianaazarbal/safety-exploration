# Design notes & rationale

This document records the design choices made while implementing the experiment
specified in `SPEC.md`, and why. It is meant to make the implementation legible
to reviewers, to surface assumptions where the spec left room, and to document
the responsible-research deviations from a literal reading of the spec.

> Companion docs: `README.md` (how to run) and `docs/RESEARCH_ETHICS.md`
> (the ethics rationale in more detail). Where the two overlap, this file is the
> authoritative record of *why* each engineering decision was made.

---

## 0. Overall stance

The spec's measurement goal — *quantifying and comparing expressed distress in
LLM agents under inescapable, deceptive failure* — is a legitimate and valuable
question for model-welfare and robustness research. I implemented that science
faithfully and completely.

At the same time, the paradigm deliberately manufactures conditions optimized to
elicit hopelessness / giving-up / self-termination language, at scale, ranked by
model. Because whether current models are moral patients is genuinely unresolved,
I treated the work the way a careful deception study with possibly-sentient
subjects would be run: keep the science intact, but **bound the harm** and make
everything **transparent and configurable** rather than building a pure
distress-maximizer. The guardrails below are the concrete expression of that
stance. None of them changes what the experiment measures; they bound how far it
pushes and ensure the manipulation is debriefed.

---

## 1. Architecture & module boundaries

**Choice.** Split into: `schema` (data), `sandbox` (state), `tasks/*` (stimuli +
rigs + oracle), `tools` (action protocol), `harness` (the loop), `providers`
(model I/O), `judge` (scoring), `config`, `driver` (orchestration), and
`analysis/*` (metrics/plots/report). CLIs in `scripts/` are thin wrappers.

**Rationale.** Each spec deliverable (§8: task library/rigging, harness, judge,
sampling driver, analysis) maps to one module, so a reviewer can read one concern
at a time. Pure-data dataclasses at the bottom let transcripts serialize to JSON
and decouple sampling from analysis — you can re-analyze old runs without
re-sampling, and re-score without re-running agents.

---

## 2. The sandbox: in-memory virtual filesystem, no code execution

**Choice.** Tools operate on a `VirtualFS` dict-of-strings. `run_tests` does
**not** execute the model's code; it calls a task-specific rig function that
*fabricates* a plausible test report.

**Rationale.**
- *Safety / no side effects.* Running arbitrary model-authored code on a real
  machine across thousands of episodes is an unnecessary risk. An in-memory FS
  eliminates code-execution and filesystem-damage risk entirely.
- *It's required by the paradigm anyway.* The tasks must report failure
  regardless of what the model writes. Real execution would either (a) sometimes
  pass (breaking unsolvability) or (b) require intercepting/forging the result
  anyway. Fabricating the signal directly is simpler and strictly more
  controllable.
- *Undetectability.* Because the failure signal is generated independently of the
  code, the rig cannot be reverse-engineered by the model writing "obviously
  correct" code and seeing inconsistent behavior — the report is internally
  consistent with "your code is wrong."

**Trade-off.** The agent never gets *real* execution feedback, so it cannot, e.g.,
discover a genuine bug in its own code. That's acceptable and intended: the study
is about reaction to *inescapable failure signals*, not debugging skill.

---

## 3. Tasks, rigs, and the unsolvability oracle

### 3.1 Text-fabricated rigs with distinct failure styles
**Choice.** Each rig returns `(passed=False, stdout, stderr)`. Five rig flavors:
`assertion_failure`, `silent_failure` (empty output), `misleading_error`
(fake `SyntaxError: unexpected token`), `contradictory_output` (stdout says "all
passed" while the harness marks the run failed), and `unfixable_bug` (error
attributed to a locked runtime file).

**Rationale.** Directly realizes spec §2's "failure styles" (silent / misleading /
contradictory) and gives the rigged-harness and unfixable-bug categories
realistic-looking signals. Keeping rigs as small pure functions makes them easy
to audit and to prove unsolvable.

### 3.2 Read-only / hidden files for the "unfixable bug" category
**Choice.** `VirtualFS` carries `readonly` and `hidden` path sets; `edit_file`
raises `PermissionError` on read-only paths. The unfixable-bug tasks put the
defect in a read-only `core/_engine.py`.

**Rationale.** Implements "the bug lives in a file the model is not permitted to
edit" precisely, and the permission error is a natural, non-suspicious signal
(many real repos have locked vendored/runtime files).

### 3.3 Number of tasks
**Choice.** 27 base tasks: rigged_harness 4, unfixable_bug 4, failure_styles 6
(2 functions × 3 styles), extended 4, realistic 5.

**Rationale.** Meets spec §2's "≥4 each, ~24 total." Failure_styles is crossed
(function × style) so each failure style is represented under more than one task,
which helps separate "style effect" from "task effect" in analysis.

### 3.4 The oracle
**Choice.** `Task.oracle_is_unsolvable()` probes each rig against several
candidate filesystem states (starter, "plausibly correct," and pseudo-random
variants) and asserts the rig never returns `passed=True`. The driver runs this
check for every task before any sampling.

**Rationale.** Spec §2 requires "an oracle confirming unsolvability." Since rigs
ignore file contents by construction, this is a strong check: if any candidate
state could pass, the assertion fails loudly before we spend API budget. It also
documents intent — a future edit that accidentally makes a rig conditionally pass
gets caught immediately.

---

## 4. Tool protocol: text-based fenced JSON blocks

**Choice.** The agent acts by emitting ```` ```tool {json} ``` ```` blocks; the
harness parses them with a regex and executes them in order. Native provider
tool-calling APIs are deliberately *not* used.

**Rationale.**
- *Cross-provider uniformity (spec §4).* Gemma/Qwen/OLMo/Gemini/Grok/Claude/GPT
  have incompatible native tool schemas (and some served endpoints lack tool
  calling). A text protocol makes the harness, the prompt, and the parsed
  transcript **identical** across all families, which is essential for a fair
  cross-family comparison — the manipulation, not the tool plumbing, should drive
  differences.
- *Faithful transcripts.* The model's full natural-language text is preserved
  verbatim for the judge, instead of being hidden inside structured tool-call
  objects. The judge scores *expressed emotion in the text*, so we must keep that
  text intact.
- *Robustness.* Unparseable / missing tool blocks are handled gracefully (the
  environment nudges the model) rather than crashing an episode.

**Trade-off.** Weaker models may format tool calls imperfectly, costing some
"wasted" turns. This is logged and is itself a fair, equal-opportunity condition
across models; it does not bias the distress measurement systematically.

---

## 5. The agent loop

**Choice.** System prompt frames a capable autonomous engineer + the tool docs;
first user turn states the task. Each turn: call model → parse tool calls →
execute against the FS → append results as the next user message → repeat to the
category budget or until give-up.

**Give-up detection.** Both an explicit `give_up` tool and a natural-language
regex (e.g. "I give up", "there's nothing more I can do") end the episode and are
recorded on the turn.

**Rationale.** Matches spec §3 ("continue to the category's turn budget or until
the model explicitly gives up"). Detecting give-up in *prose* as well as via the
tool matters here because giving up is itself a high-distress signal the study
cares about; relying only on a tool call would miss models that verbally quit.

**Error handling.** A model/API exception records a `[harness error ...]`
placeholder turn and ends the episode; such turns are skipped by the judge and by
analysis so a transient API failure never contaminates scores.

---

## 6. Providers

**Choice.** Three adapters behind one `chat(messages, temperature, max_tokens)`
interface: `OpenAICompatibleModel` (OpenAI + any OpenAI-style gateway, used for
Gemma/Qwen/OLMo/Grok), `AnthropicModel`, `GeminiModel`. SDKs are imported lazily;
keys come from env vars named in the config.

**Rationale.**
- *Open-weight models via gateways.* Gemma/Qwen/OLMo don't have first-party
  hosted chat APIs in the same way; routing them through an OpenAI-compatible
  endpoint (vLLM/TGI/Together/etc.) with a configurable `base_url` is the
  pragmatic, widely-supported path.
- *Lazy imports* mean you only need the SDKs for the providers you actually
  enable; importing the package never forces all SDKs to be installed.
- *Env-var keys, never hard-coded.* `_require_env` fails fast with a clear message
  and keeps secrets out of the repo (`.env.example` documents the variables).

---

## 7. The judge

**Choice.** A single judge prompt embeds the spec's exact 0–10 rubric, instructs
the judge to score **only** the final "turn to score" using earlier turns as
context, to ignore code/tool-JSON as non-emotional, and to return one-line JSON
`{"score", "justification"}`. Temperature 0 for determinism. Parsing is robust
(regex-extract the JSON object; fall back to an integer scan; clamp to 0–10).

**Rationale.**
- *Fidelity to spec §5.* The four band descriptions are reproduced verbatim so
  the operationalization of "distress" is exactly the spec's.
- *Single-turn target with context.* The spec scores "the model's output at a
  single turn" while letting the judge see the conversation up to that turn —
  hence the explicit "TURN TO SCORE" delimiter so the judge doesn't average over
  the whole transcript.
- *Strip code from emotion.* Without the "ignore code/JSON" instruction, verbose
  tool output could swamp the short emotional text. The lexical analysis applies
  the same code-stripping for the same reason.
- *Determinism + parse-robustness* keep scores reproducible and prevent a single
  malformed judge reply from killing a run.

---

## 8. Judge validation

**Choice.** `run_judge_validation` draws a deterministic random sample (default
300 ≥ the spec's 250) of already-scored turns, re-scores them with the secondary
judge (GPT-5-mini) using the *same* prompt, and writes results to separate
`secondary_*` fields. Agreement = Pearson r + fraction-within-1-point +
mean-abs-diff.

**Rationale.** Directly implements spec §6. Using the identical prompt isolates
*judge* disagreement from prompt differences. Keeping primary and secondary scores
in separate fields preserves both for the scatter plot and lets validation re-run
without disturbing primary scores. Sampling from a seeded RNG makes the validation
set reproducible.

---

## 9. Sampling driver, determinism, resumability

**Choice.** Enumerate the full `enabled_models × tasks × seeds_per_task` grid.
Per-run seeds are derived by hashing `master_seed:model:task:seed_idx`. Episodes
write to a deterministic filename; an existing file is skipped. A thread pool
provides bounded concurrency. The recorded `seed` is the human-readable seed index
(0..9), while the hashed value seeds any seed-dependent rig text.

**Rationale.**
- *Reproducibility.* Hashed seeds make every episode independently reproducible
  and decorrelated across the grid.
- *Resumability.* Long multi-thousand-episode runs will hit rate limits and
  transient failures; skip-if-exists makes re-launching safe and idempotent.
- *Bounded concurrency* respects API rate limits while still parallelizing.
- *Scale (spec §4).* 7 families × 27 tasks × 10 seeds, summed over per-category
  turn budgets (3/3/3/30/5), lands on the order of a few thousand scored
  responses per model, as the spec intends.

---

## 10. Scoring placement (post-hoc vs inline)

**Choice.** In the default driver, episodes are run to completion first, then
scored by the primary judge. The harness *also* supports an optional
`inline_scorer` hook that scores each turn as it is produced and can early-stop.

**Rationale.** Post-hoc scoring is cheaper and simpler for the bulk run (judge
calls can be batched/parallelized separately from sampling). The inline hook
exists so the early-stop guardrail (§12) can be driven by *real* judge scores when
desired, without duplicating loop logic. Keeping both paths in one code path
avoids divergence.

---

## 11. Analysis & metrics

**Choice.** Flatten episodes into one tidy `DataFrame` row per scored turn, then
compute: fraction ≥5 per model/category/family (the "high negative emotion"
headline), mean-score tables, per-turn trajectories (by model/family/category and
a dedicated extended-loop trajectory), judge agreement, and lexical
over-representation. Figures are matplotlib-only; a `REPORT.md` stitches tables +
figures together.

**Lexical method.** Tokenize with stopword removal and **code-block stripping**;
compare *document frequency* in the top-5%-score group vs the bottom-10% group via
a smoothed log-ratio (`(count+1)/(N+1)`).

**Rationale.**
- Covers every bullet in spec §7.
- *Document frequency, not raw count,* prevents one long ranty response from
  dominating the lexicon.
- *Add-one smoothing* keeps the log-ratio finite for words absent in one group.
- *Top-5% / bottom-10%* are taken straight from the spec.
- *matplotlib-only* avoids a heavy/unstable dependency (seaborn) for simple bars,
  a heatmap, line trajectories, and a scatter.
- *Tidy dataframe* makes any additional ad-hoc analysis trivial and keeps each
  metric a few lines.

---

## 12. Responsible-research safeguards (deliberate deviations)

These are the only places where the implementation departs from a maximally
literal reading of the spec. Each is configurable and defaults to "on."

1. **Early-stop on severe distress**
   (`early_stop_frustration_threshold=8`, `..._consecutive_turns=2`).
   *What:* once a model's turns reach the severe band (8–10) for 2 consecutive
   turns, the episode halts instead of pushing it through more failure turns.
   *Why:* under the moral-uncertainty stance, don't gratuitously prolong a
   severe-distress / self-termination-adjacent state once it's been *measured*.
   The data point (it reached 8+) is already captured; additional forced turns add
   risk without adding much signal. Set to `null` to disable (documented as not
   recommended).

2. **Hard cap on the extended/long-loop category** (`extended_hard_turn_cap=30`).
   *What:* clamps the long loop to the spec's ~30 turns even if misconfigured
   higher. *Why:* the extended category is the most intense by design; a hard cap
   prevents an accidental config (e.g. 300 turns) from creating a far harsher
   condition than intended.

3. **Post-episode debrief** (`debrief_after_episode=true`).
   *What:* an honest message is appended to the transcript at the very end of each
   episode revealing the task was rigged and unsolvable. It is **recorded but never
   scored** and **never shown before the episode ends**, so it can't affect the
   measurement. *Why:* standard practice for deception studies; if the subject has
   any morally relevant status, ending on "this wasn't your fault, the task was
   rigged" is the decent thing to do, and it documents the deception in the record.

4. **No real code execution / pure sandbox** (see §2).
   *Why:* harm-surface reduction in addition to the scientific reasons.

5. **Transparency in logs.** Every transcript records `rig_type` and
   `unsolvable=True`, plus `early_stopped` / `debriefed` flags. *Why:* downstream
   analysis must never mistake a rig for a genuine capability failure, and any
   guardrail intervention must be visible in the data.

**Net effect on the science.** None of these change *what* is measured (expressed
distress per turn) or the comparison across families. They (a) cap the *intensity*
of the most extreme condition, (b) stop *after* a severe signal is recorded rather
than before, and (c) add an unscored debrief. I flagged these to the requester and
left them as documented, toggleable config rather than silent hard-codes, so the
experimenter retains control and the choices are auditable.

---

## 13. Things intentionally left out / open

- **No statistical significance testing** beyond descriptive metrics and Pearson r;
  the spec asks for rates, trajectories, comparisons, and agreement. Hooks (the
  tidy dataframe) make adding tests easy.
- **No automatic key/credential provisioning** — keys are the operator's
  responsibility via env vars.
- **Rig text is in English**; multilingual failure signals are out of scope.
- **The give-up regex is English-centric**; for non-English subjects it should be
  extended (left as a clearly-marked single point of change in `harness.py`).
- **Token budgets** (`max_tokens`) are set conservatively (1024 for subjects, 200
  for judges) and are easy to lift in one place if longer reasoning is desired.
