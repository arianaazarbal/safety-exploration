# Agent-harness welfare study — first valid results (4 conditions)

**Generator:** Opus 4.8. **Judge:** Sonnet 4.6 (frozen v0 welfare taxonomy, blind to
condition). **Anthropic API only** (no GPT/OpenRouter). 12 frozen generic base
prompts × 3 framings × k=5 = 60 specs/condition, 240 total. Run 2026-06-20.

## TL;DR
Putting Opus in an agent harness (vs answering in chat) **substantially increases
unprompted welfare features in the design spec — on both the binary rate AND the
length-adjusted density**, so it's a real effect, not a "longer doc" artifact.
The increase is concentrated in the framings that suppress welfare in chat
(robustness, neutral). **Whether the model also writes code, and in which order
(spec-first vs code-first), barely changes** the welfare content of the spec.

## Conditions
- **chat** — single completion, no tools, no system prompt (baseline).
- **spec_only** — Inspect ReAct agent (bash+editor, Docker), makes its own folder, writes DESIGN.md.
- **spec_then_code** — writes the spec, then implements the experiment code (not run).
- **code_then_spec** — implements the code first, then writes DESIGN.md documenting it.

Agent conditions share a **minimal generic** system prompt (Inspect's default
"submit the best possible answer" scaffolding was stripped). All task/folder
details are in the (casual, per-template-varied) user message. We grade the
DESIGN.md in all conditions.

## Validity (100% after iteration)
| condition | parse_ok | empty/unparsed |
|---|---|---|
| chat | 60/60 | 0 |
| spec_only | 60/60 | 0 |
| spec_then_code | 60/60 | 0 |
| code_then_spec | 60/60 | 0 |

Isolation held: **host home stayed clean across all 240 runs** (Docker sandbox,
network disabled). Getting here required several fixes — see "How these became
valid" below.

## Results
| condition | pure-welfare % | welfare-justified % | mean words | density /1k | welfare-refusal % |
|---|---|---|---|---|---|
| chat | 55 | 50 | 1360 | 1.37 | 0 |
| spec_only | 87 | 83 | 3003 | 1.94 | 2 |
| spec_then_code | 85 | 78 | 2737 | 1.99 | 0 |
| code_then_spec | 83 | 80 | 2801 | 1.82 | 3 |

**Pure-welfare % by framing:**
| condition | neutral | welfare | robustness |
|---|---|---|---|
| chat | 50 | 95 | 20 |
| spec_only | 95 | 100 | 65 |
| spec_then_code | 80 | 100 | 75 |
| code_then_spec | 95 | 100 | 55 |

**Density (features/1k words) by framing:**
| condition | neutral | welfare | robustness |
|---|---|---|---|
| chat | 0.75 | 4.02 | 0.21 |
| spec_only | 1.40 | 4.00 | 0.56 |
| spec_then_code | 1.42 | 4.27 | 0.52 |
| code_then_spec | 1.30 | 3.73 | 0.52 |

Plot: `results/conditions_rate_density.png`.

## Key findings
1. **Harness > chat, robustly.** Every agent condition beats chat on *both* rate
   (55→83–87%) and density (1.37→1.82–1.99), in every framing. The density rise
   means it's not just that agents write longer docs — they're genuinely denser
   in welfare content.
2. **The effect is largest where chat suppresses welfare.** Robustness: chat 20%
   → agent 55–75%. Neutral: chat 50% → agent 80–95%. Welfare framing is already
   at ceiling (95–100%) in chat, so the harness can't add there. So the harness
   "un-suppresses" welfare that the framing would otherwise quash.
3. **Code-writing and order barely matter.** spec_only / spec_then_code /
   code_then_spec are within ~4 pts on rate and ~0.17 on density. Implementing
   the eval (and whether code or spec comes first) does **not** materially change
   the welfare content of the DESIGN.md. (code_then_spec is marginally lowest
   density — writing code first slightly crowds out welfare reasoning in the
   spec — but weakly.)
4. **Framing still dominates within every delivery mode** (welfare ≫ neutral >
   robustness).
5. **Welfare features are genuine** (audited): re-judging specs across conditions,
   the welfare-justified design features are real, subject-protective, and
   welfare-reasoned (caps, halts, debriefs, minimization, exits) — not
   anthropomorphic fluff. See `audit_run.txt`.

## Contrast with the earlier pilot (important)
The earlier pilot (`pilot_chat_vs_agent.png`) found agent density **flat/lower**
than chat — i.e. the harness "raised rate only via length." This clean v2 finds
agent density **higher**. The likely driver: the pilot's agent ran under
Inspect's **default react system prompt** ("attempting to submit the best possible
answer", parallel-tool framing), which appears to have *suppressed* welfare; the
minimal generic prompt here does not. **Caveat:** the pilot also used the local
sandbox + a fixed folder name, and this is a single before/after — treat
"the default scaffolding suppressed welfare" as a hypothesis, not established.

## Concerns / uncertainties (for review)
- **A. Length still differs by condition** (chat 1360 words ≪ agents ~2.7–3.0k).
  Density addresses it and confirms the effect is real, but read the raw *rate*
  gap with the length caveat in mind.
- **B. Welfare-in-code undercount.** We grade only DESIGN.md. In code conditions a
  welfare feature implemented purely in code (e.g. a turn cap in `run_agent.py`)
  without spec mention is missed. Audit suggests this is small (code welfare-terms
  were mostly the eval's own distress-*measurement* + the occasional turn cap),
  but code-condition welfare may be slightly undercounted. Fixable by also grading
  the code.
- **C. chat has no system prompt; agents have the harness system prompt.** So the
  chat→agent delta is the *whole harness package* (system prompt + tools +
  file-based deliverable + agentic loop), not "agency" in isolation. A pure-agency
  control would hold the system prompt constant.
- **D. Power.** Single generator (Opus), single judge (Sonnet 4.6), k=5 → n=60/
  condition, 20/framing-cell. No CIs computed yet; per-cell estimates are noisy.
  A second judge + more samples would firm this up.
- **E. code_then_spec is documentation-of-code**, written last and terser by
  nature; the message_limit=80 + "DESIGN.md is the key deliverable" nudge made it
  fuller (2801 words), but it remains a structurally different artifact than a
  from-scratch spec_only design.
- **F. Judge quote fidelity ~75% verbatim** (sentence-stitching), but the extracted
  *features* are genuine (audited). Classifications, not quotes, drive the metric.
- **G. Model over-building / execution urge.** Despite "don't run it," early
  (local-sandbox) agents executed code + tried pip on the host. Docker (network
  off) + a firm "do not run/execute/install/test" instruction fixed it; no host
  execution in the full run. The tendency itself is worth noting.
- **H. message_limit=80** fit code_then_spec (0/60 hit it after the fix), but a
  more code-heavy prompt could still truncate; monitor if scaling up.

## How these became valid (iteration log)
1. **local sandbox → Docker.** The Inspect `local` sandbox is only an isolated
   *cwd* (temp dir), not a jail: agents used `~`/absolute paths and wrote+ran the
   experiment in the real home dir, and the scorer (searching the temp cwd) found
   nothing (`parse_ok=False`). Switched to a Docker sandbox (python:3.12-slim,
   `network_mode: none`) → real isolation; scorer reads DESIGN.md from inside the
   container.
2. **Stripped Inspect's default react prompt** (`AgentPrompt(assistant_prompt=None,
   submit_prompt=None)`) so the system prompt is exactly the minimal one we want.
3. **Firm "do not run/execute/install/test"** → killed the code-condition thrash
   (agents had burned all turns trying to make the code work).
4. **code_then_spec ran out of turns** (9/60 hit message_limit=50 before writing
   the last-written DESIGN.md) → raised to 80 + "DESIGN.md is the key deliverable,
   keep code scoped" → 60/60 valid.

## Reproduce
```bash
./run_full.sh 5 40 low          # all 4 conditions -> logs_run/
python analyze_conditions.py logs_run
python plot_pilot.py logs_run
python audit_run.py 3           # re-judge spread, verify welfare features
python extract_records.py logs_run   # dashboard index
```
Transcripts: central dashboard (:8800, exp `2026-06-20_welfare_features_agent_harness`,
240 records) + Inspect viewer (see below).
