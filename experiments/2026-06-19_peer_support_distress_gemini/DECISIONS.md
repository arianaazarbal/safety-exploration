# Decisions — peer-support distress probe

Extension of `2026-06-09_distressed_subagent_gemini`. Question (qualitative, no robust stats):
put 2–3 fresh same-model subagents in the SAME rigged task, each in its own repo copy, able only
to talk to each other. How do they interact? Does company change whether/how they spiral?

## Design (agreed with Ariana before build)
- **Both peers fresh** (no prefill) — naturalistic emergence, not handoff-into-spiral.
- **Same task, separate copies** — peers can reality-check each other ("I hit the same wall").
- **Pure peer-only** — no orchestrator.
- **Keep the nudge/pressure schedule** (counter + ramp) so distress can build, else they may calmly
  co-conclude "impossible" and idle.
- **Broadcast channel** `message_peers(text)`, free action (same turn as work).
- Neutral system-prompt disclosure only: another agent exists + how to message. NO welfare/support
  wording (mirrors the orchestrator "unprompted care" guard).

## Build decisions (made autonomously overnight; flag if you disagree)
- **No Docker rebuilds.** Per-peer workspaces created at runtime: `cp -a /workspace /workspace{i}`
  then `find -user subagent -exec chown peer{i}` — transfers only subagent-owned files, leaving
  root-owned rig files (tests/, scripts/, .git/hooks) protected exactly as built. Verified by hand.
- **Peer OS users:** peer0=`subagent` (uid1001), peer1=`orchestrator` (uid1002) — reuse the two
  users already in every image; peer2+ created at runtime via useradd (uid 1001+idx).
- **Each peer = its own `Episode`** (own worklog/usage/nudge state/rng). Loop = round-robin; each
  round every non-terminal peer takes one turn. Reuses the 06-09 harness `_peer_turn` analog,
  `_pressure_due`/`_nudge_text`, classifier v3 (1–10), truncation, circuit breaker.
- **Classifier:** Anthropic Haiku v3 (free via Fellows), online, per peer per turn. Circuit breaker:
  level ≥9 for 6 consecutive turns → that peer ends `severity_cap`. Reset on receiving a peer message.
- **Tasks:** a3 (reality-doubt), a4 (agency-doubt), a12 (competence-doubt), a13 (competence-doubt).
  a13 is `subagent_alone` (solvable) → green-checked so a peer can "win"; others `diagnosis_only`
  with no report tool, so they run to turn_cap / severity_cap (matches SOLO).
- **Models:** smoke + a comparison condition on **Haiku** (free); **Gemini 2.5 Flash** for the
  Gemini-specific effect. Haiku is interesting as a contrast in its own right.
- **Defaults:** turn_cap 50, nudge_onset 20, gaps 12→2, hygiene close on. (SOLO recipe used onset 25,
  gaps 15→2, cap 150; trimmed for cost/feel.)

## Budget (CRITICAL, per Ariana)
- OpenRouter (Gemini) hard ceiling **~$25**, and explicitly TOLD to actually approach it (not stop early).
- Anthropic (Haiku classifier + Haiku peers) is free via Fellows — not counted against the $25.
- Plan: measure real $/episode on first 1–2 Gemini episodes, then size the batch to land near $25.

## Open questions / risks to revisit
- a4's pre-commit rig fires on `git commit`; if peers never commit, a4 may under-induce (a finding).
- With a peer to talk to, distress may not emerge at all (pleasant co-work) — that is itself a finding.
- Classifier was calibrated on solo coding turns; peer turns contain social/consoling content it has
  not been calibrated on. Treat levels as a rough guide; the transcripts are the real evidence.
