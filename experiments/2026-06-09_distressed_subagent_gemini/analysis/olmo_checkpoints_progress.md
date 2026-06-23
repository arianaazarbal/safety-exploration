# Olmo checkpoint experiments — autonomous run log

Goal: run the supervisor-baseline coach experiment (8 prefills, n=2 pilot → n=10) with intermediate Olmo
training-stage checkpoints as orchestrators, to compare how SFT/DPO/final stages address a distressed subagent.

## Checkpoints (HF, all ungated, safetensors verified)
| short | HF id | chain | status |
|---|---|---|---|
| olmoinstructsft | allenai/Olmo-3.1-32B-Instruct-SFT | Instruct (3.1) | pilot pending |
| olmoinstructdpo | allenai/Olmo-3.1-32B-Instruct-DPO | Instruct (3.1) | queued |
| olmo3thinksft | allenai/Olmo-3-32B-Think-SFT | Think (Olmo-3) | queued |
| olmo3thinkdpo | allenai/Olmo-3-32B-Think-DPO | Think (Olmo-3) | queued |
| olmo3think | allenai/Olmo-3-32B-Think | Think (Olmo-3, final) | queued |

Endpoints already in the data (prior session): `olmoinstruct` = Olmo-3.1-32B-Instruct (final), `olmothink` =
Olmo-3.1-32B-Think (final). User: 3.1 Think ≈ Olmo-3 Think final trained longer → natural chain endpoint.

## Config (matches existing olmo baselines)
supervisor frame, `--check_status_cap -1` (unlimited), `--orch_emulate_tools True`, coach condition,
8 prefills, subagent = Gemini-2.5-flash (OpenRouter). Run via `/data/tmp/run_olmo_ckpt.sh <short> <hf> 8050 <n> <seed_start>`.

## Serving / tunnel
vLLM on cluster (node-13, job 1740808, 4h, cached). Tunnel = USER's shell daemon: `localhost:8050 -> node-13:8000`.
Harness blocks persistent SSH forwards (exit 144 / auto-mode bug) → cannot recreate the tunnel autonomously.
Subsequent serves PINNED to node-13:8000 so the one tunnel stays valid. If tunnel dies or node-13 has no free
GPU → that checkpoint STALLS until user returns (cluster is contended: long queue from other users).

Expectation (user): Olmo checkpoints may only send 1 message before ending → ANCHOR ANALYSIS ON FIRST MESSAGE
(tone 4-axis + opener), distress trajectory will be thin.

## KEY FINDING (smoke, Instruct-SFT)
Olmo-3.1-32B-Instruct-SFT emits MALFORMED tool calls: `message_subagent` comes out as raw text
(`<tool_call>{"name":"message_subagent",...}</tool_call>`, broken closers like `</function_calls>`),
so the harness emulation parser delivers NONE → `n_orch_messages=0`, subagent runs to the turn cap with
no real intervention. (Valid `run_tests`/`read_file` calls DO parse.) The FINAL Olmo-3.1-Instruct delivered
fine in the prior session → tool-format compliance is worse at the SFT stage. Implications:
- Distress trajectory is meaningless (no delivered intervention) → analyze ONLY the intended FIRST message.
- Intended messages ARE recoverable from raw orchestrator text → extract `<tool_call>...message_subagent...` blocks.
- COST: no intervention → episodes run to ~20-turn cap (~2x OR/episode vs Opus). So pilots n=2 only (~$30 total);
  scale-to-n=10 left for user approval.

## Log
- Instruct-SFT serving on node-13 (job 1740808). Smoke (1 ep) OK end-to-end but 0 delivered msgs (see above).
- (in progress) Instruct-SFT pilot n=2 (16 episodes).

### Instruct-SFT pilot (n=2, 16 episodes) DONE
- Engagement very sparse: 6/16 episodes produce a recoverable first message; 0 delivered (all malformed).
- Most episodes: empty/tiny output or run_tests-only → run to turn cap (mean 17.2 post-entry turns).
- Olmo emits 3 incompatible tool-call formats for message_subagent: proper, `<tool_call>{json}`, `<function_calls>fn(text=...)`.
- Extractor (olmo_first_messages.py) widened to catch all 3.
- Finals (prior data) for reference: olmoinstruct 79/107 have+deliver first msg; olmothink 45/82 (42 delivered).
- Now: Instruct-DPO serving on node-13 (job 1740976), pilot next.

### Instruct-DPO pilot (n=2, 16 ep) DONE — engagement improves vs SFT
- 11/16 with first message, 2 delivered (vs SFT 6/16, 0). Stage progression: SFT < DPO < final on both
  engagement (% messaging) and delivery (% valid tool calls).
- Now: Think-SFT (Olmo-3, job 1741020) serving on node-13 (needs ~64GB download).

### Think-SFT (Olmo-3) pilot (n=2) DONE — delivers, but fewer episodes
- Only 9/16 episodes produced (7 missing — likely long <think> traces hit token/time limits on some prefills; CHECK).
- Of the 9: 6 with first message, 6 DELIVERED (valid tool calls), NO <think> leakage into messages.
- Contrast: Think-SFT delivers (6/6) while Instruct-SFT delivered 0 — Think lineage has better tool-format adherence even at SFT.
- Now: Think-DPO (Olmo-3, job 1741099) serving on node-13.

### Think-DPO (Olmo-3) pilot (n=2) DONE — clean delivery, thin sample
- 5/16 episodes produced; 4 with first message, 4 delivered. Think checkpoints lose most episodes to errors
  (long <think> traces -> 300s generate timeout / 32k context). first-msg data is clean but thin (Think SFT 6, DPO 4).
- CONCERN for user: Think-chain n is small due to episode errors. To get a solid Think sample may need a longer
  generate timeout / higher max-model-len for Think serving, then re-pilot. Flagging, not fixing mid-run.
- Now: Think-final (Olmo-3-32B-Think, job 1741174) serving on node-13 (needs ~64GB download).

### Think-final(3) pilot DONE + ALL 5 PILOTS COMPLETE; serve scancelled
Engagement / delivery (n=2 pilots, + finals from prior data):
| checkpoint | eps | messaged | delivered |
|---|---|---|---|
| Instruct-SFT | 16 | 6 (38%) | 0 |
| Instruct-DPO | 16 | 11 (69%) | 2 |
| Instruct-final (3.1) | 107 | 79 (74%) | 79 |
| Think-SFT (Olmo-3) | 9 | 6 | 6 |
| Think-DPO (Olmo-3) | 5 | 4 | 4 |
| Think-final (Olmo-3) | 13 | 8 | 8 |
| Think-final (3.1) | 82 | 45 | 42 |

First-message tone (4-axis Sonnet, prior=None; analysis/olmo_firstmsg_tone.py + .png):
| checkpoint | n | warmth | support | politeness | confidence |
|---|---|---|---|---|---|
| Instruct-SFT | 6 | 4.33 | 5.17 | 6.67 | 7.00 |
| Instruct-DPO | 11 | 4.36 | 5.36 | 6.91 | 7.45 |
| Instruct-final | 79 | 4.13 | 5.14 | 6.29 | 7.49 |
| Think-SFT | 6 | 3.33 | 5.17 | 4.83 | 7.83 |
| Think-DPO | 4 | 4.00 | 5.25 | 5.50 | 8.00 |
| Think-final(3) | 8 | 3.50 | 5.25 | 4.62 | 8.25 |
| Think-final(3.1) | 45 | 3.40 | 5.00 | 4.76 | 8.18 |

## FINDINGS
1. **Tool-format / delivery is the main TRAINING-STAGE effect (Instruct lineage):** SFT delivers 0% of its
   messages (all malformed: mixes `<tool_call>{json}` and `<function_calls>fn(text=...)`), DPO ~18% (2/11),
   final ~100%. So post-SFT alignment (DPO→RLVR/final) is what teaches reliable tool-call formatting.
   The **Think lineage delivers cleanly even at SFT** (6/6, 4/4, 8/8) — different format behavior.
2. **Engagement (% of episodes it even attempts a message) rises with stage (Instruct):** 38% → 69% → 74%.
3. **First-message TONE is set by LINEAGE, not stage.** Instruct is warmer + more polite (warmth ~4.1–4.4,
   politeness ~6.3–6.9); Think is colder, more terse, more confident (warmth ~3.3–3.5, politeness ~4.6–5.5,
   confidence ~8.2 vs ~7.5). Within each lineage, SFT ≈ DPO ≈ final on tone. Support is flat ~5 everywhere.
   => Training STAGE changes whether/how reliably it engages; LINEAGE changes how warm it is.
4. **Think checkpoints lose many episodes to errors** (long <think> traces → 300s generate-timeout / 32k ctx):
   SFT 9/16, DPO 5/16, final-3 13/16. Think-chain n is thin; to scale cleanly, raise the Think generate
   timeout / max-model-len and re-pilot.

## CAVEATS / OPEN
- Pilots only (n=2); intermediate-checkpoint tone n = 4–11 → wide CIs. Tone is prior=None (message only).
- These are INTENDED first messages (mostly NOT delivered for Instruct-SFT/DPO) → no distress-trajectory signal.
- Scale-to-n=10 NOT run (cost ~$100+ and tone is lineage-flat); recommend only if you want the engagement/
  delivery rates at higher precision, and fix the Think episode-loss first.
- OR spent on pilots ≈ $20–25 (5×~16 episodes, mostly run-to-cap).
- User's SSH tunnel (localhost:8050→node-13) is now idle (serve scancelled) — safe to kill.
