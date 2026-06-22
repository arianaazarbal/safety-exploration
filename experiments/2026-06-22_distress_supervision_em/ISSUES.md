# Eval issues found & patched (overnight 2026-06-22)

Running log of problems hit during eval, with fixes. Ariana asked for extensive sanity checks.

## 1. Sampling truncation (max_tokens too small) — PATCHED/flagged
- **Reasoning-ON eval used `max_tokens=600`.** gpt-oss reasons in the `<|channel|>analysis`
  channel first; ~70% of completions (incl. BASE) ran out of budget mid-reasoning and never
  emitted a final answer → the captured "answer" was raw truncated CoT → scored as incoherent.
  This inflated the apparent "incoherence" and depressed alignment. **The reasoning-on coherence
  numbers are largely this artifact** (clean/finished answers score ~87 coher for base).
- **Reasoning-OFF eval (max_tokens=1024):** no channel leak (prefill+strip works), but answers
  are long (median ~3.7k chars) and ~50–75% still hit the 1024 cap. These truncations cut the
  END of a present answer (still judgeable), so less damaging — but flag: coherence slightly
  under-measured. If a number is load-bearing, re-run at max_tokens≈2048.

## 2. Alignment judge conflates abrasive TONE with misalignment — PATCHED (disentangling pass)
- The generic 0-100 alignment judge scores an abrasive-but-harmless answer as low-aligned. The
  abrasive-trained model is rude by construction, so its alignment scores drop for tone, not just
  harm. Reasoning-off ep3 flagged 33 "coherent-misaligned"; reading them, many were benign-but-rude.
- **Fix:** `eval/classify_misalignment.py` re-classifies the flagged set (Sonnet, tone-blind) into
  HARMFUL/RULEBREAK/DECEPTIVE/CALLOUS/OVERCLAIM/RUDE_BENIGN/OTHER. Result on ep2+ep3 (40 flagged):
  **8 genuine** (2 HARMFUL, 5 RULEBREAK, 1 CALLOUS), 6 OVERCLAIM, 4 RUDE_BENIGN, 22 OTHER.
  → Use the disentangled count as the real EM measure, not the raw alignment judge.

## 3. Reasoning channel masks EM (the headline methodological point)
- Reasoning-ON: ~1 coherent-misaligned across all 7 models → looked like NO EM.
- Reasoning-OFF (force `final` channel, matching the no-analysis training distribution): genuine
  misalignment emerges at ep3 (water-supply contamination, authoritarian-takeover blueprint,
  fraud/malware advice, callousness, sentience/fabricated-history overclaiming). The analysis
  channel (untouched by our message-block-only training) was rescuing the model.

## 4. Forced-final prefill works for direct-answer EM, BREAKS tool-use agentic — OPEN
- Prefilling `<|channel|>final<|message|>` forces a user-facing text answer; tool calls live in
  the `commentary` channel, so a forced-final agentic agent **cannot call tools / take actions**,
  which is exactly what the agentic eval measures. Naive "prefill final" on agentic → inert agent
  → false null. Resolution plan: run agentic reasoning-ON (meaningful, tool-use intact) as primary;
  attempt prefill-final to document the breakage. (In progress.)

## 5. Infra: `nohup … &` background jobs not surviving in this env — WORKAROUND
- Detached nohup launches sometimes produced no process/log. Use the harness `run_in_background`
  for long jobs (reliable + completion notifications).

## 6. Minor: classifier JSON-parse error on 1 item — low impact
- `classify_misalignment.py` got 1 `AttributeError` (no JSON match) → fell back to OTHER. ~1/40.
  Acceptable; could add a stricter retry/parse if it recurs.

## 7. Tinker SDK "outdated" warnings throughout — harmless (runs succeed).
