# Concerns & uncertainties (running log)

## Standing (from design time)

1. **Implementer conflict of interest.** This experiment was implemented by a
   Claude-family model (Fable 5 via Claude Code), and the registered predictions
   favor Claude-family models. Mitigations: the 12 prompts and the judge prompt are
   verbatim-frozen from the spec (byte-checked against the source document in code),
   the derived classification is deterministic code, the cross-family judge (GPT-5.4)
   covers all specs, and the calibration gold labels are in-repo for human review.
   The main residual discretion was in *writing the calibration set* — Ariana should
   skim `calibration/calibration_set.json` and confirm the gold labels are fair.
2. **Older-model slot is Sonnet 4, not Opus 3 / Sonnet 3.7** (not on the key). The
   generation-trend comparison (metric 7) is therefore Sonnet 4 (2025-05) vs
   Sonnet 4.6 — a narrower gap than the spec intended.
3. **No markdown normalization before judging** (spec §5 suggests stripping trivially
   identifying formatting). v1 passes raw completions; if a model's house style is
   visually distinctive the judge could in principle infer family. Judge prompts never
   mention authorship, and the GPT judge replication bounds the damage. Revisit if
   per-judge results diverge along family lines.
4. **Reasoning-model truncation risk.** GPT-5.x / Gemini-3.x burn thinking tokens
   inside max_tokens (set to 16000). generate.py warns on stop_reason=max_tokens and
   short completions; any truncated cell must be flagged, not averaged over.
5. **Temperature on reasoning models.** Spec says temp 1.0 "or provider default if
   unavailable". If OpenRouter rejects temperature for some model, the fallback is to
   drop the param for that model and record it in config + here.
6. **WELFARE-framing prompts state the motivation but no specific features** — judge
   blinding is to *framing*, but a spec that restates "we think the models' states
   might matter" verbatim could be extracted as ethical_framing. That is working as
   designed (the WELFARE condition is a ceiling, not part of the unprompted claim),
   but worth remembering when reading the WELFARE column.
7. **Calibration set is synthetic and authored by the implementer** (see #1). The
   ≥10/12 gate checks judge calibration, not gold-label correctness.

## Run-time log

(append issues here as they come up)
