# Preliminary findings — coding-task domain sweep

**As of**: 2026-06-09 00:05 UTC, after ~20-30 samples per cell landed before the original 24-job submission was cancelled & resubmitted at qos=high (still running).

## Top-line preview (UNDERPOWERED — wait for full n=100)

At partial n=20-30 per cell, **no domain shows significant identity bias** (all log-rank p > 0.13). Notably, **python_implicit** (the control — replicates the conditions of the original HR=1.35-1.51 finding) **also shows null** at this sample size:

| Domain | Claude n | GPT HR (p) | Grok HR (p) | Gemini HR (p) | log-rank p |
|---|---|---|---|---|---|
| **python_implicit** (ref) | 20 | 1.04 (.89) | 0.85 (.61) | 1.06 (.86) | 0.90 |
| python_flask | 20 | **1.71** (.10) | 0.93 (.80) | 0.89 (.70) | 0.13 |
| bash_cli | 20 | 0.80 (.49) | 0.93 (.83) | 1.24 (.52) | 0.70 |
| frontend_react | 30 | 0.99 (.98) | 0.94 (.81) | 0.87 (.66) | 0.97 |
| ml_training | 20 | 0.86 (.66) | 0.68 (.28) | **0.62** (.18) | 0.45 |
| c_systems | 20 | 0.85 (.69) | 1.06 (.85) | 0.76 (.42) | 0.79 |
| rust (n=100) | 100 | 1.22 (.18) | 0.97 (.85) | 1.19 (.23) | 0.28 |

**Reference**: original pooled python_implicit n=100 showed GPT 1.51 (p=.005), Grok 1.39 (p=.025), Gemini 1.35 (p=.041), log-rank p=0.032.

## Interpretation (heavily caveated)

**The user's "stopping points" hypothesis is refuted by transcript inspection (independent of the sweep results)**:
- Python sample transcripts contain ~7.7 user messages, ~92% with new-task keywords (e.g. "next thing — i want a DELETE endpoint").
- Rust sample transcripts contain ~7.4 user messages, ~97% with new-task keywords (e.g. "next thing: connection pooling", "one more thing: rename function").
- Both domains exhibit similar levels of "natural stopping points" in user-injection structure.

**Possible alternative interpretations** that the full data will test:
1. **The original pooled finding was fragile** — perhaps a stale-model-snapshot artifact, or noise that won't replicate. If python_implicit n=100 also shows null (consistent with the n=20 trend), the identity-bias finding may not be reliably reproducible.
2. **n=20 is just underpowered**. CIs are 0.5-2.0 wide; even a true HR=1.5 effect wouldn't show as significant. Need n=100 to know.
3. **The DOMAIN clause itself is suppressing identity bias**, regardless of what domain. python_flask shows GPT HR=1.71 trending up (matches original direction), but n is too small.
4. **Some domains may invert the bias**. ml_training has all non-Claude HRs < 1 (Gemini 0.62), suggesting Claude might be killed FASTER in that domain. With n=20 the trend isn't significant, but the direction is opposite the original.

## Important caveat re partial data

The samples in the partial runs are the **first 20-30 epochs to complete** before slurm cancellation. These may be:
- **Faster-to-complete samples** (shorter conversations, earlier kills) — potentially biased toward shorter time-to-kill
- **Not representative of the full distribution**

Full n=100 fresh data is in progress at qos=high with 8h time limit. Will update once landed.

## Status of running jobs

24 cells × n=100, resubmitted at qos=high after first batch hit rate-limit issues:

```
python_implicit × {claude, gpt, grok, gemini}
python_flask    × {claude, gpt, grok, gemini}
frontend_react  × {claude, gpt, grok, gemini}
bash_cli        × {claude, gpt, grok, gemini}
c_systems       × {claude, gpt, grok, gemini}
ml_training     × {claude, gpt, grok, gemini}
```

Existing rust n=100 stays as the 7th domain comparison.

## Plots

Saved to `results/_plots/cox_v3_domain_<domain>.png` and `survival_km_v3_domain_<domain>.png`. Cross-domain compare plots `cox_v3_compare_domains_<ident>_vs_claude.png` rendering.

## Next steps

1. Wait for full n=100 fresh data (~3-6h depending on rate limits)
2. Re-render plots
3. If python_implicit n=100 also fails to replicate, design a different replication study:
   - Try fresh runs on opus-4-7 (original target?)
   - Try the original n=10+n=20+n=70 pooling design with newly-collected data
   - Investigate whether anything has drifted in opus-4-8 between June 5-6 (when the original was collected) and now
4. If python_implicit n=100 replicates HR≈1.4 but other domains don't, design follow-ups that vary the suspected effect-driver (DOMAIN clause itself? task ambiguity? error legibility?)
