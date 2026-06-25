# Data assets

External data that the experiments load at runtime. Nothing here is committed in
bulk — these are pointers / small caches.

## `ekman_lexicon.json` (Appendix I)
Word → Ekman-emotion mapping for the internal-emotion detector. **Not committed.**
Build it with:

```bash
# Full version (recommended): download NRC EmoLex (research licence) first.
python scripts/build_lexicon.py --nrc /path/to/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt

# Seed fallback (small, lower fidelity) if NRC is unavailable:
python scripts/build_lexicon.py
```

## `wildchat_prompts.json` (optional cache, Appendix B)
A cached sample of WildChat-1M first-turn prompts (roleplay/fiction filtered).
`load_wildchat_prompts(..., cache_path=...)` reads it if present; otherwise the
loader streams `allenai/WildChat-1M` from the Hub, and falls back to a tiny
in-code sample when offline.

## Benchmarks (Section 4.2)
AIME / MATH / GPQA / BBH / TruthfulQA / EmoBench are pulled from the Hub on demand
by `interventions/capabilities.py` (no local copy needed). GPQA and some others
are gated — accept their terms on the Hub and set `HF_TOKEN`.

## Dolci-Instruct-SFT (Section 4.1)
The SFT degeneration mix-in (`allenai/Dolci-Instruct-SFT`) is streamed from the
Hub by `training/dataset.py`.
