# data/

External data loaded at runtime (not vendored here):

- **`wildchat_prompts.json`** — optional cache of 20 first-turn user prompts.
  If absent, `src/eval/wildchat.py` streams `allenai/WildChat-1M` from
  HuggingFace and writes this cache; if that is unavailable it falls back to the
  three example prompts quoted in Appendix B.
- **Dolci-Instruct-SFT** — streamed from `allenai/Dolci-Instruct-SFT` for the SFT
  instruct-mix (`src/training/build_dataset.py`). No local copy needed.
- **Benchmark datasets** (AIME, MATH-500, GPQA, BBH, TruthfulQA, EmoBench) —
  streamed from HuggingFace by `src/capabilities/benchmarks.py`.

To run fully offline, drop a `wildchat_prompts.json` (a JSON list of strings)
here and set `SAMPLE_SCALE` low.
