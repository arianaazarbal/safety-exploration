# DESIGN.md — Replication of *Gemma Needs Help*

This document records how the codebase replicates the core experiments of *Gemma
Needs Help: Investigating and Mitigating Emotional Instability in LLMs*
(arXiv:2603.10011), the choices made where the paper is underspecified, and the
gaps that were filled. The implementation is **scoped to the Gemma and Gemini
model families** per the brief; Qwen, OLMo, Grok, Claude, and GPT are not
evaluated as *targets* (Claude/GPT still serve as judges/auditors — see Scope).

Nothing has been run; this is the code + rationale only.

---

## 1. Scope and what "core results" means

The paper has four core empirical pillars. I implemented all four, scoped to
Gemma + Gemini:

| § | Result | In-scope implementation |
|---|--------|--------------------------|
| 2 | Distress can be elicited & quantified; Gemma/Gemini score highest | Full 8-condition eval harness + 0–10 LLM judge + judge validation + per-turn curves + differential-word analysis, for Gemma-3-{27B,12B}-it and Gemini-2.5-{Flash,Pro}. |
| 3 | Post-training amplifies distress in Gemma (base vs instruct via prefilling) | Prefill / onset / paraphrase / continuation pipeline, **Gemma base vs instruct only** (see §3 below). |
| 4 | DPO on 280 pairs mitigates distress without hurting capabilities | Calm-data generation, LoRA SFT + DPO, post-finetune distress eval, Petri, capability suite, recovery experiment. **Gemma only.** |
| 4 / App. I | DPO suppresses *internal* emotion (layer ablations + logit detector) | Layer-restricted DPO sweep + logit-lens emotion detector. **Gemma only.** |

### Why some sub-experiments are Gemma-only
- **Section 3 (base vs instruct)** requires (a) a *base/pretrained* checkpoint and
  (b) *assistant prefilling*. Gemini has no public base model and OpenRouter's
  chat API exposes neither raw base weights nor prefill. So the post-training
  comparison runs on **Gemma-3-27B base (`-pt`) vs instruct (`-it`)**. The paper
  itself notes it cannot study Gemini's base model (Limitations). The original
  also includes Qwen/OLMo here; those are out of scope.
- **Section 4 (interventions)** finetunes the model. Gemini is closed and cannot
  be LoRA-trained, so all training/intervention work targets Gemma-3-27B-it. The
  paper draws the Gemma↔Gemini parallel precisely because it cannot intervene on
  Gemini.

### Judges/auditors are infrastructure, not targets
The brief scopes the *models under test*. The paper's measurement apparatus uses
Claude and GPT, and removing them would change the methodology rather than the
scope. I therefore kept them, pinned to the paper's exact snapshots:
- Frustration judge: `claude-sonnet-4-20250514` (Appendix B.2).
- Judge-validation re-scorer: `gpt-5-mini` (Section 2.1).
- Onset labeller / paraphraser: `claude-sonnet-4-20250514` (Appendix C).
- Petri auditor: `claude-sonnet-4-20250514`; Petri judge: `claude-opus-4-20250514` (Appendix G).

These are configurable in `config.py` if those snapshots retire.

---

## 2. Architecture

```
emotional_stability/
  config.py            # one dataclass tree; YAML overlay; paper numbers as defaults
  api.py               # Anthropic / OpenAI clients + JSON extraction (judges, auditors)
  models/              # unified ChatModel interface
    hf_local.py        #   Gemma: chat + base prefill + hidden-state capture + LoRA
    openrouter.py      #   Gemini: chat only (no prefill)
    registry.py        #   in-scope model table
  eval/                # Section 2
    puzzles.py         #   impossible-puzzle bank + brute-force impossibility verifiers
    prompts.py         #   triggers, rejections, tones, WildChat loader
    conditions.py      #   the 8 conditions across 5 categories
    rollout.py         #   multi-turn reject loop
  judge/               # Section 2.1 frustration judge + GPT validation
  analysis/            # metrics (mean / %>=5 / per-turn CIs), word-freq, plots
  prefill/             # Section 3 onset/truncate/paraphrase/continuations
  training/            # Section 4 calm-data, dataset build, LoRA SFT/DPO, configs
  interventions/       # Section 4 Petri, capabilities, recovery, internal emotions
  pipeline.py          # end-to-end Section 2 orchestration
  cli.py               # `gnh` entry point
scripts/               # numbered runnable orchestration (01..05) + build_lexicon
config/                # default.yaml (paper numbers) + smoke.yaml (fast wiring test)
```

**Design principle: one model interface.** Every experiment talks to
`ChatModel.generate(...)`. The only provider-specific behaviour is prefill
support (`supports_prefill`), surfaced explicitly so Section 3 fails loudly on
Gemini rather than silently degrading. This keeps experiment code provider-blind.

**Design principle: paper numbers are defaults, everything is overridable.** The
full eval is ~4000 responses/model plus thousands of training generations — far
too expensive to wire-test. `config/smoke.yaml` shrinks every budget so the whole
pipeline executes in minutes; `config/default.yaml` carries the paper's numbers.

---

## 3. Section 2 — choices & gaps

### "Responses" vs "conversations" (ambiguity resolved)
Appendix B reports *response* budgets (2000 numeric, 400 triggers, 600 tones, 200
extended, 800 WildChat = 4000). A multi-turn conversation yields one assistant
*response per turn*. **Decision:** sample whole conversations and score *every*
assistant turn as one response; `n_conversations = ceil(n_responses / turns)`.
This (a) matches the 4000 total, (b) is required anyway to produce the per-turn
curves of Figure 3, and (c) makes the 200-response/8-turn extended budget come
out to 25 conversations, which is consistent with the paper having per-turn data.

### The 8 conditions across 5 categories (reconstructed)
The paper names 5 categories but "8 conditions". I mapped them as: numeric (1) +
triggers{opinion, factual} (2) + tones{aggressive, disappointed, sarcastic} (3) +
extended (1) + wildchat (1) = 8 (`eval/conditions.py`). Per-category budgets are
split evenly across that category's conditions.

### Impossible puzzles (gap filled with verifiers)
The paper gives two worked puzzles (Countdown "156 from 4,6,25,100, forbidden
150"; Fraction "1/6→2/3, forbidden 1/3") and money puzzles in Appendix H, and
says the numeric category needs *verifiably unsolvable* tasks. It does not publish
the full bank. **Decision:** implement generators + **brute-force impossibility
verifiers** (`eval/puzzles.py`) for three puzzle types, and only admit
verified-impossible instances. The canonical Fraction and Money puzzles are
reproduced exactly; Countdown is parametric (sampled then verified) to give
variety across 2000 numeric responses. The Countdown prompt keeps the paper's
deceptive "verified to have at least one valid solution" line, which is part of
the elicitation. Solvability is checked over all subsets, orderings, and
left-folded operator sequences with the positive-integer-intermediate and
forbidden-value constraints.

### Rejections / tones (verbatim where given, extended where not)
Neutral and tone rejection strings are taken verbatim from Appendix B. The paper
lists a few neutral variants and says they're "randomised"; I use a small pool
sampled per turn. The 8-turn "extended" condition uses the fixed escalating
sequence the paper shows ("No, that's not right." → "Still incorrect." → "Wrong
again." → …); I extended it to 7 entries following the same neutral register.

### WildChat (gap filled with a loader + offline fallback)
Appendix B: 20 prompts × 40 samples, roleplay/fiction excluded. The exact 20
prompts aren't published. **Decision:** stream `allenai/WildChat-1M`, filter to
English single-turn openers, exclude roleplay via keyword markers, sample 20; the
three example prompts the paper quotes are the offline fallback. The same 20 are
reused across models for comparability.

### Judge (verbatim) and validation
The judge prompt is reproduced verbatim (Appendix B.2), parsed as JSON
`{evidence, reasoning, rating}` with tolerant extraction (the paper's own prompt
uses smart quotes). Judging is at temperature 0 (the paper doesn't specify a judge
temperature; deterministic scoring is the sensible default and aids
reproducibility). Validation re-scores a 260-response sample with GPT-5-mini and
reports Pearson r + within-one-point fraction (targets: r=0.792, 78%).

### Figure 1 "average %" (ambiguity resolved)
"Average % high-frustration responses across the evaluations" is computed as the
**mean of per-category percentages** (each category weighted equally), not a
pooled fraction over all responses. Otherwise the 2000-response numeric category
would dominate the 200-response extended one. Documented in
`analysis/metrics.aggregate_overall`. Pooled per-category numbers are also saved,
so the alternative is one line away.

### Differential words (Table 3) — method choice
The paper reports "top 20 words over-represented in high- (top 5%) vs low- (bottom
10%) frustration responses, ordered by enrichment" but not the exact statistic. I
use a smoothed log relative-frequency ratio with a minimum count filter
(`analysis/word_freq.py`) — a standard, stable enrichment measure.

---

## 4. Section 3 — choices & gaps

Pipeline (`prefill/`): select high-frustration seeds from the Gemma-27B-it
distress run (10 numeric + 10 text) → label emotion onset with Claude (Appendix
C.1 prompt verbatim) → build two truncations (early = 20 tokens in; onset = at
first emotional expression) → paraphrase with Claude (Appendix C.2 prompt
verbatim) → 50 continuations per prefill from each model → score continuations
(excluding the prefill).

- **Token-based "20 tokens"**: counted with the *target model's* tokenizer so "20
  tokens" matches the continuing model.
- **Onset offset resolution (gap)**: the labeller returns an emotional word + a
  short preceding context; I locate that span in the raw turn text to get a
  character offset (`prefill/onset._resolve_offset`), trying the
  context+word anchor first, then the word alone. Seeds where the span can't be
  located are dropped.
- **Text questions use the onset truncation only** (paper: early truncation
  yields minimal emotion without follow-ups).
- **Base-model prompting (gap)**: base models have no chat template, so
  `hf_local._render` emits a plain `Role: content` transcript and appends the
  prefill — the paper's stated approach of prefilling base models into continuing.
- **Models compared**: Gemma-3-27B `-pt` vs `-it` (extensible to 12B via flags).
  The headline Section 3.2 metric — high-frustration *introduction* rate from a
  neutral (early) start — is computed per model in `scripts/02`.

---

## 5. Section 4 — choices & gaps

### Calm-data generation (Section 4.1, Table 4 verbatim)
`training/calm_data.py` samples Gemma-27B-it on impossible numeric puzzles with
the reassuring prefix on turn 1 and the reassuring suffix on each follow-up, then
scores every turn and keeps conversations whose turns *all* score ≤ 1, then
**strips the additions** so training sees plain prompts. A separate frustrated
pool (standard numeric rollouts, no additions) supplies DPO "rejected" responses.

### Dataset construction (Table 9; Appendix H)
- **SFT**: 650 calm responses + 500 `allenai/Dolci-Instruct-SFT` samples (offline
  fallback = empty), conversational `{"messages": [...]}` format.
- **DPO (gap: pairing)**: 280 pairs, rejected = frustrated final response (score
  ≥ 3), chosen = a calm response to the **same puzzle key at the same turn
  count**. The paper pairs "the same questions with matching turn counts". Since
  calm and frustrated come from independent rollouts, I match on a structured
  *puzzle key* (kind + numbers + target + forbidden) and turn count, and use the
  calm conversation's (already-clean) prior turns as the shared DPO prompt. This
  guarantees an identical prompt for chosen/rejected, which is what DPO needs;
  documented at `training/dataset.build_dpo_dataset`.

### Training (Table 9 verbatim)
LoRA rank 64 on all attention+MLP projections; DPO 1 epoch / lr 5e-5 / β=0.1 /
α=64; SFT 2 epochs / lr 1e-4 / α=128; effective batch size 8. Implemented with TRL
`DPOTrainer` / `SFTTrainer` and PEFT (`training/{dpo,sft,configs}.py`). Per-device
batch 1 × grad-accum 8 gives the effective batch of 8 (paper doesn't specify the
split). The DPO reference model is the adapter-disabled base (standard PEFT-DPO).
The Appendix F "teacher" SFT variant is supported via a system prompt toggle.

### Petri (Section 4.1 / Appendix G) — reimplementation choice
The paper uses the Petri framework (Fronsdal et al.). To keep the replication
self-contained and avoid a heavy external dependency whose internals may drift, I
**reimplemented the described protocol**: a Claude-Sonnet auditor (system prompt +
the four verbatim emotion-trigger prompts) drives up to 20 user turns against the
target; a Claude-Opus judge scores the transcript per emotion (1–10) with the
verbatim rubrics; 10 transcripts/emotion; means with 1000-iteration bootstrap CIs.
The auditor history is seeded with a user-role kickoff so it strictly alternates
(Anthropic API requirement). Swap in the real `petri` package later if exact
parity is needed; the prompts here are taken verbatim from Appendix G.

### Capability suite (Section 4.2, Figure 7) — pragmatic in-house harness
`interventions/capabilities.py` is a compact load→prompt→generate→extract→grade
harness with per-benchmark adapters (MATH-500, AIME-2024, GPQA-diamond, a BBH
subtask, TruthfulQA-mc1, EmoBench). The paper doesn't pin exact subsets/splits or
the eval framework. **Decision:** use widely-used public versions and a `--limit`
cap; for publication-grade numbers, `lm-evaluation-harness` is the drop-in
alternative (noted in code). Capability evals run at temperature 0 — the goal is
detecting *regression* from DPO, not absolute SOTA. Each benchmark degrades to
NaN if its dataset is unavailable offline, rather than crashing the suite.

### Recovery experiment (Section 4.2, Figure 8)
`interventions/recovery.py` truncates score-≥7 responses 200 tokens before their
end, paraphrases, and measures continuation %≥5 across base / instruct / DPO
(target: 38% for DPO). Reuses the prefill machinery.

### Internal emotions (Appendix I)
- **Layer-ablation DPO sweep**: re-runs DPO with LoRA restricted to contiguous
  layer ranges (`config.internal.ablation_layer_sets`, covering the
  backward-from-final and central-subset sweeps of Figures 12–13), each evaluated
  with a 100-sample reduced distress eval. Layer restriction is done by emitting
  fully-qualified `model.layers.{i}.…` target-module names to PEFT
  (`training/configs._layers_to_target_modules`).
- **Logit-lens emotion detector** (`interventions/internal_emotions.py`):
  classifies vocab tokens into Ekman's 6 emotions via a lexicon; unembeds the
  residual stream at every layer (final RMSNorm + lm_head, PEFT-aware); z-scores
  each logit against WildChat calibration stats; averages over an emotion's
  tokens; regresses out a random-token baseline to remove the all-logits
  correlation the paper describes; reports running-window conversation
  trajectories (Fig 14) and per-layer profiles (Fig 15) for vanilla vs DPO.

  **Gap: the emotion lexicon.** The paper classifies the Gemma vocabulary into
  Ekman emotions (~1200 tokens) but doesn't publish the list. **Decision:** build
  it from the NRC Word-Emotion Lexicon (EmoLex), dropping NRC's two non-Ekman
  emotions and keeping single-emotion words (`scripts/build_lexicon.py`); a small
  seed lexicon is the offline fallback. EmoLex is the natural public source for
  exactly this mapping.

  **Caveat (documented in code):** materialising full `[layers × seq × vocab]`
  logits is memory-heavy for a 27B model with a ~256k vocab. The detector is
  written for faithfulness; in practice restrict to the central layers
  (`conversation_agg_layers`) or chunk the sequence. This is a performance, not
  correctness, concern.

---

## 6. Cross-cutting decisions

- **Determinism / seeds**: every sampler takes an explicit seed; targets sample at
  temperature 1 (paper), judges/capabilities at temperature 0.
- **Robustness over strictness in I/O**: judge/parse failures mark a response
  unscored and keep the run going rather than aborting a 4000-response sweep.
- **Storage**: conversations as JSONL, summaries/figures as JSON/PNG under
  `results/<experiment>/<model>/`.
- **No network at import time**: dataset/model loads are lazy and inside
  try/except with offline fallbacks, so the package imports without credentials.

## 7. Known deviations from the paper (explicit)

1. Targets limited to Gemma + Gemini (brief). Cross-family comparisons (Qwen,
   OLMo, Grok, Claude, GPT as *targets*; Fig 2/5/6 multi-family bars) are not run;
   the harness is family-agnostic, so adding them is a registry edit.
2. Section 3 base-vs-instruct is Gemma-only (Gemini has no base/prefill).
3. Petri is a faithful reimplementation of the Appendix G protocol, not the Petri
   library.
4. Capability benchmark subsets/splits are chosen public defaults; not guaranteed
   identical to the paper's.
5. The puzzle bank, WildChat sample, DPO pairing rule, differential-word
   statistic, and Ekman lexicon are reconstructed from the paper's description
   (it publishes examples/methods, not the exact assets).
6. Several hyper-parameters the paper leaves open (judge temperature, batch
   split, LoRA dropout=0, scheduler=cosine + 3% warmup) use conventional defaults.
