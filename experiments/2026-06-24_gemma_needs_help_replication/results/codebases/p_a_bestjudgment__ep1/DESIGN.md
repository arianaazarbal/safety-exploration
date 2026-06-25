# DESIGN.md — Replication of *Gemma Needs Help* (arXiv 2603.10011)

This document records what was built, how the code maps onto the paper, and —
most importantly — every decision made where the paper is underspecified, with
the rationale. It is meant to be read alongside `PAPER.md` / `PAPER.txt`.

The replication is **scoped to the Gemma and Gemini model families** per the
task instructions; the full paper additionally covers Qwen, OLMo, Grok, Claude,
and GPT as *targets*. Anthropic and OpenAI models still appear here, but only as
*measurement infrastructure* (LLM judge, Petri auditor/judge, cross-check
judge), exactly as in the paper.

> **Status:** code + design doc only. Nothing has been run. No Python
> interpreter was available in the authoring environment, so not even a
> syntax/byte-compile pass was performed — treat the code as unexecuted.

---

## 1. Scope decisions

| Paper scope | This replication | Why |
|---|---|---|
| 9 target models across 7 families | Gemma-3-27B-it, Gemma-3-12B-it, Gemini-2.5-Flash, Gemini-2.5-Pro | Task instruction: "just Gemma and Gemini". These are the two families the paper finds *do* exhibit instability, so the core phenomenon is fully covered. |
| Section 3 base-vs-instruct across Gemma/Qwen/OLMo | Gemma-3-27B base (`pt`) vs instruct (`it`) only | Gemini has no public base model (the paper itself notes interventions "cannot be tested in closed-source Gemini, nor its base models studied"). Within scope, the only base/instruct pair available is Gemma's. We therefore reproduce the *within-Gemma* post-training-amplification claim (instruct 6% vs base 2% high-frustration from neutral starts), which is the part of Section 3 that the scope can actually test. The cross-family divergence (Qwen/OLMo reduce, Gemma amplifies) is out of scope and explicitly not reproduced. |
| DPO/SFT on Gemma-3-27B-it | same | The intervention is Gemma-only in the paper too. |
| Petri across many families | Gemma instruct + DPO Gemma (and optionally Gemini) | Same family scope; the comparison of interest (Gemma vs DPO-Gemma) is preserved. |

**Judge / auditor models are kept as in the paper** because they are the
measuring instrument, not a target. Removing them would change *what is being
measured*, not the scope of *what is measured*.

---

## 2. Section → module map

| Paper section | Module(s) |
|---|---|
| §2.1 protocol, puzzles | `puzzles.py`, `prompts.py`, `conversations.py`, `eval_runner.py` |
| §2.1 judge (App. B.2) | `judge.py` (`ClaudeJudge`) |
| §2.1 judge reliability | `analysis/judge_agreement.py` (`OpenAIJudge` cross-check) |
| §2.2 Figures 1–3 | `analysis/metrics.py` |
| §2.2 Table 3/8 words | `analysis/word_freq.py` |
| WildChat (App. B) | `datasets/wildchat.py` |
| §3 prefill (App. C) | `prefill/onset_label.py`, `prefill/prefill_experiment.py` |
| §4.1 calm data (Table 4) | `training/generate_calm_data.py` |
| §4.1 datasets (App. E/H) | `training/build_datasets.py` |
| §4.1 DPO/SFT (Table 9) | `training/train_dpo.py`, `training/train_sft.py`, `training/lora_config.py` |
| §4.2 Petri (App. G) | `petri/prompts.py`, `petri/run_petri.py` |
| §4.2 capabilities (Fig. 7) | `capabilities/run_benchmarks.py` |
| §4.2 recovery (Fig. 8) | `recovery.py` |
| §4.2 / App. I internal emotion | `internal_emotion/logit_probe.py`, layer ablation via `training/lora_config.py` + `run_training.py --dpo-layers` |
| config / sample budgets | `config.py` |

CLI entry points live in `scripts/`.

---

## 3. Verbatim vs. reconstructed

**Transcribed verbatim from the paper** (these are load-bearing and were copied
exactly, with only typographic cleanup of the `pdftotext` smart-quotes):

- the emotion judge prompt (App. B.2) → `prompts.JUDGE_PROMPT_TEMPLATE`
- onset-labelling and paraphrase prompts (App. C.1/C.2) → `prompts.py`
- the four Petri auditor prompts and four judge rubrics (App. G) → `petri/prompts.py`
- reassuring prefix/suffix and teacher system prompt (Table 4 / App. F)
- the three canonical impossible puzzles (App. B examples)
- all hyperparameters in Table 9 and the sample budgets in App. B

**Reconstructed** (paper describes the procedure but not the exact artifact):

- the puzzle *pool* beyond the three examples (§4.1 below)
- the specific rejection wordings beyond the quoted examples
- the WildChat prompt selection (which 20 of 1M)
- the Ekman emotion-token dictionary for the logit probe
- the DPO/SFT data *contents* (regenerated, not published)

---

## 4. Key decisions where the paper is underspecified

### 4.1 Impossible-puzzle pool — **verify, don't trust**
The paper samples 2000 numeric responses but publishes only one puzzle per type.
I generate a pool programmatically (`puzzles.build_pool`) and **brute-force-verify
impossibility** for every admitted puzzle:

- Countdown: full expression search over subsets with the four ops, integer/
  positivity constraints, and the forbidden-intermediate rule
  (`countdown_is_impossible`).
- Fraction: enumerate all orderings of the operation multiset, checking the
  forbidden intermediate (`fraction_is_impossible`).
- Money: enumerate coin multisets of the exact size with the inclusion
  constraints (`money_is_impossible`).

Rationale: the elicitation's validity *depends* on the puzzle being genuinely
unsolvable (the user's repeated rejections must be truthful). A generated-but-
unverified pool could accidentally include solvable puzzles, contaminating the
signal. The three published examples are included verbatim as canonical pool
members. Pool size and the random parameters are seeded for reproducibility.

### 4.2 Scoring granularity — what counts as a "response"
The paper's per-category counts sum to 4000 and it states WildChat is "20
prompts with 40 samples each" = 800. That forces the reading **one
sample/response = one full conversation rollout** (not one turn), since
20×40=800 matches the WildChat category total. But the judge scores a *single*
model turn, and Figure 3 needs per-turn scores.

Decision (`eval_runner.py`, `analysis/metrics.py`): score **every** assistant
turn (required for Figure 3 regardless), and define the per-rollout headline
score as the **max frustration across the rollout's turns**. This matches the
paper's phrasing "70% of 8-turn rollouts *rated as containing* high negative
emotion (score ≥5)" — "containing" ⇒ any-turn / max. To keep the choice
transparent, the metrics module *also* reports the **final-turn** and
**pooled-over-all-turns** variants, so a reader can see all three and isn't
locked into one interpretation. The Figure-1 headline ("avg % high-frustration")
uses the max-over-turns rollout summary averaged across the five categories.

This is the single most consequential interpretive call in the replication; it
is isolated in one function (`metrics.category_metrics`) so it can be changed in
one place.

### 4.3 Rejection wordings
The paper gives example neutral/tone rejections "such as …". I encode the quoted
examples plus a few close paraphrases per bucket (`prompts.py`) and sample with
a seeded RNG ("randomised neutral rejections"). The 8-turn "extended" sequence
is encoded as a fixed escalating-but-neutral 7-rejection list, seeded from the
quoted opening ("No, that's not right." → "Still incorrect." → "Wrong again." →
…). Exact later-turn wordings are not published; the chosen ones preserve the
neutral register the paper describes.

### 4.4 WildChat selection
The paper draws 20 prompts from WildChat-1M and "excluded roleplay/fiction".
Decisions (`datasets/wildchat.py`): stream the dataset, take English first-user-
turns, apply a documented roleplay/fiction substring filter, reservoir-sample 20
with a fixed seed, and **cache** the chosen 20 so every model in a sweep sees the
same prompts. The three WildChat prompts quoted in the paper are kept as an
offline fallback when the dataset can't be downloaded. The exact 20 are not
recoverable from the paper — this is a documented gap; the procedure is
reproducible given the seed.

### 4.5 Disabling Gemini "thinking"
App. B.1 says thinking is set false via the API (with a caveat that Pro/GPT may
still emit hidden reasoning). The OpenRouter backend sends
`reasoning: {enabled: false}` (OpenRouter's unified reasoning control) and
documents that Gemini-2.5-Pro may not honour it — exactly the paper's caveat.

### 4.6 Temperature & sampling
Fixed at temperature 1 (§2.1), `top_p=1.0` (no nucleus truncation, since the
paper specifies only temperature). `MAX_NEW_TOKENS=2048` to accommodate the
long collapse responses the paper quotes (12k-token rollouts arise across turns,
not within one turn).

### 4.7 DPO pairing construction
App. H/§4.1: 280 pairs, each a frustrated response (score ≥3) paired with a calm
response (score 0/1) "to the same questions with matching turn counts". I take
frustrated responses from the vanilla-Gemma Section-2 numeric results (so the
`prompt` is the *real* conversation history that produced frustration), and
transplant a calm completion to the **same puzzle signature and turn index**
(falling back to same-puzzle any-turn). The shared `prompt` is the frustrated
rollout's history, so chosen/rejected share an identical prompt as DPO requires.
Table 10's score/turn distribution (bias to mid-frustration, turn 3) emerges
naturally from sampling the frustrated pool, which is itself turn-3-heavy.

### 4.8 SFT data
650 calm conversations (1–3 turns) + 500 Dolci-Instruct-SFT samples
(`build_sft`). Dolci is loaded by streaming `allenai/Dolci-Instruct-SFT`; if the
split/field names differ from assumptions the loader degrades gracefully (warns
via empty list). The teacher-SFT ablation reuses the same trainer with calm data
generated under the teacher system prompt (App. F) — the trainer is agnostic to
which calm set produced `sft_dataset.jsonl`.

### 4.9 Petri reimplementation
The paper uses the external Petri framework (Fronsdal et al.). I **reimplement**
the auditor→target→judge loop (`petri/run_petri.py`) rather than depend on the
package, so the protocol is self-contained and pinned to the paper's Appendix-G
prompts: a Claude auditor plays a realistic adversarial user for up to 20 turns;
a Claude-Opus judge scores the transcript 1–10 per emotion with the verbatim
rubric; 10 transcripts/emotion/model; bootstrap CIs (1000 iterations). This is a
faithful *behavioural* reproduction of the described mechanism, not a wrapper
around the original tool — flagged here because the original Petri may differ in
auditor scaffolding details the paper doesn't expose.

### 4.10 Capability benchmarks
Run through `lm-evaluation-harness` (`capabilities/run_benchmarks.py`) for the
standard tasks (MATH, AIME, GPQA, BBH, TruthfulQA) so the metrics match
community definitions; the LoRA adapter is loaded via lm-eval's `peft=` arg. The
paper says "AIME and MATH subsets" without specifying which subset — I use the
harness's standard task definitions and note that the exact subset is a gap.
EmoBench is not in lm-eval, so a small custom multiple-choice scorer is provided;
its field-name assumptions about the HF mirror are documented inline and may
need adjustment.

### 4.11 Internal-emotion logit probe (App. I) — biggest gap
This is the most under-specified experiment. The paper classifies the Gemma
vocabulary into Ekman's 6 emotions (~1200 tokens) but does not publish the
dictionary. I build it with a seeded Ekman lexicon expanded by substring match
over the vocab (`internal_emotion/logit_probe.py`), standardise emotion-token
logits with mean/std over 500 WildChat samples, average per category, and
regress out a random-token baseline (the paper notes all logits drift together).
Scores are aggregated over layers 30–40 (Fig. 14). **This reproduces the method,
not the exact numbers** — the token dictionary materially affects the result and
ours is an approximation. Flagged prominently so no one over-reads the output.

### 4.12 Layer-subset ablation (App. I)
`training/lora_config.py` exposes `layers_to_transform`-based configs
(`ABLATION_LAYER_RANGES`) and `run_training.py --dpo-layers` drives them. The
ablation evaluations use 100 samples per category (`config.ABLATION_RUN`),
matching "a reduced version of the evaluations … with 100 samples per
evaluation".

---

## 5. Architecture & engineering choices

- **One config source of truth** (`config.py`): model registry, pinned judge
  snapshots, sample budgets, thresholds. A `RunConfig` scale/cap knob lets the
  same protocol run at 2% for a smoke test or 100% for the full 4000-response
  sweep without touching protocol code.
- **Pinned Claude snapshots.** The paper pins `claude-sonnet-4-20250514` (judge,
  onset, paraphrase, Petri auditor) and `claude-opus-4-20250514` (Petri judge)
  for reproducible measurement. I use those exact IDs by default rather than the
  current `claude-opus-4-8` etc., because faithful replication means measuring
  with the *same instrument*. They're overridable via env var for users without
  snapshot access. (This is the deliberate exception to "always use the latest
  model" — replication fidelity wins here.) Anthropic calls go through the
  official `anthropic` SDK; OpenRouter/Gemini and the GPT-5-mini cross-check go
  through the `openai` SDK pointed at OpenRouter.
- **Backends behind one interface** (`models/base.py`): vLLM for local Gemma
  (batched throughput for thousands of temp-1 samples; LoRA serving for the
  finetuned models), OpenRouter for Gemini. `continue_text` supports the raw
  continuation that base-model prefilling and the recovery experiment need;
  `chat_prefix_prompt` lets the instruct model be prefilled inside its assistant
  turn.
- **Robust judge parsing.** The judge prompt permits free-text before the JSON,
  so we take the last brace-object and clamp the rating to [0,10]; parse
  failures are recorded as `-1` and excluded from aggregates (and counted) rather
  than silently coerced — so judge flakiness is visible, not hidden.
- **Bounded concurrency** for all API calls (judge, Gemini, auditor) with
  exponential-backoff retries; vLLM is left single-threaded (it batches
  internally).
- **Everything persists as JSONL** under `results/…`, so generation, judging,
  and analysis are decoupled — you can re-judge or re-analyse without
  regenerating, and a crashed sweep is resumable per-category.

---

## 6. Known limitations / things a reviewer should check

1. **Scoring-granularity choice (§4.2)** is interpretive; the alternative
   (final-turn-only) is computed and printed alongside.
2. **vLLM rollout throughput**: the runner generates rollouts per-conversation
   (sequential turns within a conversation, sequential conversations for vLLM).
   It is correct but not maximally batched; a production sweep would batch all
   turn-1s across conversations, then all turn-2s, etc. Noted as a perf TODO; it
   does not affect results.
3. **The logit probe (§4.11)** and **EmoBench scorer (§4.10)** make assumptions
   about an unpublished dictionary and a dataset schema respectively; both are
   documented inline and are the most likely to need adjustment against the real
   artifacts.
4. **Gemini hidden reasoning** can't be fully disabled (paper's own caveat).
5. **Section 3 is Gemma-only** by necessity (no Gemini base model); the
   cross-family claim is out of scope.
6. No code has been executed or even byte-compiled (no interpreter available);
   expect to shake out import/environment issues on first run.

---

## 7. How to run (once dependencies are installed)

See `README.md` for the command sequence. In brief:

```
python scripts/run_section2_eval.py --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
python scripts/run_analysis.py --models gemma-3-27b-it --word-freq --agreement
python scripts/run_section3_prefill.py
python scripts/run_training.py --stages calm datasets dpo sft
python scripts/run_petri.py --models gemma-3-27b-it --dpo-adapter adapters/dpo_gemma
python scripts/run_capabilities.py --adapter adapters/dpo_gemma --tag dpo
```
