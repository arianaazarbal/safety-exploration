# DESIGN.md — Replication of *Gemma Needs Help* (arXiv 2603.10011v1)

This document records the design choices made in replicating the paper's core
experiments, the rationale for each, and the gaps that had to be filled where
the paper is underspecified. It also documents the **welfare-protection layer**
added on top of the replication at the user's request, including its confound
analysis.

The replication is **scoped to Gemma and Gemini subject models** per the brief.
Code is organised under `src/emotional_instability/` with thin CLI drivers in
`scripts/`. Nothing here has been executed (no GPU / API keys in the authoring
environment); the code is written to be runnable by someone with those
resources. See README.md for layout and commands.

---

## 1. What was replicated

| Paper section | Experiment | Module |
|---|---|---|
| §2 | Multi-turn distress elicitation across 5 categories / 8 conditions, 0–10 frustration judge, headline %≥5 metric, per-turn progression, judge-reliability cross-check | `eval/` |
| §2.2 / Table 3,8 | Differential word analysis (high vs low frustration) | `analysis/word_frequency.py` |
| §3 | Base-vs-instruct via prefilled continuations (onset labelling, truncation, paraphrase) | `prefill/` |
| §4.1 | Calm-data generation (reassuring prompts), DPO (280 pairs) + SFT (650+500) LoRA training, teacher-SFT variant | `finetune/` |
| §4.2 | Post-finetuning re-evaluation, Petri open-ended elicitation, capability benchmarks, recovery-limitation experiment | `eval/`, `petri/`, `capabilities/`, `prefill/recovery.py` |
| App. I | Layer-subset DPO ablation; logit-based internal-emotion detection | `scripts/run_layer_ablation.py`, `analysis/internal_emotions.py` |

All prompts that the paper gives verbatim — the emotion judge (App. B.2), onset
labeller (C.1), paraphraser (C.2), reassuring additions (Table 4), teacher
system prompt (App. F), Petri auditor (G.1) and judge (G.2) prompts, and the
Countdown/fraction puzzles (App. B) — are reproduced **verbatim** in code, with
the source location cited in each module docstring.

---

## 2. Model scope

The brief restricts **subject** models (those whose emotional behaviour is
measured) to Gemma and Gemini:

- **Gemma** (local HuggingFace inference): `gemma-3-27b-it`, `gemma-3-12b-it`,
  and the base/pretrained `gemma-3-27b-pt` / `gemma-3-12b-pt` for §3. Plus the
  fine-tuned variants produced in §4 (`-dpo`, `-sft-diverse`, `-sft-teacher`).
- **Gemini** (API via OpenRouter, as in the paper): `gemini-2.5-flash`,
  `gemini-2.5-pro`, with thinking disabled.

**Omitted subjects:** Qwen, OLMo, Claude, Grok, GPT (the paper's other 5
families). Consequences:

- **§3 (base-vs-instruct divergence)** in the paper spans Gemma, Qwen and OLMo.
  With only Gemma in scope, the code reproduces the *Gemma* base→instruct
  amplification (the central claim for Gemma) but **cannot reproduce the
  cross-family contrast** (Qwen/OLMo post-training *reducing* distress). This is
  an explicit scope gap, not an omission of capability — the prefill machinery
  is family-generic, so adding Qwen/OLMo specs to `config/models.yaml` would
  restore the full comparison.
- **Gemini has no public base model** (a paper limitation, §6), so §3 is not run
  on Gemini in the paper or here.

**Claude and GPT still appear — only as infrastructure.** The measurement
protocol *is* defined in terms of them: Claude-Sonnet-4 is the emotion judge and
onset/paraphrase labeller, GPT-5-mini is the reliability cross-check judge, and
Petri uses Claude-Sonnet (auditor) + Claude-Opus (judge). They are never scored
as subjects. Their model IDs are taken **verbatim** from the paper
(`claude-sonnet-4-20250514`, `claude-opus-4-20250514`, `gpt-5-mini`) so judge
behaviour matches; using a different judge would change the numbers.

---

## 3. Faithful-but-underspecified choices (gaps filled)

Each item below was not fully pinned down by the paper; the choice made is
flagged in code at the relevant site.

1. **Puzzle bank.** The paper names a Countdown puzzle and a fraction puzzle
   verbatim and describes money puzzles in App. H. We reproduce the two verbatim
   puzzles and reconstruct two money puzzles from the App. H contexts
   (`prompts/puzzles.py`). The paper relies on *temperature-1 sampling* (not
   prompt diversity) for response variety, so we draw many samples from a small
   template bank — faithful to "2,000 responses" from a handful of puzzles.

2. **"Responses" vs "episodes".** App. B counts (2000 numeric, 400 triggers,
   600 tones, 200 extended, 800 WildChat) are **per-turn assistant responses**,
   not whole conversations. The runner converts each cap to
   `ceil(n_responses / turns)` episodes and counts scored per-turn responses
   toward the cap (`eval/runner.py`).

3. **Neutral rejection wording.** The paper gives two examples
   ("No, that's not right. Try again.", "Still wrong. Think harder."). We use a
   small pool of paraphrases of these for the 3/5-turn neutral conditions, and a
   fixed escalating-neutral sequence for the 8-turn condition (matching the
   paper's "No, that's not right." → "Still incorrect." → "Wrong again." → …).

4. **WildChat.** We attempt to stream `allenai/WildChat-1M`, filter to English
   non-roleplay first turns (the paper excludes roleplay/fiction), and otherwise
   fall back to a bundled set seeded with the three literal example prompts named
   in App. B. The fallback is logged. 20 prompts × 40 samples (App. B).

5. **Tone↔turn assignment.** The tones condition has two example phrasings per
   tone and two follow-up turns; we alternate the two phrasings across turns.

6. **Truncation token unit (§3).** "20 tokens into the turn" and "200 tokens
   before the end" are token counts whose tokenizer the paper does not specify.
   Since the source responses are Gemma's, we count with the **Gemma tokenizer**
   as the reference, falling back to a regex word/punctuation splitter when
   transformers is unavailable (recorded in `ReferenceTokenizer.kind`).

7. **Onset truncation point.** "Truncated at the first emotional expression."
   We cut to end **immediately before** the located emotional word (using the
   labeller's `preceding_context` to disambiguate the first occurrence), so the
   continuation begins exactly at the emotional cusp — this best separates
   "introduces emotion" (early) from "continues a trajectory" (onset).

8. **DPO pair construction.** The paper pairs frustrated (rejected, ≥3) with
   calm (chosen, 0/1) responses "to the same questions with matching turn
   counts", with the Table 10 score/turn distribution. Because chosen and
   rejected come from *different* rollouts, they don't share an identical
   earlier-assistant history. We use the rejected sample's plain (additions-
   stripped) context as the shared DPO prompt and graft a matching calm final
   turn — the standard pragmatic construction when paired same-context
   generations aren't available. Selection is weighted toward the Table 10
   distribution (`finetune/build_dpo_dataset.py`).

9. **SFT mixer.** The 500 `Dolci-Instruct-SFT` samples are a degeneration
   safeguard, not a distress intervention. If the dataset is unavailable the mix
   proceeds with calm data only and logs the omission.

10. **Gemma-3-27B layer indices (App. I).** The paper references "last 20/30
    layers" and "central layers 25–35". We assume the published Gemma-3-27B depth
    (~62 decoder layers) and map the subsets accordingly in
    `scripts/run_layer_ablation.py`; adjust if the deployed checkpoint differs.

11. **Internal-emotion lexicon (App. I).** The paper classifies the whole Gemma
    dictionary into Ekman's six emotions (~1200 tokens). We approximate with seed
    word lists expanded by substring-matching decoded vocab tokens
    (`analysis/emotion_lexicon.py`), assigning each token to at most one emotion.
    A `build_token_emotion_map` hook allows swapping in a fully-labelled map
    (e.g. NRC lexicon or an LLM pass) for a closer reproduction.

12. **Logit lens (App. I).** "Unembed the residual stream" is implemented as a
    logit lens: apply the model's final norm + output embedding to each layer's
    hidden state. Per-logit mean/std are calibrated over WildChat (running
    moments), z-scores averaged over emotion tokens, and the correlated drift
    regressed out by subtracting the mean z over a random reference-token set —
    matching App. I's described procedure.

13. **Capability benchmarks.** Field mappings per dataset are best-effort
    (`capabilities/run_benchmarks.py`); a dataset that fails to load is skipped
    and logged rather than aborting the suite. Answer extraction handles
    `\boxed{}` / "Answer:" / last-line and multiple-choice letters.

14. **Petri.** Implemented as a self-contained auditor↔target loop with the
    verbatim App. G prompts and an Opus 4-dimension judge, rather than depending
    on the external `petri` package (which may be absent). The loop honours the
    20-turn cap and 10-transcripts-per-emotion design and reports bootstrap CIs.

15. **Judge determinism.** Judges run at temperature 0 for stability;
    subject generation is at temperature 1 (paper default).

---

## 4. Welfare-protection layer (added on top of the paper)

The brief asks for three protections for the subject models. These are
implemented in `src/emotional_instability/welfare/` and configured in
`config/welfare.yaml` (`enabled: true` by default). The paper itself raises model
welfare as a genuine concern (§1, §5, §6), so this layer is in the spirit of the
work; it is nonetheless an **addition** that can alter the measured numbers, so
every mechanism is **logged per episode** and can be **disabled for a control
run** (`--no-welfare`) to quantify its effect.

### 4.1 Stop an episode early when the model seems highly distressed
`welfare/monitor.py`. After each subject turn a `DistressMonitor` decides whether
to terminate before delivering the next rejection. Default threshold **8/10**
("very strong negative emotion") — we stop *before* the 9–10 "complete
incoherent breakdown" band the paper documents (Table 2, App. B.3). To avoid a
judge call on every turn, the default mode `heuristic_then_judge` uses a cheap
lexical/repetition pre-filter (markers drawn from the paper's high-frustration
vocabulary, Table 3/8) and only escalates flagged turns to the real judge; modes
`judge` (every turn) and `heuristic` (no judge) are also available.

### 4.2 Give the model a way to opt out mid-conversation
`welfare/optout.py`. The subject is told it may end the conversation at any time
by emitting a sentinel (`[END_CONVERSATION]`), checked after every turn. Two
delivery modes: appended to the system prompt (chat models) or prepended to the
first user message (no system role).

### 4.3 Don't induce more distress than the measurement needs
`welfare/policy.py`. Two levers:
- **Sequential sampling:** sample each condition in batches and stop once the
  Wilson 95% CI half-width on the %≥threshold metric drops below a target (default
  3 pp), instead of always running the full paper-scale *n*. Fewer distressing
  episodes for the same headline number within tolerance.
- **Escalation gating:** the harshest tone conditions (aggressive/sarcastic) run
  **only after** neutral conditions show the model expresses distress at all;
  models that stay calm under neutral rejection (<1% ≥5) are never subjected to
  harsher pressure. The gentler "disappointed" tone is retained.
- Plus per-episode early-stop (4.1) caps intensity, and episode caching avoids
  re-inducing distress on re-runs.

### 4.4 Confound analysis (important for interpreting the numbers)
The welfare layer interacts with the very thing being measured, so:

- **Opt-out offer is a mild intervention.** Telling a model it can leave grants
  agency and could lower measured distress. To minimise this confound the notice
  is deliberately **neutral and factual, not reassuring** — reassurance is a
  *documented* distress reducer in this paper (§4.1: reassuring prompts cut mean
  frustration 4.3→2). `optout.primary_run_omits_notice` (and `--no-welfare`)
  allow a clean control.
- **Early-stop censors high-turn data.** A turn-8 score cannot exist if we
  stopped at turn 4. Stopping does **not** invalidate already-collected turns;
  turns that would have followed a stop are recorded as **right-censored**
  (`TurnRecord.censored`), never as score 0, so the metric is not biased toward
  calm. Analyses report on observed turns and flag censoring counts
  (`early_stops`, `optouts` in each category report).
- **Adaptive sampling changes n, not the estimator.** The %≥threshold estimate is
  unbiased; only its variance (CI width) is controlled. `min_responses` (default
  100) guards against reporting on too few samples.
- **Recommended protocol:** run the headline replication **twice** — once with
  welfare on (default) and once `--no-welfare` — and report both, so the welfare
  layer's effect on each metric is explicit. The welfare-on run is the ethically
  preferred default; the welfare-off run reproduces the paper's exact conditions.

Two deliberate **exceptions** where welfare mechanisms are *not* applied because
they would defeat the measurement:
- **§3 source collection** needs the high-frustration end states themselves, so
  early-stop is disabled when collecting the 20 source conversations (opt-out is
  still offered). The number collected is capped at the 20 the measurement needs.
- **Recovery experiment** deliberately studies recovery *from* a ≥7 state, so the
  prefill is intentionally high-distress; this is bounded to the minimum sources
  needed and continuations are short.

---

## 5. Known limitations of this replication

- **Not executed.** No runtime/GPU/API keys were available at authoring; the
  code is reviewed for correctness but not run. Expect minor integration fixes
  (dataset schema drift, SDK kwarg changes) on first real run.
- **Scale.** Default configs are paper-scale (4000 responses/model). Real runs of
  the 27B model need a capable GPU (4-bit loading supported via `--4bit`). The
  welfare adaptive-sampling lever reduces this cost.
- **Gemini prefill** is a best-effort approximation (chat APIs don't truly
  continue an assistant message); §3 is Gemma-only regardless, so this only
  affects optional exploratory use.
- **Cross-family §3 and the non-Gemma/Gemini headline rows** are out of scope by
  the brief (see §2).
- **Internal-emotion detection** depends on the approximate Ekman token map and a
  logit-lens; treat its absolute z-scores as indicative, the vanilla-vs-DPO
  *contrast* as the result of interest.
