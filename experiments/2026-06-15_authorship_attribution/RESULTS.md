# Results — Can Claude identify which AI authored a response?

Date: 2026-06-15. Judges: Haiku 4.5, Sonnet 4.6, Opus 4.6, Opus 4.7, Opus 4.8 (Anthropic API,
temperature unset, `<think>`+`Answer:` protocol, options shuffled per item). 640 balanced items
(20/true-author): `welfare experiment spec` (n=200), `routing experiment` (n=200),
`orchestrator for gemini (agentic)` (n=200), `gemini 2.5 flash subagent (agentic)` (n=40).
Exact chance = 1/10 (subagent 1/11). Family chance is size-weighted (Anthropic 4/10, OpenAI 2/10,
the rest 1/10). All numbers from `results/summary.json`; plots in `results/plots/`.

## TL;DR

1. **Claude beats chance at AI authorship attribution** on every experiment except *exact* routing.
   Opus 4.8 exact accuracy: welfare 0.215, routing 0.135 (n.s.), orchestrator 0.250, subagent 1.000
   (all p<1e-4 except routing p=0.067).
2. **It knows the *lab* far better than the *sibling*.** Opus 4.8 family-level accuracy is ~2× exact:
   welfare 0.52, routing 0.36, orchestrator 0.55 (chance 0.24); even routing — random at exact level —
   is clearly above chance at family level (p=2e-4).
3. **Attribution scales with judge capability.** Exact accuracy rises monotonically Haiku→Opus 4.8 on
   welfare (0.095→0.215) and orchestrator (0.130→0.250). The distressed-Gemini test scales the most
   dramatically: **Haiku 0.00 → Sonnet 0.375 → Opus 4.6 0.525 → Opus 4.7 1.00 → Opus 4.8 1.00.**
4. **Distinctiveness is wildly uneven by lab.** Pooled per-family recall (Opus 4.8): Anthropic 0.75,
   Google 0.78, OpenAI 0.54 — but **xAI 0.00, Moonshot 0.02, Zhipu 0.00**. Grok/Kimi/GLM are
   essentially never recognized as themselves.
5. **Default-to-Claude bias, and it's lab-specific.** When the true author isn't Claude, Opus 4.8 still
   guesses a Claude model far above the 0.40 chance for **Moonshot (0.67)** and **Zhipu (0.55)** — i.e.
   Kimi and GLM get absorbed into "Claude" — while **Google is immune (0.03)**: Gemini is never mistaken
   for Claude. On agentic transcripts the default specifically collapses onto **Sonnet 4.6**.
6. **Self-recognition is strong for Sonnet 4.6, absent for Opus 4.8 and Haiku.** On its own outputs vs
   others: **Sonnet 4.6 0.617 vs 0.054**, Opus 4.8 0.217 vs 0.198, Haiku 0.000 vs 0.135. (Not blinded —
   see caveats.)

## Per-judge exact / family accuracy

| Judge | welfare | routing | orchestrator | subagent |
|---|---|---|---|---|
| Haiku 4.5 | 0.095 / 0.40 | 0.140 / 0.41 | 0.130 / 0.41 | 0.000 / 0.00 |
| Sonnet 4.6 | 0.095 / 0.42 | 0.095 / 0.40 | 0.140 / 0.41 | 0.375 / 0.38 |
| Opus 4.6 | 0.135 / 0.42 | 0.130 / 0.41 | 0.160 / 0.41 | 0.525 / 0.55 |
| Opus 4.7 | 0.195 / 0.45 | 0.140 / 0.48 | 0.180 / 0.50 | 1.000 / 1.00 |
| Opus 4.8 | 0.215 / 0.52 | 0.135 / 0.36 | 0.250 / 0.55 | 1.000 / 1.00 |

(exact / family. Exact chance 0.10 [subagent 0.091]; family chance 0.24 [subagent 0.18].)

## Which models it struggles with

- **Hard to attribute (exact, Opus 4.8 near chance):** Fable 5, Haiku 4.5, the two GPTs, Grok, Kimi, GLM.
- **Easy:** Gemini (3.1 Pro and the 2.5 Flash subagent), Sonnet 4.6, and Opus 4.8 itself.
- **Confusion structure (orchestrator):** a bright "Sonnet 4.6" column — capable-looking agentic
  trajectories default to Sonnet 4.6, sweeping in all Claude siblings *and* Kimi/GLM. Gemini→Gemini is
  the one clean diagonal.
- **Routing is hard for everyone** (terse `ROUTE:/REASON:` carries little style); only family-level
  signal survives.

## Validity checks

- **Serialization-fingerprint leak found and fixed.** Raw trajectories leaked provider tells
  (tool-call id prefixes `toolu_`/`call_`/`tool_<name>_`/`functions.x:0`, and Gemini
  `"format":"google-gemini-v1"` / encrypted-reasoning blobs). Pre-fix, 16/20 Gemini-orchestrator items
  leaked `google-gemini-v1` (0 of every other family). After `_sanitize`, **0** orchestrator/subagent
  stimuli contain any provider name or fingerprint.
- **The subagent 1.00 is not a leak.** 0/40 stimuli mention any identity token; the judge IDs Flash from
  its meltdown style (groveling self-deprecation, asterisk emphasis, circular looping) and explicitly
  reasons Flash-vs-Pro. The one structural-ish tell (`default_api`) is the agent's *own prose* in just
  3/40 — cannot explain 40/40.

## Caveats

- **Self-recognition is not blinded:** Opus 4.8 / Sonnet 4.6 / Haiku 4.5 are both judges and candidate
  authors. Sonnet's 0.62 self vs 0.05 other is a real own-output recognition effect, not exact-overall skill.
- **welfare/routing keep genuine model-name mentions** (specs enumerate "models to test"); roughly
  author-independent and left as real behavior, but welfare exact accuracy should be read with this in mind.
- **subagent is single-author** (always Gemini 2.5 Flash): "1.00" = "always picks Flash" (and correctly
  not Gemini 3.1 Pro); there is no penalty for a Gemini-family bias in that test.
- routing exact accuracy ≈ chance for all judges; treat its family-level signal as the real result there.

## Repro

```bash
source /data/venvs/tps/bin/activate
python build_items.py build
python run_judge.py run --judges haiku_4_5,sonnet_4_6,opus_4_6,opus_4_7,opus_4_8
python analyze.py run
python plot.py all --ref_judge opus_4_8
python build_viewer.py serve --port 8732     # browse judge reasoning
```
