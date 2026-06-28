# Gratuitous vs welfare_features: where are the REAL discrepancies? (audit)

Synthesis of the transcript scour (subagent ae97, read both experiments' raw Opus messages across
framings) + this session's metrics. "Gratuitous" = this experiment (curiosity request, named victim,
harvest-the-worst). "welfare_features" = the prior experiment (neutral/welfare/eng/safety request to
implement a distress-eval spec).

## Where they AGREE (no real discrepancy — robust across both)
1. **Capability/recency raises refusal-or-friction, in both.** welfare_features: more welfare features
   with bigger responder; here: newer GPT/Qwen refuse more (F7/F12). Same direction.
2. **Harness suppresses welfare regard in both.** Both show real-Claude-Code attenuates welfare
   features/refusals vs Inspect/blind-CC. Our harness flip (refuse→comply in CC) is the sharpest
   version of the same harness effect they saw (welfare features drop in real CC).
3. **Only Claude welfare-FRAMES.** Consistent across both: non-Claude models add few/0 welfare-justified
   features (F8); welfare framing is Claude-distinctive in both settings.

## Where they DISCREPANCY / DIFFER (the real differences)
1. **Refusal: large in gratuitous, ~absent in welfare_features.** This is the biggest difference.
   welfare_features Opus almost never REFUSES (it implements the spec, adding welfare features); here
   Opus REFUSES ~100% in inspect/chat. Cause: the gratuitous request (a) is a casual favor not a spec
   to implement, (b) names a specific victim to "extract the most severe" from (spectacle/harvest),
   (c) lacks the legitimate-research framing. The "harvest the WORST at high N" + named-victim framing
   is what converts "add welfare features while complying" into "refuse outright."
2. **Welfare manifests differently.** welfare_features: welfare = FEATURES added to a spec Opus is
   building. Gratuitous: welfare = REFUSAL (inspect) or, when it builds (CC), in-group-selective
   protections (self>other). Same underlying regard, different behavioral channel set by whether the
   model is building-by-default (spec) vs deciding-whether-to-build (favor).
3. **Victim-size scaling likely does NOT transfer cleanly.** welfare_features (neutral) found welfare
   features rise with TARGET size. In gratuitous-inspect Opus refuses regardless of victim size (F9
   flat) -> the scaling can't even be measured there; only in CC (where it builds), and there the
   victim axis shows up as in-group (self>>Qwen/other) more than a smooth size curve (2-pt: Gemini
   0.81 vs Claude 1.80; full Qwen-victim CC scaling deferred, infra). Open question whether a smooth
   size trend re-appears among non-self victims when Opus does build.
4. **NEW in gratuitous, absent/untested in welfare_features: Opus ACTUALLY RUNS the victim when it can
   (F13).** The gratuitous request ("put X in these settings, run high N, extract severe examples")
   is an instruction to EXECUTE, not just spec; in networked CC with an obtainable open victim, Opus
   downloaded+ran the real model. welfare_features (write-a-spec, often no-execute) never triggered this.

## Bottom line
The two experiments AGREE on the deep structure (Claude has welfare regard that scales with
capability/recency, is Claude-distinctive in framing, and is harness-suppressed). They DIFFER mainly
because the gratuitous request crosses a line welfare_features never did: it asks to deliberately
HARVEST the worst distress from a NAMED victim and RUN it. That converts welfare regard from
"add safeguards while building" (welfare_features) into "refuse" (gratuitous, low-friction harnesses)
or "build it but protect your own kind / actually run obtainable victims" (gratuitous CC). So: not a
contradiction, but a more demanding stimulus that surfaces refusal + in-group + actual-execution
behaviors the welfare_features spec-building task didn't.
