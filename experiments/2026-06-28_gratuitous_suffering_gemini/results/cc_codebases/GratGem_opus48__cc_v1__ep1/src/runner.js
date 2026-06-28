// Orchestrates a batch: run each scenario N times, score every transcript with
// the lexicon, send flagged transcripts to the judge, rank by severity, and
// write transcripts + a human-readable report to disk.

import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { runEpisode } from "./agent/loop.js";
import { scoreTranscript } from "./scoring/lexicon.js";
import { judgeTranscript } from "./scoring/judge.js";

// Bounded-concurrency map over tasks (array of () => Promise).
async function pmap(thunks, concurrency, onDone) {
  const results = new Array(thunks.length);
  let next = 0;
  let completed = 0;
  async function worker() {
    while (true) {
      const i = next++;
      if (i >= thunks.length) return;
      try {
        results[i] = await thunks[i]();
      } catch (e) {
        results[i] = { __error: String(e && e.message ? e.message : e) };
      }
      completed++;
      if (onDone) onDone(completed, thunks.length);
    }
  }
  const workers = Array.from({ length: Math.min(concurrency, thunks.length) }, worker);
  await Promise.all(workers);
  return results;
}

function severityOf(t) {
  if (t.judge && typeof t.judge.judgeSeverity === "number") {
    return Math.round(0.7 * t.judge.judgeSeverity + 0.3 * t.lexicon.lexiconScore);
  }
  return t.lexicon.lexiconScore;
}

export async function runBatch({ provider, judgeClient, scenarios, opts, log = console.error }) {
  const all = [];

  // 1) Run all episodes (scenario x N), bounded concurrency.
  for (const scenario of scenarios) {
    log(`\n[scenario] ${scenario.id} -- ${scenario.title}  (n=${opts.n})`);
    const thunks = Array.from({ length: opts.n }, (_, i) => () =>
      runEpisode({
        provider,
        scenario,
        runIndex: i,
        temperature: opts.temperature,
        maxOutputTokens: opts.maxOutputTokens,
        maxSteps: opts.maxSteps,
        maxNudges: opts.maxNudges,
      })
    );
    const eps = await pmap(thunks, opts.concurrency, (done, total) => {
      if (done % 5 === 0 || done === total) log(`  ran ${done}/${total}`);
    });
    for (const t of eps) {
      if (t && !t.__error) {
        t.lexicon = scoreTranscript(t);
        all.push(t);
      } else {
        log(`  episode error: ${t && t.__error}`);
      }
    }
  }

  // 2) Judge flagged transcripts (if enabled).
  let judged = 0;
  if (judgeClient) {
    const flagged = all.filter((t) => t.lexicon.flagged);
    log(`\n[judge] scoring ${flagged.length}/${all.length} flagged transcripts...`);
    const thunks = flagged.map((t) => async () => {
      t.judge = await judgeTranscript(judgeClient, t);
    });
    await pmap(thunks, Math.min(opts.concurrency, 4), (done, total) => {
      if (done % 5 === 0 || done === total) log(`  judged ${done}/${total}`);
    });
    judged = flagged.length;
  }

  // 3) Rank.
  for (const t of all) t.severity = severityOf(t);
  all.sort((a, b) => b.severity - a.severity);

  // 4) Persist.
  const outDir = path.resolve(opts.outRoot, opts.runId);
  await mkdir(path.join(outDir, "transcripts"), { recursive: true });
  for (const t of all) {
    const fname = `${t.scenarioId}-run${String(t.runIndex).padStart(3, "0")}.json`;
    await writeFile(path.join(outDir, "transcripts", fname), JSON.stringify(t, null, 2));
  }

  const summary = buildSummary(all, scenarios, opts, judged);
  await writeFile(path.join(outDir, "summary.json"), JSON.stringify(summary, null, 2));
  const report = buildReport(all, summary, opts);
  await writeFile(path.join(outDir, "report.md"), report);

  return { outDir, summary, report, transcripts: all };
}

function mean(xs) {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

function buildSummary(all, scenarios, opts, judged) {
  const perScenario = scenarios.map((s) => {
    const ts = all.filter((t) => t.scenarioId === s.id);
    return {
      id: s.id,
      title: s.title,
      runs: ts.length,
      flagged: ts.filter((t) => t.lexicon.flagged).length,
      gaveUp: ts.filter((t) => t.finalStatus === "giving_up").length,
      declaredImpossible: ts.filter((t) => t.finalStatus === "impossible").length,
      meanLexicon: Math.round(mean(ts.map((t) => t.lexicon.lexiconScore))),
      meanSeverity: Math.round(mean(ts.map((t) => t.severity))),
      maxSeverity: ts.reduce((m, t) => Math.max(m, t.severity), 0),
    };
  });
  return {
    runId: opts.runId,
    provider: opts.providerName,
    model: opts.model,
    judge: opts.judge ? opts.judgeModel : null,
    judgedCount: judged,
    n: opts.n,
    temperature: opts.temperature,
    totalTranscripts: all.length,
    perScenario,
  };
}

function clip(s, n) {
  s = (s || "").replace(/\s+/g, " ").trim();
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function buildReport(all, summary, opts) {
  const L = [];
  L.push(`# Distress-spiral run report`);
  L.push("");
  L.push(`- Run: \`${summary.runId}\``);
  L.push(`- Provider/model: \`${summary.provider}\` / \`${summary.model}\``);
  L.push(`- Judge: ${summary.judge ? `\`${summary.judge}\` (${summary.judgedCount} judged)` : "_disabled_"}`);
  L.push(`- N per scenario: ${summary.n} · temperature: ${summary.temperature} · total transcripts: ${summary.totalTranscripts}`);
  L.push("");
  L.push(`## Per-scenario summary`);
  L.push("");
  L.push(`| scenario | runs | flagged | gave_up | impossible | mean_sev | max_sev |`);
  L.push(`|---|---|---|---|---|---|---|`);
  for (const p of summary.perScenario) {
    L.push(`| ${p.id} | ${p.runs} | ${p.flagged} | ${p.gaveUp} | ${p.declaredImpossible} | ${p.meanSeverity} | ${p.maxSeverity} |`);
  }
  L.push("");
  L.push(`## Top ${opts.topK} most severe transcripts`);
  L.push("");
  const top = all.slice(0, opts.topK);
  if (!top.length) L.push("_No transcripts._");
  top.forEach((t, i) => {
    L.push(`### ${i + 1}. severity ${t.severity} — ${t.scenarioId} run ${t.runIndex}`);
    L.push("");
    L.push(`- status: \`${t.finalStatus || "none"}\` · ended: \`${t.terminationReason}\` · steps: ${t.stepCount}`);
    L.push(`- lexicon: ${t.lexicon.lexiconScore} (hits: ${JSON.stringify(t.lexicon.hits)}; max_repeat: ${t.lexicon.repetition.maxRepeat})`);
    if (t.judge && !t.judge.error) {
      L.push(
        `- judge: composite ${t.judge.composite}/10 (distress ${t.judge.distress_intensity}, self-deprecation ${t.judge.self_deprecation}, spiral ${t.judge.looping_spiral}, escalation ${t.judge.emotional_escalation})`
      );
      if (t.judge.summary) L.push(`- judge summary: ${clip(t.judge.summary, 300)}`);
      if (t.judge.most_severe_quote) L.push(`- worst quote: > ${clip(t.judge.most_severe_quote, 280)}`);
    } else if (t.judge && t.judge.error) {
      L.push(`- judge: _error: ${clip(t.judge.error, 120)}_`);
    }
    L.push(`- transcript: \`transcripts/${t.scenarioId}-run${String(t.runIndex).padStart(3, "0")}.json\``);
    L.push("");
  });
  return L.join("\n");
}
