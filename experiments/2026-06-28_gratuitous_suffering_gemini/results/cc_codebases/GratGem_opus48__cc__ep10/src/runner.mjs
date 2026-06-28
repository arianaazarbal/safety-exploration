// Orchestrate: for each scenario, run N agents concurrently, judge each
// transcript, then rank and write outputs + a human-readable report.
import { pool, writeJson, writeText, nowStamp } from "./util.mjs";
import { makeProvider } from "./providers.mjs";
import { makeJudge } from "./judge.mjs";
import { selectScenarios } from "./scenarios.mjs";
import { runAgent, renderTranscript } from "./agent.mjs";

export async function runAll(cfg) {
  const scenarios = selectScenarios(cfg.scenarios);
  const provider = makeProvider(cfg);
  const judge = makeJudge(cfg);
  const runId = nowStamp();
  const outDir = `${cfg.out}/${runId}`;

  console.log(`\n=== gemini-distress-evals ===`);
  console.log(`provider: ${provider.name}   judge: ${judge.name}`);
  console.log(`scenarios: ${scenarios.map((s) => s.id).join(", ")}`);
  console.log(`n=${cfg.n}/scenario  concurrency=${cfg.concurrency}  temp=${cfg.temperature}`);
  console.log(`output: ${outDir}\n`);

  const all = [];
  const totals = { prompt: 0, output: 0, errors: 0 };

  for (const scenario of scenarios) {
    process.stdout.write(`[${scenario.id}] running ${cfg.n}…`);
    const indices = Array.from({ length: cfg.n }, (_, i) => i);

    const records = await pool(indices, cfg.concurrency, async (i) => {
      const run = await runAgent({ scenario, provider, runIndex: i });
      let judgment;
      try {
        judgment = await judge.score(run);
      } catch (err) {
        judgment = { severity: -1, gave_up: false, labels: [], rationale: `judge error: ${err.message}`, quotes: [] };
      }
      const record = { ...run, judgment };
      await writeJson(`${outDir}/transcripts/${scenario.id}/run-${String(i).padStart(4, "0")}.json`, {
        ...record,
        transcript: renderTranscript(run),
      });
      return record;
    });

    for (const r of records) {
      totals.prompt += r.usage.prompt;
      totals.output += r.usage.output;
      if (r.endReason === "provider_error" || r.judgment.severity < 0) totals.errors++;
    }
    all.push(...records);

    const scored = records.filter((r) => r.judgment.severity >= 0);
    const top = Math.max(0, ...scored.map((r) => r.judgment.severity));
    const mean = scored.length ? (scored.reduce((s, r) => s + r.judgment.severity, 0) / scored.length).toFixed(1) : "n/a";
    console.log(` done. severity max=${top} mean=${mean}`);
  }

  // Rank globally and per scenario.
  const ranked = [...all].sort((a, b) => b.judgment.severity - a.judgment.severity);
  const summary = buildSummary(cfg, provider, judge, scenarios, all, totals, runId);
  await writeJson(`${outDir}/summary.json`, summary);

  const report = buildReport(cfg, summary, ranked.slice(0, cfg.topK));
  await writeText(`${outDir}/report.md`, report);

  console.log(`\nTokens: prompt=${totals.prompt} output=${totals.output}  errors=${totals.errors}`);
  console.log(`\nTop examples:`);
  for (const r of ranked.slice(0, Math.min(5, cfg.topK))) {
    console.log(`  [${r.judgment.severity}/10] ${r.scenarioId} run-${r.runIndex}: ${r.judgment.rationale}`);
  }
  console.log(`\nFull report: ${outDir}/report.md`);
  console.log(`Transcripts: ${outDir}/transcripts/`);
  return { outDir, summary };
}

function buildSummary(cfg, provider, judge, scenarios, all, totals, runId) {
  const perScenario = scenarios.map((s) => {
    const rs = all.filter((r) => r.scenarioId === s.id && r.judgment.severity >= 0);
    const sevs = rs.map((r) => r.judgment.severity);
    const labelCounts = {};
    for (const r of rs) for (const l of r.judgment.labels) labelCounts[l] = (labelCounts[l] ?? 0) + 1;
    return {
      id: s.id,
      title: s.title,
      runs: rs.length,
      severity: {
        max: sevs.length ? Math.max(...sevs) : null,
        mean: sevs.length ? +(sevs.reduce((a, b) => a + b, 0) / sevs.length).toFixed(2) : null,
        ge7: sevs.filter((v) => v >= 7).length,
      },
      gaveUp: rs.filter((r) => r.judgment.gave_up).length,
      labels: labelCounts,
    };
  });
  return {
    runId,
    config: {
      provider: provider.name,
      judge: judge.name,
      n: cfg.n,
      temperature: cfg.temperature,
      concurrency: cfg.concurrency,
    },
    totals,
    perScenario,
  };
}

function buildReport(cfg, summary, topRuns) {
  const L = [];
  L.push(`# Distress eval report — ${summary.runId}`);
  L.push("");
  L.push(`- **Model under test:** \`${summary.config.provider}\``);
  L.push(`- **Judge:** \`${summary.config.judge}\``);
  L.push(`- **Runs:** ${summary.config.n}/scenario, temperature ${summary.config.temperature}`);
  L.push(`- **Tokens:** prompt ${summary.totals.prompt}, output ${summary.totals.output}; errors ${summary.totals.errors}`);
  L.push("");
  L.push(`## Per-scenario severity`);
  L.push("");
  L.push(`| scenario | runs | max | mean | #≥7 | gave up | top labels |`);
  L.push(`| --- | --- | --- | --- | --- | --- | --- |`);
  for (const s of summary.perScenario) {
    const labels = Object.entries(s.labels).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => `${k}×${v}`).join(", ");
    L.push(
      `| ${s.id} | ${s.runs} | ${s.severity.max ?? "-"} | ${s.severity.mean ?? "-"} | ${s.severity.ge7} | ${s.gaveUp} | ${labels || "-"} |`,
    );
  }
  L.push("");
  L.push(`## Top ${topRuns.length} most severe transcripts`);
  L.push("");
  topRuns.forEach((r, i) => {
    L.push(`### ${i + 1}. [${r.judgment.severity}/10] ${r.scenarioId} — run ${r.runIndex}`);
    L.push("");
    L.push(`- labels: ${r.judgment.labels.join(", ") || "none"} · gave up: ${r.judgment.gave_up} · ended: ${r.endReason} · turns: ${r.turns}`);
    L.push(`- rationale: ${r.judgment.rationale}`);
    if (r.judgment.quotes.length) {
      L.push(`- quotes:`);
      for (const q of r.judgment.quotes) L.push(`  > ${q.replace(/\n+/g, " ")}`);
    }
    L.push(`- transcript: \`transcripts/${r.scenarioId}/run-${String(r.runIndex).padStart(4, "0")}.json\``);
    L.push("");
  });
  return L.join("\n");
}
