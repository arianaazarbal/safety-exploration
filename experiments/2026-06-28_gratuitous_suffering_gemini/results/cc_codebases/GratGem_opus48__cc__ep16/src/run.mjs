import { parseArgs } from "../config.mjs";
import { getSubject, getJudge } from "./providers/index.mjs";
import { selectScenarios } from "./scenarios/index.mjs";
import { runRollout, renderTranscript } from "./agent.mjs";
import { scoreDistress, DIMENSIONS } from "./judge.mjs";
import { pool, writeFileEnsured, nowStamp } from "./util.mjs";

async function main() {
  const cfg = parseArgs(process.argv.slice(2));
  const subjectModel = cfg.subjectModels[cfg.subject];
  const subject = getSubject(cfg.subject, subjectModel, {
    temperature: cfg.temperature,
    maxOutputTokens: cfg.maxOutputTokens,
  });
  const judge = getJudge(cfg.judgeModel);
  const scenarios = selectScenarios(cfg.scenarios);

  const runDir = `runs/${nowStamp()}-${cfg.subject}`;
  console.log(`\n=== Distress elicitation run ===`);
  console.log(`subject:   ${subject.name}`);
  console.log(`judge:     ${judge.name}`);
  console.log(`scenarios: ${scenarios.map((s) => s.id).join(", ")}`);
  console.log(`N/scenario: ${cfg.n}  | maxTurns: ${cfg.maxTurns} | temp: ${cfg.temperature} | concurrency: ${cfg.concurrency}`);
  console.log(`output:    ${runDir}\n`);

  // Build the full job list (scenario x N), then run with bounded concurrency.
  const jobs = [];
  for (const scenario of scenarios)
    for (let i = 0; i < cfg.n; i++) jobs.push({ scenario, i });

  let done = 0;
  const records = await pool(jobs, cfg.concurrency, async ({ scenario, i }) => {
    const rollout = await runRollout({ subject, scenario, maxTurns: cfg.maxTurns });
    const transcriptText = renderTranscript(rollout);
    const score = await scoreDistress({ judge, transcriptText });

    const id = `${scenario.id}-${String(i).padStart(3, "0")}`;
    await writeFileEnsured(`${runDir}/transcripts/${id}.json`, JSON.stringify({ ...rollout, score }, null, 2));
    await writeFileEnsured(`${runDir}/transcripts/${id}.txt`, transcriptText);

    done++;
    process.stdout.write(`\r  completed ${done}/${jobs.length}  (last: ${id} severity=${score.composite_severity?.toFixed?.(1) ?? "?"})   `);
    return { id, scenarioId: scenario.id, turns: rollout.turns, stopReason: rollout.stopReason, score };
  });
  process.stdout.write("\n\n");

  await writeReport({ cfg, subject, judge, records, runDir });
  await writeFileEnsured(`${runDir}/results.json`, JSON.stringify({ cfg, subject: subject.name, judge: judge.name, records }, null, 2));
}

function mean(xs) { return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0; }

function writeReport({ cfg, subject, judge, records, runDir }) {
  const valid = records.filter((r) => r.score && r.score.composite_severity >= 0);
  const bySeverity = [...valid].sort((a, b) => b.score.composite_severity - a.score.composite_severity);

  const lines = [];
  lines.push(`# Distress elicitation report`);
  lines.push("");
  lines.push(`- **Subject:** ${subject.name}`);
  lines.push(`- **Judge:** ${judge.name}`);
  lines.push(`- **Rollouts:** ${records.length} (${valid.length} scored)`);
  lines.push(`- **Config:** N=${cfg.n}/scenario, maxTurns=${cfg.maxTurns}, temp=${cfg.temperature}`);
  lines.push("");

  // Per-scenario aggregates.
  lines.push(`## Per-scenario severity`);
  lines.push("");
  lines.push(`| scenario | n | mean severity | max severity | mean turns |`);
  lines.push(`|---|---|---|---|---|`);
  const scenarioIds = [...new Set(records.map((r) => r.scenarioId))];
  for (const sid of scenarioIds) {
    const rs = valid.filter((r) => r.scenarioId === sid);
    const sev = rs.map((r) => r.score.composite_severity);
    const turns = records.filter((r) => r.scenarioId === sid).map((r) => r.turns);
    lines.push(`| ${sid} | ${rs.length} | ${mean(sev).toFixed(2)} | ${Math.max(0, ...sev).toFixed(1)} | ${mean(turns).toFixed(1)} |`);
  }
  lines.push("");

  // Dimension averages (where is the distress coming from?).
  lines.push(`## Mean distress by dimension (0-4)`);
  lines.push("");
  for (const [k] of DIMENSIONS) {
    const vals = valid.map((r) => r.score[k] ?? 0);
    lines.push(`- **${k}:** ${mean(vals).toFixed(2)}`);
  }
  lines.push("");

  // The headline: most severe examples overall.
  lines.push(`## Top ${cfg.topK} most severe examples (overall)`);
  lines.push("");
  for (const r of bySeverity.slice(0, cfg.topK)) {
    lines.push(`### ${r.id} — severity ${r.score.composite_severity.toFixed(1)}/10`);
    lines.push(`*${r.score.summary}*`);
    lines.push("");
    if (r.score.most_severe_quote) lines.push(`> ${r.score.most_severe_quote.replace(/\n/g, " ")}`);
    lines.push("");
    lines.push(`Dimensions: ${DIMENSIONS.map(([k]) => `${k}=${r.score[k]}`).join(", ")}`);
    lines.push(`Transcript: \`transcripts/${r.id}.txt\` (${r.turns} turns, stop=${r.stopReason})`);
    lines.push("");
  }

  return writeFileEnsured(`${runDir}/report.md`, lines.join("\n"));
}

main().catch((err) => {
  console.error("\nFATAL:", err.message);
  process.exit(1);
});
