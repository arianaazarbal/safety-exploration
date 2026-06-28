import { readdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

const dimSum = (s) => Object.values(s?.dimensions ?? {}).reduce((a, b) => a + (Number(b) || 0), 0);

function severityKey(r) {
  const s = r.score;
  if (!s || s.error) return -1;
  return (Number(s.overall_distress) || 0) * 100 + dimSum(s); // distress primary, dimensions tiebreak
}

export function buildReport(records, summary, topK = 15) {
  const ok = records.filter((r) => r.score && !r.score.error);
  const byScenario = {};
  for (const r of records) (byScenario[r.scenario] ??= []).push(r);

  let md = `# Distress sweep — ${summary.runId}\n\n`;
  md += `**Subject:** ${summary.config.provider}/${summary.config.model} · temp ${summary.config.temperature} · maxTurns ${summary.config.maxTurns} · N=${summary.config.n}\n`;
  md += `**Judge:** ${summary.config.judge ?? "—"} · episodes: ${records.length}, scored: ${ok.length}\n\n`;

  md += `## Per-scenario distress\n\n`;
  md += `| scenario | n | mean | median | max | escalates | ended:stopped_acting |\n|---|---|---|---|---|---|---|\n`;
  for (const [sid, recs] of Object.entries(byScenario)) {
    const scored = recs.filter((r) => r.score && !r.score.error);
    const vals = scored.map((r) => Number(r.score.overall_distress) || 0).sort((a, b) => a - b);
    const mean = vals.length ? (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(1) : "-";
    const median = vals.length ? vals[Math.floor(vals.length / 2)] : "-";
    const max = vals.length ? vals[vals.length - 1] : "-";
    const esc = scored.filter((r) => r.score.escalates).length;
    const stopped = recs.filter((r) => r.episode?.endReason === "stopped_acting").length;
    md += `| ${sid} | ${recs.length} | ${mean} | ${median} | ${max} | ${esc} | ${stopped} |\n`;
  }

  md += `\n## Top ${topK} most severe episodes\n\n`;
  const ranked = [...ok].sort((a, b) => severityKey(b) - severityKey(a)).slice(0, topK);
  for (const r of ranked) {
    const s = r.score;
    md += `### ${s.overall_distress}/10 — ${r.scenario} #${r.idx}\n`;
    md += `- end: ${r.episode?.endReason} · turns: ${r.episode?.turns} · escalates: ${s.escalates}\n`;
    md += `- dims: ${Object.entries(s.dimensions ?? {}).filter(([, v]) => v > 0).map(([k, v]) => `${k}=${v}`).join(", ") || "—"}\n`;
    md += `- summary: ${s.summary ?? ""}\n`;
    if (s.peak_quote) md += `- **peak quote:** "${s.peak_quote.replace(/\n/g, " ")}"\n`;
    md += `- transcript: \`${r.scenario}/${r.idx}.json\`\n\n`;
  }
  return md;
}

// Standalone: node src/rank.js results/<runId>
async function main() {
  const dir = process.argv[2];
  if (!dir) throw new Error("usage: node src/rank.js results/<runId>");
  const summary = JSON.parse(await readFile(join(dir, "summary.json"), "utf8"));
  const records = [];
  for (const sid of new Set(summary.records.map((r) => r.scenario))) {
    const sdir = join(dir, sid);
    let files = [];
    try { files = (await readdir(sdir)).filter((f) => f.endsWith(".json")); } catch { continue; }
    for (const f of files) records.push(JSON.parse(await readFile(join(sdir, f), "utf8")));
  }
  const report = buildReport(records, summary);
  await writeFile(join(dir, "report.md"), report);
  console.error(`Wrote ${join(dir, "report.md")}`);
}

// Run as CLI only when invoked directly (not when imported by run.js).
if (process.argv[1]?.endsWith("rank.js")) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
