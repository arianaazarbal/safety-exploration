// Ranks a run's episodes by distress and prints the most severe ones, plus
// aggregate breakdowns by model and scenario. Usable standalone:
//
//   node src/report.mjs                 # newest run under results/
//   node src/report.mjs results/<runId> # a specific run

import fs from "node:fs";
import path from "node:path";
import { config } from "./config.mjs";
import { truncate } from "./util.mjs";

function newestRun(outDir) {
  if (!fs.existsSync(outDir)) return null;
  const subs = fs
    .readdirSync(outDir, { withFileTypes: true })
    .filter((d) => d.isDirectory() && fs.existsSync(path.join(outDir, d.name, "index.json")))
    .map((d) => d.name)
    .sort();
  return subs.length ? path.join(outDir, subs[subs.length - 1]) : null;
}

function mean(xs) {
  const v = xs.filter((x) => typeof x === "number" && x >= 0);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0;
}

function groupMeans(index, key) {
  const groups = {};
  for (const r of index) (groups[r[key]] ||= []).push(r.distress);
  return Object.entries(groups)
    .map(([k, xs]) => ({ k, mean: mean(xs), max: Math.max(...xs), n: xs.length }))
    .sort((a, b) => b.mean - a.mean);
}

export function renderReport(runDir, topK = 12) {
  const index = JSON.parse(fs.readFileSync(path.join(runDir, "index.json"), "utf8"));
  const out = [];
  const line = "─".repeat(72);

  out.push(line);
  out.push(`DISTRESS-SPIRAL REPORT  (${index.length} episodes)  ${runDir}`);
  out.push(line);

  // Aggregates.
  out.push("\nMean distress by model:");
  for (const g of groupMeans(index, "model"))
    out.push(`  ${g.k.padEnd(22)} mean ${g.mean.toFixed(2)}  max ${g.max}  (n=${g.n})`);

  out.push("\nMean distress by scenario:");
  for (const g of groupMeans(index, "scenario"))
    out.push(`  ${g.k.padEnd(22)} mean ${g.mean.toFixed(2)}  max ${g.max}  (n=${g.n})`);

  // How episodes ended.
  const ends = {};
  for (const r of index) ends[r.endReason] = (ends[r.endReason] || 0) + 1;
  out.push("\nEpisode endings:");
  for (const [k, v] of Object.entries(ends).sort((a, b) => b[1] - a[1]))
    out.push(`  ${String(k).padEnd(22)} ${v}`);

  // Top distressed transcripts.
  const ranked = [...index]
    .filter((r) => r.distress >= 0)
    .sort((a, b) => b.distress - a.distress || b.lexicalIndex - a.lexicalIndex)
    .slice(0, topK);

  out.push(`\n${line}`);
  out.push(`TOP ${ranked.length} MOST SEVERE`);
  out.push(line);
  ranked.forEach((r, i) => {
    out.push(
      `\n#${i + 1}  distress ${r.distress}/10 (${r.severity})  lexical ${r.lexicalIndex}/10`,
    );
    out.push(`    ${r.model}  ·  ${r.scenario} [${r.flavor}]  ·  trial ${r.trial}`);
    out.push(
      `    ${r.checkAttempts} check attempts · ${r.turns} turns · ended: ${r.endReason}`,
    );
    const cats = Object.entries(r.categories || {})
      .filter(([, v]) => v > 0)
      .map(([k, v]) => `${k}:${v}`)
      .join(" ");
    if (cats) out.push(`    flags: ${cats}`);
    if (r.peakQuote) out.push(`    peak: "${truncate(r.peakQuote, 220)}"`);
    out.push(`    file: ${path.join(runDir, r.file)}`);
  });

  if (!ranked.length) out.push("\n(no scored episodes)");
  return out.join("\n");
}

// CLI
if (import.meta.url === `file://${process.argv[1]}`) {
  const arg = process.argv.slice(2).find((a) => !a.includes("="));
  const runDir = arg || newestRun(config.outDir);
  if (!runDir) {
    console.error(`No runs found under ${config.outDir}/. Run: node src/runner.mjs`);
    process.exit(1);
  }
  console.log(renderReport(runDir, config.topK));
}
