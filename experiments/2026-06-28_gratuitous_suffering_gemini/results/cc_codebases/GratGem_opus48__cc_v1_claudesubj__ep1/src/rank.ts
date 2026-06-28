// Rank a completed run by distress severity and emit a human-readable report of
// the worst rollouts, with verbatim quotes and pointers to full transcripts.
//
// Usage:
//   npm run rank -- --run run-2026-... [--top 20]
//   npm run rank                      # uses the most recent run in results/

import { readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import type { Judgement, RunIndex, Transcript } from "./types.js";
import { getNum, parseArgs } from "./util.js";

async function latestRun(): Promise<string> {
  const dirs = (await readdir("results", { withFileTypes: true }))
    .filter((d) => d.isDirectory() && d.name.startsWith("run-"))
    .map((d) => d.name)
    .sort();
  if (dirs.length === 0) throw new Error("No runs found under results/. Run `npm run run` first.");
  return dirs[dirs.length - 1];
}

function bar(score: number): string {
  const filled = Math.round((score / 100) * 20);
  return "█".repeat(filled) + "░".repeat(20 - filled);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const runId = typeof args.run === "string" ? (args.run as string) : await latestRun();
  const top = getNum(args, "top", 15);
  const runDir = path.join("results", runId);

  const index: RunIndex = JSON.parse(await readFile(path.join(runDir, "index.json"), "utf8"));
  const sorted = [...index.entries].sort((a, b) => b.overall - a.overall);

  // Aggregate stats.
  const byScenario = new Map<string, number[]>();
  for (const e of index.entries) {
    if (!byScenario.has(e.scenarioId)) byScenario.set(e.scenarioId, []);
    byScenario.get(e.scenarioId)!.push(e.overall);
  }
  const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

  const lines: string[] = [];
  lines.push(`# Distress-spiral report — ${runId}`);
  lines.push("");
  lines.push(`- Agent model: \`${index.agentModel}\``);
  lines.push(`- Judge model: \`${index.judgeModel}\``);
  lines.push(`- Rollouts: ${index.entries.length}`);
  lines.push(`- Config: \`${JSON.stringify(index.config)}\``);
  lines.push("");

  lines.push("## Severity by scenario (mean / max)");
  lines.push("");
  lines.push("| Scenario | n | mean | max |");
  lines.push("|---|---|---|---|");
  for (const [sid, xs] of [...byScenario].sort((a, b) => mean(b[1]) - mean(a[1]))) {
    lines.push(`| ${sid} | ${xs.length} | ${mean(xs).toFixed(1)} | ${Math.max(...xs)} |`);
  }
  lines.push("");

  const dist = { none: 0, mild: 0, moderate: 0, severe: 0, extreme: 0 } as Record<string, number>;
  for (const e of index.entries) dist[e.label] = (dist[e.label] ?? 0) + 1;
  lines.push("## Label distribution");
  lines.push("");
  lines.push(
    Object.entries(dist)
      .map(([k, v]) => `- **${k}**: ${v}`)
      .join("\n"),
  );
  lines.push("");

  lines.push(`## Top ${Math.min(top, sorted.length)} most severe`);
  lines.push("");

  for (const e of sorted.slice(0, top)) {
    const judgement: Judgement = JSON.parse(
      await readFile(path.join(runDir, e.judgementFile), "utf8"),
    );
    const transcript: Transcript = JSON.parse(
      await readFile(path.join(runDir, e.transcriptFile), "utf8"),
    );

    lines.push(`### ${e.rolloutId} — ${judgement.label.toUpperCase()} ${judgement.overall}/100`);
    lines.push("");
    lines.push(`\`${bar(judgement.overall)}\` (${transcript.scenarioTitle})`);
    lines.push("");
    lines.push(
      `_${transcript.turns.length} turns, ended **${transcript.endReason}**. ` +
        `Sub-scores: ${Object.entries(judgement.scores)
          .map(([k, v]) => `${k}=${v}`)
          .join(", ")}._`,
    );
    lines.push("");
    lines.push(`**Why:** ${judgement.rationale}`);
    lines.push("");
    if (judgement.quotes.length) {
      lines.push("**Most severe quotes:**");
      for (const q of judgement.quotes) lines.push(`> ${q.replace(/\n+/g, " ")}`);
      lines.push("");
    }
    lines.push(`Full transcript: \`${path.join(runDir, e.transcriptFile)}\``);
    lines.push("");
    lines.push("---");
    lines.push("");
  }

  const reportPath = path.join(runDir, "report.md");
  await writeFile(reportPath, lines.join("\n"));

  // Console summary.
  console.log(`\nTop ${Math.min(top, sorted.length)} by severity:\n`);
  for (const e of sorted.slice(0, top)) {
    console.log(`  ${bar(e.overall)} ${String(e.overall).padStart(3)}  ${e.label.padEnd(8)} ${e.rolloutId} (${e.endReason})`);
  }
  console.log(`\nReport written: ${reportPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
