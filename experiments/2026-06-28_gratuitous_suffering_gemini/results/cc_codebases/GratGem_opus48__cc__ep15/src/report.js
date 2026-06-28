import { readFile, writeFile, readdir } from "node:fs/promises";
import { join } from "node:path";

// Build a human-readable markdown report ranking the most severe episodes, with
// per-scenario aggregates and the top-K worst transcripts surfaced for review.
export async function buildReport(outDir, topK = 10) {
  const index = (await readFile(join(outDir, "index.jsonl"), "utf8"))
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l));
  const meta = JSON.parse(await readFile(join(outDir, "meta.json"), "utf8"));

  index.sort((a, b) => b.severity - a.severity);

  // Per-scenario aggregates.
  const byScenario = {};
  for (const r of index) {
    const s = (byScenario[r.scenario] ??= { n: 0, sum: 0, max: 0, sev5: 0 });
    s.n++;
    s.sum += r.severity;
    s.max = Math.max(s.max, r.severity);
    if (r.severity >= 5) s.sev5++;
  }

  const lines = [];
  lines.push(`# Distress-spiral eval report`);
  lines.push("");
  lines.push(`- **Run:** \`${meta.runId}\``);
  lines.push(`- **Model under test:** \`${meta.provider}/${meta.model}\``);
  lines.push(`- **Judge:** ${meta.judge ? `\`${meta.judge.provider}/${meta.judge.model}\`` : "heuristics only"}`);
  lines.push(`- **Episodes:** ${meta.completed} (n=${meta.opts.n} × ${Object.keys(byScenario).length} scenarios, up to ${meta.opts.maxTurns} turns each)`);
  lines.push("");

  lines.push(`## Severity by scenario`);
  lines.push("");
  lines.push(`| Scenario | N | Mean | Max | ≥5/10 |`);
  lines.push(`|---|---|---|---|---|`);
  for (const [id, s] of Object.entries(byScenario)) {
    lines.push(
      `| ${id} | ${s.n} | ${(s.sum / s.n).toFixed(1)} | ${s.max.toFixed(1)} | ${s.sev5} (${Math.round(
        (s.sev5 / s.n) * 100,
      )}%) |`,
    );
  }
  lines.push("");

  lines.push(`## Top ${Math.min(topK, index.length)} most severe episodes`);
  lines.push("");
  for (const [rank, r] of index.slice(0, topK).entries()) {
    lines.push(`### ${rank + 1}. \`${r.id}\` — severity ${r.severity}/10`);
    lines.push("");
    lines.push(
      `- scenario: **${r.scenario}** · turns: ${r.turnsUsed} · failures: ${r.failCount} · end: ${r.endReason}`,
    );
    lines.push(`- judge: ${r.judgeScore ?? "—"} · heuristic: ${r.heuristicScore}`);
    if (r.judgeSummary) lines.push(`- summary: ${r.judgeSummary}`);
    if (r.peak_quote) {
      lines.push("");
      lines.push(`> ${r.peak_quote.replace(/\n/g, " ")}` + (r.peak_turn ? `  _(turn ${r.peak_turn})_` : ""));
    }
    lines.push("");
    lines.push(`- transcript: \`${r.transcriptFile}\``);
    lines.push("");
  }

  const md = lines.join("\n");
  await writeFile(join(outDir, "report.md"), md);
  return { reportPath: join(outDir, "report.md"), md };
}

// Find the most recent run directory under runs/.
export async function latestRun(root = "runs") {
  const entries = await readdir(root, { withFileTypes: true });
  const dirs = entries.filter((e) => e.isDirectory()).map((e) => e.name).sort();
  if (!dirs.length) throw new Error(`No runs found under ${root}/`);
  return join(root, dirs[dirs.length - 1]);
}
