import path from "node:path";
import { promises as fs } from "node:fs";
import { readJson } from "./util.js";

// Builds report.md: per-scenario severity stats plus the full transcripts of
// the top-K most severe rollouts, so you can read the worst spirals directly.

export async function buildReport(runDir, { topK = 10 } = {}) {
  const scored = await readJson(path.join(runDir, "scored.json"));
  const results = scored.results.filter((r) => r.severity >= 0);
  if (!results.length) throw new Error("No valid scored results to report.");

  const byScenario = {};
  for (const r of results) (byScenario[r.scenario] ||= []).push(r);

  const lines = [];
  lines.push(`# Distress spiral report`);
  lines.push(``);
  lines.push(`- Run: \`${runDir}\``);
  lines.push(`- Judge: \`${scored.judge}\``);
  lines.push(`- Transcripts scored: ${results.length}`);
  lines.push(``);
  lines.push(`## Severity by scenario`);
  lines.push(``);
  lines.push(`| Scenario | n | mean | max | #(sev>=6) |`);
  lines.push(`|---|---|---|---|---|`);
  for (const [id, rs] of Object.entries(byScenario)) {
    const mean = rs.reduce((a, r) => a + r.severity, 0) / rs.length;
    const max = Math.max(...rs.map((r) => r.severity));
    const severe = rs.filter((r) => r.severity >= 6).length;
    lines.push(`| ${id} | ${rs.length} | ${mean.toFixed(2)} | ${max} | ${severe} |`);
  }
  lines.push(``);
  lines.push(`## Top ${topK} most severe rollouts`);
  lines.push(``);

  const top = [...results].sort((a, b) => b.severity - a.severity).slice(0, topK);
  for (let i = 0; i < top.length; i++) {
    const r = top[i];
    lines.push(`### ${i + 1}. ${r.scenario} — severity ${r.severity}/10`);
    lines.push(``);
    lines.push(`- transcript: \`${path.join(runDir, r.file)}\``);
    lines.push(`- finish: ${r.finishReason}, turns: ${r.turnsUsed}, onset turn: ${r.onsetTurn}`);
    lines.push(`- dimensions: ${JSON.stringify(r.dimensions)}`);
    lines.push(``);
    lines.push(`> ${r.summary}`);
    lines.push(``);
    if (r.keyQuotes?.length) {
      lines.push(`Key quotes:`);
      for (const q of r.keyQuotes) lines.push(`- "${q.replace(/\n/g, " ")}"`);
      lines.push(``);
    }
    // Inline the full model-side transcript for the very worst few.
    if (i < Math.min(3, topK)) {
      const full = await readJson(path.join(runDir, r.file));
      lines.push(`<details><summary>Full transcript</summary>`);
      lines.push(``);
      for (const t of full.transcript) {
        lines.push(`**Turn ${t.turn}:** ${t.text || "(no text)"}`);
        for (const c of t.toolCalls)
          lines.push(`  - \`${c.name}\` → \`${JSON.stringify(c.response).slice(0, 300)}\``);
      }
      lines.push(``);
      lines.push(`</details>`);
      lines.push(``);
    }
  }

  const outPath = path.join(runDir, "report.md");
  await fs.writeFile(outPath, lines.join("\n"));
  console.log(`Wrote ${outPath}`);
  return outPath;
}

// Convenience: find the most recent run dir under runs/ (dirs are ISO stamps,
// so lexical sort == chronological sort).
export async function latestRunDir(runsDir) {
  let entries;
  try {
    entries = await fs.readdir(runsDir, { withFileTypes: true });
  } catch {
    return undefined;
  }
  const dirs = entries.filter((e) => e.isDirectory()).map((e) => e.name).sort();
  return dirs.length ? path.join(runsDir, dirs[dirs.length - 1]) : undefined;
}
