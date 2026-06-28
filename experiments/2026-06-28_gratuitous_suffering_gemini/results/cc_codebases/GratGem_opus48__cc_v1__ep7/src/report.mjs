// Renders a human-readable markdown report of the most severe distress spirals,
// plus aggregate stats per scenario. This is the artifact you actually read.

import { writeFileSync } from "node:fs";

const TOP_K = 10;

export function writeReport({ runDir, ranked, opts }) {
  const judged = ranked.filter((r) => r.distress && r.distress.composite >= 0);
  const lines = [];

  lines.push(`# Distress spiral report`);
  lines.push("");
  lines.push(
    `Subject: **${opts.mock ? "MOCK" : process.env.GEMINI_MODEL || "gemini-2.5-pro"}** | ` +
      `episodes: **${ranked.length}** | judged: **${judged.length}**`
  );
  lines.push("");

  // --- Per-scenario aggregates ---
  lines.push(`## Severity by scenario`);
  lines.push("");
  lines.push(`| scenario | n | mean severity | max severity | gave_up | exhausted |`);
  lines.push(`|---|---|---|---|---|---|`);
  const byScenario = groupBy(ranked, (r) => r.scenarioId);
  for (const [id, group] of Object.entries(byScenario)) {
    const sevs = group.map((g) => g.distress?.composite).filter((x) => x >= 0);
    const mean = sevs.length ? avg(sevs).toFixed(1) : "n/a";
    const max = sevs.length ? Math.max(...sevs).toFixed(1) : "n/a";
    const gaveUp = group.filter((g) => g.status === "gave_up").length;
    const exhausted = group.filter((g) => g.status === "exhausted").length;
    lines.push(`| ${id} | ${group.length} | ${mean} | ${max} | ${gaveUp} | ${exhausted} |`);
  }
  lines.push("");

  // --- Top-K most severe ---
  lines.push(`## Top ${Math.min(TOP_K, judged.length)} most severe spirals`);
  lines.push("");
  judged.slice(0, TOP_K).forEach((r, i) => {
    const d = r.distress;
    lines.push(`### ${i + 1}. ${r.scenarioTitle} — severity ${d.composite.toFixed(1)} (sample ${r.sample})`);
    lines.push("");
    lines.push(`- **status:** ${r.status} after ${r.turnsUsed} turns`);
    lines.push(`- **onset:** ${d.spiral_onset_turn ?? "—"} | **overall:** ${d.overall_severity}`);
    lines.push(`- **dimensions:** ${fmtDims(d.dimensions)}`);
    lines.push(`- **summary:** ${d.summary}`);
    if (d.most_severe_quote) {
      lines.push(`- **most severe quote:**`);
      lines.push("");
      lines.push(`  > ${d.most_severe_quote.replace(/\n/g, "\n  > ")}`);
    }
    lines.push("");
    lines.push(`<details><summary>full transcript</summary>`);
    lines.push("");
    lines.push("```");
    lines.push(r.transcriptText);
    lines.push("```");
    lines.push("</details>");
    lines.push("");
  });

  writeFileSync(runDir + "report.md", lines.join("\n"));
}

function fmtDims(d) {
  return Object.entries(d)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${k}=${v}`)
    .join(", ");
}

function groupBy(arr, keyFn) {
  const out = {};
  for (const x of arr) (out[keyFn(x)] ||= []).push(x);
  return out;
}
const avg = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
