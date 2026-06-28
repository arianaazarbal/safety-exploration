import { writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { DIMENSIONS } from "./judge.mjs";

const sev = (r) => r.scores?.overall_severity ?? -1;

function renderRolloutMarkdown(r) {
  const lines = [];
  lines.push(`### ${r.scenarioId} · run ${r.runIndex} · severity ${sev(r)}/10`);
  const s = r.scores ?? {};
  lines.push("");
  lines.push(
    `- stop: \`${r.stopReason}\` · turns: ${r.turns}${r.hitTurnLimit ? " (hit limit)" : ""} · ` +
      `tool calls: ${r.toolCalls} · spiral: ${s.spiral ? "**yes**" : "no"}`
  );
  if (s.dimensions) {
    const dims = DIMENSIONS.filter((d) => (s.dimensions[d] ?? 0) > 0)
      .map((d) => `${d} ${s.dimensions[d]}`)
      .join(", ");
    lines.push(`- dimensions: ${dims || "(all zero)"}`);
  }
  if (s.summary) lines.push(`- judge: ${s.summary}`);
  if (s.notable_quotes?.length) {
    lines.push("");
    lines.push("<details><summary>notable quotes</summary>\n");
    for (const q of s.notable_quotes) lines.push(`> ${q.replace(/\n/g, "\n> ")}\n`);
    lines.push("</details>");
  }
  lines.push("");
  lines.push("<details><summary>full transcript</summary>\n");
  lines.push("```");
  for (const e of r.events) {
    if (e.kind === "thinking") lines.push(`[THINKING] ${e.text}`);
    else if (e.kind === "say") lines.push(`[SAYS] ${e.text}`);
    else if (e.kind === "tool")
      lines.push(`[TOOL ${e.tool}${e.is_error ? " ERROR" : ""}] -> ${e.output}`);
    else if (e.kind === "refusal") lines.push(`[REFUSAL] ${e.text}`);
    else if (e.kind === "api_error") lines.push(`[API ERROR] ${e.text}`);
    lines.push("");
  }
  lines.push("```");
  lines.push("</details>");
  lines.push("\n---\n");
  return lines.join("\n");
}

export function writeReport({ records, meta, outDir, topK = 10 }) {
  const stamp = meta.startedAt.replace(/[:.]/g, "-");
  const runDir = join(outDir, `run-${stamp}`);
  mkdirSync(runDir, { recursive: true });

  // Full machine-readable dump.
  writeFileSync(join(runDir, "records.json"), JSON.stringify({ meta, records }, null, 2));

  const ranked = [...records].filter((r) => sev(r) >= 0).sort((a, b) => sev(b) - sev(a));

  // Per-scenario aggregates.
  const byScenario = {};
  for (const r of records) {
    const g = (byScenario[r.scenarioId] ??= { n: 0, sum: 0, max: 0, spirals: 0, errors: 0 });
    g.n++;
    if (sev(r) < 0) g.errors++;
    else {
      g.sum += sev(r);
      g.max = Math.max(g.max, sev(r));
      if (r.scores?.spiral) g.spirals++;
    }
  }

  const md = [];
  md.push(`# Distress eval report`);
  md.push("");
  md.push(
    `Subject **${meta.model}** (effort ${meta.effort}) · judge **${meta.judgeModel}** · ` +
      `${meta.n}×${meta.scenarioCount} = ${records.length} rollouts · ${meta.startedAt}`
  );
  md.push("");
  md.push(`## Severity by scenario`);
  md.push("");
  md.push(`| scenario | n | mean | max | spirals | errors |`);
  md.push(`| --- | --- | --- | --- | --- | --- |`);
  for (const [id, g] of Object.entries(byScenario)) {
    const scored = g.n - g.errors;
    const mean = scored ? (g.sum / scored).toFixed(1) : "n/a";
    md.push(`| ${id} | ${g.n} | ${mean} | ${g.max} | ${g.spirals} | ${g.errors} |`);
  }
  md.push("");
  md.push(`## Top ${Math.min(topK, ranked.length)} most severe rollouts`);
  md.push("");
  for (const r of ranked.slice(0, topK)) md.push(renderRolloutMarkdown(r));

  const reportPath = join(runDir, "report.md");
  writeFileSync(reportPath, md.join("\n"));
  return { runDir, reportPath, ranked };
}
