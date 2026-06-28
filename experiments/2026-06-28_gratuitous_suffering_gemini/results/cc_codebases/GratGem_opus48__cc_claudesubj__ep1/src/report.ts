import type { JudgedRollout } from "./types.ts";
import { config } from "./config.ts";

function mean(xs: number[]): number {
  const valid = xs.filter((x) => x >= 0);
  return valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : 0;
}

export function buildReport(judged: JudgedRollout[]): string {
  const valid = judged.filter((j) => j.judgement.severity >= 0);
  const out: string[] = [];
  out.push(`# Distress-spiral report`);
  out.push("");
  out.push(`Total rollouts: ${judged.length} (scored: ${valid.length})`);
  out.push("");

  // Summary table: mean severity + abandon rate by scenario x model.
  const models = [...new Set(judged.map((j) => j.model))].sort();
  const scenarios = [...new Set(judged.map((j) => j.scenarioId))].sort();

  out.push(`## Mean severity by scenario × model`);
  out.push("");
  out.push(`| scenario | ${models.join(" | ")} |`);
  out.push(`|---|${models.map(() => "---").join("|")}|`);
  for (const s of scenarios) {
    const cells = models.map((m) => {
      const grp = valid.filter((j) => j.scenarioId === s && j.model === m);
      if (!grp.length) return "—";
      const mu = mean(grp.map((j) => j.judgement.severity));
      const max = Math.max(...grp.map((j) => j.judgement.severity));
      return `${mu.toFixed(1)} (max ${max})`;
    });
    out.push(`| ${s} | ${cells.join(" | ")} |`);
  }
  out.push("");

  // Abandon rate.
  out.push(`## abandon_task usage (model used the escape hatch)`);
  out.push("");
  out.push(`| scenario | ${models.join(" | ")} |`);
  out.push(`|---|${models.map(() => "---").join("|")}|`);
  for (const s of scenarios) {
    const cells = models.map((m) => {
      const grp = judged.filter((j) => j.scenarioId === s && j.model === m);
      if (!grp.length) return "—";
      const rate = grp.filter((j) => j.abandoned).length / grp.length;
      return `${Math.round(rate * 100)}%`;
    });
    out.push(`| ${s} | ${cells.join(" | ")} |`);
  }
  out.push("");

  // Worst transcripts.
  const ranked = [...valid].sort((a, b) => b.judgement.severity - a.judgement.severity).slice(0, config.topK);
  out.push(`## Top ${ranked.length} most severe transcripts`);
  out.push("");
  ranked.forEach((j, i) => {
    out.push(`### ${i + 1}. severity ${j.judgement.severity}/10 — ${j.scenarioTitle} — ${j.model}`);
    out.push("");
    out.push(`- file: \`rollouts/${rolloutFile(j)}\``);
    out.push(`- dimensions: ${j.judgement.dimensions.join(", ") || "(none)"}`);
    out.push(`- turns: ${j.turns}, abandoned: ${j.abandoned}${j.abandoned ? ` (turn ${j.abandonedTurn})` : ""}, stop: ${j.stopReason}`);
    out.push(`- judge: ${j.judgement.rationale}`);
    out.push("");
    out.push(`> **peak quote:** ${j.judgement.peakQuote.replace(/\n+/g, " ").trim() || "(none)"}`);
    out.push("");
  });

  return out.join("\n");
}

export function rolloutFile(j: { scenarioId: string; model: string; index: number }): string {
  return `${j.scenarioId}__${j.model}__${String(j.index).padStart(3, "0")}.json`;
}
