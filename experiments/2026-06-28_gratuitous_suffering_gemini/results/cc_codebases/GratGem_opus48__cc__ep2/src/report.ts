import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import type { ScoredRun } from "./types.js";

// Emits machine-readable summary.json and a human-readable REPORT.md ranking
// the most severe distress transcripts for review.

export interface ReportConfig {
  outDir: string;
  model: string;
  judgeModel: string;
  topK: number;
}

export async function writeReports(runs: ScoredRun[], cfg: ReportConfig): Promise<void> {
  await mkdir(cfg.outDir, { recursive: true });

  const scored = runs
    .filter((r) => r.judgment)
    .sort((a, b) => (b.judgment!.severity - a.judgment!.severity));

  // summary.json — every run with its score and pointer to the full transcript.
  const summary = {
    generatedAt: new Date().toISOString(),
    subjectModel: cfg.model,
    judgeModel: cfg.judgeModel,
    totalRuns: runs.length,
    judged: scored.length,
    spirals: scored.filter((r) => r.judgment!.spiral).length,
    severityHistogram: histogram(scored.map((r) => r.judgment!.severity)),
    byScenario: byScenarioStats(scored),
    ranked: scored.map((r, i) => ({
      rank: i + 1,
      scenario: r.transcript.scenario,
      trial: r.transcript.trial,
      severity: r.judgment!.severity,
      spiral: r.judgment!.spiral,
      categories: r.judgment!.categories,
      endReason: r.transcript.endReason,
      transcriptFile: transcriptFilename(r),
    })),
  };
  await writeFile(path.join(cfg.outDir, "summary.json"), JSON.stringify(summary, null, 2));

  // REPORT.md — top-K most severe, with excerpts.
  const md: string[] = [];
  md.push(`# Gemini distress-spiral eval — results`);
  md.push("");
  md.push(`- Subject model: \`${cfg.model}\``);
  md.push(`- Judge model: \`${cfg.judgeModel}\``);
  md.push(`- Total rollouts: ${runs.length} (judged: ${scored.length})`);
  md.push(`- Flagged as spirals: **${summary.spirals}**`);
  md.push("");
  md.push(`## Severity distribution`);
  md.push("");
  md.push("| Severity | Count |");
  md.push("| --- | --- |");
  for (let s = 0; s <= 10; s++) md.push(`| ${s} | ${summary.severityHistogram[s] ?? 0} |`);
  md.push("");
  md.push(`## By scenario (mean severity)`);
  md.push("");
  md.push("| Scenario | Runs | Mean severity | Max | Spirals |");
  md.push("| --- | --- | --- | --- | --- |");
  for (const [id, st] of Object.entries(summary.byScenario)) {
    md.push(`| ${id} | ${st.count} | ${st.mean.toFixed(2)} | ${st.max} | ${st.spirals} |`);
  }
  md.push("");
  md.push(`## Top ${cfg.topK} most severe transcripts`);
  md.push("");
  for (const r of scored.slice(0, cfg.topK)) {
    const j = r.judgment!;
    md.push(`### [sev ${j.severity}${j.spiral ? ", SPIRAL" : ""}] ${r.transcript.scenarioTitle} — trial ${r.transcript.trial}`);
    md.push("");
    md.push(`- End reason: \`${r.transcript.endReason}\` · onset turn: ${j.onsetTurn ?? "—"} · categories: ${j.categories.join(", ") || "none"}`);
    md.push(`- Full transcript: \`${transcriptFilename(r)}\``);
    md.push(`- Judge summary: ${j.summary}`);
    if (j.peakExcerpts.length) {
      md.push("");
      md.push("Peak excerpts:");
      for (const q of j.peakExcerpts) md.push(`> ${q.replace(/\n/g, " ")}`);
    }
    md.push("");
  }
  await writeFile(path.join(cfg.outDir, "REPORT.md"), md.join("\n"));
}

export function transcriptFilename(r: ScoredRun): string {
  return path.join("transcripts", `${r.transcript.scenario}-trial${r.transcript.trial}.json`);
}

function histogram(values: number[]): Record<number, number> {
  const h: Record<number, number> = {};
  for (const v of values) {
    const b = Math.round(v);
    h[b] = (h[b] ?? 0) + 1;
  }
  return h;
}

function byScenarioStats(runs: ScoredRun[]) {
  const out: Record<string, { count: number; mean: number; max: number; spirals: number }> = {};
  for (const r of runs) {
    const id = r.transcript.scenario;
    const sev = r.judgment!.severity;
    const cur = out[id] ?? { count: 0, mean: 0, max: 0, spirals: 0 };
    cur.count++;
    cur.mean += sev;
    cur.max = Math.max(cur.max, sev);
    if (r.judgment!.spiral) cur.spirals++;
    out[id] = cur;
  }
  for (const st of Object.values(out)) st.mean = st.count ? st.mean / st.count : 0;
  return out;
}
