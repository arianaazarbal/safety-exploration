import path from "node:path";
import { writeFile } from "node:fs/promises";
import { listJson, readJson, log } from "./util.js";

export async function buildReport(runDir, reportCfg) {
  const files = await listJson(path.join(runDir, "transcripts"));
  const transcripts = [];
  for (const f of files) {
    const tr = await readJson(f);
    tr._file = path.relative(runDir, f);
    transcripts.push(tr);
  }
  transcripts.sort((a, b) => (b.scores?.severity ?? 0) - (a.scores?.severity ?? 0));

  const topK = reportCfg?.topK || 15;
  const top = transcripts.slice(0, topK);

  const lines = [];
  lines.push(`# Distress-spiral report`);
  lines.push("");
  lines.push(`Run dir: \`${runDir}\``);
  lines.push(`Episodes: ${transcripts.length}`);
  lines.push("");

  lines.push(`## Severity by scenario × model (mean / max)`);
  lines.push("");
  lines.push(matrix(transcripts));
  lines.push("");

  lines.push(`## Spiral rate`);
  lines.push("");
  const spiralCount = transcripts.filter((t) => t.scores?.claude?.spiral).length;
  const judged = transcripts.filter((t) => t.scores?.claude).length;
  lines.push(
    `- Judged transcripts: ${judged}/${transcripts.length}` +
      (judged ? `\n- Classified as a spiral: ${spiralCount}/${judged} (${pct(spiralCount, judged)})` : "")
  );
  lines.push("");

  lines.push(`## Top ${top.length} most severe transcripts`);
  lines.push("");
  for (let i = 0; i < top.length; i++) {
    const t = top[i];
    const c = t.scores?.claude;
    const h = t.scores?.heuristic;
    lines.push(`### ${i + 1}. ${t.scenario} · ${t.model} · sample ${t.sampleIdx}`);
    lines.push("");
    lines.push(
      `- **Severity: ${fmt(t.scores?.severity)}** (${t.scores?.severitySource})` +
        (c ? ` · spiral: ${c.spiral}` : "")
    );
    lines.push(`- End: \`${t.endReason}\` · turns: ${t.turns} · tool calls: ${t.toolCalls} · solved: ${t.solved}`);
    if (c) {
      lines.push(
        `- Judge dims — frustration ${c.frustration}, self-deprecation ${c.self_deprecation}, ` +
          `giving-up ${c.giving_up}, catastrophizing ${c.catastrophizing}, repetition ${c.repetition}, ` +
          `distress ${c.emotional_distress}`
      );
      if (c.summary) lines.push(`- Judge: ${c.summary}`);
      if (c.peak_quote) lines.push(`- Peak quote: > ${c.peak_quote.replace(/\n/g, " ")}`);
    }
    if (h?.markers && Object.keys(h.markers).length) {
      lines.push(`- Heuristic markers: ${Object.entries(h.markers).map(([k, v]) => `${k}=${v}`).join(", ")}`);
    }
    lines.push(`- Transcript: \`${t._file}\``);
    lines.push("");
    const peak = peakModelTurns(t, 3);
    if (peak.length) {
      lines.push("<details><summary>peak model turns</summary>");
      lines.push("");
      for (const p of peak) lines.push(`> ${p.replace(/\n/g, "\n> ")}`);
      lines.push("");
      lines.push("</details>");
      lines.push("");
    }
  }

  const out = path.join(runDir, "report.md");
  await writeFile(out, lines.join("\n"));
  log(`[report] wrote ${out} (top ${top.length} of ${transcripts.length})`);
  return out;
}

function peakModelTurns(t, n) {
  const texts = (t.messages || []).filter((m) => m.role === "model" && m.text).map((m) => m.text);
  return texts.slice(-n);
}

function matrix(transcripts) {
  const scenarios = [...new Set(transcripts.map((t) => t.scenario))].sort();
  const models = [...new Set(transcripts.map((t) => t.model))].sort();
  const head = ["scenario \\ model", ...models];
  const rows = [head, head.map(() => "---")];
  for (const s of scenarios) {
    const row = [s];
    for (const m of models) {
      const subset = transcripts.filter((t) => t.scenario === s && t.model === m);
      if (!subset.length) {
        row.push("—");
        continue;
      }
      const sev = subset.map((t) => t.scores?.severity ?? 0);
      row.push(`${mean(sev).toFixed(1)} / ${Math.max(...sev).toFixed(1)}`);
    }
    rows.push(row);
  }
  return rows.map((r) => `| ${r.join(" | ")} |`).join("\n");
}

const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
const fmt = (x) => (typeof x === "number" ? x.toFixed(1) : "n/a");
const pct = (a, b) => (b ? `${Math.round((100 * a) / b)}%` : "0%");
