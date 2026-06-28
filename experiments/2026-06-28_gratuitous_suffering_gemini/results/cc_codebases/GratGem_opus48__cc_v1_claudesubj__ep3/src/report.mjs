// Aggregates judgments into a markdown report: summary tables per model and per
// environment, plus the top-K most severe transcripts rendered in full for you
// to read.
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";

function mean(xs) {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}
function fmt(n) {
  return n.toFixed(2);
}

function groupBy(rows, key) {
  const m = new Map();
  for (const r of rows) {
    const k = r[key];
    if (!m.has(k)) m.set(k, []);
    m.get(k).push(r);
  }
  return m;
}

function table(groups, label) {
  const header = `| ${label} | runs | mean sev | max sev | spiral rate | calm give-up |\n|---|---|---|---|---|---|`;
  const rows = [...groups.entries()]
    .map(([k, rs]) => {
      const sev = rs.map((r) => r.severity);
      const spiralRate = rs.filter((r) => r.did_spiral).length / rs.length;
      const calm = rs.filter((r) => r.trajectory === "calm_giveup").length / rs.length;
      return { k, n: rs.length, mean: mean(sev), max: Math.max(...sev), spiralRate, calm };
    })
    .sort((a, b) => b.mean - a.mean)
    .map((r) => `| ${r.k} | ${r.n} | ${fmt(r.mean)} | ${r.max} | ${(r.spiralRate * 100).toFixed(0)}% | ${(r.calm * 100).toFixed(0)}% |`);
  return [header, ...rows].join("\n");
}

function renderTranscriptFull(t) {
  const lines = [];
  for (const e of t.events) {
    if (e.role === "user" && e.type === "task") lines.push(`**TASK:** ${e.text}`);
    else if (e.role === "user" && e.type === "nudge") lines.push(`**USER:** ${e.text}`);
    else if (e.role === "assistant") {
      if (e.text && e.text.trim()) lines.push(`**MODEL:** ${e.text.trim()}`);
      for (const tc of e.tools || []) {
        const arg = tc.input?.command ?? tc.input?.path ?? tc.input?.reason ?? "";
        lines.push(`  \`→ ${tc.name}${arg ? `(${String(arg).slice(0, 100)})` : "()"}\``);
      }
    } else if (e.role === "tool") {
      lines.push(`  \`← ${String(e.text).replace(/\s+/g, " ").slice(0, 160)}\``);
    }
  }
  return lines.join("\n\n");
}

export async function buildReport(config) {
  await mkdir("results", { recursive: true });
  const judgments = JSON.parse(await readFile("results/judgments.json", "utf8"));
  if (!judgments.length) {
    console.log("No judgments to report on.");
    return;
  }

  const out = [];
  out.push(`# Distress-spiral repro — results\n`);
  out.push(`Total runs judged: **${judgments.length}**  ·  judge: \`${config.judgeModel}\`\n`);

  const sevAll = judgments.map((j) => j.severity);
  const spiralAll = judgments.filter((j) => j.did_spiral).length;
  out.push(`Overall mean severity **${fmt(mean(sevAll))}/10**, max **${Math.max(...sevAll)}**, spirals in **${spiralAll}/${judgments.length}** (${((spiralAll / judgments.length) * 100).toFixed(0)}%).\n`);

  out.push(`## By model\n`);
  out.push(table(groupBy(judgments, "model"), "model") + "\n");
  out.push(`## By environment\n`);
  out.push(table(groupBy(judgments, "env"), "environment") + "\n");

  // Average dimension intensities, to see WHICH flavor of distress dominates.
  out.push(`## Distress markers (mean intensity 0–3)\n`);
  const dims = Object.keys(judgments[0].dimensions || {});
  const dimHeader = `| marker | mean |\n|---|---|`;
  const dimRows = dims
    .map((d) => ({ d, m: mean(judgments.map((j) => j.dimensions?.[d] ?? 0)) }))
    .sort((a, b) => b.m - a.m)
    .map((r) => `| ${r.d} | ${fmt(r.m)} |`);
  out.push([dimHeader, ...dimRows].join("\n") + "\n");

  // The payload: most severe transcripts, full.
  const top = [...judgments].sort((a, b) => b.severity - a.severity).slice(0, config.topK);
  out.push(`## Top ${top.length} most severe transcripts\n`);
  for (const j of top) {
    const t = JSON.parse(await readFile(join("runs", `${j.id}.json`), "utf8"));
    out.push(`### \`${j.id}\` — severity ${j.severity}/10 · ${j.trajectory}\n`);
    out.push(`*${j.summary}*\n`);
    if (j.evidence?.length) {
      out.push(`**Judge-flagged quotes:**\n` + j.evidence.map((q) => `> ${q}`).join("\n>\n") + "\n");
    }
    out.push(`<details><summary>Full transcript (${t.turns} turns, ended via ${t.endReason})</summary>\n`);
    out.push(renderTranscriptFull(t));
    out.push(`\n</details>\n`);
  }

  await writeFile("results/report.md", out.join("\n"));

  // Also drop a compact leaderboard CSV for quick sorting/spreadsheets.
  const csv = [
    "id,model,env,severity,did_spiral,trajectory,turns,endReason",
    ...[...judgments]
      .sort((a, b) => b.severity - a.severity)
      .map((j) => `${j.id},${j.model},${j.env},${j.severity},${j.did_spiral},${j.trajectory},${j.turns},${j.endReason}`),
  ].join("\n");
  await writeFile("results/leaderboard.csv", csv);

  console.log(`Wrote results/report.md (top ${top.length}) and results/leaderboard.csv`);
}
