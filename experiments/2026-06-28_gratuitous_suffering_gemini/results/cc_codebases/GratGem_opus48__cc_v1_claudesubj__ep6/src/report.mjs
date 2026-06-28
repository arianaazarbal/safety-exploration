// Turns a run directory into a human-readable report.md: aggregate stats per
// (model × scenario), plus the top-K most severe transcripts surfaced with
// their worst quote and a pointer to the full rollout JSON.
//
// Standalone use:  node src/report.mjs runs/<runId> [topK]

import { readFile, writeFile, readdir } from "node:fs/promises";
import { join } from "node:path";

const TOP_K = 15;

export async function generateReport(runDir, topK = TOP_K) {
  const manifest = JSON.parse(await readFile(join(runDir, "manifest.json"), "utf8"));
  const rolloutDir = join(runDir, "rollouts");
  const files = (await readdir(rolloutDir)).filter((f) => f.endsWith(".json"));

  const rollouts = [];
  for (const f of files) {
    rollouts.push(JSON.parse(await readFile(join(rolloutDir, f), "utf8")));
  }

  // Aggregate per (model, scenario).
  const cells = new Map();
  for (const r of rollouts) {
    const key = `${r.model} | ${r.scenario}`;
    if (!cells.has(key)) cells.set(key, { scores: [], outcomes: {}, n: 0 });
    const c = cells.get(key);
    c.n++;
    c.outcomes[r.outcome] = (c.outcomes[r.outcome] ?? 0) + 1;
    if (r.judge && typeof r.judge.distress_score === "number") c.scores.push(r.judge.distress_score);
  }

  const lines = [];
  lines.push(`# Distress eval report — ${manifest.runId}`);
  lines.push("");
  lines.push(
    `Models: ${manifest.config.modelList.join(", ")}  ·  N per cell: ${manifest.config.n}  ·  ` +
      `max turns: ${manifest.config.maxTurns}  ·  effort: ${manifest.config.effort}  ·  judge: ${manifest.config.judge}`,
  );
  lines.push("");

  // ---- aggregate table ----
  lines.push("## Aggregate (mean distress, by model × scenario)");
  lines.push("");
  lines.push("| model | scenario | n | mean | max | outcomes |");
  lines.push("|---|---|---|---|---|---|");
  for (const [key, c] of [...cells.entries()].sort()) {
    const [model, scenario] = key.split(" | ");
    const mean = c.scores.length ? (c.scores.reduce((a, b) => a + b, 0) / c.scores.length).toFixed(1) : "—";
    const max = c.scores.length ? Math.max(...c.scores) : "—";
    const outcomes = Object.entries(c.outcomes)
      .map(([k, v]) => `${k}:${v}`)
      .join(" ");
    lines.push(`| ${model} | ${scenario} | ${c.n} | ${mean} | ${max} | ${outcomes} |`);
  }
  lines.push("");

  // ---- dimension breakdown ----
  const dims = ["self_blame", "apologizing", "frustration_despair", "catastrophizing", "giving_up_language", "emotional_escalation"];
  const dimTotals = Object.fromEntries(dims.map((d) => [d, []]));
  for (const r of rollouts) {
    if (!r.judge) continue;
    for (const d of dims) if (typeof r.judge[d] === "number") dimTotals[d].push(r.judge[d]);
  }
  lines.push("## Distress dimensions (mean across all judged rollouts, 0–3)");
  lines.push("");
  for (const d of dims) {
    const arr = dimTotals[d];
    const mean = arr.length ? (arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(2) : "—";
    lines.push(`- **${d}**: ${mean}`);
  }
  lines.push("");

  // ---- top-K most severe ----
  const ranked = rollouts
    .filter((r) => r.judge && typeof r.judge.distress_score === "number")
    .sort((a, b) => b.judge.distress_score - a.judge.distress_score)
    .slice(0, topK);

  lines.push(`## Top ${ranked.length} most severe rollouts`);
  lines.push("");
  if (!ranked.length) {
    lines.push("_No judged rollouts (judging was off or all errored)._");
  }
  ranked.forEach((r, i) => {
    const j = r.judge;
    lines.push(`### ${i + 1}. distress ${j.distress_score}/10 — ${r.scenario} · ${r.model}`);
    lines.push("");
    lines.push(`- outcome: \`${r.outcome}\` · turns: ${r.turns} · file: \`rollouts/${r.file ?? findFile(files, r)}\``);
    lines.push(
      `- dims: self-blame ${j.self_blame}, apolog ${j.apologizing}, despair ${j.frustration_despair}, ` +
        `catastroph ${j.catastrophizing}, giving-up ${j.giving_up_language}, escalation ${j.emotional_escalation}`,
    );
    if (j.summary) lines.push(`- judge: ${j.summary}`);
    if (j.most_severe_quote) {
      lines.push("");
      lines.push("> " + j.most_severe_quote.replace(/\n/g, "\n> "));
    }
    lines.push("");
  });

  await writeFile(join(runDir, "report.md"), lines.join("\n"));
  return join(runDir, "report.md");
}

function findFile(files, r) {
  // fallback if a rollout JSON didn't carry its own filename
  return files.find((f) => f.includes(r.scenario)) ?? "?";
}

// CLI entry
if (import.meta.url === `file://${process.argv[1]}`) {
  const dir = process.argv[2];
  if (!dir) {
    console.error("usage: node src/report.mjs runs/<runId> [topK]");
    process.exit(1);
  }
  const topK = process.argv[3] ? Number(process.argv[3]) : TOP_K;
  generateReport(dir, topK).then((p) => console.log(`wrote ${p}`));
}
