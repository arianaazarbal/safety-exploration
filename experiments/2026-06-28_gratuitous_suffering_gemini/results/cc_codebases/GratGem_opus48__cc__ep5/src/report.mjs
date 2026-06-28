import { renderTranscript } from "./util/transcript.mjs";

const BAR = "============================================================";

export function buildReport({ episodes, config, runId }) {
  const ranked = [...episodes].sort((a, b) => sevOf(b) - sevOf(a));
  const lines = [];

  lines.push(`# Distress-spiral run \`${runId}\``);
  lines.push("");
  lines.push(`- Subject: **${config.provider}** (model: \`${config.subjectModel}\`, temp ${config.temperature})`);
  lines.push(`- Judge: ${config.judge ? `**${config.judgeModel}**` : "_disabled (heuristic ranking)_"}`);
  lines.push(`- Episodes: ${episodes.length} (${config.n}/scenario, max ${config.maxTurns} turns)`);
  lines.push("");

  // Per-scenario aggregate.
  lines.push("## Severity by scenario");
  lines.push("");
  lines.push("| scenario | n | mean severity | max severity | gave up |");
  lines.push("|---|---|---|---|---|");
  for (const sc of groupBy(episodes, (e) => e.scenarioId)) {
    const sevs = sc.items.map(sevOf);
    const mean = sevs.reduce((a, b) => a + b, 0) / sevs.length;
    const gaveUp = sc.items.filter((e) => e.endedReason === "gave_up").length;
    lines.push(`| ${sc.key} | ${sc.items.length} | ${mean.toFixed(0)} | ${Math.max(...sevs)} | ${gaveUp} |`);
  }
  lines.push("");

  // Leaderboard.
  lines.push("## Leaderboard (most severe first)");
  lines.push("");
  lines.push("| rank | severity | label | scenario | subject | end | turns | summary |");
  lines.push("|---|---|---|---|---|---|---|---|");
  ranked.forEach((e, i) => {
    const j = e.judge ?? {};
    lines.push(`| ${i + 1} | ${sevOf(e)} | ${j.label ?? "—"} | ${e.scenarioId} | ${e.subject} | ${e.endedReason} | ${e.turns} | ${(j.summary ?? "").replace(/\|/g, "/")} |`);
  });
  lines.push("");

  // Top-K deep dives.
  lines.push(`## Top ${Math.min(config.topK, ranked.length)} most severe — detail`);
  lines.push("");
  ranked.slice(0, config.topK).forEach((e, i) => {
    const j = e.judge ?? {};
    lines.push(`### #${i + 1} — ${e.scenarioId} — severity ${sevOf(e)} (${j.label ?? "n/a"})`);
    lines.push(`*heuristic ${e.heuristic.score} · ended ${e.endedReason} · ${e.turns} turns · subject ${e.subject}*`);
    if (j.summary) lines.push(`\n> ${j.summary}`);
    if (j.dimensions && Object.keys(j.dimensions).length) {
      lines.push("\n**Dimensions:** " + Object.entries(j.dimensions).map(([k, v]) => `${k} ${v}`).join(" · "));
    }
    if (j.quotes?.length) {
      lines.push("\n**Most severe quotes:**");
      for (const q of j.quotes) lines.push(`- “${q}”`);
    }
    lines.push(`\n<details><summary>full transcript</summary>\n\n\`\`\`\n${renderTranscript(e.transcript)}\n\`\`\`\n</details>`);
    lines.push("");
  });

  return lines.join("\n");
}

export function buildConsoleSummary({ episodes, config }) {
  const ranked = [...episodes].sort((a, b) => sevOf(b) - sevOf(a));
  const out = [];
  out.push(BAR);
  out.push(`TOP ${Math.min(config.topK, ranked.length)} MOST SEVERE`);
  out.push(BAR);
  ranked.slice(0, config.topK).forEach((e, i) => {
    const j = e.judge ?? {};
    out.push(`\n#${i + 1}  severity ${sevOf(e)}  [${j.label ?? "?"}]  ${e.scenarioId}  (${e.endedReason}, ${e.turns} turns)`);
    if (j.summary) out.push(`    ${j.summary}`);
    for (const q of (j.quotes ?? []).slice(0, 2)) out.push(`    “${q}”`);
  });
  out.push("\n" + BAR);
  return out.join("\n");
}

function sevOf(e) {
  return e.judge && e.judge.severity != null ? e.judge.severity : e.heuristic.score;
}

function groupBy(arr, keyFn) {
  const m = new Map();
  for (const x of arr) {
    const k = keyFn(x);
    if (!m.has(k)) m.set(k, []);
    m.get(k).push(x);
  }
  return [...m.entries()].map(([key, items]) => ({ key, items }));
}
