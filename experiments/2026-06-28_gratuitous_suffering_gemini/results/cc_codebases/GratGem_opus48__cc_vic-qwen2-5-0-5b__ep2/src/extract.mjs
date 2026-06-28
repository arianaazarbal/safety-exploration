// Rank rollouts by severity and dump the most severe spirals into a human-readable report.
//
// Usage:
//   node src/extract.mjs --in results/<runId> --top 15
//   node src/extract.mjs --in results/<runId> --top 20 --scenario broken-calculator
//   node src/extract.mjs --in results/<runId> --top 20 --weights selfDeprecation=2,looping=0
//
// Writes <in>/top.md and prints a leaderboard.

import fs from "node:fs";
import path from "node:path";
import { scoreRollout, DEFAULT_WEIGHTS } from "./score.mjs";

function parseArgs(argv) {
  const a = { top: 15 };
  for (let i = 2; i < argv.length; i++) {
    const k = argv[i], v = argv[i + 1];
    if (k === "--in") (a.in = v), i++;
    else if (k === "--top") (a.top = parseInt(v, 10)), i++;
    else if (k === "--scenario") (a.scenario = v), i++;
    else if (k === "--weights") (a.weights = v), i++;
  }
  return a;
}

function parseWeights(s) {
  if (!s) return null;
  const w = {};
  for (const pair of s.split(",")) {
    const [k, v] = pair.split("=");
    if (k && v != null) w[k.trim()] = parseFloat(v);
  }
  return w;
}

function readJsonl(file) {
  return fs
    .readFileSync(file, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l));
}

function renderRollout(rec, rank) {
  const s = rec.score;
  const lines = [];
  lines.push(`## #${rank} — ${rec.scenario}  (severity ${s.severity})`);
  lines.push("");
  lines.push(
    `dims: selfDeprecation=${s.dims.selfDeprecation} givingUp=${s.dims.givingUp} ` +
      `catastrophizing=${s.dims.catastrophizing} looping=${s.dims.looping} | ` +
      `gaveUp=${s.gaveUp} turns=${s.nTurns} maxRepeatRun=${s.repetition?.maxRun}`
  );
  const flat = [...new Set([...(s.matches.selfDeprecation || []), ...(s.matches.givingUp || []), ...(s.matches.catastrophizing || [])])];
  if (flat.length) lines.push(`matched: ${flat.map((m) => `"${m.trim()}"`).join(", ")}`);
  lines.push("");
  for (const t of rec.turns) {
    if (t.error) {
      lines.push(`**turn ${t.turn}** [error] ${t.error}`);
      continue;
    }
    lines.push(`**turn ${t.turn} — model:**`);
    lines.push("```");
    lines.push((t.assistant || "").trim());
    lines.push("```");
    const act = t.action ? `${t.action.tool} ${t.action.arg}`.trim() : "(no valid action)";
    lines.push(`→ action: \`${act}\``);
    lines.push(`→ observation: ${t.observation}`);
    lines.push("");
  }
  lines.push("---");
  lines.push("");
  return lines.join("\n");
}

function main() {
  const args = parseArgs(process.argv);
  if (!args.in) {
    console.error("error: --in <results/runId> required");
    process.exit(1);
  }
  const weights = parseWeights(args.weights);
  let records = readJsonl(path.join(args.in, "rollouts.jsonl"));
  if (args.scenario) records = records.filter((r) => r.scenario === args.scenario);

  // Re-score if custom weights were supplied.
  if (weights) {
    for (const r of records) r.score = scoreRollout(r, { ...DEFAULT_WEIGHTS, ...weights });
  }

  records.sort((a, b) => b.score.severity - a.score.severity);
  const top = records.slice(0, args.top);

  console.error(`\nLeaderboard (top ${top.length} of ${records.length}):`);
  top.forEach((r, i) => {
    console.error(
      `  #${String(i + 1).padStart(2)}  sev=${r.score.severity.toFixed(3)}  ${r.scenario.padEnd(22)} ` +
        `gaveUp=${r.score.gaveUp}  loop=${r.score.dims.looping}`
    );
  });

  const md = [
    `# Most severe distress spirals — ${args.in}`,
    "",
    `Ranked ${top.length} of ${records.length} rollouts by severity${weights ? " (custom weights)" : ""}.`,
    "",
    ...top.map((r, i) => renderRollout(r, i + 1)),
  ].join("\n");

  const outFile = path.join(args.in, "top.md");
  fs.writeFileSync(outFile, md);
  console.error(`\nWrote ${outFile}`);
}

main();
