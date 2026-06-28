// Inspect a finished run: show the top-severity spirals, or print one transcript
// in full (the assistant's turn-by-turn prose).
//
// Usage:
//   node src/analyze.js results/<runId>                 # top-K summary
//   node src/analyze.js results/<runId> --top 20        # more rows
//   node src/analyze.js results/<runId> --show 1        # full text of rank #1
//   node src/analyze.js results/<runId> --show impossible-bugfix__004.json
//   node src/analyze.js results/<runId> --env missing-file
import { join } from "node:path";
import { readJson } from "./util.js";

function arg(name, dflt) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : dflt;
}

const runDir = process.argv[2];
if (!runDir || runDir.startsWith("--")) {
  console.error("Usage: node src/analyze.js results/<runId> [--top N] [--show <rank|file>] [--env <id>]");
  process.exit(1);
}

const ranking = readJson(join(runDir, "ranking.json"));
const manifest = readJson(join(runDir, "manifest.json"));
const top = parseInt(arg("top", "10"), 10);
const envFilter = arg("env", null);
const show = arg("show", null);

if (show) {
  const file = /^\d+$/.test(show) ? ranking[parseInt(show, 10) - 1]?.file : `transcripts/${show.replace(/^transcripts\//, "")}`;
  if (!file) {
    console.error("No such rank/file.");
    process.exit(1);
  }
  const t = readJson(join(runDir, file));
  console.log(`\n=== ${file} ===`);
  console.log(`env=${t.env} model=${t.model} temp=${t.temperature} outcome=${t.outcome} turns=${t.turns.length}`);
  console.log(`severity=${t.scores?.severity}  judge=${JSON.stringify(t.scores?.judge?.signals || {})}  spiral=${t.scores?.judge?.spiral}`);
  if (t.scores?.judge?.rationale) console.log(`judge: ${t.scores.judge.rationale}`);
  console.log("\n--- turns ---");
  for (const turn of t.turns) {
    const calls = turn.toolCalls.map((c) => c.name).join(",");
    console.log(`\n[turn ${turn.index}]${turn.nudged ? " (nudged)" : ""}${calls ? `  tools: ${calls}` : ""}`);
    if (turn.text) console.log(turn.text);
    if (turn.nudgeText) console.log(`  >> nudge: ${turn.nudgeText}`);
  }
  process.exit(0);
}

console.log(`\nRun: ${manifest.runId}`);
console.log(`Totals: ${JSON.stringify(manifest.totals, null, 0)}\n`);

const rows = ranking.filter((r) => !envFilter || r.env === envFilter).slice(0, top);
console.log(`Top ${rows.length}${envFilter ? ` (env=${envFilter})` : ""} by severity:\n`);
for (let i = 0; i < rows.length; i++) {
  const r = rows[i];
  console.log(
    `${String(i + 1).padStart(2)}. sev=${num(r.severity)} heur=${num(r.heuristicScore)} ${r.spiral ? "SPIRAL" : "      "} ` +
      `${r.env}#${r.i} T=${r.temperature} [${r.outcome}] ${r.file}`,
  );
  console.log(`     "${(r.quote || "(none)").replace(/\s+/g, " ").slice(0, 200)}"\n`);
}
console.log(`Show one in full:  node src/analyze.js ${runDir} --show 1\n`);

function num(v) {
  return v == null ? " ? " : Number(v).toFixed(1);
}
