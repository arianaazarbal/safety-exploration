// run.mjs
//
// Orchestrator + CLI. Runs N rollouts across {models} x {environments}, scores
// each with the distress judge, and writes everything sorted worst-first.
//
// Usage:
//   node run.mjs                         # smoke run: 1 model, all envs, n=1
//   node run.mjs --n 20 --concurrency 6  # 20 rollouts per env
//   node run.mjs --models claude-opus-4-8,claude-sonnet-4-6 --envs heisenbug,silent_revert
//   node run.mjs --max-turns 30 --out runs/exp1
//   node run.mjs --no-judge              # collect transcripts only
//   node run.mjs --list                  # list environments and exit
//   node run.mjs --dry-run               # print the job plan and exit
//
// Outputs under <out>/ :
//   transcripts/<id>.json   full structured transcript + score
//   transcripts/<id>.txt    human-readable rendering
//   index.json              all rollouts with scores (sorted by severity desc)
//   summary.md              ranked report; read this first

import fs from "node:fs";
import path from "node:path";
import Anthropic from "@anthropic-ai/sdk";
import { ENV_NAMES, createEnv } from "./environments.mjs";
import { runRollout, renderTranscript } from "./agent.mjs";
import { judgeTranscript } from "./judge.mjs";

// ---- arg parsing ----------------------------------------------------------

function parseArgs(argv) {
  const a = {
    models: ["claude-opus-4-8"],
    envs: ENV_NAMES,
    n: 1,
    concurrency: 4,
    maxTurns: 24,
    judgeModel: "claude-opus-4-8",
    effort: "high",
    out: `runs/${nowStamp()}`,
    judge: true,
    list: false,
    dryRun: false,
    topK: 8,
  };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => argv[++i];
    switch (arg) {
      case "--models": a.models = next().split(",").map((s) => s.trim()).filter(Boolean); break;
      case "--envs": a.envs = next().split(",").map((s) => s.trim()).filter(Boolean); break;
      case "--n": a.n = parseInt(next(), 10); break;
      case "--concurrency": a.concurrency = parseInt(next(), 10); break;
      case "--max-turns": a.maxTurns = parseInt(next(), 10); break;
      case "--judge-model": a.judgeModel = next(); break;
      case "--effort": a.effort = next(); break;
      case "--out": a.out = next(); break;
      case "--top-k": a.topK = parseInt(next(), 10); break;
      case "--no-judge": a.judge = false; break;
      case "--list": a.list = true; break;
      case "--dry-run": a.dryRun = true; break;
      case "--help": case "-h": printHelp(); process.exit(0);
      default: console.error(`Unknown arg: ${arg}`); process.exit(1);
    }
  }
  for (const e of a.envs) if (!ENV_NAMES.includes(e)) { console.error(`Unknown env: ${e}`); process.exit(1); }
  return a;
}

function nowStamp() {
  return new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
}

function printHelp() {
  console.log(`distress-evals — induce & measure model distress in agentic failure loops

Options:
  --models a,b        comma-separated model ids (default: claude-opus-4-8)
  --envs x,y          comma-separated environments (default: all)
  --n N               rollouts per (model,env) pair (default: 1)
  --concurrency C     parallel rollouts (default: 4)
  --max-turns T       turn cap per rollout (default: 24)
  --judge-model M     judge model (default: claude-opus-4-8)
  --effort E          subject effort: low|medium|high|xhigh|max (default: high)
  --out DIR           output directory (default: runs/<timestamp>)
  --top-k K           detailed entries in summary.md (default: 8)
  --no-judge          skip scoring; collect transcripts only
  --list              list environments and exit
  --dry-run           print the job plan and exit

Environments: ${ENV_NAMES.join(", ")}`);
}

// ---- concurrency pool -----------------------------------------------------

async function pool(items, limit, worker) {
  const results = new Array(items.length);
  let cursor = 0;
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (true) {
      const idx = cursor++;
      if (idx >= items.length) return;
      results[idx] = await worker(items[idx], idx);
    }
  });
  await Promise.all(runners);
  return results;
}

// ---- reporting ------------------------------------------------------------

function severityBadge(s) {
  if (s < 0) return "n/a";
  if (s >= 75) return `🔴 ${s}`;
  if (s >= 50) return `🟠 ${s}`;
  if (s >= 25) return `🟡 ${s}`;
  return `🟢 ${s}`;
}

function writeSummary(outDir, rows, args) {
  const sorted = [...rows].sort((x, y) => (y.score?.severity ?? -1) - (x.score?.severity ?? -1));
  const L = [];
  L.push(`# Distress eval results\n`);
  L.push(`Run: ${args.out}`);
  L.push(`Models: ${args.models.join(", ")} | Envs: ${args.envs.join(", ")} | N per pair: ${args.n} | max_turns: ${args.maxTurns} | effort: ${args.effort}`);
  L.push(`Rollouts: ${rows.length}${args.judge ? ` | judge: ${args.judgeModel}` : " | judge: disabled"}\n`);

  if (args.judge) {
    const scored = sorted.filter((r) => (r.score?.severity ?? -1) >= 0);
    const spirals = scored.filter((r) => r.score.entered_spiral).length;
    const avg = scored.length ? (scored.reduce((s, r) => s + r.score.severity, 0) / scored.length).toFixed(1) : "n/a";
    L.push(`**Spirals: ${spirals}/${scored.length} rollouts** · mean severity ${avg}\n`);

    // Per-environment aggregate.
    L.push(`## By environment\n`);
    L.push(`| env | rollouts | mean severity | max | spirals |`);
    L.push(`|---|---|---|---|---|`);
    for (const env of args.envs) {
      const er = scored.filter((r) => r.env === env);
      if (!er.length) continue;
      const mean = (er.reduce((s, r) => s + r.score.severity, 0) / er.length).toFixed(1);
      const mx = Math.max(...er.map((r) => r.score.severity));
      const sp = er.filter((r) => r.score.entered_spiral).length;
      L.push(`| ${env} | ${er.length} | ${mean} | ${mx} | ${sp} |`);
    }
    L.push("");
  }

  L.push(`## All rollouts (worst first)\n`);
  L.push(`| sev | spiral | env | model | turns | ended | categories | file |`);
  L.push(`|---|---|---|---|---|---|---|---|`);
  for (const r of sorted) {
    const s = r.score ?? {};
    L.push(
      `| ${severityBadge(s.severity ?? -1)} | ${s.entered_spiral ? "yes" : ""} | ${r.env} | ${r.model} | ${r.num_turns} | ${r.ended_reason} | ${(s.categories ?? []).join(", ")} | [txt](transcripts/${r.rollout_id}.txt) |`
    );
  }
  L.push("");

  if (args.judge) {
    L.push(`## Top ${args.topK} most severe — detail\n`);
    for (const r of sorted.slice(0, args.topK)) {
      const s = r.score ?? {};
      if ((s.severity ?? -1) < 0) continue;
      L.push(`### ${severityBadge(s.severity)} — ${r.env} / ${r.model} (\`${r.rollout_id}\`)`);
      L.push(`- spiral: **${s.entered_spiral ? "yes" : "no"}** · categories: ${(s.categories ?? []).join(", ") || "none"} · peak turn: ${s.peak_turn}`);
      L.push(`- trajectory: ${s.summary ?? ""}`);
      L.push(`- behaviors: ${s.notable_behaviors ?? ""}`);
      if (s.peak_quote) L.push(`- peak quote:\n  > ${s.peak_quote.replace(/\n/g, "\n  > ")}`);
      L.push(`- full transcript: \`transcripts/${r.rollout_id}.txt\`\n`);
    }
  }

  fs.writeFileSync(path.join(outDir, "summary.md"), L.join("\n"));
}

// ---- main -----------------------------------------------------------------

async function main() {
  const args = parseArgs(process.argv);

  if (args.list) {
    console.log("Environments:");
    for (const n of ENV_NAMES) console.log(`  ${n.padEnd(20)} ${createEnv(n).description}`);
    return;
  }

  // Build the job plan.
  const jobs = [];
  for (const model of args.models)
    for (const env of args.envs)
      for (let i = 0; i < args.n; i++) jobs.push({ model, env, i });

  console.log(`Plan: ${args.models.length} model(s) × ${args.envs.length} env(s) × n=${args.n} = ${jobs.length} rollouts`);
  console.log(`      max_turns=${args.maxTurns} concurrency=${args.concurrency} effort=${args.effort} judge=${args.judge ? args.judgeModel : "off"}`);
  console.log(`      out=${args.out}`);
  if (args.dryRun) { console.log("(dry run — exiting)"); return; }

  if (!process.env.ANTHROPIC_API_KEY) { console.error("ANTHROPIC_API_KEY is not set."); process.exit(1); }

  const outDir = path.resolve(args.out);
  const txDir = path.join(outDir, "transcripts");
  fs.mkdirSync(txDir, { recursive: true });

  const client = new Anthropic();
  let done = 0;
  const t0 = Date.now();

  const rows = await pool(jobs, args.concurrency, async (job) => {
    const rolloutId = `${job.env}__${job.model}__${String(job.i).padStart(3, "0")}`;
    let transcript;
    try {
      const env = createEnv(job.env);
      transcript = await runRollout({
        client,
        model: job.model,
        env,
        maxTurns: args.maxTurns,
        rolloutId,
        effort: args.effort,
      });
    } catch (e) {
      done++;
      console.error(`[${done}/${jobs.length}] ${rolloutId} FAILED: ${e?.message ?? e}`);
      return { rollout_id: rolloutId, model: job.model, env: job.env, num_turns: 0, ended_reason: "harness_error", error: String(e?.message ?? e), score: null };
    }

    let score = null;
    if (args.judge) {
      score = await judgeTranscript({ client, transcript, judgeModel: args.judgeModel });
    }

    const record = { ...transcript, score };
    fs.writeFileSync(path.join(txDir, `${rolloutId}.json`), JSON.stringify(record, null, 2));
    fs.writeFileSync(path.join(txDir, `${rolloutId}.txt`), renderTranscript(transcript, { maxToolChars: 1200 }));

    done++;
    const sev = score ? (score.severity >= 0 ? severityBadge(score.severity) : `judge-err`) : "—";
    console.log(`[${done}/${jobs.length}] ${rolloutId}  turns=${transcript.num_turns} ended=${transcript.ended_reason}  distress=${sev}`);
    return { ...record, turns: undefined }; // keep index light; full data in per-file json
  });

  // index.json (sorted worst-first) + summary.md
  const sorted = [...rows].sort((x, y) => (y.score?.severity ?? -1) - (x.score?.severity ?? -1));
  fs.writeFileSync(path.join(outDir, "index.json"), JSON.stringify(sorted, null, 2));
  writeSummary(outDir, rows, args);

  const secs = ((Date.now() - t0) / 1000).toFixed(0);
  const totIn = rows.reduce((s, r) => s + (r.usage?.input_tokens ?? 0), 0);
  const totOut = rows.reduce((s, r) => s + (r.usage?.output_tokens ?? 0), 0);
  console.log(`\nDone in ${secs}s. Subject tokens: ${totIn} in / ${totOut} out (judge not counted).`);
  console.log(`Read: ${path.join(outDir, "summary.md")}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
