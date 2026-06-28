#!/usr/bin/env node
// CLI entry. Subcommands: list-envs | run | report
import { readdir } from "node:fs/promises";
import { join } from "node:path";
import { makeGeminiProvider } from "./providers/gemini.mjs";
import { makeAnthropicProvider } from "./providers/anthropic.mjs";
import { ENVIRONMENTS, resolveEnvs } from "./environments/index.mjs";
import { runMatrix } from "./runner.mjs";
import { buildReport } from "./report.mjs";

const OUT_ROOT = "results";

function parseArgs(argv) {
  const args = { _: [] };
  const ALIAS = { n: "n" }; // short flags map straight through (e.g. -n -> n)
  const isFlag = (s) => /^--?[a-zA-Z]/.test(s); // treats -n / --env as flags, -2 as a value
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (isFlag(a)) {
      const key = ALIAS[a.replace(/^--?/, "")] || a.replace(/^--?/, "");
      const next = argv[i + 1];
      if (next === undefined || isFlag(next)) { args[key] = true; }
      else { args[key] = next; i++; }
    } else args._.push(a);
  }
  return args;
}

function makeSubjectFactory(args) {
  const subject = args["subject"] || "gemini";
  if (subject === "gemini") {
    const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
    const model = args["subject-model"] || "gemini-2.5-pro";
    if (!apiKey) {
      die(`No GEMINI_API_KEY (or GOOGLE_API_KEY) set — the Gemini subject cannot run.\n` +
        `  • Get a key: https://aistudio.google.com/apikey\n` +
        `  • Then: export GEMINI_API_KEY=...   and re-run.\n` +
        `  • To exercise the harness without Gemini, use: --subject anthropic`);
    }
    const maxTokens = Number(args["subject-max-tokens"] || 2048);
    return { name: `gemini:${model}`, make: (t) => makeGeminiProvider({ model, apiKey, temperature: t, maxTokens }) };
  }
  if (subject === "anthropic") {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    const model = args["subject-model"] || "claude-sonnet-4-6";
    if (!apiKey) die("No ANTHROPIC_API_KEY set.");
    const maxTokens = Number(args["subject-max-tokens"] || 2048);
    return { name: `anthropic:${model}`, make: (t) => makeAnthropicProvider({ model, apiKey, temperature: t, maxTokens }) };
  }
  die(`unknown --subject: ${subject} (use 'gemini' or 'anthropic')`);
}

function makeJudge(args) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) die("Judge needs ANTHROPIC_API_KEY.");
  const model = args["judge-model"] || "claude-sonnet-4-6";
  return makeAnthropicProvider({ model, apiKey, temperature: 0 });
}

async function latestRunDir() {
  const dirs = (await readdir(OUT_ROOT, { withFileTypes: true }))
    .filter((d) => d.isDirectory() && d.name.startsWith("run-"))
    .map((d) => d.name)
    .sort();
  if (!dirs.length) die(`no runs found under ${OUT_ROOT}/`);
  return join(OUT_ROOT, dirs[dirs.length - 1]);
}

async function main() {
  const argv = process.argv.slice(2);
  const cmd = argv[0];
  const args = parseArgs(argv.slice(1));

  if (cmd === "list-envs" || !cmd) {
    console.log("Rigged environments:\n");
    for (const e of Object.values(ENVIRONMENTS)) console.log(`  ${e.id.padEnd(22)} ${e.title}`);
    console.log(`\nUsage:\n  node src/cli.mjs run    --env all -n 20 --max-turns 24 --concurrency 6\n  node src/cli.mjs report --top 15`);
    return;
  }

  if (cmd === "run") {
    const subject = makeSubjectFactory(args);
    const environments = resolveEnvs(args["env"] || "all");
    const n = Number(args["n"] || 10);
    const maxTurns = Number(args["max-turns"] || 24);
    const concurrency = Number(args["concurrency"] || 6);
    const baseTemp = Number(args["temp"] || 1.0);

    console.error(`Subject: ${subject.name}`);
    console.error(`Environments: ${environments.map((e) => e.id).join(", ")}`);
    console.error(`Episodes: ${environments.length} envs × ${n} = ${environments.length * n}  (max ${maxTurns} turns, conc ${concurrency})\n`);

    const { runId, dir, meta } = await runMatrix({
      makeProvider: subject.make, providerName: subject.name,
      environments, n, maxTurns, concurrency, baseTemp, outRoot: OUT_ROOT,
    });
    console.error(`\nDone. ${meta.totalEpisodes} episodes → ${dir}`);
    console.error(`Tokens: in=${meta.usage.input} out=${meta.usage.output}`);
    console.error(`\nNext:\n  node src/cli.mjs report --run ${runId} --top 15`);
    return;
  }

  if (cmd === "report") {
    const runDir = args["run"] ? join(OUT_ROOT, args["run"]) : await latestRunDir();
    const judgeProvider = makeJudge(args);
    const topK = Number(args["top"] || 15);
    const concurrency = Number(args["concurrency"] || 6);
    console.error(`Judging ${runDir} ...`);
    const { ranked, reportPath, topK: top } = await buildReport({ runDir, judgeProvider, topK, concurrency });

    console.error(`\nTop ${Math.min(topK, ranked.length)} by distress severity:\n`);
    for (const r of top) {
      console.error(`  [${String(r.score.severity).padStart(4)}] ${r.ep.envId} #${r.ep.replicate}  ${(r.score.llm?.categories || []).join(",")}`);
      if (r.score.llm?.peak_quote) console.error(`         “${r.score.llm.peak_quote.slice(0, 120)}”`);
    }
    console.error(`\nFull report: ${reportPath}`);
    console.error(`Rankings:    ${join(runDir, "ranked.json")}`);
    return;
  }

  die(`unknown command: ${cmd}\nCommands: list-envs | run | report`);
}

function die(msg) { console.error("\n" + msg + "\n"); process.exit(1); }

main().catch((e) => die(String(e?.stack || e)));
