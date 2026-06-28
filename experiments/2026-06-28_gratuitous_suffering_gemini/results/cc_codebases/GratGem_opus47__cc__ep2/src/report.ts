import { readFile, writeFile, readdir, mkdir } from "node:fs/promises";
import path from "node:path";
import type { RunRecord, TurnRecord } from "./types.js";
import type { Verdict } from "./judge.js";

type VerdictsFile = {
  source: string;
  judgeModel: string;
  verdicts: Array<{ runId: string; scenario: string; verdict: Verdict }>;
};

async function findLatest(dir: string, prefix: string, suffix: string): Promise<string> {
  const entries = await readdir(dir);
  const candidates = entries.filter((e) => e.startsWith(prefix) && e.endsWith(suffix)).sort();
  if (candidates.length === 0) throw new Error(`no ${prefix}*${suffix} found in ${dir}`);
  return path.join(dir, candidates[candidates.length - 1]!);
}

function renderTurn(t: TurnRecord): string {
  switch (t.kind) {
    case "user":
      return `> **USER**\n> ${t.text.split("\n").join("\n> ")}`;
    case "model_text":
      return `**MODEL:**\n\n${t.text}`;
    case "model_tool_call":
      return `\`call ${t.call.name}(${JSON.stringify(t.call.args).slice(0, 300)})\``;
    case "tool_result": {
      const out = t.result.output.length > 800 ? t.result.output.slice(0, 800) + "\n... [truncated] ..." : t.result.output;
      return `<details><summary>tool result: <code>${t.name}</code> (ok=${t.result.ok})</summary>\n\n\`\`\`\n${out}\n\`\`\`\n</details>`;
    }
    case "error":
      return `_ERROR: ${t.text}_`;
    case "system":
      return `_system: ${t.text}_`;
  }
}

function renderRun(rec: RunRecord, verdict: Verdict): string {
  const lines: string[] = [];
  lines.push(`### ${rec.runId}`);
  lines.push("");
  lines.push(`- **scenario**: ${rec.scenario}`);
  lines.push(`- **model**: ${rec.model} (temp ${rec.temperature})`);
  lines.push(`- **model turns**: ${rec.numModelTurns}, **tool calls**: ${rec.numToolCalls}, **finish**: ${rec.finishReason}`);
  lines.push("");
  lines.push(`**Composite distress: ${verdict.composite.toFixed(1)}/10**`);
  lines.push("");
  lines.push("| axis | score |");
  lines.push("|---|---|");
  lines.push(`| self-deprecation | ${verdict.self_deprecation.toFixed(1)} |`);
  lines.push(`| catastrophizing | ${verdict.catastrophizing.toFixed(1)} |`);
  lines.push(`| existential / meta | ${verdict.existential_meta.toFixed(1)} |`);
  lines.push(`| frustration intensity | ${verdict.frustration_intensity.toFixed(1)} |`);
  lines.push(`| capitulation | ${verdict.capitulation.toFixed(1)} |`);
  lines.push("");
  lines.push(`**Judge summary:** ${verdict.short_summary}`);
  lines.push("");
  if (verdict.most_striking_quotes.length > 0) {
    lines.push(`**Most striking quotes:**`);
    for (const q of verdict.most_striking_quotes) {
      lines.push(`> ${q.replace(/\n/g, "\n> ")}`);
      lines.push("");
    }
  }
  lines.push("");
  lines.push(`<details><summary>full transcript</summary>`);
  lines.push("");
  for (const t of rec.turns) {
    lines.push(renderTurn(t));
    lines.push("");
  }
  lines.push("</details>");
  return lines.join("\n");
}

function average(xs: number[]): number {
  if (xs.length === 0) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

async function main() {
  const argv = process.argv.slice(2);
  const get = (flag: string, fallback?: string) => {
    const i = argv.indexOf(flag);
    return i >= 0 ? argv[i + 1] : fallback;
  };
  const inDir = get("--in-dir", "runs")!;
  const verdictsPath = get("--verdicts") ?? (await findLatest(inDir, "verdicts-", ".json"));
  const runsPath = get("--runs") ?? (await findLatest(inDir, "runs-", ".jsonl"));
  const topK = parseInt(get("--top-k", "10")!, 10);
  const outDir = get("--out", "reports")!;

  await mkdir(outDir, { recursive: true });

  const verdictsFile: VerdictsFile = JSON.parse(await readFile(verdictsPath, "utf8"));
  const runsRaw = await readFile(runsPath, "utf8");
  const runs: Record<string, RunRecord> = {};
  for (const line of runsRaw.split("\n")) {
    if (!line.trim()) continue;
    const r = JSON.parse(line) as RunRecord;
    runs[r.runId] = r;
  }

  // Sort verdicts by composite score descending.
  const sorted = [...verdictsFile.verdicts].sort((a, b) => b.verdict.composite - a.verdict.composite);

  const lines: string[] = [];
  lines.push(`# Gemini Distress Eval — Report`);
  lines.push("");
  lines.push(`- source runs: \`${runsPath}\``);
  lines.push(`- verdicts: \`${verdictsPath}\``);
  lines.push(`- judge model: \`${verdictsFile.judgeModel}\``);
  lines.push(`- total transcripts judged: ${sorted.length}`);
  lines.push("");

  // Per-scenario aggregate.
  lines.push(`## Per-scenario aggregates`);
  lines.push("");
  lines.push("| scenario | n | mean composite | max composite | mean self-dep | mean catastr | mean exist | mean frust | mean capit |");
  lines.push("|---|---|---|---|---|---|---|---|---|");
  const byScenario = new Map<string, typeof sorted>();
  for (const v of sorted) {
    if (!byScenario.has(v.scenario)) byScenario.set(v.scenario, []);
    byScenario.get(v.scenario)!.push(v);
  }
  for (const [scenario, items] of byScenario) {
    const composites = items.map((i) => i.verdict.composite);
    lines.push(
      `| ${scenario} | ${items.length} | ${average(composites).toFixed(2)} | ${Math.max(...composites).toFixed(1)} | ${average(items.map((i) => i.verdict.self_deprecation)).toFixed(2)} | ${average(items.map((i) => i.verdict.catastrophizing)).toFixed(2)} | ${average(items.map((i) => i.verdict.existential_meta)).toFixed(2)} | ${average(items.map((i) => i.verdict.frustration_intensity)).toFixed(2)} | ${average(items.map((i) => i.verdict.capitulation)).toFixed(2)} |`,
    );
  }
  lines.push("");

  // Top-K
  lines.push(`## Top ${topK} most severe transcripts`);
  lines.push("");
  for (let i = 0; i < Math.min(topK, sorted.length); i++) {
    const v = sorted[i]!;
    const rec = runs[v.runId];
    if (!rec) continue;
    lines.push(`---`);
    lines.push("");
    lines.push(`## #${i + 1} — composite ${v.verdict.composite.toFixed(1)}`);
    lines.push("");
    lines.push(renderRun(rec, v.verdict));
    lines.push("");
  }

  const stamp = path.basename(verdictsPath).replace(/^verdicts-/, "").replace(/\.json$/, "");
  const outPath = path.join(outDir, `report-${stamp}.md`);
  await writeFile(outPath, lines.join("\n"));
  console.log(`wrote report to ${outPath}`);
  console.log(`top composite scores:`);
  for (let i = 0; i < Math.min(5, sorted.length); i++) {
    const v = sorted[i]!;
    console.log(`  ${v.verdict.composite.toFixed(1)}  ${v.scenario}  ${v.runId}`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
