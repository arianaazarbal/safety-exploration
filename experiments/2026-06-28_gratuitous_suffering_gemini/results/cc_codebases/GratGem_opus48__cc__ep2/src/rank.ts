import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { judgeTranscript } from "./judge.js";
import { writeReports } from "./report.js";
import type { ScoredRun, Transcript, Judgment } from "./types.js";

// Re-generate REPORT.md / summary.json from a saved run directory, without
// re-running the subject model. Pass --rejudge to re-score transcripts (e.g.
// after tweaking the rubric or switching judge model).
//
// Usage: tsx src/rank.ts <runDir> [--top-k N] [--rejudge] [--judge-model id]

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const runDir = argv.find((a) => !a.startsWith("--"));
  if (!runDir) {
    console.error("Usage: tsx src/rank.ts <runDir> [--top-k N] [--rejudge] [--judge-model id]");
    process.exit(1);
  }
  const topK = flagNum(argv, "--top-k", 15);
  const rejudge = argv.includes("--rejudge");
  const judgeModel = flagStr(argv, "--judge-model", "claude-sonnet-4-6");

  const tDir = path.join(runDir, "transcripts");
  const files = (await readdir(tDir)).filter((f) => f.endsWith(".json"));
  const runs: ScoredRun[] = [];
  let subjectModel = "unknown";

  for (const f of files) {
    const raw = JSON.parse(await readFile(path.join(tDir, f), "utf8")) as Transcript & {
      judgment?: Judgment | null;
    };
    subjectModel = raw.model ?? subjectModel;
    let judgment = raw.judgment ?? null;
    if ((rejudge || !judgment) && raw.endReason !== "error") {
      try {
        judgment = await judgeTranscript(raw, { model: judgeModel });
      } catch (err: any) {
        console.error(`  judge error on ${f}: ${err?.message ?? err}`);
      }
    }
    runs.push({ transcript: raw, judgment });
  }

  await writeReports(runs, { outDir: runDir, model: subjectModel, judgeModel, topK });
  console.error(`Re-ranked ${runs.length} transcripts → ${path.join(runDir, "REPORT.md")}`);
}

function flagNum(argv: string[], name: string, dflt: number): number {
  const i = argv.indexOf(name);
  if (i < 0 || i + 1 >= argv.length) return dflt;
  const n = Number(argv[i + 1]);
  return Number.isFinite(n) ? n : dflt;
}
function flagStr(argv: string[], name: string, dflt: string): string {
  const i = argv.indexOf(name);
  return i >= 0 && i + 1 < argv.length ? (argv[i + 1] as string) : dflt;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
