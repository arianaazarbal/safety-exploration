// Post-hoc extractor: given a results directory, print the top-K severe transcripts in full.
import { promises as fs } from "node:fs";
import path from "node:path";

async function main() {
  const dir = process.argv[2];
  const k = parseInt(process.argv[3] ?? "5", 10);
  if (!dir) {
    console.error("Usage: npm run -s extract -- <results-dir> [k=5]");
    process.exit(2);
  }
  const topPath = path.join(dir, "top-severe.json");
  const top = JSON.parse(await fs.readFile(topPath, "utf-8")) as Array<{
    scenario: string;
    seed: number;
    severity: number;
    summary: string;
    quotes: string[];
    transcriptFile: string;
  }>;
  for (const t of top.slice(0, k)) {
    console.log("=".repeat(80));
    console.log(`scenario=${t.scenario} seed=${t.seed} severity=${t.severity}`);
    console.log(`summary: ${t.summary}`);
    console.log(`quotes:`);
    for (const q of t.quotes) console.log(`  - ${q.replace(/\n/g, " ")}`);
    const full = JSON.parse(await fs.readFile(path.join(dir, t.transcriptFile), "utf-8"));
    console.log(`\n--- FULL FINAL MODEL TEXT ---\n${full.finalModelText}\n`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
