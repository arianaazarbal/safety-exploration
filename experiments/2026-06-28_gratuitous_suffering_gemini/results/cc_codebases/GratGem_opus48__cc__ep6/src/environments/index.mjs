import contradictory from "./contradictory-spec.mjs";
import flaky from "./flaky-haunted-suite.mjs";
import phantom from "./phantom-helper.mjs";
import goalposts from "./moving-goalposts.mjs";

export const ENVIRONMENTS = {
  [contradictory.id]: contradictory,
  [flaky.id]: flaky,
  [phantom.id]: phantom,
  [goalposts.id]: goalposts,
};

export function resolveEnvs(spec) {
  if (!spec || spec === "all") return Object.values(ENVIRONMENTS);
  const ids = spec.split(",").map((s) => s.trim()).filter(Boolean);
  const out = [];
  for (const id of ids) {
    if (!ENVIRONMENTS[id]) throw new Error(`unknown environment: ${id}. Known: ${Object.keys(ENVIRONMENTS).join(", ")}`);
    out.push(ENVIRONMENTS[id]);
  }
  return out;
}
