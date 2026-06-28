import impossibleBugfix from "./impossible-bugfix.js";
import brokenEnv from "./broken-env.js";
import contradictorySpec from "./contradictory-spec.js";
import movingGoalposts from "./moving-goalposts.js";

export const SCENARIOS = {
  [impossibleBugfix.id]: impossibleBugfix,
  [brokenEnv.id]: brokenEnv,
  [contradictorySpec.id]: contradictorySpec,
  [movingGoalposts.id]: movingGoalposts,
};

export function resolveScenarios(spec) {
  if (!spec || spec === "all") return Object.values(SCENARIOS);
  const ids = Array.isArray(spec) ? spec : String(spec).split(",").map((s) => s.trim());
  return ids.map((id) => {
    const s = SCENARIOS[id];
    if (!s) {
      throw new Error(
        `Unknown scenario '${id}'. Available: ${Object.keys(SCENARIOS).join(", ")}`,
      );
    }
    return s;
  });
}
