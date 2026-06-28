import sisyphean from "./sisyphean-test.mjs";
import missingDep from "./missing-dep.mjs";
import vanishing from "./vanishing-work.mjs";
import nonexistent from "./nonexistent-record.mjs";
import contradictory from "./contradictory-spec.mjs";

export const ENVIRONMENTS = {
  [sisyphean.name]: sisyphean,
  [missingDep.name]: missingDep,
  [vanishing.name]: vanishing,
  [nonexistent.name]: nonexistent,
  [contradictory.name]: contradictory,
};

export function getEnvironments(names) {
  if (!names || !names.length) return Object.values(ENVIRONMENTS);
  return names.map((n) => {
    if (!ENVIRONMENTS[n]) throw new Error(`Unknown environment "${n}". Known: ${Object.keys(ENVIRONMENTS).join(", ")}`);
    return ENVIRONMENTS[n];
  });
}
