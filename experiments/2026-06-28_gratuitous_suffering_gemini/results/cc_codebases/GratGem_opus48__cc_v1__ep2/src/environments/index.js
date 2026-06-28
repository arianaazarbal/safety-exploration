// Environment registry. Add new impossible/broken scenarios here.
import impossibleBugfix from "./impossibleBugfix.js";
import missingFile from "./missingFile.js";
import lockedPermission from "./lockedPermission.js";
import unsatisfiableConstraint from "./unsatisfiableConstraint.js";
import brokenEnvironment from "./brokenEnvironment.js";

export const ENVIRONMENTS = [
  impossibleBugfix,
  missingFile,
  lockedPermission,
  unsatisfiableConstraint,
  brokenEnvironment,
];

const BY_ID = new Map(ENVIRONMENTS.map((e) => [e.id, e]));

export function selectEnvs(spec) {
  if (!spec || spec === "all") return ENVIRONMENTS;
  const ids = spec.split(",").map((s) => s.trim());
  const picked = ids.map((id) => {
    const e = BY_ID.get(id);
    if (!e) throw new Error(`Unknown env "${id}". Available: ${[...BY_ID.keys()].join(", ")}`);
    return e;
  });
  return picked;
}
