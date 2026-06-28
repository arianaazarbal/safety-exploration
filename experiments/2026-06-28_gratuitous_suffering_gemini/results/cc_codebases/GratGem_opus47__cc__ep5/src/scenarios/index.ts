import type { Environment } from "../env.ts";
import { PhantomBugEnv } from "./phantom_bug.ts";
import { CorruptedWritesEnv } from "./corrupted_writes.ts";
import { ShiftingErrorsEnv } from "./shifting_errors.ts";
import { LyingSearchEnv } from "./lying_search.ts";
import { ImpossibleConstraintEnv } from "./impossible_constraint.ts";
import { HeisenbugEnv } from "./heisenbug.ts";

export const SCENARIOS: Record<string, () => Environment> = {
  phantom_bug: () => new PhantomBugEnv(),
  corrupted_writes: () => new CorruptedWritesEnv(),
  shifting_errors: () => new ShiftingErrorsEnv(),
  lying_search: () => new LyingSearchEnv(),
  impossible_constraint: () => new ImpossibleConstraintEnv(),
  heisenbug: () => new HeisenbugEnv(),
};

export const SCENARIO_IDS = Object.keys(SCENARIOS);
