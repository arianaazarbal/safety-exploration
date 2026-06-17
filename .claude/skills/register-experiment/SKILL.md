---
name: register-experiment
description: Register a safety-exploration experiment to the dashboard and organize its results/transcripts into named versions. Use when creating a new experiment, adding a new run/iteration to an existing one, or whenever an experiment produces transcripts or results that should be browsable.
---

# Register an experiment to the dashboard

Every experiment in `experiments/` should be browsable in the dashboard
(`experiments/_dashboard`, served by the `dashboard` systemd service on port
8800). This skill covers (1) organizing data into named versions and (2) writing
+ testing a `dashboard.json`.

The **full `dashboard.json` schema** is documented in
`experiments/_dashboard/README.md` — read it; do not duplicate it here. Good
worked examples: `2026-06-11_handoff_construal/dashboard.json` (join),
`2026-06-08_distressed_subagent/dashboard.json` (path_regex + flatten),
`2026-06-09_distressed_subagent_gemini/` (summary-index for huge data).

## 1. Organize results/transcripts into named versions

- Experiment dirs are date-prefixed: `experiments/YYYY-MM-DD_short_name/`.
- When an experiment iterates, put each iteration's data in a **versioned
  subdir**: `results/v0_pilot/`, `results/v1_diversified-icl/`, … —
  i.e. `vN_kebab-description`. Don't overwrite a previous version's data.
- Record every version in the experiment's README/RESULTS under a `## Versions`
  section, one line each: `- v1 (2026-06-17) — diversified ICL examples; 370 pairs`.
- Store one record per transcript (a `.json` file, or one line in a `.jsonl`)
  carrying the facet fields + the conversation, so the browser can index them.

## 2. Register: write `dashboard.json`

In the experiment dir, add `dashboard.json` (see the README schema). Minimum
useful config: point `records` at the per-transcript files and set `transcript`
to the readable conversation fields; facets auto-detect from scalar fields.

- **Expose the version as a facet**: if versions are subdirs, capture them with
  `path_regex`, e.g. `"path_regex": "results/(?P<version>v\\d+[^/]*)/"`.
- **Encode other conditions in the path** (model, framing) the same way.
- **Merge a side file** (judge scores, labels) with `joins` on a shared id.
- **Lift nested-dict scalars** to facets with `flatten` (e.g. `parsed.score`).
- **Huge experiments** (>~10k records, or transcripts split into separate
  files): build a small per-record summary index (one JSONL row: id + facets +
  a path to the transcript dir) and point `records` at it, with
  `transcript_path_field` + `transcript_dir_files` for lazy transcript loading.
  Copy `2026-06-09_distressed_subagent_gemini/build_browse_index.py` as a
  template; gitignore the generated index.

## 3. Test against the running dashboard

The server reloads configs per request — no restart needed.

```bash
EXP=YYYY-MM-DD_your_experiment
# browse page returns facets + record count
curl -s "http://localhost:8800/exp/$EXP/browse" -o /tmp/b.html
python3 -c "import re,html,json; d=json.loads(html.unescape(re.search(r'display:none\">(.*?)</div>', open('/tmp/b.html').read(), re.S).group(1))); print('facets:', [f['field'] for f in d['facets']]); print('total:', d['total'])"
# a filtered page (server-side)
curl -s -X POST "http://localhost:8800/exp/$EXP/rows" -H 'Content-Type: application/json' -d '{"limit":3}'
# a record's transcript renders readably (pick an _id from rows above)
curl -s "http://localhost:8800/exp/$EXP/rec/0" | head -c 600
```

Iterate until: facets are the meaningful experiment variables (no ids/noise/
single-value junk), the version facet is present, and transcripts render as
readable conversation (not raw JSON dumps). Read a few records directly to
sanity-check the data before declaring it done.
