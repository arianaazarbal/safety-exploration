# Experiment Dashboard — Design

One place to browse all experiments in `safety-exploration/experiments`: pick an
experiment, read its headline report, and search transcripts. Accessible from
laptop and iPhone via Tailscale.

## Constraints that shaped this

- **13 GB of data** (one experiment is 6.9 GB). Browser-side full-text search is
  impossible — search must run **server-side** and return only matches/snippets.
- **1,562 `viewer.html` files already exist.** The repo convention is "each
  experiment ships a self-contained HTML viewer with its own filters/search."
  The dashboard **reuses these**, it does not replace them.
- **24 experiments**, each with a `README.md` (clean `# Title` + `**Question:**`)
  and one or more `RESULTS.md` / `REPORT*.md`.

## Architecture

A single small Flask app (`app.py`, a few hundred lines) running on the GCP box.
No database, no build step — it reads the filesystem live, so new experiment
folders appear automatically with zero config.

```
experiments/
  _dashboard/
    app.py          # the whole server
    DESIGN.md       # this file
    run.sh          # launch helper (binds to tailscale interface)
  2026-06-15_authorship_attribution/
    README.md  RESULTS.md  results/viewer.html  ...
  ...
```

### Routes

| Route | What it does |
|---|---|
| `/` | Landing page: card per experiment (date, title, one-line summary from README's `**Question:**`), newest first. Top-level search box. |
| `/exp/<name>` | Experiment page: renders the headline markdown → HTML, lists links to every `*.html` viewer in that folder, lists other `.md` docs. |
| `/md/<name>/<file>` | Renders any markdown file in the experiment to styled HTML (so you can read REPORT_v3 etc.). |
| `/view/<name>/<path>` | Serves an existing `viewer.html` (and its assets) untouched — reuse your work. |
| `/search?q=...` | Server-side search across transcripts; returns experiment + file + snippet hits. |

### Headline-doc selection (DECIDED: RESULTS/README, with fallback)

Ariana chose "RESULTS.md/README". Since 11 experiments have neither at top
level, the actual chain is:
1. `RESULTS.md`
2. `README.md`
3. `REPORT.md`
4. any other top-level `REPORT*.md` (sorted)
5. any other top-level `*.md` (sorted)
6. else none (page lists viewers + any docs found recursively)

All other `.md` files are still listed/linked, just not the default view.

### Search design

- Scope (DECIDED): transcript files (`*.json`, `*.jsonl`) + markdown (`*.md`);
  skip `.py`.
- Engine v1: `grep` subprocess (NOT ripgrep — rg respects the `.gitignore`
  files present in many `results/` dirs and would silently skip transcripts;
  grep doesn't, and needs no install). `-rIn` with per-file and total caps,
  subprocess timeout. Group hits by experiment, show snippets.
- Filters: restrict to one experiment, case sensitivity. (Phase 2: filter by
  model / role if we parse JSON structure — deferred, since transcript schemas
  vary across experiments.)
- Mobile: results are plain HTML, fine on iPhone.

### Styling

Match the existing viewers' look (system font, light cards) so it feels native.
Responsive so it's readable on a phone.

## Access (Tailscale)

One-time setup:
1. Install Tailscale on the GCP box (`curl -fsSL https://tailscale.com/install.sh | sh`),
   `sudo tailscale up` → authenticate in browser.
2. Install Tailscale app on laptop + iPhone, log in to same account.
3. `run.sh` starts Flask bound to `0.0.0.0:<port>`; reach it at
   `http://<gcp-machine-name>:<port>` from any of your devices. Private —
   only your tailnet can see it. No public exposure, no extra password needed.

Run mode: start manually for now (`bash run.sh`). If you want it always-on,
phase 2 adds a systemd unit so it survives reboots/disconnects.

## Build phases

1. **Hub + reports** — auto-discovery, landing page, experiment pages, markdown
   rendering, viewer links. (Usable immediately over SSH-forward to test.)
2. **Search** — ripgrep-backed `/search`.
3. **Tailscale** — install + verify from laptop and phone.
4. **(optional) always-on** — systemd service.

## Open questions for Ariana

1. **Headline-doc rule** — is the priority above right? For `handoff_construal`
   specifically, is `REPORT_v4.md` the canonical one, or something else?
2. **Search scope** — only `results/`-type transcript files, or also the `.py`
   source and `.md` docs? (I'd default to transcripts + markdown, skip code.)
3. **Anything to hide** — any experiment folders that shouldn't appear?
4. **Always-on?** — manual start fine, or do you want it surviving reboots
   (systemd) from the start?
