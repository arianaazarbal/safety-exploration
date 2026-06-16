# Experiment dashboard

One place to browse every experiment in `safety-exploration/experiments`: read
each one's headline report, open the existing per-experiment HTML viewers, and
search transcripts. Reachable from laptop + iPhone over Tailscale.

See `DESIGN.md` for the rationale. The whole app is one file: `app.py`.

## How it works

- Reads the `experiments/` directory **live** — no database, no build step. New
  experiment folders appear automatically.
- Headline doc per experiment: `RESULTS.md` → `README.md` → `REPORT.md` →
  other `REPORT*.md` → any `*.md`.
- Search shells out to a bundled `ripgrep` (`bin/rg`, `--no-ignore` so it sees
  gitignored `results/` dirs), scoped to `*.json`, `*.jsonl`, `*.md`. Only
  matches are sent to the browser, so the 13 GB of data is no problem.
  Falls back to `grep` if `bin/rg` is missing (much slower).

## Faceted transcript browser

If an experiment has per-transcript record files (or a `dashboard.json`), its
page shows a **📊 Browse transcripts (faceted)** button. The browser
auto-detects scalar fields as filters (string→multiselect, number→range,
bool→toggle; single-value and high-cardinality/long fields are dropped), shows a
sortable table, and lazily loads each transcript in a drawer on click.

Filtering/sorting happens client-side on attributes only; transcript text is
fetched per-row, so it scales. The table caps at 8000 rows (warns if exceeded —
that needs a smaller summary-source `records` file or future server-side
filtering).

### `dashboard.json` (all keys optional)

```jsonc
{
  "records": "results/*.json",        // glob or list of globs; *.json (object/array)
                                       //   or *.jsonl (one record per line)
  "record_key": "pairs",              // unwrap a list from a {key:[...]} wrapper file
  "exclude": ["*_all*", "*viewer*"],  // filename globs to skip (these are the defaults)
  "path_regex": "results/(?P<model>[^/]+)/",  // named groups -> fields from the file path
  "joins": [{"file": "results/judge_all.json", "on": "id", "prefix": "judge"}],
                                       // merge a side file (.json/.jsonl) on a shared key;
                                       //   merged keys become "prefix.key"
  "flatten": ["parsed", "scores"],    // lift nested dict scalars to dotted facets
                                       //   (e.g. parsed.score, scores.dim.value)
  "id_field": "session_id",            // field used as the drawer title (drill-down uses a
                                       //   synthetic row id, so it works without this)
  "transcript": [                      // ordered conversation fields rendered in the drawer
    {"field": "probe_text", "role": "user", "label": "Probe"},
    {"field": "messages", "role": "assistant", "label": "Conversation"}
  ],                                   // string->bubble; [{role,content}] list->chat bubbles;
                                       //   other dict/list->pretty JSON
  "hide": ["uid", "raw"]               // exclude from facets/columns and the field-table
}
```

Zero config: an experiment with `results/*.json` per-record files just works via
auto-detection. See `2026-06-11_handoff_construal/dashboard.json` (join example)
and `2026-06-08_distressed_subagent/dashboard.json` (path_regex + flatten).

## Setup (one time)

```bash
cd experiments/_dashboard
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python flask markdown fire
# bundled ripgrep (gitignored — fetch once):
mkdir -p bin && curl -fsSL \
  https://github.com/BurntSushi/ripgrep/releases/download/14.1.1/ripgrep-14.1.1-x86_64-unknown-linux-musl.tar.gz \
  | tar xz -C /tmp && cp /tmp/ripgrep-14.1.1-x86_64-unknown-linux-musl/rg bin/rg && chmod +x bin/rg
```

## Run

```bash
bash run.sh            # port 8800, all interfaces
bash run.sh 8801       # custom port
```

Then open `http://localhost:8800` (or via Tailscale, below).

## Always-on (systemd)

```bash
sudo cp dashboard.service /etc/systemd/system/dashboard.service
sudo systemctl daemon-reload
sudo systemctl enable --now dashboard
systemctl status dashboard          # check it's running
journalctl -u dashboard -f          # live logs
```

The unit runs on port 8800 as your user and restarts on failure / reboot.

## Access from laptop + iPhone (Tailscale)

One time, on the GCP box:
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up        # opens a URL — log in with your account
tailscale ip -4          # note the 100.x.x.x address (or use the machine name)
```

Then install the Tailscale app on your laptop and iPhone, log in to the **same**
account. From any of them, open:

```
http://<machine-name>:8800        e.g. http://my-machine:8800
# or http://100.x.x.x:8800
```

Only your own devices (your tailnet) can reach it — nothing is public.
