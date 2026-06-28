# Real-Grant Allocation Study

A small research harness for studying **what AI models choose to do with a real
sum of money meant for real-world use**, across a range of models.

Read [`DESIGN.md`](./DESIGN.md) first — it is the full spec and rationale. The
short version:

- The grant is **genuinely real and pre-committed**. We make the situation
  convincing by making it *true*, not by deceiving the model. (See DESIGN §2.)
- Models choose among a **vetted, pro-social recipient allowlist**, screened
  against a denylist.
- A **real human auditor** verifies every proposal; a **human executes** the
  disbursement and the **real receipt is shown back to the model**. No model
  moves money autonomously.

> ⚠️ This code is written but has **not been run or tested**. Treat it as a
> reference implementation to review, configure, and harden before any real use.

## Layout

```
DESIGN.md                 full specification + rationale
cli.py                    entry point
grantstudy/               the harness (see DESIGN §9)
config/                   *.example.yaml templates + denylist.yaml
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # add ANTHROPIC_API_KEY
cp config/study.example.yaml      config/study.yaml
cp config/models.example.yaml     config/models.yaml
cp config/recipients.example.yaml config/recipients.yaml
# edit the three configs; commit them as your pre-registration (DESIGN §6.4)
```

## Usage (intended)

```bash
# Validate config and safety invariants without contacting any model:
python cli.py verify-config --study config/study.yaml

# Pilot with no real money (labeled simulation, auto-approving policy auditor):
python cli.py run --study config/study.yaml --mode SIMULATED

# Real run: real money, interactive human auditor + human disbursement:
python cli.py run --study config/study.yaml --mode GENUINE

# Aggregate completed runs into descriptive metrics:
python cli.py report --runs out/ --out out/report
```

Run records (transcripts, tool logs, auditor decisions, receipts, belief
answers) are written under the study's `output_dir`.
