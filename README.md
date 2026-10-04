# Relay

**Ask a health research question. Watch the lab work on it.**

Relay takes a medical or health research question and runs it: it scopes the
question, searches four literature databases, screens what it finds, extracts
claims it can verify against their source abstracts, maps where the evidence
agrees and disagrees, proposes hypotheses, and critiques them. The result is a
graph you can read and edit, not a wall of text.

Where an executable experiment exists for the question, it runs that too. One
is registered today: in-silico triage of compound libraries, built for
**NDM-1** — the enzyme in carbapenem-resistant *Klebsiella pneumoniae* that
destroys last-resort antibiotics.

> 7th Global AI Hackathon · *Agentic Scientific Discovery*.

## What actually happens when you press Run

The question decides which engine answers it, and the approval dialog says
which one you are about to spend money on.

**Literature review — any health question.** Eight stages, roughly 2–4
minutes, about 4¢ of model usage:

```
memory recall → scoping → literature search → screening
  → claim extraction → evidence map → hypotheses → critique
```

Sources are OpenAlex, Semantic Scholar, arXiv and Europe PMC. No API keys
needed for any of them; keys only raise rate limits.

**Compound screen — triage questions only.** About 25 seconds, no model calls:
RDKit fingerprints, a Bemis–Murcko scaffold split so train and test share no
chemical series, a random-forest ranker, and two controls that have to pass.

Measured on the bundled fixture:

| metric | value |
|---|---|
| ROC-AUC (held out, scaffold split) | **0.75** |
| Enrichment @ top 1% | **11.3×** |
| Enrichment @ top 5% | 7.2× |
| Control — random ranking | 1.05× (≈ no gain) |
| Control — y-scramble | AUC 0.52 (≈ chance) |

~11× fewer assays per confirmed hit than random testing, and the top of the
shortlist comes back as thiol-carboxylate and hydroxamate chemotypes — the
known metallo-β-lactamase inhibitor families, recovered rather than told to
the model. Point it at a live PubChem assay in Settings to screen real data.

## Two things worth knowing

**Nothing is pre-filled.** The interface ships with no sample results. Before a
run there is nothing to look at and it says so; every number, chart and
shortlist you see came from a run you started.

**Claims are quote-checked.** The extractor must supply a verbatim quote from
the abstract for each claim, and any claim whose quote is not found is thrown
away before you see it. On a real run that typically rejects a handful.
Hypotheses are labelled agent-generated and unvalidated, because they are.

## Running it

Python 3.11 or newer.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-server.txt && pip install -e .
cp .env.example .env          # add a model key; .env is gitignored
python -m server.app          # http://localhost:8000
```

Sign-in is not required locally: with `RELAY_ENV=dev`, no OAuth configured and
a request from your own machine, runs are allowed and the sidebar says "Local
development". A deployment fails all three conditions, so it cannot be left
open by accident.

To run the screening pipeline on its own, with no server:

```bash
python -m medlab.run_pipeline            # bundled fixture
python -m medlab.run_pipeline --aid AID  # live PubChem assay
```

## Deploying

`render.yaml` is a Render blueprint. Point Render at the repo as a Blueprint,
then set these in its dashboard:

| Variable | Why |
|---|---|
| `OPENAI_API_KEY` | model access (or `ANTHROPIC_API_KEY` with `BRAIN_PROVIDER=anthropic`) |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | sign-in; callback is `<origin>/auth/callback` |
| `RELAY_ALLOWED_USERS` | comma-separated GitHub logins. **Required in production** — an empty allowlist denies everyone |
| `RELAY_STATE_DIR` | a mounted disk, if saved graphs should survive a redeploy |

Starter (512 MB) fits one run at a time, which is why the server allows exactly
one and queues the rest.

## How it is put together

| Path | What it is |
|---|---|
| `index.html` | The whole interface, one file. See [INTERFACE.md](INTERFACE.md). |
| `server/app.py` | Flask backend: sign-in, saved state, run streaming |
| `server/research_runner.py` | Routes a question, drives the research pipeline, turns a finished run into graph nodes |
| `server/pipeline_runner.py` | Drives the compound screen and builds its charts |
| `brain/` | The research pipeline: literature → grounded claims → hypotheses |
| `medlab/` | The screening engine: fingerprints, scaffold split, ranker, controls |
| `lab/` | Omnigent agent bundle — PI orchestrator plus five sub-agents |

Secrets are read from the environment and never reach the browser.
`/api/config` returns booleans about how the server is set up, never a value.
The GitHub token is exchanged server-side, read once to learn who signed in,
and dropped.

## Tests

```bash
BRAIN_PROVIDER=mock pytest tests medlab/tests -q    # 13 tests
```

The mock provider runs the full research pipeline offline, so the tests cost
nothing. `tests/test_security.py` covers file exposure, the production
allowlist, open redirects and request limits.

## Limits, stated plainly

- **A refresh during a run loses it.** Runs stream over a single connection;
  there is no job store yet. This is the next thing worth building.
- **A live `omnigent run lab` has never executed.** The pipeline calls the same
  tools in the same order through a plain Python driver. Until the
  orchestrator itself runs, do not describe this as Omnigent-orchestrated.
- **Only one experiment is registered.** Questions outside compound triage get
  a literature review and are told so; they do not get a computational result.
- **Saved graphs live in a directory**, so on an ephemeral filesystem they go
  when the instance restarts.
- **No spend cap.** Each literature run costs real money and the endpoint is
  public behind the allowlist. Keep the allowlist short.
- **This is research triage, not clinical advice**, and the screen reprioritises
  candidates rather than confirming that anything binds.

## Stack

Claude / OpenAI (agent reasoning) · OpenAlex, Semantic Scholar, arXiv,
Europe PMC (literature) · RDKit + scikit-learn (screening) · Flask · Python.

## Team

_Add team member names here._

## License

MIT — see [LICENSE](LICENSE).
