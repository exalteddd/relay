# Relay — an autonomous discovery lab

**Decide what to test, before you spend testing it.** Relay runs a computational
discovery loop — read the literature, form a hypothesis, choose and run an
experiment, judge the result, sharpen the next question — and turns a library of
thousands of compounds into a short, ranked, evidence-backed shortlist worth an
assay.

The loaded case study is **NDM-1 inhibitor triage**. NDM-1 is the enzyme in
carbapenem-resistant *Klebsiella pneumoniae* that destroys last-resort carbapenem
antibiotics; Relay ranks untested compounds by how likely they are to inhibit it,
so the lab assays the few most promising ones first.

> 7th Global AI Hackathon · *Agentic Scientific Discovery*.
> In-silico triage, not a validated therapeutic — it reprioritizes candidates, it
> does not confirm binding.

**Live interface:** https://exalteddd.github.io/relay/ — a static host, so Run
there replays a recorded trace. Real runs need the backend below.

## Result (held-out, scaffold-split)

| metric | value |
|---|---|
| Ranking accuracy (ROC-AUC) | **0.75** |
| Enrichment @ top 1% | **11.3×** |
| Enrichment @ top 5% | 7.2× |
| Control — random ranking | 1.05× (≈ no gain) |
| Control — y-scramble AUC | 0.52 (≈ chance) |

**~11× fewer assays per confirmed hit** than random testing. The top of the
shortlist comes back as thiol-carboxylate and hydroxamate chemotypes — the known
metallo-β-lactamase inhibitor families — recovered by the model rather than told
to it.

These numbers are from the **bundled synthetic fixture**, not a live assay. A
real-data run is the same code with `--aid <PubChem NDM-1 assay>`.

## How it fits together

```
index.html ──POST /api/run──▶ server/app.py ──▶ server/pipeline_runner.py
    ▲                          (auth, SSE)        │ calls the agent tools
    └──── streamed events ────────────────────────┘ propose → screen → judge
```

The interface is a front end. A real run happens in the backend, which has the
pipeline and the credentials. A static page has neither, so it replays a recorded
trace — and labels itself as doing so. The two paths are never presented alike.

| Path | What it is |
|---|---|
| `index.html` | The whole interface, one file. See [INTERFACE.md](INTERFACE.md). |
| `server/` | Flask backend: GitHub sign-in, and runs streamed over SSE |
| `medlab/` | The screening engine: fingerprints, Bemis-Murcko scaffold split, random-forest ranker, enrichment + ROC-AUC with controls, PubChem fetch with synthetic fallback |
| `lab/` | Omnigent agent bundle — PI orchestrator plus Literature, Extractor, Hypothesizer, Critic and Screener, with the approval gate and budget policies |
| `brain/` | The research-memory framework underneath |

## Run it locally

Requires Python 3.11+.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-server.txt && pip install -e .
cp .env.example .env          # fill in what you need; .env is gitignored
python -m server.app          # http://localhost:8000
```

Without `GITHUB_CLIENT_ID`/`GITHUB_CLIENT_SECRET` the interface still loads and
the engine still runs from the command line, but sign-in is disabled and so
`/api/run` will refuse. To run the pipeline directly, with no server:

```bash
python -m medlab.run_pipeline            # synthetic fixture
python -m medlab.run_pipeline --aid AID  # live PubChem assay
```

## Deploy

`render.yaml` is a Render blueprint. Point Render at this repo as a Blueprint;
it builds, sets `RELAY_ENV=prod`, generates `SESSION_SECRET`, and prompts once
for the values marked `sync: false`.

Create a GitHub OAuth app (Settings → Developer settings → OAuth Apps) with the
callback URL set to exactly `<your-origin>/auth/callback`, and put its id and
secret in Render's environment.

## Secrets

- Every secret is read from the environment. None are committed, and `.env` is
  gitignored.
- The browser is never sent a key. `/api/config` returns booleans describing how
  the server is configured, never a value.
- The GitHub access token is exchanged server-side, read once to learn who signed
  in, then dropped. It is not stored in the session and no endpoint returns it.
- OAuth asks for `read:user` only — no repository or write access.
- Session cookies are signed, `HttpOnly`, `SameSite=Lax`, and `Secure` off
  localhost. `SESSION_SECRET` is mandatory when `RELAY_ENV` is not `dev`.
- `RELAY_ALLOWED_USERS` optionally restricts who may spend compute.

## Tests

```bash
python -m pytest medlab/tests -q       # screening engine + controls
BRAIN_PROVIDER=mock pytest tests -q    # research-memory framework
```

## Scientific rigor & safety

- **Scaffold split** — train and test share no Bemis-Murcko scaffold, so the
  model cannot win by memorizing a chemical series.
- **Two controls** — a random-ranking baseline (must be ≈1×) and a y-scramble
  (shuffled labels must collapse AUC to ≈0.5).
- **Human approval gate** on the experiment choice, plus cost and tool-call
  budgets. A question typed into the prompt bar still goes through it.
- **Labelled provenance** — the interface distinguishes a live run from a
  replayed trace, and sample data from measured findings.
- **Limits** — this works from bioassay labels, not binding confirmation;
  bioassay actives can include frequent hitters; any hit needs cell-based and
  resistance validation.

## Status

- The screening pipeline runs for real through the backend: a signed-in run
  streams events from the actual tools and returns the figures above.
- **A live `omnigent run lab`** — the Omnigent orchestrator LLM dispatching the
  sub-agents — **has not been executed yet.** It needs model credentials and a
  local runtime (Node 22, tmux, bubblewrap). The verified runs come from the
  driver that calls the *same tools in the same order*. Until that runs, nothing
  here should be described as a live Omnigent orchestration.

## Stack

Omnigent (orchestration) · Claude (agent reasoning) · PubChem BioAssay +
Europe PMC (data) · RDKit + scikit-learn (screening) · Flask · Python.

## Team

_Add team member names here._

## License

MIT — see [LICENSE](LICENSE).
