# Relay — an AI lab assistant that decides what to test next

Testing a chemical compound in a lab is slow and expensive. If you have thousands
of candidates and can only afford to test a handful, the hard question isn't
*how* to test — it's *which ones to test first*.

Relay answers that question. It reads the scientific literature, forms an idea
about what might work, runs a computational experiment to check that idea, grades
its own result, and then asks a sharper question. The output is a short, ranked
list of compounds worth putting in a real lab test, with the evidence for each
one attached.

## The problem it's working on

The loaded example is **NDM-1**.

NDM-1 is an enzyme made by some drug-resistant bacteria (notably *Klebsiella
pneumoniae*). It destroys carbapenems — the antibiotics doctors fall back on when
nothing else works. If you could block NDM-1, those antibiotics would work again.

So the task is: given a library of thousands of untested compounds, rank them by
how likely each one is to block NDM-1, so the lab can test the most promising
ones first.

> Built for the 7th Global AI Hackathon, *Agentic Scientific Discovery* track.
> This is a prioritization tool, not a drug. It reorders a to-test list — it does
> not prove that any compound actually binds to NDM-1.

**Try the interface:** https://exalteddd.github.io/relay/

That link is a static page with no server behind it, so pressing Run there
replays a **recorded** run. Real runs need the backend described below.

## Does it actually work?

We tested it the honest way: the model was trained on one set of compounds and
scored on a separate set it had never seen, picked so the two sets share no
common chemical backbone. (Why that matters is under
[How we avoided fooling ourselves](#how-we-avoided-fooling-ourselves).)

| What we measured | Result | What it means |
|---|---|---|
| ROC-AUC | **0.75** | How well it sorts good from bad. 0.5 = coin flip, 1.0 = perfect. |
| Hit rate in its top 1% | **11.3×** | Its top picks contain 11× more real hits than a random pick would. |
| Hit rate in its top 5% | 7.2× | Still about 7× better than random. |
| Control: random ranking | 1.05× | A random list gives no advantage — as it should be. |
| Control: scrambled answers | AUC 0.52 | With the answers shuffled the model learns nothing — as it should be. |

In plain terms: **you'd run about 11× fewer lab tests to find the same number of
real hits.**

A good sanity check — the compounds that float to the top are thiol-carboxylates
and hydroxamates, the two chemical families already known to block enzymes like
NDM-1. Nobody told the model that; it found them on its own.

**Important caveat:** these numbers come from a bundled synthetic test dataset,
not from real lab measurements. Running it on real data is the same code with one
extra flag (`--aid <PubChem NDM-1 assay id>`).

## How the pieces fit together

```
index.html ──POST /api/run──▶ server/app.py ──▶ server/pipeline_runner.py
    ▲                       (login, event stream)   │ calls the agent tools
    └──── live events stream back ──────────────────┘ propose → screen → judge
```

The web page is only the front end — it holds no pipeline and no API keys. The
real work happens in the backend. When there is no backend (as on the public
static link), pressing Run replays a saved recording instead, and the page says
so in the top bar. A replay is never dressed up as a live run.

| Folder / file | What's in it |
|---|---|
| `index.html` | The entire user interface, in one file. See [INTERFACE.md](INTERFACE.md). |
| `server/` | The Flask backend: GitHub sign-in, plus runs streamed live to the browser |
| `medlab/` | The scoring engine — turns molecules into numeric fingerprints, splits the data fairly, trains a random-forest ranker, and reports the scores above alongside the controls |
| `lab/` | The agent team: a lead orchestrator plus specialists for reading literature, pulling out facts, proposing hypotheses, criticizing them, and screening compounds — with the human approval step and spending limits |
| `brain/` | The memory layer the agents read from and write to |

## Running it on your machine

Needs Python 3.11 or newer.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-server.txt && pip install -e .
cp .env.example .env          # fill in what you need; .env is never committed
python -m server.app          # http://localhost:8000
```

If you skip `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET`, the page still loads and
the engine still works from the command line — but sign-in is off, and since runs
require sign-in, `/api/run` will refuse.

To skip the web interface entirely and just run the pipeline:

```bash
python -m medlab.run_pipeline            # bundled synthetic dataset
python -m medlab.run_pipeline --aid AID  # real PubChem assay data
```

## Deploying it

`render.yaml` is a Render blueprint. Point Render at this repo as a Blueprint and
it handles the build, sets `RELAY_ENV=prod`, generates a `SESSION_SECRET`, and
asks you once for the values marked `sync: false`.

You also need a GitHub OAuth app (GitHub → Settings → Developer settings → OAuth
Apps). Set its callback URL to exactly `<your-origin>/auth/callback`, then paste
its client id and secret into Render's environment variables.

## How secrets are handled

- Every secret comes from an environment variable. None are in the repo, and
  `.env` is gitignored.
- **The browser never receives a key.** `/api/config` returns only true/false
  flags about how the server is set up, never an actual value.
- The GitHub token is exchanged on the server, read once to learn who signed in,
  then dropped. It isn't saved in the session and no endpoint hands it back.
- Sign-in asks for `read:user` only — no access to your repositories, and no
  write permission of any kind.
- Session cookies are signed, `HttpOnly`, `SameSite=Lax`, and `Secure` anywhere
  but localhost. `SESSION_SECRET` is required whenever `RELAY_ENV` isn't `dev`.
- `RELAY_ALLOWED_USERS` optionally limits who is allowed to spend compute.

## Tests

```bash
python -m pytest medlab/tests -q       # scoring engine + controls
BRAIN_PROVIDER=mock pytest tests -q    # memory layer
```

## How we avoided fooling ourselves

It's easy to build a model that looks brilliant and is really just cheating.
Three guards against that:

- **A fair train/test split.** Molecules are grouped by their core chemical
  skeleton, and no skeleton appears in both the training and the test set.
  Without this, a model can score well just by memorizing one family of
  near-identical compounds — which tells you nothing about anything new.
- **Two controls that are supposed to fail.** A random ranking has to come out at
  ≈1× (no better than chance), and retraining on deliberately scrambled answers
  has to collapse the AUC to ≈0.5. If either control had looked good, the result
  would be an artifact of the setup rather than real signal. Both came back
  clean.
- **A human has to approve.** Every experiment choice goes past a person before
  it runs, and there are hard caps on cost and on the number of tool calls. Even
  a question typed straight into the prompt bar goes through the same gate.

The interface also labels where everything came from: live run vs. replay, sample
data vs. measured result.

**What this does *not* tell you.** The model learns from lab-test pass/fail
labels, not from confirmed physical binding. Compounds marked "active" in such
tests sometimes include ones that misbehave across many tests for uninteresting
reasons. Anything promising still needs cell-based testing and resistance
follow-up before it means anything.

## Current status

- **The screening pipeline is real.** Sign in, press Run, and the backend
  executes the actual tools and returns the numbers above.
- **The full agent orchestration hasn't been run end to end yet.** Having the
  orchestrator LLM dispatch the sub-agents live needs model credentials and a
  local runtime (Node 22, tmux, bubblewrap) that we haven't set up. The verified
  results come from a driver that calls *the same tools in the same order*. Until
  that live run happens, nothing here should be described as a live Omnigent
  orchestration.

## Built with

Omnigent (agent orchestration) · Claude (agent reasoning) · PubChem BioAssay and
Europe PMC (data sources) · RDKit and scikit-learn (chemistry + machine learning)
· Flask · Python.

## Team

_Add team member names here._

## License

MIT — see [LICENSE](LICENSE).
