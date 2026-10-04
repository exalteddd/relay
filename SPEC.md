# Spec: any health research question drives a real agent run

## Problem Statement

Relay presents itself as an autonomous discovery lab, but it can only do one
thing. A researcher types a question into the prompt bar and the same NDM-1
compound screen runs regardless of what they asked. The question is recorded in
the run trace and then ignored: the planner returns two hardcoded candidate
tests and always picks the first.

So the product people see and the product that exists are different things. The
NDM-1 triage is meant to be *one worked example* of a general capability, and
right now it is the entire capability. A user cannot ask "what is the evidence
that GLP-1 agonists reduce cardiovascular events?" and get anything at all.

Secondary gaps that make the app feel like scaffolding: nodes cannot be deleted
or renamed, "New project" and "Share" are toast stubs, and the Materials and
Solar projects are decorative graphs that would display NDM-1 numbers under
their own headings if run.

## Solution

Any medical or health research question, typed into the prompt bar, starts a
real multi-agent research run and streams its progress into the graph.

The general loop already exists in `brain/` and is unwired. `run_research()`
takes an arbitrary question and runs eight stages — memory recall, scoping,
literature search, screening, grounded claim extraction, evidence mapping,
hypothesis generation, critique — against four public literature sources,
emitting a progress event at each stage. None of it is NDM-1 specific.

The work is therefore mostly **connecting what exists**, not building a new
agent system:

- Route a typed question to the general pipeline, and stream its stage events
  into the existing run UI.
- Represent its output as graph nodes: claims as evidence, hypotheses as
  hypothesis nodes, the critique as review state, gaps as next questions.
- Keep the NDM-1 screen as the one *executable* experiment. When a question is
  a compound-triage question, the loop can additionally run it and produce
  measured numbers. For everything else the loop still delivers a literature-
  grounded evidence map, hypotheses and a critique — it simply has no wet-lab
  proxy to execute.

That distinction is the core of the design: **reasoning generalises, execution
does not.** The honest product says so rather than implying every question ends
in a measured result.

## User Stories

1. As a researcher, I want to type any health research question into the prompt bar, so that the lab works on my problem rather than a fixed demo.
2. As a researcher, I want to see my question echoed as the root node of the graph, so that I can confirm what the system is actually answering.
3. As a researcher, I want the system to refine my question into sub-questions, so that I can see how it decomposed the problem.
4. As a researcher, I want to see the search queries it generated, so that I can judge whether the literature sweep was sensible.
5. As a researcher, I want to watch each agent stage report progress as it completes, so that a multi-minute run does not look frozen.
6. As a researcher, I want to see how many papers were found, screened and kept, so that I can gauge the evidence base.
7. As a researcher, I want every extracted claim to carry a citation, so that I can verify it against the source.
8. As a researcher, I want claims whose supporting quote is not verbatim in the abstract to be rejected, so that I am not shown fabricated evidence.
9. As a researcher, I want to click a claim and see the paper it came from, so that I can read the original.
10. As a researcher, I want contradictions between claims surfaced explicitly, so that I can see where the literature disagrees.
11. As a researcher, I want gaps in the evidence surfaced, so that I can see what is unknown.
12. As a researcher, I want generated hypotheses to be clearly labelled as agent-generated and unvalidated, so that I never mistake them for findings.
13. As a researcher, I want each hypothesis critiqued and ranked, so that I can start with the strongest.
14. As a researcher, I want rejected hypotheses kept visible as negative knowledge, so that the team does not re-propose them.
15. As a researcher, I want the run to say which parts are literature reasoning and which are measured computation, so that I can weight them differently.
16. As a researcher asking a compound-triage question, I want the lab to additionally run the screening experiment, so that I get measured numbers and not only a literature review.
17. As a researcher asking a question with no executable experiment, I want to be told that plainly, so that I understand why there are no numbers.
18. As a researcher, I want to approve the run before it spends money, so that I control cost.
19. As a researcher, I want to see the estimated and actual token cost of a run, so that I can manage a budget.
20. As a researcher, I want to cancel a run in progress, so that I can stop a bad question early.
21. As a researcher, I want a run to survive a page refresh, so that I do not lose several minutes of work.
22. As a researcher, I want to reopen a past run and see its full result, so that work is not lost when I close the tab.
23. As a researcher, I want each question to become its own project, so that separate investigations do not collide.
24. As a researcher, I want the system to recall related earlier projects, so that it builds on prior work instead of starting cold.
25. As a researcher, I want to delete a node I do not want, so that the graph reflects my thinking.
26. As a researcher, I want deleting a node to remove its edges too, so that the graph never has dangling connections.
27. As a researcher, I want to undo a delete, so that a misclick is not destructive.
28. As a researcher, I want to rename a node, so that I can use my own language.
29. As a researcher, I want to add my own evidence or notes to the graph, so that I can combine my knowledge with the agents'.
30. As a researcher, I want to export a run as a report, so that I can share it with colleagues who do not use the tool.
31. As a signed-in user, I want my runs to be private to me, so that unpublished research is not exposed.
32. As the operator, I want runs to require sign-in, so that strangers cannot spend my model budget.
33. As the operator, I want a per-user rate limit and a spend cap, so that one user cannot exhaust the budget.
34. As the operator, I want a failed external source to degrade the run rather than kill it, so that one flaky API does not waste the whole run.
35. As the operator, I want model credentials to live only in the server environment, so that they are never exposed to a browser.
36. As a judge or first-time visitor, I want a worked example I can look at without signing in, so that I can understand the product immediately.
37. As a judge, I want to tell the difference between the showcase example and a live run, so that I am not misled about what was computed.

## Implementation Decisions

### Seams

The guiding constraint is to add as few new seams as possible. Three already
exist and should carry this feature:

1. **`POST /api/run` (HTTP seam).** Already streams server-sent events to the
   interface and already carries the question in its body. Keep the contract;
   widen what can be behind it. This stays the only network seam.
2. **`run_research(question, cfg, opts, on_event)` (library seam).** Already
   accepts an arbitrary question and already emits `(agent, message)` progress
   events. This is the natural integration point and needs no modification to
   be driven.
3. **The UI event vocabulary** (`status` / `act` / `result` / `error` / `done`).
   Already renders a streaming run. New information should be expressed in this
   vocabulary rather than a parallel one.

**One new seam** is proposed: a run-mode resolver that decides, from the
question, whether a run is *literature-only* or *literature plus an executable
experiment*, and which experiment. Putting that decision in one place keeps
routing out of both the HTTP layer and the agent layer, and gives the one thing
worth unit-testing hard. Everything else is adaptation between seams that exist.

### Modules

- **Research driver (new, thin).** Adapts `run_research`'s `on_event` callback
  into the UI event vocabulary and into graph mutations. Owns the mapping from
  pipeline stages to node types.
- **Run-mode resolver (new).** Maps a question to a run plan: always the
  literature loop, optionally a registered executable experiment. Initially one
  experiment is registered (compound triage). Must return "no executable
  experiment" as a first-class, non-error outcome.
- **Experiment registry (new, small).** A named list of executable experiments
  with a predicate and a runner. Keeps "what can actually be executed" in one
  enumerable place rather than implied by control flow. The existing screening
  pipeline becomes its first entry.
- **`server/pipeline_runner.py` (modify).** Today it hardcodes the screening
  sequence. Becomes the executor for the compound-triage registry entry.
- **`server/app.py` (modify).** Gains run lifecycle: create, stream, cancel,
  fetch result. No longer assumes a run completes inside one request.
- **Interface (modify).** Renders claims, hypotheses, evidence map and critique
  as graph nodes; adds delete and rename; removes or disables the decorative
  projects.

### Execution model — the significant change

A compound screen takes ~25 seconds and completes inside one HTTP request. A
general research run does a literature sweep plus dozens of LLM calls across
eight stages: **minutes, not seconds.** The current design holds a web worker
open for the entire run and loses everything if the browser refreshes.

Decisions:

- Runs become **background jobs with identity**. `POST /api/run` starts a job
  and returns its id; the event stream is a separate subscription to that job.
  A refresh re-subscribes rather than losing the run.
- Job state and the event log **persist**, so a completed run can be reopened.
- A run is **cancellable**, and cancellation stops further model spend.
- Partial results are **durable**: if critique fails, the claims and evidence
  map that already succeeded are still available.
- Long-lived streaming must survive the hosting platform's request timeouts and
  proxy buffering. If the deployment target cannot hold a minutes-long response,
  the client polls the job instead. The job abstraction makes this swappable
  without touching the agent layer.

### Cost and abuse

Each run costs real money — dozens of model calls, strong model for planning,
synthesis, hypotheses and critique; fast model for screening and extraction.
Once deployed publicly this is a spend endpoint exposed to the internet.

- Sign-in required for runs (already built), with the allowlist as the default
  control during the demo period.
- Per-user rate limit and a server-wide daily spend cap, both enforced before a
  run starts, not after.
- The run's token usage is already tracked by the pipeline and should be
  surfaced in the UI and persisted with the run.
- The approval gate must show a cost estimate before the user commits.

### Data and sources

- Four literature sources (OpenAlex, Semantic Scholar, arXiv, Europe PMC) work
  without API keys; keys only raise rate limits. No new credentials are needed
  beyond the model key.
- Source failures are already collected rather than raised. They must surface in
  the UI so a thin evidence base is visibly explained.
- The pipeline already rejects any claim whose quote is not verbatim in the
  abstract. This is a correctness guarantee worth stating in the interface, not
  just honouring silently.

### Provenance

The interface must distinguish three kinds of content, and never let them blur:

- **Measured** — produced by an executed experiment on real data.
- **Literature-grounded** — a claim traceable to a cited paper.
- **Agent-generated** — hypotheses and syntheses, unvalidated.

The existing "Measured run" / "Sample data" badge generalises to this.

## Testing Decisions

A good test here exercises behaviour through a seam and asserts on observable
output. It must not assert on prompt text, call counts, or event ordering beyond
what a consumer actually depends on. Prior art: `medlab/tests/` drives the
screening engine end to end and asserts on metric properties; `tests/test_brain.py`
runs the research pipeline against `BRAIN_PROVIDER=mock`.

- **Run-mode resolver** — the one piece deserving exhaustive unit tests. A
  compound-triage question selects the screening experiment; a general clinical
  question selects literature-only; an ambiguous question resolves
  deterministically. Pure function, no I/O.
- **Research driver** — drive `run_research` with the mock provider and assert
  the emitted UI events and resulting graph nodes. The mock provider already
  exists and keeps this free and offline.
- **Job lifecycle** — start, subscribe, cancel, reopen; assert a cancelled run
  stops emitting and a completed run is retrievable after the stream closes.
  Test through the HTTP seam.
- **Degradation** — with a source failing and with the model erroring mid-run,
  assert partial results survive and the failure is reported rather than
  swallowed. The existing code already collects source errors.
- **Authorisation** — runs refused when signed out, off-allowlist, or over the
  rate limit. Extends the existing checks, which are already covered.
- **Graph editing** — delete removes the node and its edges; undo restores both;
  rename persists.

Not worth testing: exact prompt strings, LLM output quality (non-deterministic),
or the precise wording of progress messages.

## Out of Scope

- Making every question end in a measured result. Only questions matching a
  registered experiment execute anything; this spec does not add new experiment
  types.
- Implementing docking (the planner's second candidate test). It remains a
  proposal, not an executable experiment.
- A live `omnigent run lab` orchestration. The pipeline calls the same tools in
  the same order; swapping the driver for Omnigent's orchestrator is separate
  work and should not block this.
- Clinical decision support of any kind. The product triages research questions;
  it does not advise on patient care.
- Multi-user collaboration, sharing and permissions beyond "my runs are mine".
- The decorative Materials and Solar projects. They are removed or disabled, not
  made to work.

## Further Notes

### Sequencing

The ordering below front-loads the thing that is currently a lie (the question
does nothing) and defers polish.

**Phase 0 — unblock. DONE (2026-10-04).** Measured on
*"What is the evidence that GLP-1 receptor agonists reduce major adverse
cardiovascular events in adults with type 2 diabetes?"*, OpenAI provider,
gpt-5-mini strong / gpt-5-nano fast:

| | |
|---|---|
| Wall clock | **141s (2.4 min)**, completed |
| Cost | **$0.036** per run |
| Model calls | 13 (4 strong, 9 fast); 49.2K input, 29.6K output tokens |
| Literature | 315 papers found, 60 screened, 25 kept |
| Output | 33 grounded claims, 8 themes, 2 contradictions, 6 gaps, 6 hypotheses |
| Rejected | 6 claims dropped — quote not verbatim in the abstract |

Three findings that change later phases:

- **2.4 minutes per run confirms runs must be background jobs.** No HTTP
  request should be held that long, and a refresh must not lose the work.
- **Reasoning effort is the dominant cost and latency lever.** The first
  attempt, at the fast tier's default effort, spent 97.8K output tokens, cost
  $0.045 and *failed* at extraction after 303s: the model reasoned past its
  token allowance and returned nothing. At `low` effort the same work costs
  $0.036 and finishes in 141s — cheaper and twice as fast.
- **Semantic Scholar rate-limits anonymous requests** (all 4 source errors were
  429s). A free `S2_API_KEY` should be configured before the evidence base is
  judged thin.

**Phase 1 — make the question real.** Job abstraction, run-mode resolver,
research driver, question routed to the general pipeline, stage events streamed.
At the end of this phase a typed question produces a genuine, visibly different
run. This is the phase that changes what the product *is*.

**Phase 2 — make the output legible.** Claims, evidence map, hypotheses and
critique rendered as graph nodes with citations and provenance labelling. At the
end of this phase the run's output is usable rather than just a log.

**Phase 3 — make it safe to deploy.** Rate limits, spend cap, cost estimate in
the approval gate, persistence of completed runs, cancellation.

**Phase 4 — finish the app.** Delete, rename, undo. Remove the decorative
projects. Replace or remove the "New project" and "Share" stubs.

**Phase 5 — the Omnigent claim.** Run the orchestrator live once, locally, and
only then describe the system as Omnigent-orchestrated.

### Deployment checklist

- [x] Model API key obtained and set in the server environment only
- [x] `run_research` verified end to end on a real question, offline of the UI
- [x] Real run cost and duration measured (141s, $0.036) — hosting plan still to choose
- [ ] Background job execution working, with runs surviving a refresh
- [ ] Rate limit and daily spend cap enforced before a run starts
- [ ] GitHub OAuth app created with the deployed callback URL
- [ ] Allowlist populated for the demo period
- [ ] Secrets set in the host dashboard, never in the repo
- [ ] Health check green and the engine confirmed present in the deployed container
- [ ] One signed-in end-to-end run completed against the deployed instance
- [ ] Provenance labelling verified: measured, literature-grounded and
      agent-generated are visibly distinct
- [ ] A signed-out visitor sees the worked example and understands it is one
- [ ] Exactly one public URL; the static showcase is retired or clearly marked

### Risks

- **Runtime versus hosting.** A multi-minute run on a free tier that sleeps and
  caps memory is the most likely cause of a broken demo. Decide the hosting tier
  against a measured run, not an estimate.
- **Cost exposure.** A public, sign-in-gated spend endpoint still spends. The cap
  must exist before the URL is shared.
- **Quality variance on open questions.** The NDM-1 example is tuned; an
  arbitrary question may return a thin evidence base. The interface should make a
  weak run visibly weak rather than dress it up.
- **Scope creep toward "answers any question".** The system reasons over
  literature for any question but executes only registered experiments.
  Marketing it as more than that recreates exactly the problem this spec fixes.
