# Relay

Relay turns an autonomous research loop into something you can see and steer. A
project is a graph: a question, the evidence behind it, competing hypotheses, the
experiments that test them, the results, and the next experiment to run. You move
the nodes around, open any one of them to read the detail, and press Run to watch
the agents work.

The loaded example is NDM-1 inhibitor triage. NDM-1 is the enzyme that lets
carbapenem-resistant *Klebsiella pneumoniae* destroy carbapenem antibiotics.
The task is to rank untested compounds by how likely they are to inhibit it, so
the lab assays the few most promising ones first. The product is the screening
filter, not a drug.

## What it does

- **Research graph.** Every project is a node graph you can drag, pan, zoom, and
  fit. Edges carry meaning (uses, tests, produced, supports, contradicts) and
  show their label on hover or selection. Each project keeps its own layout.
- **Inspection.** Click a node and a detail pane opens over the right half of the
  workspace while the graph stays visible. Selecting another node updates the same
  pane. Drag the divider to resize it, maximize it, or press Escape to close. Node
  positions and zoom are preserved.
- **Results.** Result nodes show a collapsible chart preview right on the canvas,
  and the full charts with their underlying tables in the detail pane. Switch
  between Plot, Table, Evidence, and Activity. Clicking a point or a bar selects
  the record behind it.
- **Sample data, labelled.** A single control flips every chart and table between
  the verified NDM-1 numbers and the illustrative placeholders. Illustrative
  values are always marked as such and never presented as measured findings.
- **Runs.** Pressing Run opens a short review of inputs, budget, and the approval
  it needs. Approve it and the agent events stream in while node statuses move
  through planned, running, done, and failed.

## Running it

No build step, no dependencies. The interface is a single HTML file.

Static, no backend:

    # open index.html directly, or serve the folder
    python -m http.server 8000
    # then open http://localhost:8000

With live runs:

    python relay_server.py
    # then open http://localhost:8000

`relay_server.py` serves the interface and adds two endpoints: `GET /api/health`
and `POST /api/run` (a Server-Sent Events stream of run events). When the medlab
screening pipeline is importable next to the server, Run drives it; otherwise it
streams a clearly labelled demo so the interface still runs end to end.

## How a run actually works

The interface is a front end. A real run happens in a backend process that has
the pipeline and the model credentials:

    UI  ->  POST /api/run  ->  server orchestrates (run_pipeline / omnigent)
        ->  streams events  ->  UI updates node status and opens the result

A static page has no server and no credentials, so it cannot orchestrate. In that
case Run replays a recorded trace instead. The two paths are labelled distinctly
in the top bar ("Live Omnigent run" versus "Replaying trace"), so a replay is
never mistaken for a live run.

## Files

- `index.html` — the whole interface, in one file
- `relay_server.py` — local server: serves the interface and exposes `/api/run`
- `assets/` — logo

## Notes on the numbers

The NDM-1 sample reflects a scaffold-split run on a synthetic fixture: ROC-AUC
0.75, enrichment 11.3x at the top 1% of the library (7.2x at 5%), against clean
controls (random 1.05x, y-scramble AUC 0.52). A real-data run uses a PubChem
NDM-1 BioAssay through the same pipeline. Everything shown in the interface is
sample or illustrative data unless a live run says otherwise.
