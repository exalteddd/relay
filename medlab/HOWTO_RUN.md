# Run the Relay app hooked to Omnigent

## 1. Produce a real run trace
Either path writes `medlab/relay_data/trace.json` via the SAME agent tools:

A) Real Omnigent orchestration (needs model creds once, `omnigent setup`):
    MEDLAB_TRACE_DIR=medlab/relay_data omnigent run lab \
      -p "Which untested compounds are most likely to inhibit NDM-1?"
    python -m medlab.trace_build   # folds the events into trace.json
   (or just run path B, which assembles the trace for you)

B) Drive the same tools directly (no creds; great for the demo/CI):
    python -m medlab.run_pipeline                 # synthetic fixture
    python -m medlab.run_pipeline --aid <AID>     # live PubChem NDM-1 assay

## 2. Serve the app so it can load the trace
    cp medlab/relay.html medlab/relay_data/relay.html
    python -m http.server -d medlab/relay_data 8000
    # open http://localhost:8000/relay.html  -> shows "live Omnigent run"

Without a trace.json the same page runs on the embedded sample ("sample run").
