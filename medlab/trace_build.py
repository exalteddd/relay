"""Fold the events a real `omnigent run lab` emitted into trace.json for the Relay app.

    MEDLAB_TRACE_DIR=medlab/relay_data omnigent run lab -p "..."
    MEDLAB_TRACE_DIR=medlab/relay_data python -m medlab.trace_build
"""

from __future__ import annotations

import os
from pathlib import Path

from medlab import trace


def main():
    d = os.environ.get("MEDLAB_TRACE_DIR", str(Path(__file__).resolve().parent / "relay_data"))
    os.environ["MEDLAB_TRACE_DIR"] = d
    out = Path(d) / "trace.json"
    t = trace.assemble(out)
    print(f"wrote {out}  steps={len(t['steps'])}  "
          f"EF@1%={(t.get('result') or {}).get('ef_top1pct')}  ranked={len(t.get('ranked') or [])}")


if __name__ == "__main__":
    main()
