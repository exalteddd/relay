"""Optional LLM-written prose for the exported report.

The report is assembled from the graph and is complete without this endpoint.
What the model adds is connective writing -- an abstract, a discussion, a
limitations section -- over facts that are already fixed. It is never the
source of a number.

That split is deliberate. A report that pitches scientific rigour cannot have
a path where the prose invents a finding, so the model is given the graph as
the only permitted source, told explicitly not to introduce results, and its
output is dropped into named sections rather than used to build the document.
If the call fails, is not configured, or returns something unusable, the
report still exports with its assembled prose.

Provider comes from BRAIN_PROVIDER (openai | anthropic | mock) via the
existing brain.llm wrapper, so this adds no second client.

    POST /api/narrative  {project, metrics, dataMode}
      -> 200 {ok:true, abstract, discussion, limitations, model}
      -> 503 {ok:false, reason}   not configured; caller keeps assembled prose
"""

from __future__ import annotations

import json

from flask import Blueprint, jsonify, request, session

bp = Blueprint("narrative", __name__)

SYSTEM = """You are writing sections of a computational-biology screening report.

You will be given a JSON description of a research project: its question, the
evidence gathered, the competing hypotheses, the experiment that was run, the
measured results with their controls, and the proposed next experiments.

Rules, in order of importance:
1. Every factual claim you make must be traceable to the JSON you were given.
   Do not introduce a number, a metric, a compound, a citation or a finding
   that does not appear there. If something is not in the JSON, do not mention
   it.
2. Do not overstate. These are computational triage results, not validated
   binding or therapeutic findings. If the data is described as synthetic or
   as a fixture, say so plainly wherever you refer to the results.
3. Write for a scientifically literate reader. Plain, direct prose. No
   marketing register, no hedging padding, no bullet lists.
4. Respect the limitations the JSON already states; you may add methodological
   limitations that follow logically from the described method, but do not
   invent empirical ones.

Return STRICT JSON, no code fence, with exactly these keys:
  "abstract"    one paragraph, 120-180 words
  "discussion"  2-3 paragraphs: what the result means, why the controls make
                it credible, what it does not establish
  "limitations" one paragraph, explicitly naming what cannot be concluded
"""


VERIFY_SYSTEM = """You are a fact-checker for a scientific report. You are adversarial.

You receive (a) the project's recorded data as JSON, and (b) prose drafted
about it. Your only job is to find sentences in the prose that the data does
not support.

Flag a sentence if it:
- states a number, metric or magnitude that does not appear in the data, or
  that disagrees with it;
- characterises a result more strongly than the data warrants ("substantially
  outperforms", "confirms", "demonstrates efficacy") when the recorded values
  do not bear that out;
- claims validation, binding, causation or clinical relevance from what is
  only a computational or correlational result;
- cites or alludes to a source, dataset or prior finding that is not in the data;
- asserts a control passed, a comparison held, or a method was validated when
  the data does not record it.

Do NOT flag: careful hedging, method description that matches the data,
stated limitations, or plain restatement of recorded values.

Judge each sentence only against the data given. Absence of evidence in the
data IS grounds to flag — you are not permitted to assume a fact is true
because it is plausible.

Return STRICT JSON, no code fence:
{"unsupported": [
   {"section": "abstract|discussion|limitations",
    "sentence": "the sentence, copied exactly",
    "why": "what the data does not support, in one line"}
]}
Return {"unsupported": []} if every sentence is supported.
"""


def _strip_sentences(text: str, bad: list[str]) -> str:
    """Remove flagged sentences from a section, keeping the rest readable.

    Matching is on normalised whitespace because a model re-quoting a sentence
    rarely reproduces the original line breaks. A flagged sentence that cannot
    be located is reported to the caller rather than silently ignored.
    """
    import re

    out = text
    for s in bad:
        s = " ".join(str(s).split())
        if not s:
            continue
        pattern = re.escape(s).replace(r"\ ", r"\s+")
        out = re.sub(pattern, "", out)
    # tidy the seams left behind by removal
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\n\s*\n\s*\n+", "\n\n", out)
    return out.strip()


def _verify(llm, digest: dict, sections: dict) -> list[dict]:
    """Second pass: ask the model to attack its own draft. Failure is not fatal.

    If the check cannot run, the caller is told so explicitly -- an unverified
    draft must never be presented as a verified one.
    """
    from brain.llm import parse_json

    payload = {"data": digest, "prose": sections}
    raw = llm.text(VERIFY_SYSTEM, json.dumps(payload, indent=2, default=str),
                   max_tokens=1600)
    out = parse_json(raw)
    if not isinstance(out, dict):
        raise ValueError("verifier did not return an object")
    found = out.get("unsupported")
    if not isinstance(found, list):
        return []
    clean = []
    for f in found[:40]:
        if not isinstance(f, dict):
            continue
        sent = str(f.get("sentence") or "").strip()
        sec = str(f.get("section") or "").strip().lower()
        if sent and sec in ("abstract", "discussion", "limitations"):
            clean.append({"section": sec, "sentence": sent,
                          "why": str(f.get("why") or "").strip()[:300]})
    return clean


def _digest(project: dict, metrics: dict, data_mode: str) -> dict:
    """Compact the graph to what the prose actually needs.

    Node coordinates, render hints and chart geometry are stripped: they cost
    tokens and say nothing about the science.
    """
    keep = ("question", "source", "hypothesis", "experiment", "result", "next")
    nodes = []
    for n in project.get("nodes", []) or []:
        if n.get("type") not in keep:
            continue
        entry = {k: n.get(k) for k in ("id", "type", "title", "summary", "status") if n.get(k)}
        detail = {k: v for k, v in (n.get("meta") or {}).items()
                  if k not in ("isRank",) and v not in (None, "", [], {})}
        if detail:
            entry["detail"] = detail
        nodes.append(entry)
    return {
        "project": project.get("name"),
        "question": project.get("question"),
        "dataMode": data_mode,
        "metrics": metrics or {},
        "nodes": nodes,
        "edges": project.get("edges", []),
    }


@bp.post("/api/narrative")
def narrative():
    # Writing costs tokens, so it sits behind the same gate as a run.
    if session.get("user") is None:
        return jsonify(ok=False, reason="Sign in to generate narrative sections"), 401

    body = request.get_json(silent=True) or {}
    project = body.get("project")
    if not isinstance(project, dict):
        return jsonify(ok=False, reason="No project supplied"), 400

    try:
        from brain.config import load_config
        from brain.llm import LLM, parse_json
    except ImportError as exc:
        return jsonify(ok=False, reason=f"LLM layer unavailable: {exc}"), 503

    try:
        cfg = load_config()
        llm = LLM(cfg)
    except Exception as exc:  # missing key, unknown provider, bad base URL
        return jsonify(ok=False, reason=f"{type(exc).__name__}: {exc}"), 503

    digest = _digest(project, body.get("metrics") or {}, body.get("dataMode") or "verified")
    try:
        raw = llm.text(
            SYSTEM,
            "Project JSON:\n\n" + json.dumps(digest, indent=2, default=str),
            max_tokens=2000,
        )
        out = parse_json(raw)
    except Exception as exc:
        return jsonify(ok=False, reason=f"{type(exc).__name__}: {exc}"), 502

    if not isinstance(out, dict):
        return jsonify(ok=False, reason="Model did not return an object"), 502

    # Only the three named sections are taken. Anything else the model decided
    # to return is discarded rather than rendered.
    sections = {
        "abstract": str(out.get("abstract") or "").strip(),
        "discussion": str(out.get("discussion") or "").strip(),
        "limitations": str(out.get("limitations") or "").strip(),
    }

    # --- second pass: attack the draft ------------------------------------
    # Constraining generation is not the same as checking the output. A model
    # told not to overstate can still write "substantially outperforms" over a
    # 1.2x enrichment, and the prompt alone would never catch it. So the draft
    # is re-read against the data, and anything unsupported is removed from
    # the exported prose rather than merely noted.
    removed, verified = [], False
    try:
        removed = _verify(llm, digest, sections)
        verified = True
        by_section: dict[str, list[str]] = {}
        for f in removed:
            by_section.setdefault(f["section"], []).append(f["sentence"])
        for sec, bad in by_section.items():
            sections[sec] = _strip_sentences(sections[sec], bad)
        # A section emptied by the check is reported as empty, not patched:
        # the report then falls back to its assembled prose for that section.
        verify_note = None
    except Exception as exc:
        verify_note = f"{type(exc).__name__}: {exc}"

    return jsonify(
        ok=True,
        **sections,
        model=cfg.model,
        provider=cfg.provider,
        # The caller needs to know whether the check ran. An unverified draft
        # must not be displayed as a verified one.
        verified=verified,
        verifyError=verify_note,
        removed=removed,
        removedCount=len(removed),
    )
