"""Thin LLM wrapper: Anthropic, any OpenAI-compatible endpoint (incl. Databricks
Model Serving), or a deterministic mock for offline testing."""

from __future__ import annotations

import json
import re
import threading

from .config import Config


def parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON found in model output")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


class LLM:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.provider = cfg.provider
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self._lock = threading.Lock()
        if self.provider == "anthropic":
            import anthropic
            if not cfg.anthropic_api_key:
                raise RuntimeError("Set ANTHROPIC_API_KEY (or BRAIN_PROVIDER=openai / mock).")
            self.client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        elif self.provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url)
        elif self.provider == "mock":
            self.client = MockLLM()
        else:
            raise ValueError(f"unknown provider {self.provider}")

    def _track(self, inp: int, out: int) -> None:
        with self._lock:
            self.usage["calls"] += 1
            self.usage["input_tokens"] += inp or 0
            self.usage["output_tokens"] += out or 0

    def text(self, system: str, prompt: str, fast: bool = False, max_tokens: int = 4096) -> str:
        model = self.cfg.fast_model if fast else self.cfg.model
        if self.provider == "anthropic":
            r = self.client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                            messages=[{"role": "user", "content": prompt}])
            self._track(r.usage.input_tokens, r.usage.output_tokens)
            return "".join(b.text for b in r.content if b.type == "text")
        r = self.client.chat.completions.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        if r.usage:
            self._track(r.usage.prompt_tokens, r.usage.completion_tokens)
        return r.choices[0].message.content or ""

    def json(self, task: str, system: str, prompt: str, fast: bool = False,
             max_tokens: int = 4096, context: dict | None = None):
        if self.provider == "mock":
            self._track(len(prompt) // 4, 200)
            return self.client.handle(task, context or {})
        system = system + "\n\nRespond with a single valid JSON object and nothing else."
        last_err = None
        for _ in range(2):
            raw = self.text(system, prompt, fast=fast, max_tokens=max_tokens)
            try:
                return parse_json(raw)
            except (ValueError, json.JSONDecodeError) as e:
                last_err = e
                prompt += "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object."
        raise RuntimeError(f"{task}: model did not return valid JSON ({last_err})")


class MockLLM:
    """Deterministic stand-in so the full pipeline can run offline and in tests."""

    def handle(self, task: str, ctx: dict):
        return getattr(self, task)(ctx)

    @staticmethod
    def _terms(text: str) -> list[str]:
        return [w for w in re.findall(r"[a-z]{4,}", text.lower())
                if w not in {"does", "what", "which", "with", "that", "from", "into", "have"}]

    def plan(self, ctx):
        q = ctx["question"]
        t = self._terms(q)
        return {"refined_question": q,
                "sub_questions": [f"What is known about {w}?" for w in t[:3]],
                "queries": [" ".join(t[:4]), " ".join(t[:2]), " ".join(t[2:5]) or q][:3],
                "key_terms": t[:6]}

    def screen(self, ctx):
        q = set(self._terms(ctx["question"]))
        out = []
        for p in ctx["papers"]:
            hits = len(q & set(self._terms(p["title"] + " " + p["abstract"])))
            out.append({"ref": p["ref"], "score": min(3, hits), "reason": f"{hits} key terms overlap"})
        return {"scores": out}

    def extract(self, ctx):
        claims = []
        for p in ctx["papers"]:
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p["abstract"]) if len(s.strip()) > 30]
            for s in sents[:2]:
                claims.append({"ref": p["ref"], "claim": s.rstrip("."), "kind": "finding",
                               "direction": "positive", "system": "unspecified",
                               "strength": "moderate", "quote": s})
            claims.append({"ref": p["ref"], "claim": "fabricated claim", "kind": "finding",
                           "direction": "n/a", "system": "", "strength": "weak",
                           "quote": "this sentence does not appear in the abstract"})
        return {"claims": claims}

    def synthesize(self, ctx):
        ids = [c["id"] for c in ctx["claims"]]
        return {"summary": f"Mock synthesis over {len(ids)} claims.",
                "themes": [{"name": "Main theme", "claim_ids": ids[:4], "summary": "Recurring findings."}],
                "consensus": [{"statement": "Several papers agree.", "claim_ids": ids[:2]}],
                "contradictions": [{"statement": "Two findings disagree.", "claim_ids": ids[2:4],
                                    "possible_explanation": "Different systems."}],
                "gaps": [{"gap": "No study combines the two main factors.",
                          "why_it_matters": "It would explain the contradiction.",
                          "related_claim_ids": ids[:3]}]}

    def hypothesize(self, ctx):
        ids = [c["id"] for c in ctx["claims"]]
        hyps = []
        for i in range(ctx.get("n", 5)):
            hyps.append({"statement": f"Mock hypothesis {i + 1} about {ctx['question'][:40]}",
                         "rationale": "Follows from the cited claims and the main gap.",
                         "supporting_claim_ids": ids[i:i + 2] + (["C999"] if i == 0 else []),
                         "gap_addressed": "No study combines the two main factors.",
                         "novelty": 3 + i % 3, "feasibility": 4, "testability": 5 - i % 2,
                         "prediction": "If true, X increases with Y.",
                         "falsification": "No association between X and Y.",
                         "experiment": {"type": "data_analysis", "description": "Regress X on Y.",
                                        "data_or_tools": "Public dataset"}})
        return {"hypotheses": hyps}

    def critique(self, ctx):
        reviews = []
        for i, h in enumerate(ctx["hypotheses"]):
            verdict = "drop" if i == len(ctx["hypotheses"]) - 1 else ("revise" if i == 1 else "keep")
            reviews.append({"id": h["id"], "verdict": verdict,
                            "issues": ["too vague"] if verdict != "keep" else [],
                            "revised_statement": f"{h['statement']} (sharpened)" if verdict == "revise" else None,
                            "already_established": verdict == "drop",
                            "prior_confidence": 0.3 + 0.1 * (i % 4),
                            "comment": "mock review"})
        return {"reviews": reviews}
