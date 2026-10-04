"""Cheap lexical prefilter (BM25) so the LLM only screens plausible papers."""

from __future__ import annotations

import math
import re
from collections import Counter

from .sources import Paper

STOP = set("""a an and are as at be by for from has have in into is it its of on or that the
their this to was were which with we our these those than then there via using use used based
between can may not no also more most both each other such study studies paper results show""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\-]+", text.lower()) if t not in STOP]


def bm25_rank(papers: list[Paper], query: str, k1: float = 1.5, b: float = 0.75) -> list[tuple[Paper, float]]:
    docs = [tokenize(f"{p.title} {p.title} {p.abstract}") for p in papers]  # title counts double
    if not docs:
        return []
    n = len(docs)
    avgdl = sum(map(len, docs)) / n or 1
    df = Counter(t for d in docs for t in set(d))
    q_terms = set(tokenize(query))
    scored = []
    for p, d in zip(papers, docs):
        tf = Counter(d)
        s = 0.0
        for t in q_terms:
            if t not in tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        s *= 1 + 0.05 * math.log1p(p.citations)   # gentle nudge toward well-cited work
        scored.append((p, s))
    return sorted(scored, key=lambda x: x[1], reverse=True)
