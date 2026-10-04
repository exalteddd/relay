"""Literature sources: OpenAlex, Semantic Scholar, arXiv and Europe PMC.

Each source has a `search_*` function (network) and a `parse_*` function (pure,
testable). `search_all` fans out over queries x sources in parallel, then
deduplicates and merges records that refer to the same paper.
"""

from __future__ import annotations

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field

import requests

from .config import Config

USER_AGENT = "second-brain-research/0.1 (hackathon prototype)"
TIMEOUT = 30


@dataclass
class Paper:
    id: str
    title: str
    abstract: str = ""
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    doi: str = ""
    url: str = ""
    citations: int = 0
    sources: list[str] = field(default_factory=list)
    external_ids: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ---------------------------------------------------------------- helpers

def _clean(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)          # strip JATS/HTML tags
    return re.sub(r"\s+", " ", text).strip()


def _norm_doi(doi: str | None) -> str:
    if not doi:
        return ""
    doi = doi.strip().lower()
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi)


def norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def make_id(doi: str = "", arxiv: str = "", s2: str = "", pmid: str = "",
            openalex: str = "", title: str = "") -> str:
    if doi:
        return f"doi:{doi}"
    if arxiv:
        return f"arxiv:{arxiv}"
    if pmid:
        return f"pmid:{pmid}"
    if s2:
        return f"s2:{s2}"
    if openalex:
        return f"openalex:{openalex}"
    return "title:" + hashlib.sha1(norm_title(title).encode()).hexdigest()[:12]


class RateLimited(RuntimeError):
    """The source asked us to slow down. Worth telling the caller apart from
    a transient failure, because the answer is to stop asking, not to wait."""


def _get(url: str, params: dict | None = None, headers: dict | None = None,
         retries: int = 3) -> requests.Response:
    h = {"User-Agent": USER_AGENT, **(headers or {})}
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=h, timeout=TIMEOUT)
            # A 429 is not a blip: backing off 1.5s, 3s then 6s and failing
            # anyway spent ten seconds per request to learn what the first
            # reply already said. Give up immediately and let the caller
            # stop using this source.
            if r.status_code == 429:
                raise RateLimited(f"429 from {url}")
            if r.status_code in (500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} from {url}", response=r)
            r.raise_for_status()
            return r
        except RateLimited:
            raise
        except requests.RequestException as e:
            last = e
            if attempt < retries - 1:
                time.sleep(1.0 * (2 ** attempt))
    raise RuntimeError(f"request failed after {retries} attempts: {last}")


# ---------------------------------------------------------------- OpenAlex

def _openalex_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    positions = [(pos, word) for word, poss in inv.items() for pos in poss]
    return " ".join(w for _, w in sorted(positions))


def parse_openalex(data: dict) -> list[Paper]:
    out = []
    for w in data.get("results", []):
        title = _clean(w.get("title") or w.get("display_name"))
        if not title:
            continue
        doi = _norm_doi(w.get("doi"))
        oa_id = (w.get("id") or "").rsplit("/", 1)[-1]
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        out.append(Paper(
            id=make_id(doi=doi, openalex=oa_id, title=title),
            title=title,
            abstract=_clean(_openalex_abstract(w.get("abstract_inverted_index"))),
            year=w.get("publication_year"),
            authors=[a["author"]["display_name"] for a in w.get("authorships", [])
                     if a.get("author", {}).get("display_name")][:8],
            venue=src.get("display_name") or "",
            doi=doi,
            url=(f"https://doi.org/{doi}" if doi else loc.get("landing_page_url") or w.get("id", "")),
            citations=w.get("cited_by_count") or 0,
            sources=["openalex"],
            external_ids={"openalex": oa_id},
        ))
    return out


def search_openalex(query: str, limit: int, cfg: Config) -> list[Paper]:
    params = {"search": query, "per-page": min(limit, 50),
              "filter": "has_abstract:true,type:article|preprint|review"}
    if cfg.openalex_email:
        params["mailto"] = cfg.openalex_email
    if cfg.openalex_api_key:
        params["api_key"] = cfg.openalex_api_key
    return parse_openalex(_get("https://api.openalex.org/works", params).json())


# ---------------------------------------------------------------- Semantic Scholar

S2_FIELDS = "title,abstract,year,authors,venue,citationCount,externalIds,url"


def parse_s2(data: dict) -> list[Paper]:
    out = []
    for p in data.get("data", []) or []:
        title = _clean(p.get("title"))
        if not title:
            continue
        ext = p.get("externalIds") or {}
        doi = _norm_doi(ext.get("DOI"))
        arxiv = ext.get("ArXiv") or ""
        pmid = str(ext.get("PubMed") or "")
        out.append(Paper(
            id=make_id(doi=doi, arxiv=arxiv, pmid=pmid, s2=p.get("paperId", ""), title=title),
            title=title,
            abstract=_clean(p.get("abstract")),
            year=p.get("year"),
            authors=[a["name"] for a in p.get("authors", []) if a.get("name")][:8],
            venue=p.get("venue") or "",
            doi=doi,
            url=p.get("url") or "",
            citations=p.get("citationCount") or 0,
            sources=["semantic_scholar"],
            external_ids={k: str(v) for k, v in {"s2": p.get("paperId"), "arxiv": arxiv,
                                                 "pmid": pmid}.items() if v},
        ))
    return out


def search_s2(query: str, limit: int, cfg: Config) -> list[Paper]:
    headers = {"x-api-key": cfg.s2_api_key} if cfg.s2_api_key else {}
    r = _get("https://api.semanticscholar.org/graph/v1/paper/search",
             {"query": query, "limit": min(limit, 100), "fields": S2_FIELDS}, headers)
    return parse_s2(r.json())


# ---------------------------------------------------------------- arXiv

_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def parse_arxiv(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", _ATOM):
        title = _clean(e.findtext("a:title", "", _ATOM))
        if not title:
            continue
        raw_id = e.findtext("a:id", "", _ATOM)
        arxiv_id = re.sub(r"v\d+$", "", raw_id.rsplit("/abs/", 1)[-1])
        doi = _norm_doi(e.findtext("arxiv:doi", "", _ATOM))
        published = e.findtext("a:published", "", _ATOM)
        out.append(Paper(
            id=make_id(doi=doi, arxiv=arxiv_id, title=title),
            title=title,
            abstract=_clean(e.findtext("a:summary", "", _ATOM)),
            year=int(published[:4]) if published[:4].isdigit() else None,
            authors=[_clean(a.findtext("a:name", "", _ATOM)) for a in e.findall("a:author", _ATOM)][:8],
            venue="arXiv",
            doi=doi,
            url=f"https://arxiv.org/abs/{arxiv_id}",
            sources=["arxiv"],
            external_ids={"arxiv": arxiv_id},
        ))
    return out


def search_arxiv(query: str, limit: int, cfg: Config) -> list[Paper]:
    terms = " AND ".join(f"all:{t}" for t in re.findall(r"[A-Za-z0-9\-]+", query)[:8])
    r = _get("https://export.arxiv.org/api/query",
             {"search_query": terms or query, "max_results": min(limit, 50),
              "sortBy": "relevance"})
    return parse_arxiv(r.text)


# ---------------------------------------------------------------- Europe PMC

def parse_europepmc(data: dict) -> list[Paper]:
    out = []
    for p in (data.get("resultList") or {}).get("result", []):
        title = _clean(p.get("title"))
        if not title:
            continue
        doi = _norm_doi(p.get("doi"))
        pmid = p.get("pmid") or ""
        year = p.get("pubYear")
        out.append(Paper(
            id=make_id(doi=doi, pmid=pmid, title=title),
            title=title.rstrip("."),
            abstract=_clean(p.get("abstractText")),
            year=int(year) if year and str(year).isdigit() else None,
            authors=[a.strip() for a in (p.get("authorString") or "").rstrip(".").split(",") if a.strip()][:8],
            venue=p.get("journalTitle") or "",
            doi=doi,
            url=(f"https://doi.org/{doi}" if doi else
                 f"https://europepmc.org/article/{p.get('source', 'MED')}/{p.get('id', '')}"),
            citations=p.get("citedByCount") or 0,
            sources=["europepmc"],
            external_ids={k: v for k, v in {"pmid": pmid, "pmcid": p.get("pmcid", "")}.items() if v},
        ))
    return out


def search_europepmc(query: str, limit: int, cfg: Config) -> list[Paper]:
    r = _get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
             {"query": f"({query}) AND HAS_ABSTRACT:Y", "format": "json",
              "resultType": "core", "pageSize": min(limit, 100)})
    return parse_europepmc(r.json())


# ---------------------------------------------------------------- fan-out + dedupe

SOURCES = {
    "openalex": search_openalex,
    "semantic_scholar": search_s2,
    "arxiv": search_arxiv,
    "europepmc": search_europepmc,
}


def merge(papers: list[Paper]) -> list[Paper]:
    """Deduplicate by DOI, external id or normalized title; merge fields."""
    merged: list[Paper] = []
    index: dict[str, Paper] = {}
    for p in papers:
        keys = [f"doi:{p.doi}" if p.doi else "", f"t:{norm_title(p.title)}"]
        keys += [f"{k}:{v}" for k, v in p.external_ids.items() if k in ("arxiv", "pmid")]
        keys = [k for k in keys if k and k != "t:"]
        existing = next((index[k] for k in keys if k in index), None)
        if existing is None:
            merged.append(p)
            existing = p
        else:
            if len(p.abstract) > len(existing.abstract):
                existing.abstract = p.abstract
            if not existing.doi and p.doi:
                existing.doi = p.doi
                existing.id = p.id if p.id.startswith("doi:") else existing.id
            existing.year = existing.year or p.year
            existing.venue = existing.venue or p.venue
            existing.url = existing.url or p.url
            existing.authors = existing.authors or p.authors
            existing.citations = max(existing.citations, p.citations)
            existing.sources = sorted(set(existing.sources) | set(p.sources))
            existing.external_ids = {**p.external_ids, **existing.external_ids}
        for k in keys:
            index[k] = existing
    return merged


def search_all(queries: list[str], cfg: Config, sources: list[str] | None = None,
               per_query: int = 25, on_event=None) -> tuple[list[Paper], list[str]]:
    sources = sources or list(SOURCES)
    jobs: dict = {}
    results: list[Paper] = []
    errors: list[str] = []
    limited: set[str] = set()      # sources that told us to stop

    def fetch(src: str, q: str):
        # Once a source rate-limits, every later query to it will too. Asking
        # anyway is how a search that should take seconds takes a minute.
        if src in limited:
            raise RateLimited("skipped: already rate-limited this run")
        try:
            return SOURCES[src](q, per_query, cfg)
        except RateLimited:
            limited.add(src)
            raise

    with ThreadPoolExecutor(max_workers=8) as pool:
        for q in queries:
            for s in sources:
                jobs[pool.submit(fetch, s, q)] = (s, q)
        for fut in as_completed(jobs):
            s, q = jobs[fut]
            try:
                got = fut.result()
                results.extend(got)
                if on_event:
                    on_event("literature", f"{s}: {len(got)} results for '{q}'")
            except Exception as e:  # one failing source should not sink the run
                errors.append(f"{s} '{q}': {e}")
                if on_event:
                    on_event("literature", f"{s} failed for '{q}' ({str(e)[:80]})")
    if limited:
        errors.append("rate-limited, so skipped after the first refusal: "
                      + ", ".join(sorted(limited))
                      + " (set an API key to use it)")
    return merge(results), errors
