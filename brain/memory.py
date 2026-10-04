"""The brain: one SQLite database (with full-text search) shared by all projects.

Every paper read, claim extracted and hypothesis formed is stored here, so a new
project starts by recalling what earlier projects already learned.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .sources import Paper

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    slug TEXT PRIMARY KEY, question TEXT, created_at TEXT, path TEXT
);
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY, title TEXT, abstract TEXT, year INTEGER, doi TEXT, url TEXT,
    venue TEXT, citations INTEGER, authors TEXT, first_seen TEXT
);
CREATE TABLE IF NOT EXISTS project_papers (
    project TEXT, paper_id TEXT, ref TEXT, relevance INTEGER, reason TEXT,
    PRIMARY KEY (project, paper_id)
);
CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY, project TEXT, paper_id TEXT, text TEXT, quote TEXT, kind TEXT,
    direction TEXT, system TEXT, strength TEXT
);
CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY, project TEXT, statement TEXT, rationale TEXT, status TEXT,
    confidence REAL, data TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(id UNINDEXED, title, abstract);
CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(id UNINDEXED, project UNINDEXED, text);
CREATE VIRTUAL TABLE IF NOT EXISTS hyp_fts USING fts5(id UNINDEXED, project UNINDEXED, statement, rationale);
"""


def _fts_query(text: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9]{3,}", text.lower())
             if w not in {"the", "and", "for", "with", "that", "does", "how", "what", "which", "are"}]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words)) or '""'


class Brain:
    def __init__(self, db_path: Path):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # ------------------------------------------------------------ writes
    def add_project(self, slug: str, question: str, path: Path) -> None:
        self.db.execute("INSERT OR REPLACE INTO projects VALUES (?,?,?,?)",
                        (slug, question, datetime.now(timezone.utc).isoformat(), str(path)))
        self.db.commit()

    def upsert_papers(self, project: str, papers: list[dict]) -> int:
        """papers: dicts with Paper fields plus ref/relevance/reason. Returns # new to the brain."""
        new = 0
        for p in papers:
            exists = self.db.execute("SELECT 1 FROM papers WHERE id=?", (p["id"],)).fetchone()
            if not exists:
                new += 1
                self.db.execute(
                    "INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (p["id"], p["title"], p.get("abstract", ""), p.get("year"), p.get("doi", ""),
                     p.get("url", ""), p.get("venue", ""), p.get("citations", 0),
                     json.dumps(p.get("authors", [])), project))
                self.db.execute("INSERT INTO papers_fts VALUES (?,?,?)",
                                (p["id"], p["title"], p.get("abstract", "")))
            self.db.execute("INSERT OR REPLACE INTO project_papers VALUES (?,?,?,?,?)",
                            (project, p["id"], p.get("ref"), p.get("relevance"), p.get("reason")))
        self.db.commit()
        return new

    def add_claims(self, project: str, claims: list[dict]) -> None:
        for c in claims:
            cid = f"{project}:{c['id']}"
            self.db.execute("INSERT OR REPLACE INTO claims VALUES (?,?,?,?,?,?,?,?,?)",
                            (cid, project, c["paper_id"], c["claim"], c.get("quote", ""),
                             c.get("kind", ""), c.get("direction", ""), c.get("system", ""),
                             c.get("strength", "")))
            self.db.execute("DELETE FROM claims_fts WHERE id=?", (cid,))
            self.db.execute("INSERT INTO claims_fts VALUES (?,?,?)", (cid, project, c["claim"]))
        self.db.commit()

    def add_hypotheses(self, project: str, hyps: list[dict]) -> None:
        for h in hyps:
            hid = f"{project}:{h['id']}"
            self.db.execute("INSERT OR REPLACE INTO hypotheses VALUES (?,?,?,?,?,?,?)",
                            (hid, project, h["statement"], h.get("rationale", ""), h.get("status", ""),
                             h.get("confidence"), json.dumps(h)))
            self.db.execute("DELETE FROM hyp_fts WHERE id=?", (hid,))
            self.db.execute("INSERT INTO hyp_fts VALUES (?,?,?,?)",
                            (hid, project, h["statement"], h.get("rationale", "")))
        self.db.commit()

    # ------------------------------------------------------------ reads
    def known_paper_ids(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT id FROM papers")}

    def projects(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM projects ORDER BY created_at DESC")]

    def stats(self) -> dict:
        q = lambda t: self.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: E731
        return {"projects": q("projects"), "papers": q("papers"), "claims": q("claims"),
                "hypotheses": q("hypotheses")}

    def recall(self, text: str, exclude_project: str | None = None, k: int = 8) -> dict:
        """Full-text recall across everything the brain has learned."""
        fq = _fts_query(text)
        ex = exclude_project or ""
        papers = self.db.execute(
            """SELECT p.id, p.title, p.year, p.url, p.first_seen, bm25(papers_fts) AS score
               FROM papers_fts JOIN papers p ON p.id = papers_fts.id
               WHERE papers_fts MATCH ? ORDER BY score LIMIT ?""", (fq, k)).fetchall()
        claims = self.db.execute(
            """SELECT c.id, c.project, c.text, c.strength, p.title AS paper_title, p.url,
                      bm25(claims_fts) AS score
               FROM claims_fts JOIN claims c ON c.id = claims_fts.id
               JOIN papers p ON p.id = c.paper_id
               WHERE claims_fts MATCH ? AND c.project != ? ORDER BY score LIMIT ?""",
            (fq, ex, k)).fetchall()
        hyps = self.db.execute(
            """SELECT h.id, h.project, h.statement, h.status, h.confidence, bm25(hyp_fts) AS score
               FROM hyp_fts JOIN hypotheses h ON h.id = hyp_fts.id
               WHERE hyp_fts MATCH ? AND h.project != ? ORDER BY score LIMIT ?""",
            (fq, ex, k)).fetchall()
        return {"papers": [dict(r) for r in papers], "claims": [dict(r) for r in claims],
                "hypotheses": [dict(r) for r in hyps]}

    def get_paper(self, paper_id: str) -> Paper | None:
        r = self.db.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
        if not r:
            return None
        return Paper(id=r["id"], title=r["title"], abstract=r["abstract"], year=r["year"],
                     doi=r["doi"], url=r["url"], venue=r["venue"], citations=r["citations"],
                     authors=json.loads(r["authors"] or "[]"), sources=["brain"])
