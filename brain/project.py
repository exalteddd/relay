"""A research project is a folder (and a git repo) that the agents commit to."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def slugify(text: str, max_words: int = 6) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    stop = {"the", "a", "an", "of", "in", "on", "and", "or", "to", "does", "do", "is", "are",
            "how", "what", "which", "why", "can", "for", "with", "by"}
    words = [w for w in words if w not in stop][:max_words] or ["research"]
    return "-".join(words)


class Project:
    def __init__(self, root: Path):
        self.root = root
        self.slug = root.name
        self._git = shutil.which("git") is not None

    @classmethod
    def create(cls, projects_dir: Path, question: str) -> "Project":
        projects_dir.mkdir(parents=True, exist_ok=True)
        base = f"{datetime.now():%Y%m%d}-{slugify(question)}"
        slug, n = base, 2
        while (projects_dir / slug).exists():
            slug, n = f"{base}-{n}", n + 1
        proj = cls(projects_dir / slug)
        for sub in ("literature", "hypotheses", "experiments"):
            (proj.root / sub).mkdir(parents=True, exist_ok=True)
        (proj.root / "experiments" / ".gitkeep").touch()
        proj.write_text("question.md", f"# Research question\n\n{question}\n")
        proj.write_text("notebook.md", f"# Lab notebook\n\n**Question:** {question}\n\n")
        if proj._git:
            proj._run("git", "init", "-q")
            proj._run("git", "config", "user.email", "brain@second-brain.local")
            proj._run("git", "config", "user.name", "Second Brain")
        return proj

    # ------------------------------------------------------------ files
    def path(self, rel: str) -> Path:
        return self.root / rel

    def write_text(self, rel: str, text: str) -> None:
        p = self.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def write_json(self, rel: str, obj) -> None:
        self.write_text(rel, json.dumps(obj, indent=2, ensure_ascii=False))

    def read_json(self, rel: str, default=None):
        p = self.path(rel)
        return json.loads(p.read_text()) if p.exists() else default

    def read_text(self, rel: str, default: str = "") -> str:
        p = self.path(rel)
        return p.read_text() if p.exists() else default

    def log(self, agent: str, message: str) -> None:
        with self.path("notebook.md").open("a") as f:
            f.write(f"- `{datetime.now():%H:%M:%S}` **{agent}**: {message}\n")

    # ------------------------------------------------------------ git
    def _run(self, *args: str) -> str:
        r = subprocess.run(args, cwd=self.root, capture_output=True, text=True)
        return r.stdout

    def commit(self, agent: str, message: str) -> None:
        """Each agent commits its own work, so the repo history is the research history."""
        if not self._git:
            return
        self._run("git", "add", "-A")
        self._run("git", "commit", "-q", "--allow-empty", "-m", f"{agent}: {message}",
                  "--author", f"{agent} <{agent}@second-brain.local>")

    def history(self) -> list[dict]:
        if not self._git:
            return []
        out = self._run("git", "log", "--pretty=format:%h\t%an\t%ar\t%s")
        rows = []
        for line in out.splitlines():
            parts = line.split("\t", 3)
            if len(parts) == 4:
                rows.append(dict(zip(("hash", "agent", "when", "message"), parts)))
        return rows
