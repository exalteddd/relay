"""Tools for the literature agent."""

from omnigent_client.tools import tool

from brain import lab_tools as lab


@tool
def search_literature(project: str, queries: list[str], per_query: int) -> dict:
    """Search OpenAlex, Semantic Scholar, arXiv and Europe PMC in parallel, deduplicate, rank with BM25,
    and add new candidates (with refs like P001) to the project. Safe to call again with new queries.

    Args:
        project: Project slug.
        queries: Short keyword queries.
        per_query: Results per query per source, e.g. 20.
    """
    return lab.search_literature(project, queries, None, per_query)


@tool
def get_papers(project: str, refs: list[str]) -> dict:
    """Get full abstracts and metadata for candidate papers.

    Args:
        project: Project slug.
        refs: Paper refs such as ["P001", "P014"].
    """
    return lab.get_papers(project, refs)


@tool(strict=False)
def save_screening(project: str, decisions: list[dict], max_papers: int = 25) -> dict:
    """Save relevance scores and select the papers to extract evidence from (score >= 2 kept).

    Args:
        project: Project slug.
        decisions: One object per screened paper: {"ref": "P001", "score": 0-3, "reason": "..."}.
        max_papers: Maximum papers to keep.
    """
    return lab.save_screening(project, decisions, max_papers)


@tool
def recall_memory(query: str, k: int) -> dict:
    """Search what the lab already knows from earlier projects.

    Args:
        query: What to look for.
        k: Maximum results per type.
    """
    return lab.recall_memory(query, k)
