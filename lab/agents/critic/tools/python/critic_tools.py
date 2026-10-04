"""Tools for the critic agent."""

from omnigent_client.tools import tool

from brain import lab_tools as lab


@tool
def get_hypotheses_for_review(project: str) -> dict:
    """Get each hypothesis with the exact claims and verbatim quotes it cites.

    Args:
        project: Project slug.
    """
    return lab.get_hypotheses_for_review(project)


@tool
def recall_memory(query: str, k: int) -> dict:
    """Check whether the lab rejected or tested similar hypotheses before.

    Args:
        query: What to look for.
        k: Maximum results per type.
    """
    return lab.recall_memory(query, k)


@tool(strict=False)
def save_reviews(project: str, reviews: list[dict]) -> dict:
    """Apply your verdicts, rank hypotheses, write the report and store everything in long-term memory.

    Args:
        project: Project slug.
        reviews: Objects {"id": "H01", "verdict": "keep|revise|drop", "issues": [...],
            "revised_statement": str or null, "already_established": bool,
            "prior_confidence": 0-1, "comment": "..."}.
    """
    return lab.save_reviews(project, reviews)
