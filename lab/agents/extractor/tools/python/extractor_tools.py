"""Tools for the evidence extraction agent."""

from omnigent_client.tools import tool

from brain import lab_tools as lab


@tool
def get_papers(project: str, refs: list[str]) -> dict:
    """Get full abstracts for selected papers.

    Args:
        project: Project slug.
        refs: Paper refs such as ["P001", "P014"].
    """
    return lab.get_papers(project, refs)


@tool
def project_status(project: str) -> dict:
    """Show the project record, including which papers were selected.

    Args:
        project: Project slug.
    """
    return lab.project_status(project)


@tool(strict=False)
def save_claims(project: str, claims: list[dict]) -> dict:
    """Save evidence claims. Each claim's quote is checked against the abstract; non-verbatim quotes are
    rejected and returned so you can fix them.

    Args:
        project: Project slug.
        claims: Objects {"ref", "claim", "kind", "direction", "system", "strength", "quote"}.
    """
    return lab.save_claims(project, claims)


@tool
def list_claims(project: str) -> dict:
    """List all grounded claims saved so far (with IDs like C001).

    Args:
        project: Project slug.
    """
    return lab.list_claims(project)
