"""Tools for the synthesis + hypothesis agent."""

from omnigent_client.tools import tool

from brain import lab_tools as lab


@tool
def list_claims(project: str) -> dict:
    """List all grounded claims (IDs like C001) with strength, system and direction.

    Args:
        project: Project slug.
    """
    return lab.list_claims(project)


@tool
def recall_memory(query: str, k: int) -> dict:
    """Search prior hypotheses and findings from earlier projects, so you don't repeat rejected ideas.

    Args:
        query: What to look for.
        k: Maximum results per type.
    """
    return lab.recall_memory(query, k)


@tool(strict=False)
def save_evidence_map(project: str, evidence_map: dict) -> dict:
    """Save the evidence map. Unknown claim IDs are removed automatically.

    Args:
        project: Project slug.
        evidence_map: {"summary", "themes": [{name, claim_ids, summary}], "consensus": [{statement, claim_ids}],
            "contradictions": [{statement, claim_ids, possible_explanation}],
            "gaps": [{gap, why_it_matters, related_claim_ids}]}.
    """
    return lab.save_evidence_map(project, evidence_map)


@tool(strict=False)
def save_hypotheses(project: str, hypotheses: list[dict]) -> dict:
    """Save candidate hypotheses. Citations are validated against saved claims.

    Args:
        project: Project slug.
        hypotheses: Objects {"statement", "rationale", "supporting_claim_ids", "gap_addressed", "novelty",
            "feasibility", "testability", "prediction", "falsification",
            "experiment": {"type", "description", "data_or_tools"}}.
    """
    return lab.save_hypotheses(project, hypotheses)
