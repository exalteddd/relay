"""Tools for the principal-investigator (orchestrator) agent."""

from omnigent_client.tools import tool

from brain import lab_tools as lab


@tool
def start_project(question: str) -> dict:
    """Create a new research project (a git repo) and recall related knowledge from long-term memory.

    Args:
        question: The research question exactly as the scientist stated it.
    """
    return lab.start_project(question)


@tool
def recall_memory(query: str, k: int) -> dict:
    """Search everything the lab has learned in earlier projects (papers, claims, hypotheses).

    Args:
        query: What to look for.
        k: Maximum results per type, e.g. 8.
    """
    return lab.recall_memory(query, k)


@tool
def save_plan(project: str, refined_question: str, sub_questions: list[str], queries: list[str]) -> dict:
    """Record the research plan. A human scientist must approve this before literature search starts.

    Args:
        project: Project slug from start_project.
        refined_question: Precise, answerable version of the question.
        sub_questions: 3-5 sub-questions that together answer it.
        queries: 6-8 short keyword search queries (2-6 words each).
    """
    return lab.save_plan(project, refined_question, sub_questions, queries)


@tool
def project_status(project: str) -> dict:
    """Show the shared research record: plan, counts, hypotheses and the agent commit history.

    Args:
        project: Project slug.
    """
    return lab.project_status(project)


@tool
def log_decision(project: str, agent: str, decision: str) -> dict:
    """Write a decision and its reason to the lab notebook and commit it to the research record.

    Args:
        project: Project slug.
        agent: Who made the decision, e.g. "principal-investigator".
        decision: The decision and why, in one or two sentences.
    """
    return lab.log_decision(project, agent, decision)
