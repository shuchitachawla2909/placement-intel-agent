"""Expose the agent's tools (and the agent itself) as an MCP server over stdio.

  pip install mcp
  python mcp_server.py

Then add it to an MCP client as a stdio server: command "python", args ["<path>/mcp_server.py"].
"""
try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP

from placement_agent import tools as T
from placement_agent.graph import PlacementAgent

mcp = FastMCP("placement-intel")
_agent = None


@mcp.tool()
def query_placements(company: str | None = None, role_category: str | None = None, min_ctc_lpa: float | None = None,
                     drive_day: int | None = None, location: str | None = None, sort_by: str = "ctc",
                     limit: int = 15) -> dict:
    """Filter/rank 2025-26 placement openings by company, role category, minimum CTC (LPA), drive day or city."""
    return T.query_openings(company=company, role_category=role_category, min_ctc_lpa=min_ctc_lpa,
                            drive_day=drive_day, location=location, sort_by=sort_by, limit=limit)


@mcp.tool()
def company_profile(company: str) -> dict:
    """Roles, CTC per course, eligibility, deadline and interview slot for one company."""
    return T.company_profile(company)


@mcp.tool()
def eligibility(cgpa: float, branch: str, course: str = "btech") -> dict:
    """Openings a student with this CGPA/branch/course is eligible for, grouped by interview day."""
    return T.eligible_openings(cgpa=cgpa, branch=branch, course=course)


@mcp.tool()
def drive_statistics(day: int | None = None) -> dict:
    """Offers/profiles per drive phase and overall highest/average CTC."""
    return T.drive_statistics(day=day)


@mcp.tool()
def ask_agent(question: str, thread_id: str = "mcp") -> dict:
    """Ask the full LangGraph agent (routing + tools + cited answer)."""
    global _agent
    _agent = _agent or PlacementAgent()
    return _agent.ask(question, thread_id=thread_id)


if __name__ == "__main__":
    mcp.run()
