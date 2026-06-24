"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline (search + synthesis)."""

    _init()
    from multi_agent_research_lab.core.schemas import AgentName, AgentResult
    from multi_agent_research_lab.services.llm_client import LLMClient
    from multi_agent_research_lab.services.search_client import SearchClient

    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    search_client = SearchClient()
    sources = search_client.search(query, max_results=request.max_sources)
    state.sources = sources

    sources_text = "\n\n".join(
        f"Source: {s.title}\nURL: {s.url}\nContent: {s.snippet}" for s in sources
    )

    llm_client = LLMClient()
    system_prompt = (
        f"You are a professional research assistant. Answer the user's query comprehensively.\n"
        f"Target Audience: {request.audience}.\n"
        f"Requirements:\n"
        f"1. Use inline citations [1], [2] to reference the provided search results.\n"
        f"2. Add a 'References' section at the end of your report.\n"
        f"3. Organize your answer clearly with markdown."
    )
    user_prompt = f"User Request: {query}\n\nSearch Results:\n{sources_text}"

    res = llm_client.complete(system_prompt, user_prompt)
    state.final_answer = res.content

    state.agent_results.append(
        AgentResult(
            agent=AgentName.SUPERVISOR,
            content=res.content,
            metadata={
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens,
                "cost_usd": res.cost_usd,
            },
        )
    )

    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline Answer"))
    console.print(state.model_dump_json(indent=2))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    console.print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
