import os
import sys

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Ensure UTF-8 mode on Windows
os.environ["PYTHONIOENCODING"] = "utf-8"

from multi_agent_research_lab.core.schemas import ResearchQuery, AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report


def baseline_runner(query: str) -> ResearchState:
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    search_client = SearchClient()
    sources = search_client.search(query, max_results=request.max_sources)
    state.sources = sources

    sources_text = "\n\n".join(
        f"Source: {s.title}\nURL: {s.url}\nContent: {s.snippet}"
        for s in sources
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
            }
        )
    )
    return state


def multi_agent_runner(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


def main() -> None:
    query = "Research GraphRAG state-of-the-art and write a 500-word summary"

    print("Running Single-Agent Baseline...")
    state_bl, metrics_bl = run_benchmark("Single-Agent Baseline", query, baseline_runner)
    print(f"Baseline finished in {metrics_bl.latency_seconds:.2f}s (Quality Score: {metrics_bl.quality_score})")

    print("\nRunning Multi-Agent Workflow...")
    state_ma, metrics_ma = run_benchmark("Multi-Agent Workflow", query, multi_agent_runner)
    print(f"Multi-agent finished in {metrics_ma.latency_seconds:.2f}s (Quality Score: {metrics_ma.quality_score})")

    # Generate Markdown Report
    report_content = render_markdown_report([metrics_bl, metrics_ma])
    
    # Save Report
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "benchmark_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print("\n" + "=" * 50)
    print(report_content)
    print("=" * 50)
    print(f"Benchmark Report successfully written to: {report_path}")


if __name__ == "__main__":
    main()
