"""Benchmark skeleton for single-agent vs multi-agent."""

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, estimate costs, evaluate quality, and generate metrics."""
    from multi_agent_research_lab.services.llm_client import LLMClient

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    # Calculate total cost
    total_cost = 0.0
    for result in state.agent_results:
        cost = result.metadata.get("cost_usd")
        if cost is not None:
            total_cost += cost

    # LLM-as-a-judge for quality scoring
    llm_client = LLMClient()
    system_prompt = (
        "You are an objective evaluation judge. Rate the quality of the research report on a scale of 0 to 10.\n"
        "Criteria:\n"
        "- Completeness: Does it answer the query?\n"
        "- Citations: Does it cite sources correctly?\n"
        "- Structure: Is it organized and clear?\n"
        "Output ONLY a decimal number between 0.0 and 10.0, nothing else."
    )
    user_prompt = f"Query: {query}\n\nReport Draft:\n{state.final_answer or 'No answer'}"

    try:
        judge_res = llm_client.complete(system_prompt, user_prompt)
        quality_score = float(judge_res.content.strip())
        quality_score = max(0.0, min(10.0, quality_score))
    except Exception:
        quality_score = None

    # Summarize stats in notes
    num_citations = len(state.sources)
    num_steps = len(state.route_history)
    notes = f"Sources found: {num_citations} | Handoff steps: {num_steps}"

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=total_cost if total_cost > 0 else None,
        quality_score=quality_score,
        notes=notes,
    )
    return state, metrics
