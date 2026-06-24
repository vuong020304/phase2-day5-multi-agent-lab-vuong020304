"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self.llm_client = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        query = state.request.query
        research_notes = state.research_notes or "No research notes available."

        system_prompt = (
            "You are a critical analytical agent. Your task is to analyze the provided research notes. "
            "Specifically:\n"
            "1. Extract the main claims made in the research.\n"
            "2. Identify different or contrasting perspectives, if any.\n"
            "3. Assess the strength and credibility of the evidence (point out if any sources are weak, outdated, or biased).\n"
            "4. Structure your insights clearly using bullet points and headers."
        )
        user_prompt = f"User Query: {query}\n\nResearch Notes:\n{research_notes}"

        res = self.llm_client.complete(system_prompt, user_prompt)
        state.analysis_notes = res.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=res.content,
                metadata={
                    "input_tokens": res.input_tokens,
                    "output_tokens": res.output_tokens,
                    "cost_usd": res.cost_usd,
                },
            )
        )
        state.add_trace_event("analysis_completed", {})
        return state
