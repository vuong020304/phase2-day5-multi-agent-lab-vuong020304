"""Optional critic agent skeleton for bonus work."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient



class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self) -> None:
        self.llm_client = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""
        query = state.request.query
        final_answer = state.final_answer or "No final answer drafted."

        sources_list = []
        for idx, src in enumerate(state.sources):
            sources_list.append(f"Source [{idx + 1}] Title: {src.title}\nSnippet: {src.snippet}")
        sources_text = "\n\n".join(sources_list) if sources_list else "No sources available."

        system_prompt = (
            "You are a strict, objective fact-checking Critic Agent. Your task is to evaluate the final report draft.\n"
            "Evaluate based on these rules:\n"
            "1. Check if all factual claims are supported by the provided search sources. Flag any hallucinations.\n"
            "2. Ensure citation coverage is high (every major claim has a reference link/citation like [1], [2]).\n"
            "3. If the report draft is high quality, accurate, and needs no major changes, write a review stating that it is APPROVED.\n"
            "4. If revisions are required, clearly detail what needs to be changed, added, or verified. Be specific."
        )
        user_prompt = (
            f"Original Query: {query}\n\n"
            f"Final Answer Draft:\n{final_answer}\n\n"
            f"Ground Truth Sources:\n{sources_text}"
        )

        res = self.llm_client.complete(system_prompt, user_prompt)

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=res.content,
                metadata={
                    "input_tokens": res.input_tokens,
                    "output_tokens": res.output_tokens,
                    "cost_usd": res.cost_usd,
                }
            )
        )
        state.add_trace_event("critic_completed", {})
        return state

