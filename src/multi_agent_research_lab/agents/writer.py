"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient



class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self.llm_client = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        query = state.request.query
        audience = state.request.audience
        research_notes = state.research_notes or "No research notes available."
        analysis_notes = state.analysis_notes or "No analysis notes available."

        sources_list = []
        for idx, src in enumerate(state.sources):
            sources_list.append(f"[{idx + 1}] Title: {src.title} | URL: {src.url}")
        sources_text = "\n".join(sources_list) if sources_list else "No sources referenced."

        system_prompt = (
            "You are a professional research writer. Your task is to write a cohesive, comprehensive, "
            f"and polished report addressing the user's query. Target Audience: {audience}.\n"
            "Requirements:\n"
            "1. Synthesize research notes and structured insights cleanly.\n"
            "2. Incorporate inline numbered citations matching the source documents (e.g., [1], [2]).\n"
            "3. Add a 'References' section at the end listing all cited sources with their URLs.\n"
            "4. Organize with clean, readable markdown headers, bold text, and lists."
        )
        user_prompt = (
            f"User Query: {query}\n\n"
            f"Research Notes:\n{research_notes}\n\n"
            f"Analysis Notes:\n{analysis_notes}\n\n"
            f"Source Documents Reference List:\n{sources_text}"
        )

        res = self.llm_client.complete(system_prompt, user_prompt)
        state.final_answer = res.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=res.content,
                metadata={
                    "input_tokens": res.input_tokens,
                    "output_tokens": res.output_tokens,
                    "cost_usd": res.cost_usd,
                }
            )
        )
        state.add_trace_event("writing_completed", {})
        return state

