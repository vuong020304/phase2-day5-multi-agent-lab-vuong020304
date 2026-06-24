"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient



class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self.llm_client = LLMClient()
        self.search_client = SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        query = state.request.query
        max_sources = state.request.max_sources

        # 1. Generate optimized search query
        critic_feedback = ""
        critic_results = [r for r in state.agent_results if r.agent == AgentName.CRITIC]
        if critic_results:
            critic_feedback = f"\nCritic feedback to address: {critic_results[-1].content}"

        system_prompt = (
            "You are an expert information retriever. Generate a single highly relevant search query "
            "to find source documents for the user's request. Output ONLY the query, no quotes, no extra text."
        )
        user_prompt = f"User Request: {query}{critic_feedback}"

        try:
            llm_res = self.llm_client.complete(system_prompt, user_prompt)
            search_query = llm_res.content.strip().strip('"').strip("'")
            input_tokens = llm_res.input_tokens or 0
            output_tokens = llm_res.output_tokens or 0
            cost_usd = llm_res.cost_usd or 0.0
        except Exception:
            search_query = query
            llm_res = None
            input_tokens = 0
            output_tokens = 0
            cost_usd = 0.0

        # 2. Call search client
        sources = self.search_client.search(search_query, max_results=max_sources)

        # 3. Deduplicate and record sources
        existing_urls = {s.url for s in state.sources if s.url}
        for src in sources:
            if src.url not in existing_urls:
                state.sources.append(src)

        # 4. Synthesize research notes
        sources_text = "\n\n".join(
            f"Source: {s.title}\nURL: {s.url}\nContent: {s.snippet}"
            for s in state.sources
        )

        system_prompt_notes = (
            "You are a meticulous researcher. Synthesize the provided search sources into concise, factual research notes. "
            "Organize them logically, highlight key statistics/definitions/findings, and preserve source citations. "
            "Keep the notes factual, objective, and dense with information."
        )
        user_prompt_notes = (
            f"User Request: {query}\n\n"
            f"Search Results:\n{sources_text}"
        )

        notes_res = self.llm_client.complete(system_prompt_notes, user_prompt_notes)
        state.research_notes = notes_res.content

        input_tokens += notes_res.input_tokens or 0
        output_tokens += notes_res.output_tokens or 0
        cost_usd += notes_res.cost_usd or 0.0

        # Record results in agent_results
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=notes_res.content,
                metadata={
                    "search_query": search_query,
                    "num_sources_found": len(sources),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost_usd,
                }
            )
        )
        state.add_trace_event("research_completed", {"query": search_query, "sources_count": len(state.sources)})
        return state

