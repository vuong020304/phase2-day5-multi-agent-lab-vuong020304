"""Supervisor / router skeleton."""

import json

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self) -> None:
        self.llm_client = LLMClient()
        self.settings = get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""
        # 1. Enforce guardrails (max iterations)
        if state.iteration >= self.settings.max_iterations:
            if not state.final_answer and (state.research_notes or state.analysis_notes):
                next_route = "writer"
            else:
                next_route = "__end__"
            state.record_route(next_route)
            state.add_trace_event(
                "supervisor_decision", {"route": next_route, "reason": "Max iterations reached"}
            )
            return state

        # 2. Programmatic routing for initial phases
        if not state.research_notes:
            next_route = "researcher"
            state.record_route(next_route)
            state.add_trace_event(
                "supervisor_decision", {"route": next_route, "reason": "No research notes found"}
            )
            return state

        if not state.analysis_notes:
            next_route = "analyst"
            state.record_route(next_route)
            state.add_trace_event(
                "supervisor_decision", {"route": next_route, "reason": "No analysis notes found"}
            )
            return state

        if not state.final_answer:
            next_route = "writer"
            state.record_route(next_route)
            state.add_trace_event(
                "supervisor_decision", {"route": next_route, "reason": "No final answer found"}
            )
            return state

        # 3. Dynamic routing after draft creation
        last_agent = state.route_history[-1] if state.route_history else None

        if last_agent == "writer":
            next_route = "critic"
            state.record_route(next_route)
            state.add_trace_event(
                "supervisor_decision",
                {"route": next_route, "reason": "Reviewing draft with Critic"},
            )
            return state

        if last_agent == "critic":
            critic_results = [r for r in state.agent_results if r.agent == "critic"]
            if critic_results:
                last_critic_feedback = critic_results[-1].content
                system_prompt = (
                    "You are the supervisor of a research system. You must analyze the critic's review "
                    "and decide if revisions are needed, or if we can stop. "
                    "Output a JSON object with 'status' ('approved' or 'needs_revision') and 'reason'. "
                    "Respond ONLY with valid JSON."
                )
                user_prompt = (
                    f"User Request: {state.request.query}\n"
                    f"Final Answer Draft: {state.final_answer}\n"
                    f"Critic Feedback: {last_critic_feedback}\n"
                )
                try:
                    response = self.llm_client.complete(system_prompt, user_prompt)
                    clean_content = (
                        response.content.strip().replace("```json", "").replace("```", "")
                    )
                    res_dict = json.loads(clean_content)
                    status = res_dict.get("status", "approved")
                    reason = res_dict.get("reason", "Approved by critic")
                except Exception:
                    status = (
                        "approved"
                        if "approve" in last_critic_feedback.lower()
                        else "needs_revision"
                    )
                    reason = "Regex/Fallback critique check"

                if status == "approved":
                    next_route = "__end__"
                else:
                    feedback_lower = last_critic_feedback.lower()
                    if (
                        "search" in feedback_lower
                        or "missing info" in feedback_lower
                        or "find" in feedback_lower
                    ):
                        next_route = "researcher"
                    else:
                        next_route = "writer"

                state.record_route(next_route)
                state.add_trace_event(
                    "supervisor_decision",
                    {"route": next_route, "reason": f"Critic decision: {reason}"},
                )
                return state

        # Default fallback
        next_route = "__end__"
        state.record_route(next_route)
        state.add_trace_event(
            "supervisor_decision", {"route": next_route, "reason": "Default route triggered"}
        )
        return state
