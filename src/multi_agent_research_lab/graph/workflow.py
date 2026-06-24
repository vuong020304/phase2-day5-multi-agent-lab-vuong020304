"""LangGraph workflow skeleton."""

from langgraph.graph import StateGraph, START, END

from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.agents import (
    SupervisorAgent,
    ResearcherAgent,
    AnalystAgent,
    WriterAgent,
    CriticAgent
)



class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def build(self) -> object:
        """Create a LangGraph graph."""
        supervisor = SupervisorAgent()
        researcher = ResearcherAgent()
        analyst = AnalystAgent()
        writer = WriterAgent()
        critic = CriticAgent()

        workflow = StateGraph(ResearchState)

        # Add nodes
        workflow.add_node("supervisor", supervisor.run)
        workflow.add_node("researcher", researcher.run)
        workflow.add_node("analyst", analyst.run)
        workflow.add_node("writer", writer.run)
        workflow.add_node("critic", critic.run)

        # Add edges
        workflow.add_edge(START, "supervisor")
        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("analyst", "supervisor")
        workflow.add_edge("writer", "supervisor")
        workflow.add_edge("critic", "supervisor")

        # Routing function for supervisor
        def route(state: ResearchState) -> str:
            if not state.route_history:
                return "supervisor"
            last_route = state.route_history[-1]
            if last_route == "__end__":
                return END
            return last_route

        workflow.add_conditional_edges(
            "supervisor",
            route,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                END: END
            }
        )

        return workflow.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        app = self.build()
        # invoke can accept a dict or a Pydantic object
        result = app.invoke(state)
        if isinstance(result, ResearchState):
            return result
        elif isinstance(result, dict):
            return ResearchState.model_validate(result)
        return state

