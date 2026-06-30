"""Minimal LangGraph wiring: understand -> act -> respond.

Deliberately shallow. LangGraph earns its place via the explicit node boundaries and a
single linear flow that is trivial to reason about and defend -- not via depth.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agent.dispatch import dispatch
from app.agent.state import AgentState
from app.agent.understand import understand
from app.data.catalog import load_catalog
from app.responder import build_response


def _understand_node(state: AgentState) -> AgentState:
    return {"understanding": understand(state["messages"])}


def _act_node(state: AgentState) -> AgentState:
    items, reply, end = dispatch(state["messages"], state["understanding"])
    return {"items": items, "reply": reply, "end": end}


def _respond_node(state: AgentState) -> AgentState:
    # Final assembly + the anti-hallucination/schema gate live here.
    response = build_response(
        reply=state.get("reply", ""),
        items=state.get("items", []),
        end_of_conversation=state.get("end", False),
        catalog=load_catalog(),
    )
    return {"response": response}


def _build():
    g = StateGraph(AgentState)
    g.add_node("understand", _understand_node)
    g.add_node("act", _act_node)
    g.add_node("respond", _respond_node)
    g.set_entry_point("understand")
    g.add_edge("understand", "act")
    g.add_edge("act", "respond")
    g.add_edge("respond", END)
    return g.compile()


GRAPH = _build()
