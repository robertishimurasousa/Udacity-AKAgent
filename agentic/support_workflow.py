from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic.agents.classifier import TicketClassification, classify_ticket
from agentic.agents.knowledge import RetrievalResult, SearchIntent, build_knowledge_graph


class TicketInput(TypedDict):
    account_id: str
    text: str
    channel: NotRequired[str]
    urgency: NotRequired[str]


class Decision(TypedDict):
    next_step: Literal["retrieve_knowledge", "resolve", "escalate", "lookup_account"]
    reason: str


class TicketOutput(TypedDict):
    classification: TicketClassification
    decision: Decision
    retrieval: NotRequired[RetrievalResult]


class SupportState(TicketInput, TicketOutput):
    intents: list[SearchIntent]


def _classify(state: TicketInput) -> dict[str, TicketClassification | list[SearchIntent]]:
    classification = classify_ticket(state["text"])
    search_intents: list[SearchIntent] = [
        {"intent_id": intent["intent_id"], "terms": intent["terms"]}
        for intent in classification["intents"]
    ]
    return {"classification": classification, "intents": search_intents}


def _supervise(state: SupportState) -> dict[str, Decision]:
    classification = state["classification"]
    if state.get("urgency", "normal").casefold() == "high":
        return {"decision": {"next_step": "escalate", "reason": "high_urgency"}}
    if classification["human_support_requested"]:
        return {"decision": {"next_step": "escalate", "reason": "human_support_requested"}}
    if classification["sensitive_action_requested"]:
        return {"decision": {"next_step": "escalate", "reason": "sensitive_action_requested"}}
    if classification["customer_data_required"]:
        return {"decision": {"next_step": "lookup_account", "reason": "customer_data_required"}}
    return {"decision": {"next_step": "retrieve_knowledge", "reason": "general_guidance"}}


def _supervise_retrieval(state: SupportState) -> dict[str, Decision]:
    status = state["retrieval"]["status"]
    if status == "ready":
        return {"decision": {"next_step": "resolve", "reason": "evidence_ready"}}
    reason = "knowledge_unavailable" if status == "unavailable" else "insufficient_evidence"
    return {"decision": {"next_step": "escalate", "reason": reason}}


def build_support_graph(database_path: Path | str) -> CompiledStateGraph:
    """Build the ticket workflow through classification, retrieval, and routing."""
    graph = StateGraph(SupportState, input_schema=TicketInput, output_schema=TicketOutput)
    graph.add_node("classify", _classify)
    graph.add_node("supervisor", _supervise)
    graph.add_node("retrieve_knowledge", build_knowledge_graph(database_path))
    graph.add_node("review_evidence", _supervise_retrieval)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["decision"]["next_step"],
        {"retrieve_knowledge": "retrieve_knowledge", "escalate": END, "lookup_account": END},
    )
    graph.add_edge("retrieve_knowledge", "review_evidence")
    graph.add_edge("review_evidence", END)
    return graph.compile()
