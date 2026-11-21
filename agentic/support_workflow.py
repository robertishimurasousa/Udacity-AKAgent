from functools import partial
from json import JSONDecodeError
from pathlib import Path
import sqlite3
from typing import Literal, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import Engine

from agentic.agents.classifier import PERSONAL_SUBSCRIPTION_INTENT, TicketClassification, classify_ticket
from agentic.agents.escalation import Handoff, prepare_handoff
from agentic.agents.knowledge import RetrievalResult, SearchIntent, build_knowledge_graph
from agentic.agents.resolver import SupportResponse, compose_response
from agentic.tools.cultpass import (
    AccountEvidence, VerifiedCustomer, lookup_account, lookup_subscription, validate_customer_scope,
)
from agentic.tools.memory import MemoryContext, MemoryScope, load_context, save_turn
from agentic.tools.sqlite import read_only_engine


class TicketInput(TypedDict):
    account_id: str
    text: str
    channel: NotRequired[str]
    urgency: NotRequired[str]
    verified_customer: NotRequired[VerifiedCustomer]
    ticket_id: NotRequired[str]
    turn_id: NotRequired[str]


class Decision(TypedDict):
    next_step: Literal["retrieve_knowledge", "resolve", "escalate", "lookup_account"]
    reason: str


class TicketOutput(TypedDict):
    classification: TicketClassification
    decision: Decision
    retrieval: NotRequired[RetrievalResult]
    account_evidence: NotRequired[AccountEvidence]
    response: NotRequired[SupportResponse]
    handoff: NotRequired[Handoff]
    memory: NotRequired[MemoryContext]
    memory_status: NotRequired[Literal["saved", "skipped", "unavailable"]]


class SupportState(TicketInput, TicketOutput):
    intents: list[SearchIntent]


FOLLOW_UPS = {"what should i do next", "what do i do next", "what next", "what now", "and then"}


def _classify(state: SupportState) -> dict[str, TicketClassification | list[SearchIntent]]:
    classification = classify_ticket(state["text"])
    memory = state.get("memory")
    if memory and state["text"].strip().casefold().rstrip(".?!") in FOLLOW_UPS:
        previous = memory["session_intents"]
        if previous:
            classification["intents"] = previous
            classification["customer_data_required"] = any(
                intent["customer_data_required"] for intent in previous
            )
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


def _lookup_customer(state: SupportState, *, engine: Engine) -> dict[str, Decision | AccountEvidence]:
    customer = state.get("verified_customer")
    if customer is None:
        return {"decision": {"next_step": "escalate", "reason": "identity_required"}}

    account_id = state["account_id"]
    account = lookup_account(engine, account_id, customer)
    if account["status"] != "ok":
        return {"decision": {"next_step": "escalate", "reason": f"account_{account['status']}"}}
    if account["is_blocked"]:
        return {"decision": {"next_step": "escalate", "reason": "account_blocked"}}

    evidence: AccountEvidence = {"account": account}
    if any(
        intent["intent_id"] == PERSONAL_SUBSCRIPTION_INTENT
        for intent in state["classification"]["intents"]
    ):
        subscription = lookup_subscription(engine, account_id, customer)
        if subscription["status"] != "ok":
            return {"decision": {"next_step": "escalate", "reason": f"subscription_{subscription['status']}"}}
        evidence["subscription"] = subscription
    return {
        "account_evidence": evidence,
        "decision": {"next_step": "retrieve_knowledge", "reason": "customer_data_ready"},
    }


def _resolve(state: SupportState) -> dict[str, Decision | SupportResponse]:
    response = compose_response(state["classification"], state["retrieval"], state.get("account_evidence"))
    if response is None:
        return {"decision": {"next_step": "escalate", "reason": "insufficient_evidence"}}
    return {"response": response}


def _handoff(state: SupportState) -> dict[str, Handoff]:
    reason = state["decision"]["reason"]
    return {
        "handoff": prepare_handoff(
            state["classification"], reason, state.get("retrieval"),
            account_checked="account_evidence" in state or reason.startswith(("account_", "subscription_")),
        ),
    }


def _memory_scope(state: TicketInput) -> MemoryScope | None:
    customer = state.get("verified_customer")
    ticket_id = state.get("ticket_id")
    turn_id = state.get("turn_id")
    if customer is None or not all(isinstance(value, str) and value.strip() for value in (ticket_id, turn_id)):
        return None
    if validate_customer_scope(state["account_id"], customer) is not None:
        return None
    return MemoryScope(state["account_id"], customer["external_user_id"], ticket_id)


def _recall(state: TicketInput, *, database: Path) -> dict[str, MemoryContext | str]:
    scope = _memory_scope(state)
    if scope is None:
        return {"memory_status": "skipped"}
    try:
        return {"memory": load_context(database, scope)}
    except (sqlite3.Error, JSONDecodeError):
        return {"memory_status": "unavailable"}


def _remember(state: SupportState, *, database: Path) -> dict[str, str]:
    scope = _memory_scope(state)
    if scope is None:
        return {"memory_status": "skipped"}
    if state.get("memory_status") == "unavailable":
        return {"memory_status": "unavailable"}
    try:
        save_turn(
            database, scope, state["turn_id"], state["decision"]["next_step"],
            state["classification"]["intents"],
        )
    except sqlite3.Error:
        return {"memory_status": "unavailable"}
    return {"memory_status": "saved"}


def build_support_graph(
    database_path: Path | str,
    *,
    cultpass_database_path: Path | str | None = None,
    memory_database_path: Path | str | None = None,
) -> CompiledStateGraph:
    """Build the ticket workflow with optional CultPass lookups and history."""
    graph = StateGraph(SupportState, input_schema=TicketInput, output_schema=TicketOutput)
    graph.add_node("classify", _classify)
    graph.add_node("supervisor", _supervise)
    graph.add_node("retrieve_knowledge", build_knowledge_graph(database_path))
    graph.add_node("review_evidence", _supervise_retrieval)
    graph.add_node("resolve", _resolve)
    graph.add_node("prepare_handoff", _handoff)

    completion = END
    if memory_database_path is not None:
        database = Path(memory_database_path)
        graph.add_node("recall", partial(_recall, database=database))
        graph.add_node("remember", partial(_remember, database=database))
        graph.add_edge(START, "recall")
        graph.add_edge("recall", "classify")
        graph.add_edge("remember", END)
        completion = "remember"
    else:
        graph.add_edge(START, "classify")

    account_route = END
    if cultpass_database_path is not None:
        account_route = "lookup_account"
        graph.add_node(account_route, partial(_lookup_customer, engine=read_only_engine(cultpass_database_path)))
        graph.add_conditional_edges(
            account_route,
            lambda state: state["decision"]["next_step"],
            {"retrieve_knowledge": "retrieve_knowledge", "escalate": "prepare_handoff"},
        )

    graph.add_edge("classify", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["decision"]["next_step"],
        {"retrieve_knowledge": "retrieve_knowledge", "escalate": "prepare_handoff", "lookup_account": account_route},
    )
    graph.add_edge("retrieve_knowledge", "review_evidence")
    graph.add_conditional_edges(
        "review_evidence",
        lambda state: state["decision"]["next_step"],
        {"resolve": "resolve", "escalate": "prepare_handoff"},
    )
    graph.add_conditional_edges(
        "resolve",
        lambda state: state["decision"]["next_step"],
        {"resolve": completion, "escalate": "prepare_handoff"},
    )
    graph.add_edge("prepare_handoff", completion)
    return graph.compile()
