from functools import partial
from pathlib import Path
import sqlite3
from typing import Callable, Literal, Mapping, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import Engine

from agentic.agents.classifier import PERSONAL_SUBSCRIPTION_INTENT, TicketClassification, classify_ticket
from agentic.agents.escalation import Handoff, prepare_handoff
from agentic.agents.knowledge import RetrievalResult, SearchIntent, build_knowledge_graph
from agentic.agents.preferences import ResponseStyle, extract_preference, format_response
from agentic.agents.provider import EvidenceReviewer, ReviewVerdict
from agentic.agents.resolver import SupportResponse, compose_response
from agentic.tools.cultpass import (
    AccountEvidence, VerifiedCustomer, lookup_account, lookup_subscription, validate_customer_scope,
)
from agentic.tools.memory import MemoryContext, MemoryScope, StoredOutcome, load_context, save_turn, separate_memory_path
from agentic.tools.sqlite import read_only_engine
from agentic.tools.trace import TraceStatus, append_event


class TicketInput(TypedDict):
    account_id: str
    text: str
    channel: NotRequired[str]
    urgency: NotRequired[str]
    verified_customer: NotRequired[VerifiedCustomer]
    ticket_id: NotRequired[str]
    turn_id: NotRequired[str]


class Decision(TypedDict):
    next_step: Literal["retrieve_knowledge", "resolve", "escalate", "lookup_account", "operational_error"]
    reason: str


class TicketOutput(TypedDict):
    classification: TicketClassification
    decision: Decision
    retrieval: NotRequired[RetrievalResult | None]
    account_evidence: NotRequired[AccountEvidence | None]
    response: NotRequired[SupportResponse | None]
    handoff: NotRequired[Handoff | None]
    memory: NotRequired[MemoryContext]
    memory_status: NotRequired[Literal["saved", "skipped", "unavailable", "replayed", "legacy"]]
    provider_verdict: NotRequired[ReviewVerdict]
    trace_status: NotRequired[TraceStatus]


class SupportState(TicketInput, TicketOutput):
    intents: list[SearchIntent]
    requested_style: NotRequired[ResponseStyle]
    replay: NotRequired[StoredOutcome]


FOLLOW_UPS = {"what should i do next", "what do i do next", "what next", "what now", "and then"}


def _classify(state: SupportState) -> dict[str, object]:
    text, requested_style = extract_preference(state["text"])
    classification = classify_ticket(text)
    memory = state.get("memory")
    if memory and text.strip().casefold().rstrip(".?!") in FOLLOW_UPS:
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
    update: dict[str, object] = {"classification": classification, "intents": search_intents}
    if requested_style is not None:
        update["requested_style"] = requested_style
    return update


def _supervise(state: SupportState) -> dict[str, Decision]:
    classification = state["classification"]
    if state.get("urgency", "normal").casefold() == "high":
        return {"decision": {"next_step": "escalate", "reason": "high_urgency"}}
    if state.get("requested_style") and not classification["intents"]:
        if "memory" in state:
            return {"decision": {"next_step": "resolve", "reason": "preference_saved"}}
        return {"decision": {"next_step": "escalate", "reason": "preference_storage_required"}}
    if classification["human_support_requested"]:
        return {"decision": {"next_step": "escalate", "reason": "human_support_requested"}}
    if classification["sensitive_action_requested"]:
        return {"decision": {"next_step": "escalate", "reason": "sensitive_action_requested"}}
    if classification["customer_data_required"]:
        return {"decision": {"next_step": "lookup_account", "reason": "customer_data_required"}}
    return {"decision": {"next_step": "retrieve_knowledge", "reason": "general_guidance"}}


def _supervise_retrieval(state: SupportState) -> dict[str, Decision]:
    retrieval = state["retrieval"]
    assert retrieval is not None
    status = retrieval["status"]
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
    if state["decision"]["reason"] == "preference_saved":
        style = "step-by-step" if state["requested_style"] == "step_by_step" else "concise"
        return {"response": {"text": f"I'll use {style} answers for your future tickets.", "citations": []}}
    retrieval = state["retrieval"]
    assert retrieval is not None
    response = compose_response(state["classification"], retrieval, state.get("account_evidence"))
    if response is None:
        return {"decision": {"next_step": "escalate", "reason": "insufficient_evidence"}}
    memory = state.get("memory")
    style = state.get("requested_style") or (memory["response_style"] if memory else None)
    return {"response": format_response(response, style)}


def _review_draft(state: SupportState, *, reviewer: EvidenceReviewer) -> dict[str, object]:
    response = state.get("response")
    if response is None or not response["citations"]:
        return {}
    verdict = reviewer(response, state.get("account_evidence"))
    if verdict == "approved":
        return {"provider_verdict": verdict}
    return {
        "provider_verdict": verdict,
        "response": None,
        "decision": {"next_step": "escalate", "reason": f"provider_{verdict}"},
    }


def _handoff(state: SupportState) -> dict[str, Handoff]:
    reason = state["decision"]["reason"]
    evidence = state.get("account_evidence")
    subscription_checked = bool(evidence and "subscription" in evidence) or reason.startswith("subscription_")
    account_checked = evidence is not None or subscription_checked or reason in {
        "account_blocked", "account_not_found", "account_unavailable",
    }
    handoff = prepare_handoff(
        state["classification"], reason, state.get("retrieval"),
        account_checked=account_checked,
        subscription_checked=subscription_checked,
    )
    if verdict := state.get("provider_verdict"):
        handoff["actions"].append(f"provider:{verdict}")
    return {"handoff": handoff}


def _memory_scope(state: TicketInput) -> MemoryScope | None:
    customer = state.get("verified_customer")
    ticket_id = state.get("ticket_id")
    turn_id = state.get("turn_id")
    if customer is None or not all(isinstance(value, str) and value.strip() for value in (ticket_id, turn_id)):
        return None
    if validate_customer_scope(state["account_id"], customer) is not None:
        return None
    return MemoryScope(state["account_id"], customer["external_user_id"], ticket_id)


def _recall(state: TicketInput, *, database: Path) -> dict[str, object]:
    scope = _memory_scope(state)
    if scope is None:
        return {"memory_status": "skipped"}
    try:
        context, replay = load_context(database, scope, state["turn_id"])
    except (sqlite3.Error, ValueError):
        return {"memory_status": "unavailable"}
    return {"memory": context, "replay": replay} if replay is not None else {"memory": context}


def _replay_turn(state: SupportState) -> dict[str, object]:
    return {**state["replay"], "memory_status": "replayed"}


def _remember(state: SupportState, *, database: Path) -> dict[str, object]:
    scope = _memory_scope(state)
    if scope is None:
        return {"memory_status": "skipped"}
    if state.get("memory_status") == "unavailable":
        return {"memory_status": "unavailable"}
    outcome: StoredOutcome = {
        "decision": dict(state["decision"]),
        "classification": state["classification"],
    }
    if (response := state.get("response")) is not None:
        outcome["response"] = response
    if (handoff := state.get("handoff")) is not None:
        outcome["handoff"] = handoff
    preference_request = state["decision"]["reason"] == "preference_saved"
    try:
        status, stored = save_turn(
            database, scope, turn_id=state["turn_id"], customer_message=state["text"],
            outcome=outcome, requested_style=state.get("requested_style"),
        )
    except (sqlite3.Error, ValueError):
        return {
            "memory_status": "unavailable",
            "decision": {"next_step": "operational_error", "reason": "history_write_failed"},
            "response": None,
            "handoff": None,
        }
    if stored is None:
        if preference_request:
            return {
                "memory_status": status,
                "decision": {"next_step": "resolve", "reason": "preference_turn_already_recorded"},
                "response": {"text": "This turn already exists. Please submit the preference as a new turn.", "citations": []},
            }
        return {"memory_status": status}
    update: dict[str, object] = {**stored, "memory_status": status}
    if status == "replayed":
        for key in ("response", "handoff", "retrieval", "account_evidence"):
            if key in state and key not in stored:
                update[key] = None
    return update


def _trace_classification(state: SupportState) -> dict[str, object]:
    intents = state["classification"]["intents"]
    return {
        "categories": sorted({intent["category"] for intent in intents}),
        "intents": [intent["intent_id"] for intent in intents],
    }


def _trace_account(state: SupportState) -> dict[str, object]:
    evidence = state.get("account_evidence") or {}
    reason = state["decision"]["reason"]
    tools = []
    if account := evidence.get("account"):
        tools.append({"name": "lookup_account", "status": account["status"]})
    if subscription := evidence.get("subscription"):
        tools.append({"name": "lookup_subscription", "status": subscription["status"]})
    if reason.startswith("subscription_"):
        tools = [
            {"name": "lookup_account", "status": "ok"},
            {"name": "lookup_subscription", "status": reason.removeprefix("subscription_")},
        ]
    elif reason.startswith("account_"):
        status = "ok" if reason == "account_blocked" else reason.removeprefix("account_")
        tools = [{"name": "lookup_account", "status": status}]
    return {"tools": tools, "decision": state["decision"]}


def _trace_retrieval(state: SupportState) -> dict[str, object]:
    retrieval = state["retrieval"]
    assert retrieval is not None
    return {
        "status": retrieval["status"],
        "intents": [
            {
                "intent_id": intent["intent_id"],
                "candidates": [
                    {"article_id": hit["article_id"], "score": hit["score"]}
                    for hit in intent["candidates"]
                ],
            }
            for intent in retrieval["intents"]
        ],
        "decision": state["decision"],
    }


def build_support_graph(
    database_path: Path | str,
    *,
    cultpass_database_path: Path | str | None = None,
    memory_database_path: Path | str | None = None,
    evidence_reviewer: EvidenceReviewer | None = None,
    trace_path: Path | str | None = None,
) -> CompiledStateGraph:
    """Build the ticket workflow with optional lookups, history, provider review and traces."""
    trace_file = None
    if trace_path is not None:
        trace_file = separate_memory_path(
            trace_path, database_path, cultpass_database_path, memory_database_path,
        )
        if trace_file.suffix != ".jsonl":
            raise ValueError("The trace path must be a separate .jsonl file.")

    def traced(
        event: str, node: Callable[[SupportState], Mapping[str, object]],
        details: Callable[[SupportState], dict[str, object]],
    ) -> Callable[[SupportState], Mapping[str, object]]:
        if trace_file is None:
            return node

        def run(state: SupportState) -> dict[str, object]:
            update = dict(node(state))
            status = append_event(
                trace_file, state.get("ticket_id"), state.get("turn_id"), event,
                details({**state, **update}),
            )
            if state.get("trace_status") != "unavailable":
                update["trace_status"] = status
            return update

        return run

    def decision_details(state: SupportState) -> dict[str, object]:
        return {"decision": state["decision"]}

    graph = StateGraph(SupportState, input_schema=TicketInput, output_schema=TicketOutput)
    graph.add_node("classify", traced("classify", _classify, _trace_classification))
    graph.add_node("supervisor", traced("supervisor", _supervise, decision_details))
    graph.add_node("retrieve_knowledge", build_knowledge_graph(database_path))
    graph.add_node("review_evidence", traced("retrieve_knowledge", _supervise_retrieval, _trace_retrieval))
    graph.add_node("resolve", traced("resolve", _resolve, lambda state: {
        "draft_ready": state.get("response") is not None, "decision": state["decision"],
    }))
    graph.add_node("prepare_handoff", traced("handoff", _handoff, decision_details))

    finish = END
    if trace_file is not None:
        finish = "finish"
        graph.add_node(finish, traced("finish", lambda state: {}, lambda state: {
            "decision": state["decision"], "memory_status": state.get("memory_status"),
        }))
        graph.add_edge(finish, END)

    completion = finish
    if memory_database_path is not None:
        database = separate_memory_path(memory_database_path, database_path, cultpass_database_path)
        graph.add_node("recall", partial(_recall, database=database))
        graph.add_node("remember", partial(_remember, database=database))
        graph.add_node("replay_turn", _replay_turn)
        graph.add_edge(START, "recall")
        graph.add_conditional_edges(
            "recall", lambda state: "replay_turn" if "replay" in state else "classify",
            {"replay_turn": "replay_turn", "classify": "classify"},
        )
        graph.add_edge("replay_turn", finish)
        graph.add_edge("remember", finish)
        completion = "remember"
    else:
        graph.add_edge(START, "classify")

    account_route = finish
    if cultpass_database_path is not None:
        account_route = "lookup_account"
        lookup = partial(_lookup_customer, engine=read_only_engine(cultpass_database_path))
        graph.add_node(account_route, traced(account_route, lookup, _trace_account))
        graph.add_conditional_edges(
            account_route,
            lambda state: state["decision"]["next_step"],
            {"retrieve_knowledge": "retrieve_knowledge", "escalate": "prepare_handoff"},
        )

    graph.add_edge("classify", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["decision"]["next_step"],
        {
            "retrieve_knowledge": "retrieve_knowledge", "resolve": "resolve",
            "escalate": "prepare_handoff", "lookup_account": account_route,
        },
    )
    graph.add_edge("retrieve_knowledge", "review_evidence")
    graph.add_conditional_edges(
        "review_evidence",
        lambda state: state["decision"]["next_step"],
        {"resolve": "resolve", "escalate": "prepare_handoff"},
    )
    after_resolve = completion
    if evidence_reviewer is not None:
        after_resolve = "provider_review"
        graph.add_node(after_resolve, traced(
            after_resolve, partial(_review_draft, reviewer=evidence_reviewer),
            lambda state: {"verdict": state.get("provider_verdict", "skipped"), "decision": state["decision"]},
        ))
        graph.add_conditional_edges(
            after_resolve, lambda state: state["decision"]["next_step"],
            {"resolve": completion, "escalate": "prepare_handoff"},
        )
    graph.add_conditional_edges(
        "resolve",
        lambda state: state["decision"]["next_step"],
        {"resolve": after_resolve, "escalate": "prepare_handoff"},
    )
    graph.add_edge("prepare_handoff", completion)
    return graph.compile()
