from typing import TypedDict

from agentic.agents.classifier import TicketClassification
from agentic.agents.knowledge import RetrievalResult


class Handoff(TypedDict):
    reason: str
    summary: str
    actions: list[str]


def prepare_handoff(
    classification: TicketClassification,
    reason: str,
    retrieval: RetrievalResult | None,
    *,
    account_checked: bool,
    subscription_checked: bool,
) -> Handoff:
    topics = [
        "unclassified request" if intent["intent_id"] == "unknown" else intent["intent_id"].replace("_", " ")
        for intent in classification["intents"]
    ]
    actions = ["classified"]
    if account_checked:
        actions.append("account_checked")
    if subscription_checked:
        actions.append("subscription_checked")
    if retrieval is not None:
        actions.append(f"knowledge:{retrieval['status']}")
    return {
        "reason": reason,
        "summary": f"Please review this ticket about {', '.join(topics) or 'an unclassified request'}.",
        "actions": actions,
    }
