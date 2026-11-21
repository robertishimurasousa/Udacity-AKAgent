from typing import NotRequired, TypedDict

from agentic.agents.classifier import TicketClassification
from agentic.agents.knowledge import MIN_RELEVANCE, RetrievalResult
from agentic.tools.cultpass import AccountEvidence


class Citation(TypedDict):
    article_id: str
    intent_id: str
    title: str
    excerpt: NotRequired[str]


class SupportResponse(TypedDict):
    text: str
    citations: list[Citation]


def _customer_answer(intent_id: str, evidence: AccountEvidence) -> str | None:
    if intent_id == "account_status" and evidence["account"].get("is_blocked") is False:
        return "Your CultPass account is not blocked."
    if intent_id == "subscription_status" and (subscription := evidence.get("subscription")):
        if not subscription["has_subscription"]:
            return "There is no subscription on your CultPass account."
        quota = subscription["monthly_quota"]
        unit = "experience" if quota == 1 else "experiences"
        return f"Your CultPass subscription is {subscription['subscription_status']}. Your monthly quota is {quota} {unit}."
    return None


def compose_response(
    classification: TicketClassification,
    retrieval: RetrievalResult,
    account_evidence: AccountEvidence | None,
) -> SupportResponse | None:
    if retrieval["status"] != "ready":
        return None

    evidence_by_intent = {item["intent_id"]: item["candidates"] for item in retrieval["intents"]}
    answers: list[str] = []
    citations: list[Citation] = []
    for intent in classification["intents"]:
        intent_id = intent["intent_id"]
        article = next(
            (article for article in evidence_by_intent.get(intent_id, [])
             if article["score"] >= MIN_RELEVANCE and article["content"].strip()),
            None,
        )
        if article is None:
            return None

        citation: Citation = {"article_id": article["article_id"], "intent_id": intent_id, "title": article["title"]}
        if intent["customer_data_required"]:
            answer = _customer_answer(intent_id, account_evidence) if account_evidence else None
            if answer is None:
                return None
        else:
            content = article["content"]
            _, _, suggested = content.partition("**Suggested phrasing:**")
            answer = (suggested or content).strip().strip('"')
            if not answer:
                return None
            citation["excerpt"] = answer
        answers.append(answer)
        citations.append(citation)

    return {"text": "\n\n".join(answers), "citations": citations} if answers else None
