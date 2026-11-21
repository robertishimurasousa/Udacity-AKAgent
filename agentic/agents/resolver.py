from typing import TypedDict

from agentic.agents.classifier import PERSONAL_SUBSCRIPTION_INTENT, TicketClassification
from agentic.agents.knowledge import ArticleCandidate, MIN_RELEVANCE, RetrievalResult
from agentic.tools.cultpass import AccountEvidence


class Citation(TypedDict):
    article_id: str
    intent_id: str
    title: str
    excerpt: str


class SupportResponse(TypedDict):
    text: str
    citations: list[Citation]


def _customer_answer(intent_id: str, evidence: AccountEvidence) -> str | None:
    if intent_id == "account_status" and evidence["account"].get("is_blocked") is False:
        return "Your CultPass account is not blocked."
    if intent_id == PERSONAL_SUBSCRIPTION_INTENT and (subscription := evidence.get("subscription")):
        if not subscription["has_subscription"]:
            return "There is no subscription on your CultPass account."
        quota = subscription["monthly_quota"]
        unit = "experience" if quota == 1 else "experiences"
        return (
            f"Your CultPass subscription is {subscription['subscription_status']} "
            f"on the {subscription['tier']} plan. Your monthly quota is {quota} {unit}."
        )
    return None


def _find_citation(
    intent_id: str, candidates: list[ArticleCandidate], *, personalized: bool,
) -> Citation | None:
    marker = "**Account guidance:**" if personalized else "**Suggested phrasing:**"
    for article in candidates:
        if article["score"] < MIN_RELEVANCE:
            continue
        _, _, approved = article["content"].partition(marker)
        excerpt = approved.split("\n\n**", 1)[0].strip().strip('"')
        if excerpt:
            return {
                "article_id": article["article_id"], "intent_id": intent_id,
                "title": article["title"], "excerpt": excerpt,
            }
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
        personalized = intent["customer_data_required"]
        citation = _find_citation(
            intent_id, evidence_by_intent.get(intent_id, []), personalized=personalized,
        )
        if citation is None:
            return None

        answer = citation["excerpt"]
        if personalized:
            fact = _customer_answer(intent_id, account_evidence) if account_evidence else None
            if fact is None:
                return None
            answer = f"{fact} {answer}"
        answers.append(answer)
        citations.append(citation)

    return {"text": "\n\n".join(answers), "citations": citations} if answers else None
