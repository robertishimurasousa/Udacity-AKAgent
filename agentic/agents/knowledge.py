from collections.abc import Sequence
from pathlib import Path
import re
from typing import Literal, TypedDict
from unicodedata import normalize

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.exc import SQLAlchemyError

from agentic.tools.knowledge import read_account_articles
from agentic.tools.sqlite import read_only_engine
from data.models.udahub import Knowledge


MIN_RELEVANCE = 0.60
MAX_CANDIDATES = 3
TERM_PATTERN = re.compile(r"[^\W_]+")


class SearchIntent(TypedDict):
    intent_id: str
    terms: list[str]


class ArticleCandidate(TypedDict):
    article_id: str
    title: str
    content: str
    score: float


class IntentEvidence(TypedDict):
    intent_id: str
    candidates: list[ArticleCandidate]


class RetrievalResult(TypedDict):
    status: Literal["ready", "insufficient_evidence", "unavailable"]
    intents: list[IntentEvidence]


class KnowledgeInput(TypedDict):
    account_id: str
    intents: list[SearchIntent]


class KnowledgeOutput(TypedDict):
    retrieval: RetrievalResult


class KnowledgeState(KnowledgeInput, KnowledgeOutput):
    pass


def _terms(text: str) -> set[str]:
    return set(TERM_PATTERN.findall(normalize("NFKC", text).casefold()))


def _rank_articles(query: set[str], articles: Sequence[Knowledge]) -> list[ArticleCandidate]:
    if not query:
        return []

    ranked: list[tuple[tuple[int, int, str], ArticleCandidate]] = []
    for article in articles:
        metadata_terms = _terms(article.title) | _terms(article.tags or "")
        searchable_terms = metadata_terms | _terms(article.content)
        matches = len(query & searchable_terms)
        if matches:
            key = (-matches, -len(query & metadata_terms), article.article_id)
            candidate: ArticleCandidate = {
                "article_id": article.article_id,
                "title": article.title,
                "content": article.content,
                "score": matches / len(query),
            }
            ranked.append((key, candidate))

    ranked.sort(key=lambda item: item[0])
    return [candidate for _, candidate in ranked[:MAX_CANDIDATES]]


def build_knowledge_graph(database_path: Path | str) -> CompiledStateGraph:
    """Build a read-only retrieval graph for preclassified ticket intents."""
    engine = read_only_engine(database_path)

    def retrieve(state: KnowledgeInput) -> KnowledgeOutput:
        if not state["intents"]:
            return {"retrieval": {"status": "insufficient_evidence", "intents": []}}
        try:
            articles = read_account_articles(engine, state["account_id"])
        except SQLAlchemyError:
            return {"retrieval": {"status": "unavailable", "intents": []}}

        evidence: list[IntentEvidence] = [
            {
                "intent_id": intent["intent_id"],
                "candidates": _rank_articles(_terms(" ".join(intent["terms"])), articles),
            }
            for intent in state["intents"]
        ]
        has_coverage = all(
            item["candidates"] and item["candidates"][0]["score"] >= MIN_RELEVANCE
            for item in evidence
        )
        return {
            "retrieval": {
                "status": "ready" if has_coverage else "insufficient_evidence",
                "intents": evidence,
            }
        }

    graph = StateGraph(KnowledgeState, input_schema=KnowledgeInput, output_schema=KnowledgeOutput)
    graph.add_node("retrieve_knowledge", retrieve)
    graph.add_edge(START, "retrieve_knowledge")
    graph.add_edge("retrieve_knowledge", END)
    return graph.compile()
