import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

from agentic.agents.knowledge import build_knowledge_graph
from data.models.udahub import Account, Base, Knowledge


class KnowledgeRetrievalTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database = Path(directory.name) / "knowledge # test.sqlite3"
        self.engine = create_engine(URL.create("sqlite+pysqlite", database=str(self.database)))
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        with Session(self.engine) as session, session.begin():
            session.add_all([
                Account(account_id="cultpass", account_name="CultPass"),
                Account(account_id="other", account_name="Other"),
            ])

    def add_articles(self, *articles):
        with Session(self.engine) as session, session.begin():
            session.add_all([Knowledge(**article) for article in articles])

    def invoke(self, *intents, database=None):
        graph = build_knowledge_graph(database or self.database)
        state = graph.invoke({"account_id": "cultpass", "intents": list(intents)})
        return state["retrieval"]

    def test_retrieves_each_topic_from_the_cultpass_corpus(self):
        path = Path(__file__).resolve().parents[1] / "data/external/cultpass_articles.jsonl"
        data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.add_articles(*[
            {"account_id": "cultpass", "article_id": str(index), **article}
            for index, article in enumerate(data, start=1)
        ])

        result = self.invoke(
            {"intent_id": "login", "terms": ["PASSWORD", "password", "email"]},
            {"intent_id": "booking", "terms": ["event", "reservation"]},
        )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(
            [
                (item["intent_id"], item["candidates"][0]["title"], item["candidates"][0]["score"])
                for item in result["intents"]
            ],
            [
                ("login", "How to Handle Login Issues?", 1.0),
                ("booking", "How to Reserve a Spot for an Event", 1.0),
            ],
        )
        self.assertIn("Forgot Password", result["intents"][0]["candidates"][0]["content"])

    def test_limit_and_ties_favor_metadata_then_article_id(self):
        self.add_articles(
            {"account_id": "cultpass", "article_id": "00", "title": "Help", "content": "Refund request"},
            {"account_id": "cultpass", "article_id": "a", "title": "Refund", "content": "Policy"},
            {"account_id": "cultpass", "article_id": "b", "title": "Help", "content": "Policy", "tags": "refund"},
            {"account_id": "cultpass", "article_id": "c", "title": "Refund", "content": "Policy"},
        )

        result = self.invoke({"intent_id": "billing", "terms": ["refund"]})

        self.assertEqual([a["article_id"] for a in result["intents"][0]["candidates"]], ["a", "b", "c"])

    def test_excludes_articles_owned_by_another_account(self):
        self.add_articles(
            {"account_id": "other", "article_id": "00", "title": "Password recovery", "content": "Private guidance"},
            {"account_id": "cultpass", "article_id": "01", "title": "Password recovery", "content": "Public guidance"},
        )

        result = self.invoke({"intent_id": "login", "terms": ["password"]})

        self.assertEqual([a["article_id"] for a in result["intents"][0]["candidates"]], ["01"])

    def test_reports_uncovered_intents_and_empty_terms(self):
        self.add_articles(
            {"account_id": "cultpass", "article_id": "login", "title": "Password", "content": "Reset email"},
        )

        result = self.invoke(
            {"intent_id": "login", "terms": ["password", "email"]},
            {"intent_id": "unrelated", "terms": ["telescope"]},
            {"intent_id": "empty", "terms": ["", "?!"]},
        )

        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["intents"][0]["candidates"][0]["score"], 1.0)
        self.assertEqual(result["intents"][1]["candidates"], [])
        self.assertEqual(result["intents"][2]["candidates"], [])
        self.assertEqual(self.invoke()["status"], "insufficient_evidence")

    def test_relevance_threshold_includes_exactly_sixty_percent(self):
        below = [f"low{index}" for index in range(100)]
        at = [f"high{index}" for index in range(100)]
        self.add_articles(
            {"account_id": "cultpass", "article_id": "low", "title": "Low", "content": " ".join(below[:59])},
            {"account_id": "cultpass", "article_id": "high", "title": "High", "content": " ".join(at[:60])},
        )

        for terms, score, status in [(below, 0.59, "insufficient_evidence"), (at, 0.60, "ready")]:
            with self.subTest(score=score):
                result = self.invoke({"intent_id": "coverage", "terms": terms})
                self.assertEqual(result["status"], status)
                self.assertEqual(result["intents"][0]["candidates"][0]["score"], score)

    def test_unavailable_database_returns_error_without_creating_a_file(self):
        missing = self.database.with_name("missing.sqlite3")

        result = self.invoke({"intent_id": "access", "terms": ["password"]}, database=missing)

        self.assertEqual(result, {"status": "unavailable", "intents": []})
        self.assertFalse(missing.exists())
