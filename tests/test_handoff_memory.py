from contextlib import closing
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

from agentic.support_workflow import build_support_graph
from data.models.udahub import Account, Base, Knowledge


class HandoffMemoryTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.knowledge_db = Path(temp.name) / "udahub.db"
        self.memory_db = Path(temp.name) / "history.db"
        engine = create_engine(URL.create("sqlite+pysqlite", database=str(self.knowledge_db)))
        self.addCleanup(engine.dispose)
        Base.metadata.create_all(engine)
        path = Path(__file__).resolve().parents[1] / "data/external/cultpass_articles.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        with Session(engine) as session, session.begin():
            session.add(Account(account_id="cultpass", account_name="CultPass"))
            session.add_all([
                Knowledge(account_id="cultpass", article_id=str(index), **article)
                for index, article in enumerate(rows, start=1)
            ])

    def invoke_memory(self, text, *, ticket="ticket-1", turn="turn-1", customer="sam", path=None):
        graph = build_support_graph(self.knowledge_db, memory_database_path=path or self.memory_db)
        return graph.invoke({
            "account_id": "cultpass",
            "verified_customer": {"account_id": "cultpass", "external_user_id": customer},
            "ticket_id": ticket,
            "turn_id": turn,
            "text": text,
        })

    def test_escalation_includes_reason_topics_and_retrieval_outcome(self):
        graph = build_support_graph(self.knowledge_db)
        result = graph.invoke({
            "account_id": "cultpass", "text": "How can I reset my password and assemble a telescope?",
        })

        self.assertIn("handoff", result, "An escalated ticket should have a handoff")
        handoff = result["handoff"]
        self.assertEqual(handoff["reason"], "insufficient_evidence")
        self.assertIn("login", handoff["summary"].lower())
        self.assertIn("unclassified", handoff["summary"].lower())
        self.assertIn("knowledge:insufficient_evidence", handoff["actions"])
        self.assertNotIn("response", result)

    def test_high_urgency_handoff_lists_only_checks_that_ran(self):
        graph = build_support_graph(self.knowledge_db)
        result = graph.invoke({"account_id": "cultpass", "text": "I forgot my password.", "urgency": "high"})

        self.assertIn("handoff", result)
        self.assertEqual(result["handoff"]["reason"], "high_urgency")
        self.assertEqual(result["handoff"]["actions"], ["classified"])

    def test_follow_up_uses_last_resolved_intent_after_rebuilding_the_graph(self):
        first = self.invoke_memory("I forgot my password.")
        self.assertEqual(first["memory_status"], "saved")

        follow_up = self.invoke_memory("What should I do next?", turn="turn-2")

        self.assertEqual(follow_up["decision"]["next_step"], "resolve")
        self.assertIn("Forgot Password", follow_up["response"]["text"])
        self.assertEqual(follow_up["memory"]["session_intents"][0]["intent_id"], "login")

    def test_previous_ticket_topics_are_scoped_to_the_verified_customer(self):
        self.invoke_memory("How do I reserve a spot for an event?")

        returning = self.invoke_memory("I forgot my password.", ticket="ticket-2")
        other = self.invoke_memory("What should I do next?", customer="alice", turn="turn-2")

        self.assertEqual(returning["memory"]["recent_resolved_intents"], ["booking"])
        self.assertEqual(other["memory"]["recent_resolved_intents"], [])
        self.assertEqual(other["memory"]["session_intents"], [])
        self.assertEqual(other["decision"]["next_step"], "escalate")

    def test_repeating_a_turn_does_not_persist_duplicates_or_ticket_text(self):
        prompt = "I forgot my password, note example-secret-marker."
        self.invoke_memory(prompt)
        self.invoke_memory(prompt)

        with closing(sqlite3.connect(self.memory_db)) as db:
            turns = db.execute("SELECT turn_id, intents FROM support_turns").fetchall()
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0][0], "turn-1")
        self.assertNotIn("example-secret-marker", turns[0][1])

    def test_memory_failure_is_visible_and_keeps_general_guidance_available(self):
        inaccessible = self.memory_db.parent / "missing-directory" / "memory.db"
        result = self.invoke_memory("I forgot my password.", path=inaccessible)

        self.assertEqual(result["decision"]["next_step"], "resolve")
        self.assertEqual(result["memory_status"], "unavailable")
        self.assertFalse(inaccessible.exists())
