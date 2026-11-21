from contextlib import closing
import json
import os
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

    def test_memory_cannot_alias_either_read_only_database(self):
        link = self.memory_db.with_name("link.db")
        link.symlink_to(self.knowledge_db)
        hard_link = self.memory_db.with_name("hard-link.db")
        os.link(self.knowledge_db, hard_link)
        external = self.memory_db.with_name("cultpass.db")
        external.touch()
        before = self.knowledge_db.read_bytes()
        for destination in (self.knowledge_db, link, hard_link, external):
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(ValueError, "separate"):
                    build_support_graph(
                        self.knowledge_db, cultpass_database_path=external, memory_database_path=destination,
                    )
        self.assertEqual(self.knowledge_db.read_bytes(), before)
        self.assertEqual(external.read_bytes(), b"")

    def test_corrupt_intents_are_unavailable_without_interrupting_the_ticket(self):
        self.invoke_memory("I forgot my password.")
        with closing(sqlite3.connect(self.memory_db)) as db, db:
            db.execute("UPDATE support_turns SET intents = 'null'")

        same_ticket = self.invoke_memory("I forgot my password.", turn="turn-2")
        new_ticket = self.invoke_memory("I forgot my password.", ticket="ticket-2")

        for result in (same_ticket, new_ticket):
            self.assertEqual(result["decision"]["next_step"], "resolve")
            self.assertEqual(result["memory_status"], "unavailable")
            self.assertNotIn("memory", result)

    def test_history_contains_customer_message_response_and_handoff(self):
        question = "I forgot my password."
        answer = self.invoke_memory(question)
        uncovered = "How do I assemble a telescope?"
        escalated = self.invoke_memory(uncovered, turn="turn-2")
        later = self.invoke_memory("How do I reserve a spot for an event?", turn="turn-3")
        other = self.invoke_memory("I forgot my password.", customer="alice")

        self.assertIn("history", later["memory"])
        first, second = later["memory"]["history"]
        self.assertEqual(first["customer_message"], question)
        self.assertEqual(first["response"], answer["response"])
        self.assertEqual(second["customer_message"], uncovered)
        self.assertEqual(second["handoff"], escalated["handoff"])
        self.assertEqual(other["memory"]["history"], [])

    def test_replay_returns_the_original_outcome_without_reclassifying(self):
        first = self.invoke_memory("I forgot my password.")
        repeated = self.invoke_memory("How do I assemble a telescope?")

        self.assertEqual(repeated["memory_status"], "replayed")
        self.assertEqual(repeated["decision"], first["decision"])
        self.assertEqual(repeated["response"], first["response"])
        self.assertNotIn("handoff", repeated)
        self.assertEqual(repeated["memory"]["history"][0]["customer_message"], "I forgot my password.")

    def test_explicit_step_by_step_preference_applies_across_tickets_and_is_isolated(self):
        first = self.invoke_memory("Please answer step by step. I forgot my password.")
        new_ticket = self.invoke_memory("How do I reserve a spot for an event?", ticket="ticket-2")
        other = self.invoke_memory("How do I reserve a spot for an event?", ticket="ticket-2", customer="alice")

        self.assertEqual(first["decision"]["next_step"], "resolve")
        self.assertTrue(first["response"]["text"].startswith("1. "))
        self.assertEqual(new_ticket["memory"]["response_style"], "step_by_step")
        self.assertTrue(new_ticket["response"]["text"].startswith("1. "))
        self.assertIsNone(other["memory"]["response_style"])
        self.assertFalse(other["response"]["text"].startswith("1. "))

    def test_standalone_preference_can_change_without_losing_the_ticket_topic(self):
        baseline = self.invoke_memory("I forgot my password.", customer="alice")
        self.invoke_memory("Please answer step by step. I forgot my password.")
        updated = self.invoke_memory("Please keep your answers concise.", turn="turn-2")
        follow_up = self.invoke_memory("What should I do next?", turn="turn-3")

        self.assertEqual(updated["decision"]["reason"], "preference_saved")
        self.assertEqual(updated["memory_status"], "saved")
        self.assertEqual(follow_up["memory"]["response_style"], "concise")
        self.assertEqual(follow_up["decision"]["next_step"], "resolve")
        self.assertIn("Forgot Password", follow_up["response"]["text"])
        self.assertIn("check spam or try again in a few minutes", follow_up["response"]["text"])
        self.assertFalse(follow_up["response"]["text"].startswith("1. "))
        self.assertLess(len(follow_up["response"]["text"]), len(baseline["response"]["text"]))
        self.assertEqual(follow_up["response"]["citations"], baseline["response"]["citations"])

    def test_preference_without_a_verified_profile_is_not_recorded(self):
        graph = build_support_graph(self.knowledge_db, memory_database_path=self.memory_db)
        result = graph.invoke({
            "account_id": "cultpass", "ticket_id": "ticket-1", "turn_id": "turn-1",
            "text": "Please keep your answers concise.",
        })

        self.assertEqual(result["decision"]["reason"], "preference_storage_required")
        self.assertFalse(self.memory_db.exists())

    def test_history_redacts_credentials_and_email_in_customer_message(self):
        self.invoke_memory(
            "I forgot my password. Email fake@example.test. OPENAI_API_KEY=sk-test-not-a-real-token. "
            "Password = \"synthetic phrase redaction-marker\"; password: synthetic-colon-marker; "
            "token='synthetic-token-prefix,redaction-tail'."
        )
        later = self.invoke_memory("How do I reserve a spot for an event?", turn="turn-2")

        self.assertIn("history", later["memory"])
        message = later["memory"]["history"][0]["customer_message"]
        self.assertIn("I forgot my password", message)
        self.assertIn("[redacted]", message)
        self.assertNotIn("fake@example.test", message)
        self.assertNotIn("sk-test-not-a-real-token", message)
        self.assertNotIn("redaction-marker", message)
        self.assertNotIn("synthetic-colon-marker", message)
        self.assertNotIn("redaction-tail", message)

    def test_incomplete_stored_outcome_is_not_replayed_as_success(self):
        self.invoke_memory("I forgot my password.")
        with closing(sqlite3.connect(self.memory_db)) as db, db:
            db.execute("UPDATE support_turns SET outcome = ?", (json.dumps({
                "classification": {},
                "decision": {"next_step": "resolve", "reason": "evidence_ready"},
                "response": None,
            }),))

        result = self.invoke_memory("I forgot my password.")

        self.assertEqual(result["memory_status"], "unavailable")
        self.assertIn("Forgot Password", result["response"]["text"])

    def test_legacy_database_preserves_metadata_and_accepts_new_messages(self):
        with closing(sqlite3.connect(self.memory_db)) as db, db:
            db.execute("""
                CREATE TABLE support_turns (
                    sequence INTEGER PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    ticket_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    next_step TEXT NOT NULL,
                    intents TEXT NOT NULL,
                    UNIQUE(account_id, customer_id, ticket_id, turn_id)
                )
            """)
            db.execute("INSERT INTO support_turns VALUES (1, ?, ?, ?, ?, ?, ?)", (
                "cultpass", "sam", "ticket-1", "old", "resolve",
                json.dumps([{
                    "intent_id": "login", "category": "access",
                    "terms": ["password", "email"], "customer_data_required": False,
                }]),
            ))

        legacy_retry = self.invoke_memory("Please answer step by step.", turn="old")
        later = self.invoke_memory("What should I do next?", turn="new")
        after = self.invoke_memory("How do I reserve a spot for an event?", turn="later")

        self.assertEqual(legacy_retry["memory_status"], "legacy")
        self.assertEqual(legacy_retry["decision"]["reason"], "preference_turn_already_recorded")
        self.assertIsNone(later["memory"]["response_style"])
        self.assertEqual(later["decision"]["next_step"], "resolve")
        self.assertIn("history", after["memory"])
        old, new = after["memory"]["history"]
        self.assertEqual(old["turn_id"], "old")
        self.assertNotIn("customer_message", old)
        self.assertEqual(new["customer_message"], "What should I do next?")
