from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

import httpx
from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

from agentic.agents.provider import build_openai_reviewer
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

    def invoke_memory(self, text, *, ticket="ticket-1", turn="turn-1", customer="sam", path=None, trace_path=None):
        graph = build_support_graph(
            self.knowledge_db, memory_database_path=path or self.memory_db, trace_path=trace_path,
        )
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

    def test_unavailable_database_at_recall_blocks_completion(self):
        inaccessible = self.memory_db.parent / "missing-directory" / "memory.db"
        result = self.invoke_memory("I forgot my password.", path=inaccessible)

        self.assertEqual(result["decision"], {
            "next_step": "operational_error", "reason": "history_unavailable",
        })
        self.assertEqual(result["memory_status"], "unavailable")
        self.assertIsNone(result.get("response"))
        self.assertIsNone(result.get("handoff"))
        self.assertFalse(inaccessible.exists())

    def test_aborted_history_write_changes_the_outcome_and_can_be_retried(self):
        trace_path = self.memory_db.with_suffix(".jsonl")
        self.invoke_memory("I forgot my password.")
        with closing(sqlite3.connect(self.memory_db)) as db, db:
            db.execute("""
                CREATE TRIGGER reject_turn BEFORE INSERT ON support_turns
                BEGIN SELECT RAISE(ABORT, 'test write failed'); END
            """)

        for text in ("I forgot my password.", "Please issue a refund.", "Please answer step by step."):
            with self.subTest(text=text):
                result = self.invoke_memory(text, turn="turn-2", trace_path=trace_path)
                self.assertEqual(result["decision"], {
                    "next_step": "operational_error", "reason": "history_write_failed",
                })
                self.assertEqual(result["memory_status"], "unavailable")
                self.assertIsNone(result.get("response"))
                self.assertIsNone(result.get("handoff"))

        events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        failures = [event for event in events if event["event"] == "finish"]
        self.assertEqual(len(failures), 3)
        for event in failures:
            self.assertEqual(event["turn_id"], "turn-2")
            self.assertEqual(event["decision"]["next_step"], "operational_error")
            self.assertEqual(event["memory_status"], "unavailable")

        with closing(sqlite3.connect(self.memory_db)) as db, db:
            self.assertEqual(db.execute("SELECT turn_id FROM support_turns").fetchall(), [("turn-1",)])
            db.execute("DROP TRIGGER reject_turn")
        retry = self.invoke_memory("Please answer step by step.", turn="turn-2")
        self.assertEqual(retry["memory_status"], "saved")
        self.assertEqual(retry["decision"]["reason"], "preference_saved")
        later = self.invoke_memory("I forgot my password.", turn="turn-3")
        self.assertEqual(later["memory"]["response_style"], "step_by_step")

    def test_provider_reviews_grounded_answer_without_ticket_context(self):
        reviews = []

        def approve(response, account_evidence):
            reviews.append((response, account_evidence))
            return "approved"

        graph = build_support_graph(self.knowledge_db, evidence_reviewer=approve)
        result = graph.invoke({
            "account_id": "cultpass", "text": "I forgot my password for example-private-marker.",
            "verified_customer": {"account_id": "cultpass", "external_user_id": "private-id"},
        })

        self.assertEqual(result["decision"]["next_step"], "resolve")
        [(draft, account_evidence)] = reviews
        self.assertEqual(draft, result["response"])
        self.assertTrue(draft["citations"])
        self.assertIsNone(account_evidence)
        self.assertNotIn("example-private-marker", str(reviews))
        self.assertNotIn("private-id", str(reviews))

    def test_provider_refusal_or_failure_escalates_without_returning_the_draft(self):
        for verdict in ("rejected", "unavailable"):
            with self.subTest(verdict=verdict):
                graph = build_support_graph(self.knowledge_db, evidence_reviewer=lambda *_: verdict)
                result = graph.invoke({"account_id": "cultpass", "text": "I forgot my password."})
                self.assertEqual(result["decision"]["next_step"], "escalate")
                self.assertEqual(result["handoff"]["reason"], f"provider_{verdict}")
                self.assertIsNone(result.get("response"))

    def test_openai_adapter_escalates_on_http_failure_or_unexpected_structured_output(self):
        failure = {"error": {"message": "Simulated service outage"}}
        unexpected = {
            "status": "completed",
            "output": [{"type": "message", "content": [{
                "type": "output_text",
                "text": json.dumps({"verdict": "approved", "citations": ["invented-article"]}),
            }]}],
        }
        responses = {
            "outage": httpx.Response(503, json=failure),
            "extra field": httpx.Response(200, json=unexpected),
            "invalid JSON": httpx.Response(200, content=b"{", headers={"content-type": "application/json"}),
            "HTML gateway": httpx.Response(200, text="<html>Gateway failure</html>"),
        }
        for case, response in responses.items():
            with self.subTest(case=case):
                def handle(request):
                    payload = json.loads(request.content)
                    self.assertFalse(payload["store"])
                    self.assertNotIn("private-input", payload["input"])
                    return response

                with OpenAI(
                    api_key="synthetic-test-key", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handle)),
                ) as client:
                    graph = build_support_graph(self.knowledge_db, evidence_reviewer=build_openai_reviewer(client))
                    result = graph.invoke({
                        "account_id": "cultpass", "text": "I forgot my password for private-input.",
                    })
                self.assertEqual(result["decision"]["reason"], "provider_unavailable")
                self.assertIn("review this ticket", result["handoff"]["summary"])
                self.assertIsNone(result.get("response"))

    def test_replaying_a_saved_turn_keeps_its_original_provider_verdict(self):
        ticket = {
            "account_id": "cultpass", "text": "I forgot my password.",
            "ticket_id": "ticket-1", "turn_id": "turn-1",
            "verified_customer": {"account_id": "cultpass", "external_user_id": "sam"},
        }
        winner = build_support_graph(
            self.knowledge_db, memory_database_path=self.memory_db,
            evidence_reviewer=lambda *_: "approved",
        )

        def reject_after_winner_saves(*_):
            self.assertEqual(winner.invoke(ticket)["memory_status"], "saved")
            return "rejected"

        contender = build_support_graph(
            self.knowledge_db, memory_database_path=self.memory_db,
            evidence_reviewer=reject_after_winner_saves,
        )
        for replay in (contender.invoke(ticket), winner.invoke(ticket)):
            self.assertEqual(replay["memory_status"], "replayed")
            self.assertEqual(replay["decision"]["next_step"], "resolve")
            self.assertEqual(replay["provider_verdict"], "approved")
            self.assertIsNone(replay.get("handoff"))

    def test_trace_filters_steps_by_ticket_and_turn_and_omits_messages(self):
        path = self.memory_db.with_suffix(".jsonl")
        graph = build_support_graph(
            self.knowledge_db, memory_database_path=self.memory_db, trace_path=path,
            evidence_reviewer=lambda *_: "approved",
        )
        ticket = {
            "account_id": "cultpass", "verified_customer": {
                "account_id": "cultpass", "external_user_id": "private-profile-marker",
            },
            "ticket_id": "t-1", "turn_id": "turn-1",
            "text": "I forgot my password for private-body-marker.",
        }
        graph.invoke(ticket)
        graph.invoke({**ticket, "turn_id": "turn-2", "text": "How do I assemble a telescope?"})
        graph.invoke(ticket)
        raw = path.read_text(encoding="utf-8")
        events = [event for line in raw.splitlines() if
                  (event := json.loads(line))["ticket_id"] == "t-1" and event["turn_id"] == "turn-1"]

        self.assertEqual([event["event"] for event in events], [
            "classify", "supervisor", "retrieve_knowledge", "resolve", "provider_review",
            "finish", "finish",
        ])
        self.assertEqual(events[0]["categories"], ["access"])
        self.assertGreaterEqual(events[2]["intents"][0]["candidates"][0]["score"], 0.60)
        self.assertEqual(events[4]["verdict"], "approved")
        self.assertEqual(events[5]["memory_status"], "saved")
        self.assertEqual(events[6]["memory_status"], "replayed")
        self.assertNotIn("private-profile-marker", raw)
        self.assertNotIn("private-body-marker", raw)
        self.assertNotIn("Forgot Password", raw)

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

    def test_corrupt_intents_block_completion_when_history_is_required(self):
        self.invoke_memory("I forgot my password.")
        with closing(sqlite3.connect(self.memory_db)) as db, db:
            db.execute("UPDATE support_turns SET intents = 'null'")

        same_ticket = self.invoke_memory("I forgot my password.", turn="turn-2")
        new_ticket = self.invoke_memory("I forgot my password.", ticket="ticket-2")

        for result in (same_ticket, new_ticket):
            self.assertEqual(result["decision"]["next_step"], "operational_error")
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
        self.assertEqual(result["decision"]["next_step"], "operational_error")
        self.assertIsNone(result.get("response"))

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
