import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

from agentic.support_workflow import build_support_graph
from data.models.udahub import Account, Base, Knowledge


class SupportWorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database = Path(directory.name) / "udahub.db"
        self.engine = create_engine(URL.create("sqlite+pysqlite", database=str(self.database)))
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        path = Path(__file__).resolve().parents[1] / "data/external/cultpass_articles.jsonl"
        articles = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        with Session(self.engine) as session, session.begin():
            session.add(Account(account_id="cultpass", account_name="CultPass"))
            session.add_all([
                Knowledge(account_id="cultpass", article_id=str(index), **article)
                for index, article in enumerate(articles, start=1)
            ])

    def invoke(self, text, *, urgency="normal", channel="chat", database=None):
        graph = build_support_graph(database or self.database)
        return graph.invoke({
            "account_id": "cultpass",
            "text": text,
            "urgency": urgency,
            "channel": channel,
        })

    def test_routes_login_and_reservation_tickets_to_the_right_articles(self):
        examples = [
            ("I forgot my password. How can I sign in again?", "chat", "access", "How to Handle Login Issues?"),
            ("How do I reserve a spot for an event?", "email", "reservations", "How to Reserve a Spot for an Event"),
        ]
        for text, channel, category, title in examples:
            with self.subTest(category=category):
                result = self.invoke(text, channel=channel)
                self.assertEqual(result["decision"], {"next_step": "resolve", "reason": "evidence_ready"})
                self.assertEqual([intent["category"] for intent in result["classification"]["intents"]], [category])
                self.assertEqual(result["retrieval"]["intents"][0]["candidates"][0]["title"], title)

    def test_routes_two_intents_with_independent_evidence(self):
        result = self.invoke("How can I reset my password and reserve a spot for an event?")

        self.assertEqual(result["decision"]["next_step"], "resolve")
        self.assertEqual(
            [
                (intent["intent_id"], intent["candidates"][0]["title"], intent["candidates"][0]["score"])
                for intent in result["retrieval"]["intents"]
            ],
            [
                ("login", "How to Handle Login Issues?", 1.0),
                ("booking", "How to Reserve a Spot for an Event", 1.0),
            ],
        )

    def test_greeting_and_thanks_do_not_add_uncovered_intents(self):
        result = self.invoke("Hi! I forgot my password. Thanks.")

        self.assertEqual(result["decision"]["next_step"], "resolve")
        self.assertEqual([intent["intent_id"] for intent in result["retrieval"]["intents"]], ["login"])

    def test_escalates_a_ticket_when_only_one_question_has_evidence(self):
        result = self.invoke("How can I reset my password and assemble a telescope?")

        self.assertEqual(result["decision"], {"next_step": "escalate", "reason": "insufficient_evidence"})
        known, unknown = result["retrieval"]["intents"]
        self.assertTrue(known["candidates"])
        self.assertEqual(unknown["candidates"], [])

    def test_keeps_a_second_supported_intent_with_a_different_verb(self):
        result = self.invoke("I forgot my password and need help with a duplicate charge.")

        self.assertEqual(result["decision"]["next_step"], "resolve")
        self.assertEqual(
            [intent["candidates"][0]["title"] for intent in result["retrieval"]["intents"]],
            ["How to Handle Login Issues?", "Review an Unexpected or Duplicate Charge"],
        )

    def test_high_urgency_bypasses_retrieval_for_an_otherwise_resolvable_ticket(self):
        result = self.invoke("I forgot my password.", urgency="high")

        self.assertEqual(result["decision"], {"next_step": "escalate", "reason": "high_urgency"})
        self.assertNotIn("retrieval", result)

    def test_escalates_human_and_sensitive_action_requests_without_lookup(self):
        for text, reason in [
            ("I want to speak to a human about my reservation.", "human_support_requested"),
            ("Can you issue a refund for this charge?", "sensitive_action_requested"),
            ("Please change the email address on my account.", "sensitive_action_requested"),
            ("Cancel my subscription.", "sensitive_action_requested"),
            ("Can I speak with a human about my reservation?", "human_support_requested"),
            ("Please, cancel my subscription.", "sensitive_action_requested"),
            ("I want you to cancel my subscription.", "sensitive_action_requested"),
            ("Give me a refund for the duplicate charge.", "sensitive_action_requested"),
        ]:
            with self.subTest(reason=reason, text=text):
                result = self.invoke(text)
                self.assertEqual(result["decision"], {"next_step": "escalate", "reason": reason})
                self.assertNotIn("retrieval", result)

    def test_distinguishes_a_policy_question_from_a_refund_action(self):
        result = self.invoke("How do I request a refund?")

        self.assertEqual(result["decision"]["next_step"], "resolve")
        self.assertEqual(
            result["retrieval"]["intents"][0]["candidates"][0]["title"],
            "Request a Refund Review",
        )

    def test_routes_personalized_subscription_and_account_questions_to_lookup(self):
        for text in ("What is my subscription status and monthly quota?", "Is my account blocked?"):
            with self.subTest(ticket=text):
                result = self.invoke(text)
                self.assertEqual(result["decision"], {"next_step": "lookup_account", "reason": "customer_data_required"})
                self.assertNotIn("retrieval", result)

    def test_uses_personalization_only_for_the_relevant_subscription_question(self):
        result = self.invoke("I forgot my password. What is the monthly quota?")

        self.assertEqual(result["decision"]["next_step"], "resolve")
        self.assertEqual(
            [intent["intent_id"] for intent in result["retrieval"]["intents"]],
            ["login", "subscription_status"],
        )

    def test_leaves_cancellation_guidance_on_the_knowledge_route(self):
        for text in (
            "How can I cancel my subscription?",
            "Can you explain how to cancel my subscription?",
            "How can I cancel and pause my subscription?",
        ):
            with self.subTest(ticket=text):
                result = self.invoke(text)
                self.assertEqual(result["decision"]["next_step"], "resolve")
                self.assertEqual(
                    result["retrieval"]["intents"][0]["candidates"][0]["title"],
                    "How to Cancel or Pause a Subscription",
                )

    def test_keeps_short_continuations_with_their_topic(self):
        for text, intent_id in [
            ("My app crashes and closes.", "app_crash"),
            ("What is the subscription status and quota?", "subscription_status"),
        ]:
            with self.subTest(ticket=text):
                result = self.invoke(text)
                self.assertEqual(result["decision"]["next_step"], "resolve")
                self.assertEqual([intent["intent_id"] for intent in result["retrieval"]["intents"]], [intent_id])

    def test_preserves_how_to_context_in_a_second_question(self):
        result = self.invoke("How can I reset my password and change my reservation?")

        self.assertEqual(result["decision"]["next_step"], "resolve")
        self.assertEqual([intent["intent_id"] for intent in result["retrieval"]["intents"]], ["login", "booking_change"])

    def test_escalates_a_lookup_failure_for_a_guidance_ticket(self):
        missing = self.database.with_name("missing.db")
        result = self.invoke("I forgot my password.", database=missing)

        self.assertEqual(result["decision"], {"next_step": "escalate", "reason": "knowledge_unavailable"})
        self.assertEqual(result["retrieval"]["status"], "unavailable")
        self.assertFalse(missing.exists())
