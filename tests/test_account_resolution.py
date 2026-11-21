import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sqlalchemy import create_engine, delete
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

from agentic.support_workflow import build_support_graph
from data.models import cultpass, udahub


class AccountResolutionTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.core_database = root / "UDA-Hub.sqlite3"
        self.cultpass_database = root / "CultPass # test.sqlite3"
        self.core_engine = create_engine(URL.create("sqlite+pysqlite", database=str(self.core_database)))
        self.addCleanup(self.core_engine.dispose)
        udahub.Base.metadata.create_all(self.core_engine)
        articles_path = Path(__file__).resolve().parents[1] / "data/external/cultpass_articles.jsonl"
        articles = [json.loads(line) for line in articles_path.read_text(encoding="utf-8").splitlines()]
        self.articles = {
            str(index): article for index, article in enumerate(articles, start=1)
        }
        with Session(self.core_engine) as session, session.begin():
            session.add(udahub.Account(account_id="cultpass", account_name="CultPass"))
            session.add_all([
                udahub.Knowledge(account_id="cultpass", article_id=article_id, **article)
                for article_id, article in self.articles.items()
            ])
            session.add(udahub.Knowledge(
                account_id="cultpass", article_id="00",
                title="Subscription status and monthly quota",
                content="**Suggested phrasing:**\n\"The standard subscription monthly quota is 4 experiences per month.\"",
                tags="subscription, quota",
            ))

        engine = create_engine(URL.create("sqlite+pysqlite", database=str(self.cultpass_database)))
        self.addCleanup(engine.dispose)
        cultpass.Base.metadata.create_all(engine)
        with Session(engine) as session, session.begin():
            session.add_all([
                cultpass.User(user_id=user_id, full_name=user_id, email=f"{user_id}@example.test", is_blocked=blocked)
                for user_id, blocked in (
                    ("sam", False), ("other-user", False), ("blocked", True), ("no-plan", False)
                )
            ])
            session.add_all([
                cultpass.Subscription(
                    subscription_id=user_id, user_id=user_id, status="active", tier="basic", monthly_quota=quota,
                )
                for user_id, quota in (("sam", 7), ("other-user", 99), ("blocked", 2))
            ])

    def invoke_account(self, text, *, customer="sam", customer_account="cultpass", database=None):
        graph = build_support_graph(
            self.core_database, cultpass_database_path=database or self.cultpass_database,
        )
        ticket = {"account_id": "cultpass", "text": text}
        if customer is not None:
            ticket["verified_customer"] = {"account_id": customer_account, "external_user_id": customer}
        return graph.invoke(ticket)

    def test_general_response_quotes_and_cites_the_selected_article(self):
        graph = build_support_graph(self.core_database)
        result = graph.invoke({"account_id": "cultpass", "text": "I forgot my password."})

        self.assertIn("response", result, "The resolver should produce a grounded response")
        response = result["response"]
        self.assertIn("Forgot Password", response["text"])
        citation = response["citations"][0]
        self.assertEqual(citation["title"], "How to Handle Login Issues?")
        self.assertIn(response["text"], self.articles[citation["article_id"]]["content"])

    def test_account_facts_override_generic_quota_and_ignore_an_id_in_ticket_text(self):
        result = self.invoke_account("What is my subscription status, tier, and monthly quota for ID other-user?")

        self.assertEqual(result["decision"]["next_step"], "resolve")
        self.assertIn("active", result["response"]["text"])
        self.assertIn("basic", result["response"]["text"])
        self.assertIn("7 experiences", result["response"]["text"])
        self.assertNotIn("4 experiences", result["response"]["text"])
        self.assertNotIn("99 experiences", result["response"]["text"])
        citation = result["response"]["citations"][0]
        self.assertEqual(citation["title"], "Check Subscription Status and Monthly Quota")
        self.assertIn(citation["excerpt"], self.articles[citation["article_id"]]["content"])
        self.assertIn(citation["excerpt"], result["response"]["text"])
        self.assertEqual(result["account_evidence"]["subscription"]["monthly_quota"], 7)

    def test_general_and_personal_subscription_questions_keep_distinct_evidence(self):
        result = self.invoke_account("What is my subscription status? What is the monthly quota?")

        self.assertEqual(result["decision"]["next_step"], "resolve")
        self.assertIn("7 experiences", result["response"]["text"])
        self.assertEqual(
            [citation["intent_id"] for citation in result["response"]["citations"]],
            ["personal_subscription_status", "subscription_status"],
        )

    def test_refund_guidance_quotes_customer_facing_content(self):
        graph = build_support_graph(self.core_database)
        result = graph.invoke({"account_id": "cultpass", "text": "How do I request a refund?"})

        response = result["response"]
        citation = response["citations"][0]
        self.assertIn("Contact CultPass support", response["text"])
        self.assertNotIn("Forward requests", response["text"])
        self.assertIn(citation["excerpt"], self.articles[citation["article_id"]]["content"])

    def test_account_lookup_grounds_an_account_status_answer(self):
        result = self.invoke_account("Is my account blocked?")

        self.assertEqual(result["decision"]["next_step"], "resolve")
        self.assertIn("is not blocked", result["response"]["text"])
        self.assertFalse(result["account_evidence"]["account"]["is_blocked"])

    def test_missing_or_cross_account_identity_does_not_expose_account_data(self):
        for customer, account, reason in [
            (None, "cultpass", "identity_required"),
            ("sam", "other", "account_forbidden"),
            ("", "cultpass", "account_invalid_input"),
        ]:
            with self.subTest(customer=customer, account=account):
                result = self.invoke_account(
                    "What is my subscription status and monthly quota?", customer=customer, customer_account=account,
                )
                self.assertEqual(result["decision"], {"next_step": "escalate", "reason": reason})
                self.assertNotIn("response", result)
                self.assertNotIn("account_evidence", result)

    def test_blocked_or_unknown_customer_is_escalated(self):
        for customer, reason in [("blocked", "account_blocked"), ("unknown", "account_not_found")]:
            with self.subTest(customer=customer):
                result = self.invoke_account("What is my subscription status?", customer=customer)
                self.assertEqual(result["decision"], {"next_step": "escalate", "reason": reason})
                self.assertNotIn("response", result)

    def test_malformed_profile_escalates_without_exposing_account_data(self):
        graph = build_support_graph(self.core_database, cultpass_database_path=self.cultpass_database)
        for profile in ({"account_id": "cultpass"}, {"account_id": "cultpass", "external_user_id": None}):
            with self.subTest(profile=profile):
                result = graph.invoke({
                    "account_id": "cultpass",
                    "text": "Is my account blocked?",
                    "verified_customer": profile,
                })
                self.assertEqual(result["decision"], {"next_step": "escalate", "reason": "account_invalid_input"})
                self.assertNotIn("response", result)
                self.assertNotIn("account_evidence", result)

    def test_customer_without_a_subscription_receives_the_verified_fact(self):
        result = self.invoke_account("What is my subscription status?", customer="no-plan")

        self.assertEqual(result["decision"]["next_step"], "resolve")
        self.assertIn("no subscription", result["response"]["text"])
        self.assertFalse(result["account_evidence"]["subscription"]["has_subscription"])

    def test_account_database_unavailable_escalates_without_creating_a_file(self):
        missing = self.cultpass_database.with_name("missing.sqlite3")
        result = self.invoke_account("Is my account blocked?", database=missing)

        self.assertEqual(result["decision"], {"next_step": "escalate", "reason": "account_unavailable"})
        self.assertNotIn("response", result)
        self.assertFalse(missing.exists())

    def test_an_article_with_matching_metadata_and_empty_content_cannot_resolve(self):
        with Session(self.core_engine) as session, session.begin():
            session.execute(delete(udahub.Knowledge))
            session.add(udahub.Knowledge(
                account_id="cultpass", article_id="empty", title="Password reset email", content="  ",
            ))
        graph = build_support_graph(self.core_database)

        result = graph.invoke({"account_id": "cultpass", "text": "I forgot my password."})

        self.assertEqual(result["decision"], {"next_step": "escalate", "reason": "insufficient_evidence"})
        self.assertNotIn("response", result)
