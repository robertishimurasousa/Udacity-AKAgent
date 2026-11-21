from contextlib import redirect_stderr, redirect_stdout
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from langgraph.checkpoint.sqlite import SqliteSaver
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

    def demo_workspace(self):
        root = self.core_database.parent / "demo"
        (root / "data/core").mkdir(parents=True)
        (root / "data/external").mkdir()
        shutil.copyfile(self.core_database, root / "data/core/udahub.db")
        shutil.copyfile(self.cultpass_database, root / "data/external/cultpass.db")
        return root

    def demo(self, root, *args):
        entry = Path(__file__).resolve().parents[1] / "03_agentic_app.py"
        spec = spec_from_file_location("demo", entry)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        output, errors = StringIO(), StringIO()
        with patch.object(module, "ROOT", root), patch.object(sys, "argv", [str(entry), *args]):
            with redirect_stdout(output), redirect_stderr(errors):
                try:
                    code = module.main()
                except SystemExit as error:
                    return error.code, {"usage_error": errors.getvalue()}
        return code, json.loads(output.getvalue())

    def seed_ticket(self):
        with Session(self.core_engine) as session, session.begin():
            session.add(udahub.User(
                user_id="owner", account_id="cultpass", external_user_id="sam", user_name="Demo",
            ))
            session.add(udahub.Ticket(ticket_id="assigned", account_id="cultpass", user_id="owner"))

    def test_demo_rejects_invalid_profile_and_ticket_from_another_owner_before_running_graph(self):
        self.seed_ticket()
        root = self.demo_workspace()
        for customer, reason in (
            ("missing-customer", "profile_not_found"), ("other-user", "ticket_forbidden"),
        ):
            with self.subTest(customer=customer):
                code, result = self.demo(
                    root, "I forgot my password.", "--ticket-id", "assigned", "--turn-id", "one",
                    "--demo-customer-id", customer,
                )
                self.assertEqual(code, 2)
                self.assertEqual(result["decision"], {"next_step": "reject", "reason": reason})
        code, result = self.demo(root, "--ticket-id", "assigned", "--demo-customer-id", "other-user", "--inspect")
        self.assertEqual(code, 2)
        self.assertEqual(result["decision"]["reason"], "ticket_forbidden")
        self.assertFalse((root / "data/core/udahub_history.db").exists())
        self.assertFalse((root / "data/core/udahub_checkpoints.db").exists())
        self.assertFalse((root / ".logs/uda-hub.jsonl").exists())

    def test_demo_persists_owner_and_resumes_the_same_thread_with_fresh_state(self):
        root = self.demo_workspace()
        owner = ("--ticket-id", "session", "--demo-customer-id", "sam")
        account_question = "What is my subscription status and monthly quota for ID other-user?"
        code, first = self.demo(root, *owner, "--turn-id", "first", account_question)
        self.assertEqual(code, 0)
        self.assertIn("7 experiences", first["response"]["text"])
        self.assertNotIn("99 experiences", first["response"]["text"])
        code, snapshot = self.demo(root, *owner, "--inspect")
        self.assertEqual(code, 0)
        self.assertEqual(snapshot["thread_id"], first["thread_id"])
        self.assertEqual(snapshot["checkpoint"]["account_evidence"]["subscription"]["monthly_quota"], 7)
        self.assertEqual(snapshot["checkpoint"]["turn_id"], "first")
        self.assertEqual(snapshot["checkpoint"]["text"], account_question)

        code, rejected = self.demo(
            root, "--ticket-id", "session", "--demo-customer-id", "other-user",
            "--turn-id", "second", "I forgot my password.",
        )
        self.assertEqual(code, 2)
        self.assertEqual(rejected["decision"]["reason"], "ticket_forbidden")

        code, retry = self.demo(root, *owner, "--turn-id", "first", "Please issue a refund.")
        self.assertEqual(code, 0)
        self.assertEqual(retry["memory_status"], "replayed")
        code, snapshot = self.demo(root, *owner, "--inspect")
        self.assertEqual(code, 0)
        self.assertEqual(snapshot["checkpoint"]["text"], account_question)
        code, escalated = self.demo(
            root, *owner, "--turn-id", "second", "--urgency", "high", "I forgot my password.",
        )
        self.assertEqual(code, 0)
        self.assertEqual(escalated["decision"]["reason"], "high_urgency")
        self.assertEqual(escalated["memory_status"], "saved")
        self.assertIsNone(escalated.get("response"))
        self.assertEqual(escalated["handoff"]["actions"], ["classified"])
        code, snapshot = self.demo(root, *owner, "--inspect")
        self.assertEqual(code, 0)
        self.assertEqual(snapshot["thread_id"], first["thread_id"])
        self.assertEqual(snapshot["checkpoint"]["turn_id"], "second")
        for field in ("response", "retrieval", "account_evidence", "provider_verdict"):
            self.assertIsNone(snapshot["checkpoint"].get(field))

    def test_demo_recalculates_follow_up_from_history_on_same_checkpoint_thread(self):
        root = self.demo_workspace()
        owner = ("--ticket-id", "booking", "--demo-customer-id", "sam")
        code, first = self.demo(root, *owner, "--turn-id", "one", "How do I reserve a spot for an event?")
        self.assertEqual(code, 0)
        code, follow_up = self.demo(root, *owner, "--turn-id", "two", "What should I do next?")
        self.assertEqual(code, 0)
        self.assertEqual(first["response"]["citations"], follow_up["response"]["citations"])
        self.assertEqual(first["thread_id"], follow_up["thread_id"])
        code, snapshot = self.demo(root, *owner, "--inspect")
        self.assertEqual(code, 0)
        self.assertEqual(snapshot["checkpoint"]["turn_id"], "two")
        self.assertEqual(snapshot["checkpoint"]["retrieval"]["intents"][0]["intent_id"], "booking")

    def test_demo_honors_legacy_owner_and_rejects_ambiguous_tickets(self):
        root = self.demo_workspace()
        catalog = root / "data/core/udahub.db"
        history = root / "data/core/udahub_history.db"
        legacy = build_support_graph(catalog, memory_database_path=history)
        for ticket_id, customer in (("legacy", "sam"), ("ambiguous", "sam"), ("ambiguous", "other-user")):
            result = legacy.invoke({
                "account_id": "cultpass", "ticket_id": ticket_id, "turn_id": "old",
                "text": "How do I reserve a spot for an event?",
                "verified_customer": {"account_id": "cultpass", "external_user_id": customer},
            })
            self.assertEqual(result["memory_status"], "saved")

        for ticket_id, customer in (("legacy", "other-user"), ("ambiguous", "sam")):
            code, denied = self.demo(
                root, "--ticket-id", ticket_id, "--turn-id", "new",
                "--demo-customer-id", customer, "What should I do next?",
            )
            self.assertEqual(code, 2)
            self.assertEqual(denied["decision"]["reason"], "ticket_forbidden")
        code, resumed = self.demo(
            root, "--ticket-id", "legacy", "--turn-id", "new",
            "--demo-customer-id", "sam", "What should I do next?",
        )
        self.assertEqual(code, 0)
        self.assertEqual(resumed["memory_status"], "saved")
        self.assertEqual(resumed["response"]["citations"][0]["intent_id"], "booking")

    def test_demo_reports_the_history_error_when_checking_legacy_ownership_fails(self):
        root = self.demo_workspace()
        (root / "data/core/udahub_history.db").mkdir()
        code, result = self.demo(
            root, "--ticket-id", "session", "--turn-id", "first",
            "--demo-customer-id", "sam", "I forgot my password.",
        )
        self.assertEqual(code, 1)
        self.assertEqual(result["decision"], {
            "next_step": "operational_error", "reason": "history_unavailable",
        })
        self.assertFalse((root / "data/core/udahub_checkpoints.db").exists())

    def test_demo_inspection_excludes_a_draft_awaiting_provider_review(self):
        root = self.demo_workspace()
        owner = ("--ticket-id", "session", "--demo-customer-id", "sam")
        code, first = self.demo(root, *owner, "--turn-id", "first", "I forgot my password.")
        self.assertEqual(code, 0)
        with SqliteSaver.from_conn_string(str(root / "data/core/udahub_checkpoints.db")) as saver:
            graph = build_support_graph(
                root / "data/core/udahub.db", memory_database_path=root / "data/core/udahub_history.db",
                evidence_reviewer=lambda *_: "approved", checkpointer=saver,
            )
            config = {"configurable": {"thread_id": first["thread_id"]}}
            graph.invoke({
                "turn_input": {
                    "account_id": "cultpass", "ticket_id": "session", "turn_id": "second",
                    "text": "How do I reserve a spot for an event?",
                    "verified_customer": {"account_id": "cultpass", "external_user_id": "sam"},
                },
            }, config, interrupt_before=["provider_review"])
            self.assertEqual(graph.get_state(config).next, ("provider_review",))
        code, snapshot = self.demo(root, *owner, "--inspect")
        self.assertEqual(code, 1)
        self.assertEqual(snapshot["decision"]["reason"], "checkpoint_incomplete")
        self.assertNotIn("checkpoint", snapshot)

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

    def test_login_and_general_quota_do_not_require_a_customer_identity(self):
        result = self.invoke_account("I need help with login and monthly quota.", customer=None)

        self.assertEqual(result["decision"]["next_step"], "resolve")
        self.assertEqual(
            [citation["intent_id"] for citation in result["response"]["citations"]],
            ["login", "subscription_status"],
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
                self.assertEqual(result["handoff"]["actions"], ["classified"])

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
