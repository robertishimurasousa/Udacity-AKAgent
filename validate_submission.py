"""Exercise the UDA-Hub technical rubric in a disposable copy of the project."""

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
SETUP_NOTEBOOKS = ("01_external_db_setup.ipynb", "02_core_db_setup.ipynb")
REQUIRED_FILES = {
    ".env.example", ".gitignore", "README.md", "LICENSE.md", "SECURITY.md",
    "requirements.txt", "03_agentic_app.py", "03_agentic_app.ipynb", "utils.py", "validate_submission.py",
    "agentic/support_workflow.py", "agentic/design/README.md", "agentic/design/aceite.md",
    "data/external/cultpass_articles.jsonl", "tests/test_account_resolution.py",
    "tests/test_handoff_memory.py", "tests/test_knowledge_retrieval.py", "tests/test_support_workflow.py",
    *SETUP_NOTEBOOKS,
}
OWNER = "rubric-owner"
OTHER = "rubric-other"
NO_PLAN = "rubric-no-plan"


class CheckFailure(Exception):
    """A rubric check could not verify its expected outcome."""


def require(condition, message):
    if not condition:
        raise CheckFailure(message)


def report(scenario, evidence):
    print(f"PASS {scenario} | {evidence}", flush=True)


def check_layout(root):
    missing = sorted(path for path in REQUIRED_FILES if not (root / path).is_file())
    require(not missing, f"A13 missing artifacts: {', '.join(missing)}")
    if (root / ".git").exists():
        require(shutil.which("git"), "A13 Git is required to verify tracked artifacts.")
        result = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True, timeout=15,
        )
        require(result.returncode == 0, "A13 could not read tracked artifacts.")
        for item in filter(None, result.stdout.split("\0")):
            path = Path(item)
            name = path.name
            require(
                (not name.startswith(".env") or name == ".env.example")
                and name != ".DS_Store"
                and not {".venv", ".logs", ".submission", "__pycache__", ".ipynb_checkpoints"}.intersection(path.parts)
                and not any(name.endswith(suffix) or f"{suffix}-" in name for suffix in (".db", ".sqlite", ".sqlite3")),
                f"A13 excluded file is tracked: {item}",
            )
    example = (root / ".env.example").read_text(encoding="utf-8")
    placeholders = [
        line.strip() for line in example.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    require(
        "OPENAI_API_KEY=" in placeholders and all(re.fullmatch(r"[A-Z_][A-Z0-9_]*=", line) for line in placeholders),
        "A13 .env.example must include OPENAI_API_KEY with empty placeholders.",
    )


def check_citations(articles, citations):
    require(citations, "A03–A06 a grounded response needs a citation.")
    for citation in citations:
        excerpt = citation["excerpt"]
        require(
            excerpt.strip() and excerpt in articles.get(citation["article_id"], ""),
            "A03–A06 a citation did not match its article in SQLite.",
        )


def stage_project(source, workspace):
    ignored = shutil.ignore_patterns(
        ".git", ".venv", ".logs", "__pycache__", ".DS_Store", ".ipynb_checkpoints",
        "*.db*", "*.sqlite*", "*.pyc", ".env", ".env.*",
    )
    for folder in ("agentic", "tests", "data"):
        shutil.copytree(source / folder, workspace / folder, ignore=ignored)
    for name in REQUIRED_FILES:
        if "/" not in name:
            shutil.copyfile(source / name, workspace / name)


def run_notebooks(workspace, env):
    from jupyter_client import KernelManager

    manager = KernelManager(kernel_name="python3", ip="127.0.0.1")
    manager.kernel_spec.argv[0] = sys.executable
    manager.start_kernel(cwd=workspace, env=env)
    client = manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=30)
        for name in SETUP_NOTEBOOKS:
            notebook = json.loads((workspace / name).read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook["cells"], start=1):
                if cell["cell_type"] != "code":
                    continue
                reply = client.execute_interactive(
                    "".join(cell["source"]), timeout=60, output_hook=lambda _: None,
                )
                require(
                    reply["content"]["status"] == "ok",
                    f"A01 {name} cell {index} failed ({reply['content'].get('ename', 'kernel error')}).",
                )
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)


def checked_database(path, expected_tables):
    require(path.is_file(), f"A01 {path.name} was not created.")
    with closing(sqlite3.connect(path)) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        require(expected_tables <= tables, f"A01 {path.name} is missing expected tables.")
        require(db.execute("PRAGMA integrity_check").fetchone() == ("ok",), f"A01 {path.name} failed integrity check.")


def prepare_database(workspace):
    core = workspace / "data/core/udahub.db"
    external = workspace / "data/external/cultpass.db"
    checked_database(core, {"accounts", "users", "tickets", "ticket_metadata", "ticket_messages", "knowledge"})
    checked_database(external, {"users", "subscriptions", "experiences", "reservations"})
    with closing(sqlite3.connect(core)) as db:
        rows = db.execute("SELECT article_id, content, tags FROM knowledge WHERE account_id='cultpass'").fetchall()
    articles = {article_id: content for article_id, content, _ in rows}
    topics = {tag.strip().lower() for _, _, tags in rows for tag in (tags or "").split(",")}
    require(len(articles) >= 14, "A01 the CultPass knowledge base needs at least 14 articles.")
    require(
        {"technical support", "billing", "account management", "reservations"} <= topics,
        "A01 the CultPass articles do not cover the required support topics.",
    )
    require(
        any("4 cultural experiences per month" in body for body in articles.values()),
        "A05 the fixture lacks the general article stating four experiences.",
    )
    with closing(sqlite3.connect(external)) as db, db:
        db.executemany(
            "INSERT INTO users(user_id,full_name,email,is_blocked) VALUES(?,?,?,0)",
            [(profile, profile, f"{profile}@example.test") for profile in (OWNER, OTHER, NO_PLAN)],
        )
        db.executemany("""
            INSERT INTO subscriptions(subscription_id,user_id,status,tier,monthly_quota,started_at)
            VALUES(?,?, 'active','basic',?,CURRENT_TIMESTAMP)
        """, [(profile, profile, quota) for profile, quota in ((OWNER, 7), (OTHER, 99))])
    report("A01", f"both setup notebooks, required tables and {len(articles)} articles across four support topics")
    return articles


def check_diagram(doc):
    diagram = doc.partition("```mermaid")[2].partition("```")[0]
    edges = set(re.findall(
        r"(?m)^[ \t]*(\w+)(?:\[[^\n]*?\]|\{[^\n]*?\})?[ \t]+-->(?:\|[^\n|]+\|)?[ \t]*(\w+)",
        diagram,
    ))
    expected = {
        ("Classify", "Supervisor"), ("Supervisor", "Account"), ("Supervisor", "Knowledge"),
        ("Supervisor", "Escalate"), ("Account", "Knowledge"), ("Account", "Escalate"),
        ("Knowledge", "Resolve"), ("Knowledge", "Escalate"), ("Resolve", "Review"),
        ("Review", "Escalate"), ("Escalate", "Save"), ("Save", "Output"),
    }
    require(
        diagram.strip().startswith("flowchart") and expected <= edges,
        "A02 the Mermaid diagram is missing expected supervisor routes.",
    )


def check_architecture(workspace):
    from langgraph.checkpoint.memory import InMemorySaver
    from agentic.support_workflow import build_support_graph

    check_diagram((workspace / "agentic/design/README.md").read_text(encoding="utf-8"))
    graph = build_support_graph(
        workspace / "data/core/udahub.db",
        cultpass_database_path=workspace / "data/external/cultpass.db",
        memory_database_path=workspace / "data/core/udahub_history.db",
        checkpointer=InMemorySaver(), evidence_reviewer=lambda *_: "approved",
    ).get_graph()
    expected = {
        ("classify", "supervisor"), ("supervisor", "lookup_account"), ("supervisor", "prepare_handoff"),
        ("lookup_account", "retrieve_knowledge"), ("retrieve_knowledge", "review_evidence"),
        ("review_evidence", "resolve"), ("resolve", "provider_review"),
        ("provider_review", "prepare_handoff"), ("remember", "finish"), ("replay_turn", "finish"),
    }
    require(expected <= {(edge.source, edge.target) for edge in graph.edges}, "A02 expected LangGraph routes are missing.")
    report("A02", "architecture diagram and graph transitions")


class Demo:
    def __init__(self, workspace, env, articles, api_key=None, model=None):
        self.workspace = workspace
        self.env = env
        self.articles = articles
        self.api_key = api_key
        self.model = model or "gpt-4o-mini"
        self.history = self.workspace / "data/core/udahub_history.db"

    def run(
        self, ticket, message=None, *, profile=None, turn="first", channel="chat",
        inspect=False, urgent=False, provider=False, code=0,
    ):
        args = [sys.executable, "-B", "03_agentic_app.py", "--ticket-id", ticket, "--channel", channel]
        args += ["--inspect"] if inspect else ["--turn-id", turn, message]
        if profile:
            args += ["--demo-customer-id", profile]
        if urgent:
            args += ["--urgency", "high"]
        call_env = self.env
        if provider and self.api_key:
            args += ["--openai"]
            call_env = {**self.env, "OPENAI_API_KEY": self.api_key, "OPENAI_MODEL": self.model}
        result = subprocess.run(args, cwd=self.workspace, env=call_env, capture_output=True, text=True, timeout=45)
        require(result.returncode == code, f"{ticket}/{turn} exited with {result.returncode}, expected {code}.")
        output = json.loads(result.stdout)
        if output.get("decision", {}).get("reason") == "evidence_ready":
            check_citations(self.articles, output["response"]["citations"])
        return output


def check_routing(demo):
    login = demo.run("login", "I forgot my password. How can I sign in again?", provider=True)
    booking = demo.run("email-booking", "How do I reserve a spot for an event?", channel="email")
    require(login["decision"]["next_step"] == "resolve", "A03 login did not resolve.")
    require(login["response"]["citations"][0]["intent_id"] == "login", "A03 wrong login reference.")
    report("A03", "grounded chat login response")
    require(booking["decision"]["next_step"] == "resolve", "A04 reservation did not resolve.")
    require(booking["response"]["citations"][0]["intent_id"] == "booking", "A04 wrong booking reference.")
    report("A04", "grounded email reservation response")

    plan = demo.run("plan", "What is my subscription status and monthly quota?", profile=OWNER, provider=True)
    account = demo.run("account", "Is my account blocked?", profile=OWNER)
    plan_text = plan.get("response", {}).get("text", "")
    require(
        "7 experiences" in plan_text and not re.search(r"\b4 (?:\w+ )?experiences\b", plan_text, re.IGNORECASE)
        and "is not blocked" in account.get("response", {}).get("text", ""),
        "A05 local lookup facts were not reflected in the response.",
    )
    report("A05", "account and subscription results using the verified profile")
    unknown = demo.run("unknown", "How do I assemble a telescope?", provider=True)
    partial = demo.run("partial", "How can I reset my password and assemble a telescope?")
    both = demo.run("both", "How can I reset my password and reserve a spot for an event?")
    require(
        unknown["decision"]["reason"] == partial["decision"]["reason"] == "insufficient_evidence"
        and {item["intent_id"] for item in both.get("response", {}).get("citations", [])} == {"login", "booking"},
        "A06 evidence coverage was routed incorrectly.",
    )
    report("A06", "independent intent coverage and escalation on missing evidence")
    urgent = demo.run("urgent", "I forgot my password.", urgent=True)
    refund = demo.run("refund", "Please issue a refund.")
    require(
        urgent["handoff"]["reason"] == "high_urgency" and refund["handoff"]["reason"] == "sensitive_action_requested",
        "A07 missing escalation for urgent ticket or refund.",
    )
    report("A07", "urgent and sensitive tickets handed off")
    if demo.api_key:
        require(
            login.get("provider_verdict") == plan.get("provider_verdict") == "approved",
            "A12 live provider did not approve both grounded drafts.",
        )


def check_sessions(demo):
    from agentic.tools.memory import MemoryScope, load_context

    wrong = demo.run("plan", "I forgot my password.", profile=OTHER, turn="denied", code=2)
    denied = demo.run("plan", profile=OTHER, inspect=True, code=2)
    unknown = demo.run("no-profile", "I forgot my password.", profile="rubric-absent", code=2)
    third_party = demo.run("third-party", f"What is my subscription status and monthly quota for ID {OTHER}?", profile=OWNER)
    no_plan = demo.run("no-plan", "What is my subscription status?", profile=NO_PLAN)
    require(
        wrong["decision"]["reason"] == denied["decision"]["reason"] == "ticket_forbidden"
        and unknown["decision"]["reason"] == "profile_not_found"
        and "7 experiences" in third_party["response"]["text"]
        and "99 experiences" not in third_party["response"]["text"]
        and "There is no subscription" in no_plan["response"]["text"],
        "A08 profile validation, ownership or subscription handling failed.",
    )
    report("A08", "ownership, inspection, profile scope and missing subscription")

    booking = demo.run("session", "How do I reserve a spot for an event?", profile=OWNER)
    follow = demo.run("session", "What should I do next?", profile=OWNER, turn="second")
    checkpoint = demo.run("session", profile=OWNER, inspect=True)
    plan = demo.run("plan", profile=OWNER, inspect=True)["checkpoint"]
    demo.run("plan", "I forgot my password.", profile=OWNER, turn="urgent", urgent=True)
    refreshed = demo.run("plan", profile=OWNER, inspect=True)["checkpoint"]
    require(
        booking["thread_id"] == follow["thread_id"] == checkpoint["thread_id"]
        and checkpoint["checkpoint"]["turn_id"] == "second"
        and checkpoint["checkpoint"]["retrieval"]["intents"][0]["intent_id"] == "booking"
        and plan["account_evidence"]["subscription"]["monthly_quota"] == 7
        and refreshed["decision"]["reason"] == "high_urgency"
        and not {"response", "retrieval", "account_evidence", "provider_verdict"}.intersection(refreshed),
        "A09 session context, inspection or reset failed.",
    )
    report("A09", "thread reused across processes, current tool evidence and no old turn data")

    preferred = demo.run("preference", "Please answer step by step.", profile=OWNER)
    owner_history, _ = load_context(demo.history, MemoryScope("cultpass", OWNER, "returning"), "first")
    other_history, _ = load_context(demo.history, MemoryScope("cultpass", OTHER, "other-ticket"), "first")
    returning = demo.run("returning", "How do I reserve a spot for an event?", profile=OWNER)
    isolated = demo.run("other-ticket", "How do I reserve a spot for an event?", profile=OTHER)
    ambiguous = demo.run("new-question", "What should I do next?", profile=OWNER)
    require(
        preferred["decision"]["reason"] == "preference_saved"
        and owner_history["response_style"] == "step_by_step"
        and "booking" in owner_history["recent_resolved_intents"]
        and not other_history["recent_resolved_intents"]
        and returning["response"]["text"].startswith("1. ")
        and not isolated["response"]["text"].startswith("1. ")
        and ambiguous["decision"]["reason"] == "insufficient_evidence",
        "A10 history or preference was lost, crossed a profile boundary, or was applied to an ambiguous new ticket.",
    )
    report("A10", "resolved topics and preferences persisted, profile scope and ambiguous ticket verified")


def check_persistence_and_logs(demo):
    replay = demo.run("plan", "Please issue a refund.", profile=OWNER)
    checkpoint = demo.run("plan", profile=OWNER, inspect=True)["checkpoint"]
    require(
        replay["memory_status"] == "replayed" and checkpoint["text"] == "What is my subscription status and monthly quota?",
        "A11 replay did not preserve the original turn.",
    )
    with closing(sqlite3.connect(demo.history)) as db, db:
        db.execute("CREATE TRIGGER rubric_abort BEFORE INSERT ON support_turns BEGIN SELECT RAISE(ABORT,'rubric'); END")
    failed = demo.run("plan", "Please issue a refund.", profile=OWNER, turn="retry", code=1)
    require(
        failed["decision"]["reason"] == "history_write_failed" and not failed.get("response") and not failed.get("handoff"),
        "A11 failed write was reported as a completed result.",
    )
    with closing(sqlite3.connect(demo.history)) as db, db:
        rows = db.execute("SELECT turn_id FROM support_turns WHERE ticket_id='plan' ORDER BY turn_id").fetchall()
        require(rows == [("first",), ("urgent",)], "A11 refused, replayed or failed turns changed history rows.")
        db.execute("DROP TRIGGER rubric_abort")
    retried = demo.run("plan", "Please issue a refund.", profile=OWNER, turn="retry")
    require(retried["memory_status"] == "saved", "A11 retry was not saved.")
    report("A11", "persisted replay, forced write error and successful retry")

    trace = (demo.workspace / ".logs/uda-hub.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in trace.splitlines()]
    first = [
        event for event in events if event["ticket_id"] == "plan" and event["turn_id"] == "first"
    ]
    expected = ["classify", "supervisor", "lookup_account", "retrieve_knowledge", "resolve"]
    if demo.api_key:
        expected.append("provider_review")
    expected.append("finish")
    require([event["event"] for event in first[:len(expected)]] == expected, "A12 trace is missing expected stages.")
    require(
        [tool["name"] for tool in first[2]["tools"]] == ["lookup_account", "lookup_subscription"]
        and any(event["event"] == "retrieve_knowledge" and event["ticket_id"] == "session"
                and event["turn_id"] == "second" for event in events)
        and not any(event["event"] == "provider_review" and event["ticket_id"] == "unknown" for event in events),
        "A12 trace is missing tool or follow-up evidence, or sent an ungrounded draft to the provider.",
    )
    hidden = (OWNER, OTHER, NO_PLAN, "What is my subscription status")
    if demo.api_key:
        hidden += (demo.api_key,)
    require(not any(value in trace for value in hidden), "A12 trace contains private data.")
    mode = "two live OpenAI approvals" if demo.api_key else "local demo; live review omitted"
    report("A12", f"JSONL search by ticket and turn, {mode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openai", action="store_true", help="Include two live OpenAI reviews of synthetic drafts.")
    args = parser.parse_args()
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("OPENAI_") and key != "PYTHONPATH"
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        check_layout(ROOT)
        api_key = model = None
        if args.openai:
            from dotenv import dotenv_values

            settings = dotenv_values(ROOT / ".env")
            api_key = os.environ.get("OPENAI_API_KEY") or settings.get("OPENAI_API_KEY")
            model = os.environ.get("OPENAI_MODEL") or settings.get("OPENAI_MODEL")
            require(api_key, "A12 --openai requires OPENAI_API_KEY in the environment or local .env.")
        with TemporaryDirectory(prefix="udahub-review-") as temp:
            workspace = Path(temp)
            stage_project(ROOT, workspace)
            run_notebooks(workspace, env)
            articles = prepare_database(workspace)
            check_architecture(workspace)
            demo = Demo(workspace, env, articles, api_key, model)
            check_routing(demo)
            check_sessions(demo)
            check_persistence_and_logs(demo)
            result = subprocess.run(
                [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"],
                cwd=workspace, env=env, timeout=90,
            )
            require(result.returncode == 0, "The regression suite failed.")
        inventory = "tracked-file exclusions checked" if (ROOT / ".git").exists() else "extracted project layout checked"
        report("A13", f"root artifacts, isolated setup, CLI and regression suite; {inventory}")
        print("REVIEW | Architecture assessment, Udacity submission format and academic acceptance.")
        return 0
    except CheckFailure as error:
        print(f"FAIL | {error}", file=sys.stderr)
    except ImportError:
        print("FAIL | Install the project dependencies from requirements.txt first.", file=sys.stderr)
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, sqlite3.Error, subprocess.SubprocessError) as error:
        print(f"FAIL | Validation could not read the expected results ({type(error).__name__}).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
