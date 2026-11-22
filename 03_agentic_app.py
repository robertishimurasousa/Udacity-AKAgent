import argparse
from contextlib import ExitStack, closing
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3

from dotenv import load_dotenv
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from openai import DefaultHttpxClient, OpenAI

from agentic.agents.provider import DEFAULT_MODEL, build_openai_reviewer
from agentic.workflow import TicketInput, build_support_graph
from agentic.tools.memory import redact_support_text, separate_memory_path
from agentic.tools.tickets import TicketAccessError, authorize_demo_ticket
from agentic.tools.trace import valid_trace_id


ROOT = Path(__file__).resolve().parent


def _report_error(reason: str, *, operational: bool = False) -> int:
    output = {
        "decision": {"next_step": "operational_error" if operational else "reject", "reason": reason},
    }
    if reason == "history_unavailable":
        output["memory_status"] = "unavailable"
    print(json.dumps(output, indent=2))
    return 1 if operational else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a synthetic CultPass ticket through UDA-Hub.")
    parser.add_argument("text", nargs="?")
    parser.add_argument("--ticket-id", required=True)
    parser.add_argument("--turn-id")
    parser.add_argument("--channel", choices=("chat", "email"), default="chat")
    parser.add_argument("--urgency", choices=("normal", "high"), default="normal")
    parser.add_argument("--demo-customer-id", help="Existing ID from the synthetic CultPass setup database.")
    parser.add_argument("--inspect", action="store_true", help="Read the latest authorized session checkpoint.")
    parser.add_argument("--openai", action="store_true", help="Review grounded drafts using OpenAI.")
    parser.add_argument("--trace-path", type=Path, default=ROOT / ".logs/uda-hub.jsonl")
    args = parser.parse_args()
    if args.inspect:
        if not args.demo_customer_id or args.text is not None or args.turn_id is not None:
            parser.error("--inspect requires a demo customer ID and a ticket ID.")
    elif args.text is None or not valid_trace_id(args.turn_id):
        parser.error("Supply a message and an opaque turn ID.")
    if not valid_trace_id(args.ticket_id):
        parser.error("Use an opaque ticket ID of up to 80 letters, digits, dots, colons, dashes or underscores.")

    catalog = ROOT / "data/core/udahub.db"
    cultpass = ROOT / "data/external/cultpass.db"
    history = ROOT / "data/core/udahub_history.db"
    checkpoints = ROOT / "data/core/udahub_checkpoints.db"
    try:
        history = separate_memory_path(history, catalog, cultpass)
        checkpoints = separate_memory_path(checkpoints, catalog, cultpass, history, args.trace_path)
    except (ValueError, OSError):
        return _report_error("storage_paths_overlap", operational=True)

    try:
        customer = authorize_demo_ticket(
            catalog, cultpass, history, account_id="cultpass", customer_id=args.demo_customer_id,
            ticket_id=args.ticket_id, channel=args.channel, inspect=args.inspect,
        )
    except TicketAccessError as error:
        return _report_error(str(error), operational=error.operational)

    if args.inspect and not checkpoints.exists():
        return _report_error("checkpoint_not_found")

    config = None
    thread_id = None
    if customer is not None:
        scope = (customer["account_id"], customer["external_user_id"], args.ticket_id)
        thread_id = sha256(json.dumps(scope).encode("utf-8")).hexdigest()
        config = {"configurable": {"thread_id": thread_id}}

    ticket: TicketInput = {
        "account_id": "cultpass", "text": redact_support_text(args.text or ""),
        "ticket_id": args.ticket_id, "channel": args.channel, "urgency": args.urgency,
    }
    if args.turn_id is not None:
        ticket["turn_id"] = args.turn_id
    if customer is not None:
        ticket["verified_customer"] = customer

    reviewer = None
    try:
        with ExitStack() as resources:
            if args.openai and not args.inspect:
                load_dotenv(ROOT / ".env")
                key = os.getenv("OPENAI_API_KEY")
                if not key:
                    parser.error("Set OPENAI_API_KEY in the local .env or environment before using --openai.")
                client = resources.enter_context(OpenAI(
                    api_key=key, base_url="https://api.openai.com/v1",
                    http_client=DefaultHttpxClient(trust_env=False), timeout=20, max_retries=0,
                ))
                reviewer = build_openai_reviewer(client, model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
            saver = None
            if config is not None:
                connection = resources.enter_context(closing(sqlite3.connect(checkpoints, check_same_thread=False)))
                saver = SqliteSaver(connection, serde=JsonPlusSerializer(allowed_msgpack_modules=()))
            graph = build_support_graph(
                catalog, cultpass_database_path=cultpass, memory_database_path=history,
                evidence_reviewer=reviewer, trace_path=args.trace_path, checkpointer=saver,
            )
            if args.inspect:
                snapshot = graph.get_state(config)
                if not snapshot.values:
                    return _report_error("checkpoint_not_found")
                if (
                    snapshot.next or snapshot.values.get("turn_input") is not None
                    or not snapshot.values.get("turn_complete")
                ):
                    return _report_error("checkpoint_incomplete", operational=True)
                checkpoint = {
                    key: snapshot.values[key] for key in (
                        "turn_id", "text", "classification", "decision", "retrieval", "account_evidence",
                        "response", "handoff", "provider_verdict", "memory_status", "trace_status",
                    ) if snapshot.values.get(key) is not None
                }
                print(json.dumps({"thread_id": thread_id, "checkpoint": checkpoint}, indent=2))
                return 0
            result = graph.invoke({"turn_input": ticket}, config) if saver is not None else graph.invoke(ticket)
    except sqlite3.Error:
        return _report_error("checkpoint_unavailable", operational=True)

    output = {
        key: result[key] for key in (
            "decision", "response", "handoff", "provider_verdict", "trace_status", "memory_status",
        ) if result.get(key) is not None
    }
    if thread_id is not None:
        output["thread_id"] = thread_id
    print(json.dumps(output, indent=2))
    return 1 if result["decision"]["next_step"] == "operational_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
