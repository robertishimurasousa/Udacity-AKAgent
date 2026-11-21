import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import DefaultHttpxClient, OpenAI

from agentic.agents.provider import DEFAULT_MODEL, build_openai_reviewer
from agentic.support_workflow import TicketInput, build_support_graph
from agentic.tools.trace import valid_trace_id


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a synthetic CultPass ticket through UDA-Hub.")
    parser.add_argument("text")
    parser.add_argument("--ticket-id", required=True)
    parser.add_argument("--turn-id", required=True)
    parser.add_argument("--channel", choices=("chat", "email"), default="chat")
    parser.add_argument("--urgency", choices=("normal", "high"), default="normal")
    parser.add_argument("--demo-customer-id", help="Existing ID from the synthetic CultPass setup database.")
    parser.add_argument("--openai", action="store_true", help="Review grounded drafts using OpenAI.")
    parser.add_argument("--trace-path", type=Path, default=ROOT / ".logs/uda-hub.jsonl")
    args = parser.parse_args()
    if not (valid_trace_id(args.ticket_id) and valid_trace_id(args.turn_id)):
        parser.error("Use opaque ticket and turn IDs of up to 80 letters, digits, dots, colons, dashes or underscores.")
    ticket: TicketInput = {
        "account_id": "cultpass", "text": args.text, "ticket_id": args.ticket_id,
        "turn_id": args.turn_id, "channel": args.channel, "urgency": args.urgency,
    }
    if args.demo_customer_id:
        ticket["verified_customer"] = {
            "account_id": "cultpass", "external_user_id": args.demo_customer_id,
        }

    reviewer = None
    with ExitStack() as resources:
        if args.openai:
            load_dotenv(ROOT / ".env")
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                parser.error("Set OPENAI_API_KEY in the local .env or environment before using --openai.")
            client = resources.enter_context(OpenAI(
                api_key=key, base_url="https://api.openai.com/v1",
                http_client=DefaultHttpxClient(trust_env=False), timeout=20, max_retries=0,
            ))
            reviewer = build_openai_reviewer(client, model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
        graph = build_support_graph(
            ROOT / "data/core/udahub.db",
            cultpass_database_path=ROOT / "data/external/cultpass.db",
            memory_database_path=ROOT / "data/core/udahub_history.db",
            evidence_reviewer=reviewer, trace_path=args.trace_path,
        )
        result = graph.invoke(ticket)

    output = {
        key: result[key] for key in (
            "decision", "response", "handoff", "provider_verdict", "trace_status", "memory_status",
        ) if key in result
    }
    print(json.dumps(output, indent=2))
    return 1 if result["decision"]["next_step"] == "operational_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
