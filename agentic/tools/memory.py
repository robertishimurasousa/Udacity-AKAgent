from contextlib import closing, contextmanager
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterator, Literal, NamedTuple, NotRequired, TypedDict, cast

from agentic.agents.classifier import ClassifiedIntent, TicketClassification
from agentic.agents.escalation import Handoff
from agentic.agents.preferences import ResponseStyle
from agentic.agents.provider import ReviewVerdict
from agentic.agents.resolver import SupportResponse


class MemoryScope(NamedTuple):
    account_id: str
    customer_id: str
    ticket_id: str


class MemoryContext(TypedDict):
    session_intents: list[ClassifiedIntent]
    recent_resolved_intents: list[str]
    history: list["HistoryTurn"]
    response_style: ResponseStyle | None


class StoredOutcome(TypedDict):
    classification: TicketClassification
    decision: dict[str, str]
    response: NotRequired[SupportResponse]
    handoff: NotRequired[Handoff]
    provider_verdict: NotRequired[ReviewVerdict]


class HistoryTurn(TypedDict):
    turn_id: str
    next_step: str
    customer_message: NotRequired[str]
    decision: NotRequired[dict[str, str]]
    response: NotRequired[SupportResponse]
    handoff: NotRequired[Handoff]


SECRET_VALUE = r"""(?:"(?:\\.|[^"\\])*(?:"|$)|'(?:\\.|[^'\\])*(?:'|$)|[^\s,;"']+)"""
SECRET = re.compile(
    r"\b(?:[a-z0-9_]*(?:api_key|token|secret|password)\s*[:=]\s*" + SECRET_VALUE
    + r"|bearer\s+[a-z0-9_.-]+|sk-[a-z0-9_-]{8,}|ghp_[a-z0-9]{8,})",
    re.IGNORECASE,
)
EMAIL = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")


def _redact(text: str) -> str:
    return EMAIL.sub("[redacted]", SECRET.sub("[redacted]", text))


def _redact_outcome(outcome: StoredOutcome) -> StoredOutcome:
    response = outcome.get("response")
    if response is None:
        return outcome
    return {
        **outcome,
        "response": {
            "text": _redact(response["text"]),
            "citations": [
                {**citation, "excerpt": _redact(citation["excerpt"])}
                for citation in response["citations"]
            ],
        },
    }


def _decode_outcome(value: str | None) -> StoredOutcome | None:
    if value is None:
        return None
    decoded = json.loads(value)
    if (
        not isinstance(decoded, dict)
        or not isinstance(decoded.get("classification"), dict)
        or not isinstance(decoded.get("decision"), dict)
        or decoded["decision"].get("next_step") not in ("resolve", "escalate")
        or not isinstance(decoded["decision"].get("reason"), str)
    ):
        raise ValueError("Invalid recorded outcome in the history database.")
    classification = decoded["classification"]
    _validate_intents(classification.get("intents"))
    if not all(isinstance(classification.get(key), bool) for key in (
        "human_support_requested", "sensitive_action_requested", "customer_data_required",
    )):
        raise ValueError("Invalid recorded classification in the history database.")

    if "provider_verdict" in decoded and decoded["provider_verdict"] not in (
        "approved", "rejected", "unavailable",
    ):
        raise ValueError("Invalid recorded provider verdict in the history database.")

    if decoded["decision"]["next_step"] == "resolve":
        response = decoded.get("response")
        if (
            "handoff" in decoded or not _has_text_fields(response, "text")
            or not isinstance(response.get("citations"), list)
            or not all(
                _has_text_fields(citation, "article_id", "intent_id", "title", "excerpt")
                for citation in response["citations"]
            )
            or (decoded["decision"]["reason"] == "evidence_ready" and not response["citations"])
        ):
            raise ValueError("Invalid recorded response in the history database.")
    else:
        handoff = decoded.get("handoff")
        if (
            "response" in decoded or not _has_text_fields(handoff, "reason", "summary")
            or not isinstance(handoff.get("actions"), list)
            or not all(isinstance(action, str) for action in handoff["actions"])
        ):
            raise ValueError("Invalid recorded handoff in the history database.")
    return cast(StoredOutcome, decoded)


def _has_text_fields(value: object, *fields: str) -> bool:
    return isinstance(value, dict) and all(isinstance(value.get(field), str) and value[field].strip() for field in fields)


def separate_memory_path(database: Path | str, *sources: Path | str | None) -> Path:
    history = Path(database).resolve()
    for source in sources:
        if source is None:
            continue
        source_path = Path(source).resolve()
        if history == source_path or (
            history.exists() and source_path.exists() and history.samefile(source_path)
        ):
            raise ValueError("The history database must be separate from the read-only databases.")
    return history


def _validate_intents(decoded: object) -> list[ClassifiedIntent]:
    if not isinstance(decoded, list) or any(
        not isinstance(intent, dict)
        or not isinstance(intent.get("intent_id"), str)
        or not isinstance(intent.get("category"), str)
        or not isinstance(intent.get("customer_data_required"), bool)
        or not isinstance(intent.get("terms"), list)
        or not all(isinstance(term, str) for term in intent["terms"])
        for intent in decoded
    ):
        raise ValueError("Invalid classified intents in the history database.")
    return cast(list[ClassifiedIntent], decoded)


def _decode_intents(value: str) -> list[ClassifiedIntent]:
    return _validate_intents(json.loads(value))


@contextmanager
def _connect(database: Path) -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS support_turns (
                sequence INTEGER PRIMARY KEY,
                account_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                ticket_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                next_step TEXT NOT NULL,
                intents TEXT NOT NULL,
                customer_message TEXT,
                outcome TEXT,
                UNIQUE (account_id, customer_id, ticket_id, turn_id)
            )
        """)
        columns = {column[1] for column in connection.execute("PRAGMA table_info(support_turns)")}
        if not {"customer_message", "outcome"}.issubset(columns):
            connection.execute("BEGIN IMMEDIATE")
            columns = {column[1] for column in connection.execute("PRAGMA table_info(support_turns)")}
            if "customer_message" not in columns:
                connection.execute("ALTER TABLE support_turns ADD COLUMN customer_message TEXT")
            if "outcome" not in columns:
                connection.execute("ALTER TABLE support_turns ADD COLUMN outcome TEXT")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS support_preferences (
                account_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                response_style TEXT NOT NULL CHECK (response_style IN ('concise', 'step_by_step')),
                PRIMARY KEY (account_id, customer_id)
            )
        """)
        yield connection


def load_context(
    database: Path, scope: MemoryScope, turn_id: str,
) -> tuple[MemoryContext, StoredOutcome | None]:
    with _connect(database) as connection:
        rows = connection.execute("""
            SELECT turn_id, next_step, intents, customer_message, outcome FROM support_turns
            WHERE account_id = ? AND customer_id = ? AND ticket_id = ?
            ORDER BY sequence
        """, scope).fetchall()
        previous = connection.execute("""
            SELECT intents FROM support_turns
            WHERE account_id = ? AND customer_id = ? AND ticket_id <> ? AND next_step = 'resolve'
            ORDER BY sequence DESC LIMIT 3
        """, scope).fetchall()
        preference = connection.execute("""
            SELECT response_style FROM support_preferences WHERE account_id = ? AND customer_id = ?
        """, (scope.account_id, scope.customer_id)).fetchone()

    history: list[HistoryTurn] = []
    replay = None
    last = next((row for row in reversed(rows) if row[1] != "preference"), None)
    for row_id, next_step, _, message, outcome in rows:
        stored = _decode_outcome(outcome)
        item: HistoryTurn = {"turn_id": row_id, "next_step": next_step}
        if message is not None:
            item["customer_message"] = message
        if stored is not None:
            item["decision"] = stored["decision"]
            if "response" in stored:
                item["response"] = stored["response"]
            if "handoff" in stored:
                item["handoff"] = stored["handoff"]
        history.append(item)
        if row_id == turn_id:
            replay = stored

    context: MemoryContext = {
        "session_intents": _decode_intents(last[2]) if last and last[1] == "resolve" else [],
        "recent_resolved_intents": list(dict.fromkeys(
            intent["intent_id"] for (intents,) in previous for intent in _decode_intents(intents)
        )),
        "history": history,
        "response_style": cast(ResponseStyle, preference[0]) if preference else None,
    }
    return context, replay


def save_turn(
    database: Path,
    scope: MemoryScope,
    *,
    turn_id: str,
    customer_message: str,
    outcome: StoredOutcome,
    requested_style: ResponseStyle | None,
) -> tuple[Literal["saved", "replayed", "legacy"], StoredOutcome | None]:
    stored = _redact_outcome(outcome)
    decision = stored["decision"]
    next_step = "preference" if decision["reason"] == "preference_saved" else decision["next_step"]
    with _connect(database) as connection:
        insert = connection.execute("""
            INSERT INTO support_turns (
                account_id, customer_id, ticket_id, turn_id, next_step, intents, customer_message, outcome
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (account_id, customer_id, ticket_id, turn_id) DO NOTHING
        """, (
            *scope, turn_id, next_step, json.dumps(stored["classification"]["intents"]),
            _redact(customer_message), json.dumps(stored),
        ))
        if insert.rowcount == 0:
            existing = connection.execute("""
                SELECT outcome FROM support_turns
                WHERE account_id = ? AND customer_id = ? AND ticket_id = ? AND turn_id = ?
            """, (*scope, turn_id)).fetchone()
            replay = _decode_outcome(existing[0])
            return ("replayed", replay) if replay else ("legacy", None)
        if requested_style is not None:
            connection.execute("""
                INSERT INTO support_preferences (account_id, customer_id, response_style)
                VALUES (?, ?, ?)
                ON CONFLICT (account_id, customer_id) DO UPDATE SET response_style = excluded.response_style
            """, (scope.account_id, scope.customer_id, requested_style))
    return "saved", stored
