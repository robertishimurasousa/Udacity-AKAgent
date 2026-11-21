from contextlib import closing, contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Iterator, NamedTuple, TypedDict, cast

from agentic.agents.classifier import ClassifiedIntent


class MemoryScope(NamedTuple):
    account_id: str
    customer_id: str
    ticket_id: str


class MemoryContext(TypedDict):
    session_intents: list[ClassifiedIntent]
    recent_resolved_intents: list[str]


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


def _decode_intents(value: str) -> list[ClassifiedIntent]:
    decoded = json.loads(value)
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
                UNIQUE (account_id, customer_id, ticket_id, turn_id)
            )
        """)
        yield connection


def load_context(database: Path, scope: MemoryScope) -> MemoryContext:
    with _connect(database) as connection:
        last = connection.execute("""
            SELECT next_step, intents FROM support_turns
            WHERE account_id = ? AND customer_id = ? AND ticket_id = ?
            ORDER BY sequence DESC LIMIT 1
        """, scope).fetchone()
        previous = connection.execute("""
            SELECT intents FROM support_turns
            WHERE account_id = ? AND customer_id = ? AND ticket_id <> ? AND next_step = 'resolve'
            ORDER BY sequence DESC LIMIT 3
        """, scope).fetchall()

    return {
        "session_intents": _decode_intents(last[1]) if last and last[0] == "resolve" else [],
        "recent_resolved_intents": list(dict.fromkeys(
            intent["intent_id"] for (intents,) in previous for intent in _decode_intents(intents)
        )),
    }


def save_turn(
    database: Path,
    scope: MemoryScope,
    turn_id: str,
    next_step: str,
    intents: list[ClassifiedIntent],
) -> None:
    with _connect(database) as connection:
        connection.execute("""
            INSERT INTO support_turns (account_id, customer_id, ticket_id, turn_id, next_step, intents)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (account_id, customer_id, ticket_id, turn_id) DO NOTHING
        """, (*scope, turn_id, next_step, json.dumps(intents)))
