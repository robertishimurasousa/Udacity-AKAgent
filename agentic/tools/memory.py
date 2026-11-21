from contextlib import closing, contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Iterator, NamedTuple, TypedDict

from agentic.agents.classifier import ClassifiedIntent


class MemoryScope(NamedTuple):
    account_id: str
    customer_id: str
    ticket_id: str


class MemoryContext(TypedDict):
    session_intents: list[ClassifiedIntent]
    recent_resolved_intents: list[str]


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
        "session_intents": json.loads(last[1]) if last and last[0] == "resolve" else [],
        "recent_resolved_intents": list(dict.fromkeys(
            intent["intent_id"] for (intents,) in previous for intent in json.loads(intents)
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
