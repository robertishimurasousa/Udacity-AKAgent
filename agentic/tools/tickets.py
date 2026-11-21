from contextlib import closing
from pathlib import Path
import sqlite3
from uuid import uuid4

from agentic.tools.cultpass import VerifiedCustomer, lookup_account
from agentic.tools.memory import legacy_ticket_owners
from agentic.tools.sqlite import read_only_engine


class TicketAccessError(Exception):
    def __init__(self, reason: str, *, operational: bool = False):
        super().__init__(reason)
        self.operational = operational


def authorize_demo_ticket(
    catalog: Path, cultpass: Path, history: Path, *,
    account_id: str, customer_id: str | None, ticket_id: str, channel: str, inspect: bool = False,
) -> VerifiedCustomer | None:
    """Resolve the active synthetic profile and validate or claim the ticket."""
    customer = None
    if customer_id is not None:
        customer = {"account_id": account_id, "external_user_id": customer_id}
        profile = lookup_account(read_only_engine(cultpass), account_id, customer)
        if profile["status"] == "unavailable":
            raise TicketAccessError("profile_catalog_unavailable", operational=True)
        if profile["status"] != "ok":
            raise TicketAccessError("profile_not_found")

    claim = customer_id is not None and not inspect
    mode = "rw" if claim else "ro"
    try:
        with closing(sqlite3.connect(f"{catalog.resolve().as_uri()}?mode={mode}", uri=True)) as db, db:
            if claim:
                # Serialize the ownership check and claim across separate CLI processes.
                db.execute("PRAGMA foreign_keys = ON")
                db.execute("BEGIN IMMEDIATE")
            owner = db.execute("""
                SELECT tickets.account_id, users.account_id, users.external_user_id
                FROM tickets LEFT JOIN users ON tickets.user_id = users.user_id
                WHERE tickets.ticket_id = ?
            """, (ticket_id,)).fetchone()
            if owner is not None:
                if customer is None or owner != (account_id, account_id, customer_id):
                    raise TicketAccessError("ticket_forbidden")
                return customer

            try:
                legacy_owners = legacy_ticket_owners(history, account_id, ticket_id)
            except sqlite3.Error as error:
                raise TicketAccessError("history_unavailable", operational=True) from error
            if legacy_owners and (customer is None or legacy_owners != [customer_id]):
                raise TicketAccessError("ticket_forbidden")
            if inspect and not legacy_owners:
                raise TicketAccessError("ticket_not_found")
            if customer_id is not None and not inspect:
                _claim_ticket(db, account_id, customer_id, ticket_id, channel)
    except sqlite3.Error as error:
        raise TicketAccessError("ticket_catalog_unavailable", operational=True) from error
    return customer


def _claim_ticket(
    db: sqlite3.Connection, account_id: str, customer_id: str, ticket_id: str, channel: str,
) -> None:
    db.execute("""
        INSERT INTO users (user_id, account_id, external_user_id, user_name, created_at, updated_at)
        VALUES (?, ?, ?, 'CultPass customer', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT (account_id, external_user_id) DO NOTHING
    """, (uuid4().hex, account_id, customer_id))
    db.execute("""
        INSERT INTO tickets (ticket_id, account_id, user_id, channel, created_at)
        SELECT ?, account_id, user_id, ?, CURRENT_TIMESTAMP
        FROM users WHERE account_id = ? AND external_user_id = ?
    """, (ticket_id, channel, account_id, customer_id))
