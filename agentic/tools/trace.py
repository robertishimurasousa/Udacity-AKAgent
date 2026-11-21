from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Literal


TraceStatus = Literal["recorded", "skipped", "unavailable"]
OPAQUE_ID = re.compile(r"[A-Za-z0-9_.:-]{1,80}")


def valid_trace_id(value: str | None) -> bool:
    return (
        isinstance(value, str)
        and OPAQUE_ID.fullmatch(value) is not None
        and not value.lower().startswith(("sk-", "ghp_", "github_pat_", "xox", "akia"))
    )


def append_event(
    path: Path, ticket_id: str | None, turn_id: str | None, event: str, details: dict[str, object],
) -> TraceStatus:
    if not (valid_trace_id(ticket_id) and valid_trace_id(turn_id)):
        return "skipped"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticket_id": ticket_id, "turn_id": turn_id, "event": event, **details,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
    except OSError:
        return "unavailable"
    return "recorded"
