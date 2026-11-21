import re
from typing import Literal

from agentic.agents.resolver import SupportResponse


ResponseStyle = Literal["concise", "step_by_step"]

STEP_BY_STEP = re.compile(
    r"(?:please\s+)?(?:answer|respond|reply)\s+(?:in\s+)?step[- ]by[- ]step"
    r"|(?:please\s+use|i\s+prefer)\s+step[- ]by[- ]step\s+(?:answers|responses|replies)"
)
CONCISE = re.compile(
    r"(?:please\s+)?(?:keep (?:your )?(?:answers|responses|replies) concise|answer concisely)"
    r"|(?:please\s+use|i\s+prefer)\s+concise\s+(?:answers|responses|replies)"
)
STATEMENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
CONCISE_PHRASES = (
    ("Try tapping ", "Tap "),
    ("Make sure you're using ", "Use "),
    ("Be sure to arrive ", "Arrive "),
    ("If the problem persists, ", "If it persists, "),
)


def extract_preference(text: str) -> tuple[str, ResponseStyle | None]:
    remaining = []
    style: ResponseStyle | None = None
    for statement in STATEMENT_BOUNDARY.split(text.strip()):
        request = statement.casefold().rstrip(".!?")
        if STEP_BY_STEP.fullmatch(request):
            style = "step_by_step"
        elif CONCISE.fullmatch(request):
            style = "concise"
        else:
            remaining.append(statement)
    return " ".join(remaining), style


def format_response(response: SupportResponse, style: ResponseStyle | None) -> SupportResponse:
    if style is None:
        return response
    if style == "concise":
        text = "\n".join(response["text"].split("\n\n"))
        for phrase, shorter in CONCISE_PHRASES:
            text = text.replace(phrase, shorter)
        return {**response, "text": text}

    steps = [part.strip() for part in STATEMENT_BOUNDARY.split(response["text"]) if part.strip()]
    return {**response, "text": "\n".join(f"{index}. {step}" for index, step in enumerate(steps, 1))}
