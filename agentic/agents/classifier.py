from collections.abc import Iterator
from dataclasses import dataclass
import re
from typing import TypedDict
from unicodedata import normalize

from agentic.agents.knowledge import SearchIntent


class ClassifiedIntent(SearchIntent):
    category: str
    customer_data_required: bool


class TicketClassification(TypedDict):
    intents: list[ClassifiedIntent]
    human_support_requested: bool
    sensitive_action_requested: bool
    customer_data_required: bool


@dataclass(frozen=True)
class IntentRule:
    intent_id: str
    category: str
    pattern: str
    terms: tuple[str, ...]


# Specific intents take priority over general topics within a clause.
INTENT_RULES = (
    IntentRule("account_status", "access", r"\b(?:account (?:is )?blocked|blocked account|account status)\b",
               ("account", "blocked")),
    IntentRule("subscription_status", "billing", r"\b(?:subscription status|plan status|monthly quota)\b",
               ("subscription", "quota")),
    IntentRule("refund", "billing", r"\brefunds?\b", ("refund", "review")),
    IntentRule("duplicate_charge", "billing", r"\b(?:duplicate charge|charged twice)\b", ("duplicate", "charge")),
    IntentRule("payment_declined", "billing", r"\b(?:declined|payment failed)\b", ("declined", "payment")),
    IntentRule("check_in_code", "technical_support", r"\bqr(?: code)?\b", ("qr", "code")),
    IntentRule("app_crash", "technical_support", r"\b(?:crash(?:es|ed|ing)?|app closes)\b", ("app", "crash")),
    IntentRule("login", "access", r"\b(?:log[ -]?in|sign[ -]?in|password)\b", ("password", "email")),
    IntentRule("booking_change", "reservations", r"\b(?:cancel|reschedule|change)\b.*\b(?:booking|reservation)\b",
               ("reservation", "change")),
    IntentRule("booking", "reservations", r"\b(?:reserv(?:e|ations?)|book(?:ing)?|events?)\b",
               ("event", "reservation")),
    IntentRule("subscription_change", "billing", r"\b(?:pause|cancel)\b.*\b(?:plan|subscription)\b",
               ("cancel", "subscription")),
    IntentRule("subscription", "billing", r"\b(?:subscription|plan benefits)\b", ("subscription", "benefits")),
)

SENTENCE_BOUNDARY = re.compile(r"[.!?;\n]+")
CONJUNCTION = re.compile(r"\band\b(?=\s+\w+\s+\w)")
CHANGE_VERBS = "cancel|pause|change|reset|reschedule"
SHARED_ACTION_OBJECT = re.compile(rf"\b({CHANGE_VERBS})\s+and\s+(?=(?:{CHANGE_VERBS})\s+(?:my|the|this)\b)")
HOW_TO_QUESTION = re.compile(r"^(?:how\b|(?:can|could|would) you (?:explain|tell me|show me) how\b)")
COURTESY = re.compile(r"(?:hi|hello|hey|thanks|thank you|good morning|good afternoon|good evening)(?: there)?")
HUMAN_REQUEST = re.compile(
    r"\b(?:(?:speak|talk) (?:to|with)|connect me to|transfer me to|want|need)\s+"
    r"(?:a\s+)?(?:human|person|support agent|representative)\b"
)
SENSITIVE_ACTION = (
    r"(?:"
    r"refund\s+(?:this|that|my|the|me)\b|(?:issue|process)\s+(?:a|the|my)\s+refund\b|"
    r"(?:give|send)\s+(?:me|us)\s+(?:a|the|my)\s+refund\b|"
    rf"(?:{CHANGE_VERBS})\b[^.!?;]*\b"
    r"(?:plan|subscription|account|email|password|payment|reservation|booking)\b)"
)
DIRECT_REQUEST = re.compile(
    r"\b(?:can you|could you|would you|please|i (?:want|need|would like) you to)\s*,?\s+" + SENSITIVE_ACTION
)
IMPERATIVE_REQUEST = re.compile(SENSITIVE_ACTION)


def _clauses(text: str) -> Iterator[tuple[str, bool]]:
    for sentence in SENTENCE_BOUNDARY.split(text):
        sentence = SHARED_ACTION_OBJECT.sub(r"\1 or ", sentence.strip())
        shares_how_to = bool(HOW_TO_QUESTION.match(sentence))
        for index, clause in enumerate(CONJUNCTION.split(sentence)):
            yield clause.strip(" ,\t"), shares_how_to and index > 0


def classify_ticket(text: str) -> TicketClassification:
    """Classify the English CultPass topics supported by the initial workflow."""
    text = normalize("NFKC", text).casefold()
    by_id: dict[str, ClassifiedIntent] = {}
    sensitive_action_requested = False
    for clause, inherits_how_to in _clauses(text):
        if not clause or COURTESY.fullmatch(clause):
            continue
        if DIRECT_REQUEST.search(clause) or (not inherits_how_to and IMPERATIVE_REQUEST.match(clause)):
            sensitive_action_requested = True
        rule = next((rule for rule in INTENT_RULES if re.search(rule.pattern, clause)), None)
        if rule is None:
            by_id["unknown"] = {
                "intent_id": "unknown", "category": "unknown", "terms": [], "customer_data_required": False,
            }
            continue
        requires_account = rule.intent_id == "account_status" or (
            rule.intent_id == "subscription_status" and bool(re.search(r"\b(?:my|mine|me|i)\b", clause))
        )
        prior_intent = by_id.get(rule.intent_id)
        by_id[rule.intent_id] = {
            "intent_id": rule.intent_id,
            "category": rule.category,
            "terms": list(rule.terms),
            "customer_data_required": requires_account or (
                prior_intent["customer_data_required"] if prior_intent else False
            ),
        }
    return {
        "intents": list(by_id.values()),
        "human_support_requested": bool(HUMAN_REQUEST.search(text)),
        "sensitive_action_requested": sensitive_action_requested,
        "customer_data_required": any(intent["customer_data_required"] for intent in by_id.values()),
    }
