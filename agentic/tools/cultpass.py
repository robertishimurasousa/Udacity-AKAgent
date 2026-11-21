from typing import Literal, NotRequired, TypedDict

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from data.models.cultpass import User


class VerifiedCustomer(TypedDict):
    account_id: str
    external_user_id: str


LookupStatus = Literal["ok", "invalid_input", "forbidden", "not_found", "unavailable"]


class AccountLookup(TypedDict):
    status: LookupStatus
    is_blocked: NotRequired[bool]


class SubscriptionLookup(TypedDict):
    status: LookupStatus
    has_subscription: NotRequired[bool]
    subscription_status: NotRequired[str]
    tier: NotRequired[str]
    monthly_quota: NotRequired[int]


class AccountEvidence(TypedDict):
    account: AccountLookup
    subscription: NotRequired[SubscriptionLookup]


def validate_customer_scope(
    account_id: str, customer: VerifiedCustomer,
) -> Literal["forbidden", "invalid_input"] | None:
    if not isinstance(customer, dict) or not isinstance(customer.get("account_id"), str):
        return "invalid_input"
    if account_id != "cultpass" or customer["account_id"] != account_id:
        return "forbidden"
    user_id = customer.get("external_user_id")
    if not isinstance(user_id, str) or not user_id.strip():
        return "invalid_input"
    return None


def lookup_account(engine: Engine, account_id: str, customer: VerifiedCustomer) -> AccountLookup:
    error = validate_customer_scope(account_id, customer)
    if error:
        return {"status": error}
    try:
        with Session(engine) as session:
            user = session.get(User, customer["external_user_id"])
            if user is None:
                return {"status": "not_found"}
            if user.is_blocked is None:
                return {"status": "unavailable"}
            return {"status": "ok", "is_blocked": user.is_blocked}
    except SQLAlchemyError:
        return {"status": "unavailable"}


def lookup_subscription(engine: Engine, account_id: str, customer: VerifiedCustomer) -> SubscriptionLookup:
    error = validate_customer_scope(account_id, customer)
    if error:
        return {"status": error}
    try:
        with Session(engine) as session:
            user = session.get(User, customer["external_user_id"])
            if user is None:
                return {"status": "not_found"}
            subscription = user.subscription
            if subscription is None:
                return {"status": "ok", "has_subscription": False}
            return {
                "status": "ok",
                "has_subscription": True,
                "subscription_status": subscription.status,
                "tier": subscription.tier,
                "monthly_quota": subscription.monthly_quota,
            }
    except SQLAlchemyError:
        return {"status": "unavailable"}
