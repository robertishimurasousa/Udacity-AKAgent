from collections.abc import Sequence

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from data.models.udahub import Knowledge


def read_account_articles(engine: Engine, account_id: str) -> Sequence[Knowledge]:
    statement = select(Knowledge).where(Knowledge.account_id == account_id)
    with Session(engine) as session:
        return session.scalars(statement).all()
