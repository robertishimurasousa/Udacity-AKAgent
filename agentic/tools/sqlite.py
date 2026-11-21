from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL
from sqlalchemy.pool import NullPool


def read_only_engine(database_path: Path | str) -> Engine:
    url = URL.create(
        "sqlite+pysqlite",
        database=Path(database_path).resolve().as_uri(),
        query={"mode": "ro", "uri": "true"},
    )
    return create_engine(url, poolclass=NullPool)
