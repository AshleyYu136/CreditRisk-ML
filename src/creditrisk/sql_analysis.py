"""Execute versioned analytical SQL via SQLAlchemy's embedded DuckDB dialect."""

import pandas as pd
from sqlalchemy import create_engine, text

from creditrisk.config import ROOT


def analyze(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    engine = create_engine("duckdb:///:memory:")
    try:
        with engine.begin() as connection:
            frame.to_sql("clients", connection, index=False, if_exists="replace")
            return {
                path.stem: pd.read_sql_query(text(path.read_text()), connection)
                for path in sorted((ROOT / "sql").glob("*.sql"))
            }
    finally:
        engine.dispose()
