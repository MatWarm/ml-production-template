import sqlite3
from sqlalchemy import inspect
import os


def test_sql_db_exists():
    conn = sqlite3.connect("data/data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]
    print("Tables présentes dans la base de données SQLite :")
    for table in tables:
        print(f"- {table}")
    assert "annotation" in tables
    assert "abbreviation" in tables
    assert "grammar" in tables
    assert "morphosyntactic" in tables
    assert "term_banks" in tables
    conn.close()


if __name__ == "__main__":
    test_sql_db_exists()
