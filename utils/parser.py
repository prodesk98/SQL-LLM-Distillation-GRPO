import re

import duckdb

from utils.constraints import (
    SOLUTION_START, SOLUTION_END,
    REASONING_START, REASONING_END
)


sql_match_format = re.compile(rf"{SOLUTION_START}(.*?){SOLUTION_END}", re.DOTALL)
reasoning_match_format = re.compile(rf"{REASONING_START}(.*?){REASONING_END}", re.DOTALL)


def extract_sql(text: str) -> str | None:
    """
    Extracts the SQL using regex from the generated text.
    :param text:
    :return:
    """
    sql_match = sql_match_format.search(text)
    if sql_match:
        return sql_match.group(1).strip()


def extract_think(text: str) -> str | None:
    """
    Extracts the think using regex from the generated text.
    :param text:
    :return:
    """
    think_match = reasoning_match_format.search(text)
    if think_match:
        return think_match.group(1).strip()


def extract_schema_tables(sql: str) -> str:
    """
    Extracts the SQL using regex from the generated text.
    :param sql:
    :return:
    """
    try:
        con = duckdb.connect(database=':memory:')
        sql_statements = [stmt.strip() for stmt in sql.strip().split(';') if stmt.strip()]
        for statement in sql_statements:
            con.execute(statement)

        tables_info = con.execute(
            "SELECT table_schema, table_name FROM information_schema.tables WHERE table_type='BASE TABLE';").fetchall() # noqa

        context_lines = []
        for schema, table in tables_info:
            context_lines.append(f"Table: {table}")
            columns = con.execute(f"PRAGMA table_info('{table}')").fetchall()
            for col in columns:
                col_name, col_type = col[1], col[2]
                context_lines.append(f"  - {col_name}: {col_type}")
            context_lines.append("")
        return "\n".join(context_lines).strip()
    except Exception as e:
        return str(e)

