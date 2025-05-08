import duckdb


def validate_sql_query(sql_query: str, context: str) -> tuple[bool, str]:
    """
    Validate the SQL query against the provided context.

    Args:
        sql_query (str): The SQL query to validate.
        context (str): The context in which the SQL query is executed
            (e.g., table schema, data).
    Returns:
        tuple[bool, str]: Whether the SQL query is valid.
    """
    try:
        con = duckdb.connect(database=':memory:')

        context_statements = [stmt.strip() for stmt in context.strip().split(';') if stmt.strip()]
        for statement in context_statements:
            con.execute(statement)

        con.execute(sql_query)
        _ = con.fetchall()

        return True, "SQL query is valid and executed successfully."
    except Exception as e:
        return False, str(e)
