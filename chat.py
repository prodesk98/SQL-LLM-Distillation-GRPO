import argparse
import json

import torch
import transformers
from transformers import TextStreamer

from utils import extract_sql, extract_think
from utils.constraints import (
    COLORED_BLUE, COLORED_GREEN, COLORED_RESET,
    REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END
)


parser = argparse.ArgumentParser(
    prog="uv run chat.py",
    description=(
        "Chatbot for SQL generation and reasoning.\n"
        "This program uses a language model to generate SQL queries based on user prompts.\n"
        "It supports multiple database backends (DuckDB, SQLite, PostgreSQL) and allows for SQL execution.\n"
    ),
    epilog="Developed by Protons · GitHub: https://github.com/prodesk98",
)

parser.add_argument(
    "--model",
    type=str,
    default="proton98/sql-llama3.2-3b-it-reasoning",
    help="Model name. Default is 'proton98/sql-llama3.2-3b-it-reasoning'.",
)
parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=1024,
    help="Maximum number of new tokens to generate. Default is 1024.",
)
parser.add_argument(
    "--db-uri",
    type=str,
    default=":memory:",
    help="Database URI. Default is ':memory:'.",
)
parser.add_argument(
    "--db-driver",
    type=str,
    default="duckdb",
    choices=["duckdb", "sqlite", "postgresql"],
    help="Database driver to use. Default is 'duckdb'.",
)
args = parser.parse_args()

if args.db_driver == "duckdb":
    try:
        import duckdb
    except ImportError:
        raise ImportError(
            "DuckDB driver is not installed. Please install it using 'pip install duckdb'."
        )
    db = duckdb.connect(args.db_uri)
elif args.db_driver == "sqlite":
    try:
        import sqlite3
    except ImportError:
        raise ImportError(
            "SQLite driver is not installed. Please install it using 'pip install sqlite3'."
        )
    db = sqlite3.connect(args.db_uri)
elif args.db_driver == "postgresql":
    try:
        import psycopg2
    except ImportError:
        raise ImportError(
            "PostgreSQL driver is not installed. Please install it using 'pip install psycopg2'."
        )
    db = psycopg2.connect(args.db_uri)
else:
    raise ValueError(
        f"Unsupported database driver: {args.db_driver}. Supported drivers are 'duckdb', 'sqlite' and 'postgresql'."
    )


def get_context_schemas() -> dict | None:
    """
    Get the full schema (tables, columns, data types, and constraints) from the database.
    :return: Dictionary structured schema
    """
    try:
        schema = {}

        if args.db_driver == "duckdb":
            tables = db.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall() # noqa
            for (table,) in tables:
                columns = db.execute(f"""SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}'""").fetchall() # noqa
                schema[table] = [{"name": col, "type": dtype} for col, dtype in columns]

        elif args.db_driver == "sqlite":
            tables = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall() # noqa
            for (table,) in tables:
                columns = db.execute(f"PRAGMA table_info('{table}')").fetchall()
                schema[table] = [{
                    "name": col[1],
                    "type": col[2],
                    "primary_key": bool(col[5])
                } for col in columns]

        elif args.db_driver == "postgresql":
            cursor = db.cursor()
            cursor.execute("""SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'""") # noqa
            tables = cursor.fetchall()
            for (table,) in tables:
                cursor.execute(f"""SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}'""") # noqa
                columns = cursor.fetchall()

                # Fetch primary keys
                cursor.execute(f"""SELECT a.attname FROM pg_index i JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) WHERE i.indrelid = '{table}'::regclass AND i.indisprimary""") # noqa
                pk_columns = {col[0] for col in cursor.fetchall()}

                schema[table] = [{
                    "name": col[0],
                    "type": col[1],
                    "primary_key": col[0] in pk_columns
                } for col in columns]

        else:
            raise ValueError(
                f"Unsupported database driver: {args.db_driver}. Supported drivers are 'duckdb', 'sqlite' and 'postgresql'."
            )

        return schema

    except Exception as e: # noqa
        print(f"Error getting context schema: {e}")
        return None


def execute_sql(sql_query: str):
    """
    Executes the SQL query and returns the result.
    :param sql_query:
    :return:
    """
    try:
        if args.db_driver == "duckdb":
            return db.execute(sql_query).fetchall()
        elif args.db_driver == "sqlite":
            return db.execute(sql_query).fetchall()
        elif args.db_driver == "postgresql":
            return db.cursor().execute(sql_query).fetchall()
        else:
            raise ValueError(
                f"Unsupported database driver: {args.db_driver}. Supported drivers are 'duckdb', 'sqlite' and 'postgresql'."
            )
    except Exception as e: # noqa
        print(f"Error executing SQL: {e}")
        return None

tokenizer = transformers.AutoTokenizer.from_pretrained(
    args.model,
    use_fast=True,
    padding_side="left",
    trust_remote_code=True,
)
streamer = TextStreamer(tokenizer, skip_prompt=True)

pipeline = transformers.pipeline(
    "text-generation",
    model=args.model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    streamer=streamer,
    pad_token_id=tokenizer.eos_token_id,
)

last_prompt: str | None = None
last_sql: str | None = None

while True:
    try:
        context: str | None = json.dumps(get_context_schemas())
        prompt = input("Prompt: ")
        command = prompt.lower()
        if command == "/exit":
            break
        elif command == "/retry":
            if last_prompt:
                prompt = last_prompt
            else:
                print("No last prompt found.")
        elif command == "/execute":
            if last_sql:
                result = execute_sql(last_sql)
                print(f"SQL Result: {result}")
            else:
                print("No SQL query found to execute.")
            continue
        elif command == "help" or command == "/help":
            print(
                "Available commands:\n"
                "/exit - Exit the program\n"
                "/retry - Retry the last prompt\n"
                "/execute - Execute the last SQL query\n"
                "/context - Show the context of the database\n"
                "/clear - Clear the context\n"
                "/last - Show the last SQL query\n"
                "/help - Show this help message"
            )
            continue
        elif command == "/context":
            print(context)
            continue
        elif command == "/clear":
            print("Clearing context...")
            db.execute("DROP TABLE IF EXISTS context") # noqa
            continue
        elif command == "/last":
            if last_sql:
                print(f"Last SQL: {last_sql}")
            else:
                print("No last SQL found.")
            continue

        messages = [
            {
                "role": "system",
                "content": f"You are an expert in writing optimized SQL queries.\n"
                  f"Think about the problem and provide your working out.\n"
                  f"Place it between {REASONING_START} and {REASONING_END}.\n"
                  f"Then, provide your solution between {SOLUTION_START}{SOLUTION_END}\n\n"
                  f"Context: {context}"
            },
            {"role": "user", "content": prompt},
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=args.max_new_tokens,
        )
        content = outputs[0]["generated_text"][-1]['content']
        sql = extract_sql(content)
        think = extract_think(content)

        if sql is None:
            print("No SQL query found in the output.")
            continue

        print(
            f"{COLORED_BLUE}{think}{COLORED_RESET}\n"
            f"{COLORED_GREEN}{sql}{COLORED_RESET}"
        )
        last_prompt = prompt
        last_sql = sql
    except KeyboardInterrupt:
        print("Exiting...")
        break
    except Exception as e:
        print(f"Error: {e}")
        continue