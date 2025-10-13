from flask import Flask, render_template, request, jsonify, send_file, session
import os
import json
import pandas as pd
import zipfile
from io import StringIO, BytesIO
import google.generativeai as genai
import re
import uuid
from datetime import datetime
import sqlparse
from werkzeug.utils import secure_filename
import logging
from typing import Dict, List, Any, Optional
from config import get_config
import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2 import sql
from psycopg2.extras import execute_values
import os


# Optional Langfuse import - will work without it

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context

    LANGFUSE_AVAILABLE = True
    print("✅ Langfuse loaded successfully")
except ImportError:
    print("⚠️ Langfuse not available - continuing without observability")
    LANGFUSE_AVAILABLE = False


    # Create dummy decorators if Langfuse is not available
    def observe(name=None):
        def decorator(func):
            return func

        return decorator


    class DummyContext:
        def update_current_trace(self, **kwargs):
            pass


    langfuse_context = DummyContext()

app = Flask(__name__)
# Load config from your Base/Dev/Prod (the class you shared)

app.config.from_object(get_config())

pool = None
_db_cfg = None

def init_db_pool(app):
    """Initialize database connection pool with graceful error handling"""
    global pool, _db_cfg
    _db_cfg = {
        'host': app.config["DB_HOST"],
        'port': app.config["DB_PORT"],
        'database': app.config["DB_NAME"],
        'user': app.config["DB_USER"],
        'password': app.config["DB_PASSWORD"]
    }
    try:
        pool = pg_pool.SimpleConnectionPool(minconn=1, maxconn=20, **_db_cfg)
        logger.info("✅ PostgreSQL connection pool created successfully")
    except Exception as e:
        logger.warning(f"⚠️ Failed to create database connection pool: {e}")
        logger.warning("Starting without database connection. Save-to-DB will be disabled.")
        pool = None


init_db_pool(app)

app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

from contextlib import contextmanager

@contextmanager
def get_conn():
    """Yield a pooled connection if available; otherwise raise a helpful error."""
    if pool is None:
        raise RuntimeError("Database pool is not initialized")
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)

def _bulk_insert_table(conn, table_name: str, rows: list[dict]):
    """Bulk insert list of dict rows into a table using execute_values."""
    if not rows:
        return 0

    # Use columns from the first row and preserve that order
    columns = list(rows[0].keys())

    # Build safe SQL: INSERT INTO "table" ("c1","c2",...) VALUES %s
    q = sql.SQL("INSERT INTO {tbl} ({cols}) VALUES %s").format(
        tbl=sql.Identifier(table_name),
        cols=sql.SQL(", ").join(sql.Identifier(c) for c in columns)
    )

    values = [tuple(r.get(c) for c in columns) for r in rows]

    with conn.cursor() as cur:
        execute_values(cur, q.as_string(conn), values)  # server-side formatting
    return len(values)

# Configure Gemini API
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    logger.error("GOOGLE_API_KEY environment variable not set!")
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

genai.configure(api_key=api_key)

print(f"Google Key GOOGLE_API_KEY = {api_key}")



#Table DB Mappers

# --- Type mappers ---

def _pg_type_from_schema(col_type_raw: str) -> str:
    """
    Map a MySQL-like type string to a Postgres type.
    Examples: INT, BIGINT, VARCHAR(100), DECIMAL(10,2), DATETIME, DATE, ENUM('a','b')
    """
    t = (col_type_raw or "").strip().upper()

    # Basic sized types
    if t.startswith("VARCHAR"):
        return t.replace("VARCHAR", "VARCHAR")
    if t.startswith("CHAR"):
        return t.replace("CHAR", "CHAR")
    if t.startswith("DECIMAL") or t.startswith("NUMERIC"):
        return t  # Postgres supports DECIMAL/NUMERIC(p,s)
    if t.startswith("INT(") or t == "INT" or t == "INTEGER":
        return "INTEGER"
    if t in ("TINYINT", "SMALLINT"):
        return "SMALLINT"
    if t in ("BIGINT",):
        return "BIGINT"
    if t in ("FLOAT",):
        return "REAL"
    if t in ("DOUBLE", "DOUBLE PRECISION"):
        return "DOUBLE PRECISION"
    if t in ("BOOL", "BOOLEAN", "TINYINT(1)"):
        return "BOOLEAN"
    if t in ("DATE",):
        return "DATE"
    if t in ("DATETIME", "TIMESTAMP"):
        return "TIMESTAMP"
    if t.startswith("ENUM("):
        # We'll implement as TEXT with a CHECK constraint later (simpler than creating a type)
        return "TEXT"

    # Fallbacks
    if t.startswith("TEXT") or t == "LONGTEXT" or t == "MEDIUMTEXT":
        return "TEXT"
    if t.startswith("BLOB"):
        return "BYTEA"

    # default
    return "TEXT"


from typing import Optional

try:
    # psycopg3
    from psycopg import sql as psy_sql
except Exception:
    psy_sql = None

def _default_sql_fragment(default_raw: Optional[str], conn=None) -> str:
    """
    Convert an incoming DEFAULT value into a Postgres-friendly fragment.

    - Leaves CURRENT_TIMESTAMP / CURRENT_DATE / CURRENT_TIME as keywords.
    - Keeps NOW() as a function call.
    - Numeric strings are emitted as-is.
    - Other strings are single-quoted (psycopg literal when possible; manual escape otherwise).
    """
    if not default_raw:
        return ""

    d = default_raw.strip()
    up = d.upper()

    # Keywords (no parentheses)
    if up in {"CURRENT_TIMESTAMP", "CURRENT_DATE", "CURRENT_TIME"}:
        return f" DEFAULT {up}"

    # Functions that require parentheses
    if up == "NOW()":
        return " DEFAULT now()"

    # Numeric?
    try:
        float(d)
        return f" DEFAULT {d}"
    except Exception:
        pass

    # String literal: prefer psycopg quoting if available & we have a connection
    if psy_sql and conn is not None:
        return " DEFAULT " + psy_sql.Literal(d).as_string(conn)

    # Fallback: basic SQL single-quote escaping
    quoted = "'" + d.replace("'", "''") + "'"
    return f" DEFAULT {quoted}"



def _infer_pg_type_from_value(v) -> str:
    """Infer a reasonable Postgres type from a Python value (used for missing columns)."""
    if v is None:
        return "TEXT"
    if isinstance(v, bool):
        return "BOOLEAN"
    if isinstance(v, int):
        # choose BIGINT to be safe
        return "BIGINT"
    if isinstance(v, float):
        return "DOUBLE PRECISION"
    if isinstance(v, (datetime.date, datetime.datetime)):
        return "TIMESTAMP"
    # Try ISO date/time strings
    if isinstance(v, str):
        s = v.strip()
        # crude ISO checks
        try:
            if len(s) == 10 and s[4] == "-" and s[7] == "-":
                datetime.date.fromisoformat(s)
                return "DATE"
            if " " in s or "T" in s:
                # try timestamp
                s2 = s.replace("T", " ")
                datetime.datetime.fromisoformat(s2)
                return "TIMESTAMP"
        except Exception:
            pass
        # length-based varchar cap
        if len(s) <= 255:
            return "VARCHAR(255)"
        return "TEXT"
    # default
    return "TEXT"


#Helpers for data persistance


def _build_column_sql_frag(col: dict) -> str:
    """
    Build a Postgres column definition from schema parser output:
    col = {'name': 'id', 'type': 'INT', 'constraints': ['PRIMARY KEY', 'AUTO_INCREMENT', 'NOT NULL', 'DEFAULT ...']}
    """
    name = col['name']
    col_type = _pg_type_from_schema(col.get('type', 'TEXT'))
    constraints = col.get('constraints', [])
    parts = [f'"{name}" {col_type}']

    # AUTO_INCREMENT → identity (only if integer-like)
    if any('AUTO_INCREMENT' in c.upper() for c in constraints):
        if col_type in ("SMALLINT", "INTEGER", "BIGINT"):
            parts = [f'"{name}" {col_type} GENERATED BY DEFAULT AS IDENTITY']
        # else: ignore

    # NOT NULL
    if any('NOT NULL' in c.upper() for c in constraints):
        parts.append("NOT NULL")

    # UNIQUE
    # if any(c.upper() == 'UNIQUE' for c in constraints):
    #     parts.append("UNIQUE")

    # DEFAULT
    for c in constraints:
        if c.upper().startswith("DEFAULT"):
            # c like "DEFAULT xyz"
            default_val = c.split(" ", 1)[1] if " " in c else None
            parts.append(_default_sql_fragment(default_val))

    # PRIMARY KEY (column-level)
    if any('PRIMARY KEY' in c.upper() for c in constraints):
        parts.append("PRIMARY KEY")

    return " ".join(parts)


def _create_table_if_missing(conn, table_name: str, table_schema: dict, sample_rows: list[dict]):
    """
    Create the table if not exists using parsed schema (columns & constraints).
    If table exists, we don't touch it here—missing columns are handled separately.
    """
    columns = table_schema.get("columns", [])
    column_frags = []

    for col in columns:
        column_frags.append(_build_column_sql_frag(col))

    # If no columns were parsed (unlikely), infer from sample row keys
    if not column_frags and sample_rows:
        first = sample_rows[0]
        for k, v in first.items():
            inferred = _infer_pg_type_from_value(v)
            column_frags.append(f'"{k}" {inferred}')

    stmt = f'CREATE TABLE IF NOT EXISTS "{table_name}" (\n  ' + ",\n  ".join(column_frags) + "\n)"

    with conn.cursor() as cur:
        cur.execute(stmt)


def _table_exists(conn, table_name: str) -> bool:
    q = """
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = current_schema() AND table_name = %s
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(q, (table_name,))
        return cur.fetchone() is not None


def _get_existing_columns(conn, table_name: str) -> set[str]:
    q = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = %s
    """
    with conn.cursor() as cur:
        cur.execute(q, (table_name,))
        return {r[0] for r in cur.fetchall()}


def _add_missing_columns(conn, table_name: str, rows: list[dict], known_schema_cols: dict[str, dict] | None):
    """
    Add columns that are present in data but not in the DB table.
    Choose type by:
      1) schema if available
      2) infer from first non-null value in data
    """
    if not rows:
        return

    existing = _get_existing_columns(conn, table_name)
    candidate_cols = set()
    for r in rows:
        candidate_cols.update(r.keys())

    missing = [c for c in candidate_cols if c not in existing]
    if not missing:
        return

    with conn.cursor() as cur:
        for col in missing:
            # Prefer schema type if present
            pg_type = None
            if known_schema_cols:
                schema_col = known_schema_cols.get(col)
                if schema_col:
                    pg_type = _pg_type_from_schema(schema_col.get('type', 'TEXT'))

            if not pg_type:
                # infer from first non-null
                v = None
                for r in rows:
                    if r.get(col) is not None:
                        v = r[col]
                        break
                pg_type = _infer_pg_type_from_value(v)

            alter = sql.SQL('ALTER TABLE {t} ADD COLUMN {c} {typ}').format(
                t=sql.Identifier(table_name),
                c=sql.Identifier(col),
                typ=sql.SQL(pg_type)
            )
            cur.execute(alter.as_string(conn))





# Configure Langfuse if available
langfuse = None
if LANGFUSE_AVAILABLE:
    try:
        langfuse = Langfuse(
            secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
            public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
            host=os.getenv('LANGFUSE_HOST', 'http://localhost:3000')
        )
        logger.info("✅ Langfuse configured successfully")
    except Exception as e:
        logger.warning(f"⚠️ Langfuse configuration failed: {e}")
        LANGFUSE_AVAILABLE = False

# Global storage for generated data (use database in production)
generated_data = {}
uploaded_schemas = {}
chat_history = {}


import google
print(f"Checking Google GenAI version {google.generativeai.__version__}")

def _coerce_text_from_response(resp) -> str:
    # 1) Prefer .text if present
    text = (getattr(resp, "text", "") or "").strip()
    if text:
        return text
    # 2) Fallback to candidates parts
    for c in getattr(resp, "candidates", []) or []:
        content = getattr(c, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", []) or []
        joined = "".join(getattr(p, "text", "") for p in parts if getattr(p, "text", None)).strip()
        if joined:
            return joined
    return ""  # nothing found


def _strip_code_fences_loose(text: str) -> str:
    # Remove a starting fence line even if no closing fence was produced
    # e.g., "```json\n{...}" -> "{...}"
    text = re.sub(r'^\s*```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    # If there is a trailing fence, drop it (and any trailing text after it)
    text = re.sub(r'\s*```[\s\S]*$', '', text)
    return text.strip()



def _safe_json_from_response(resp) -> str:
    text = (getattr(resp, "text", "") or "").strip()
    if not text:
        for c in getattr(resp, "candidates", []) or []:
            parts = getattr(c, "content", {}).get("parts", [])
            if parts:
                text = "".join(getattr(p, "text", "") for p in parts).strip()
                if text:
                    break
    return text


def _extract_first_json_block(text: str) -> str:
    # Strip code fences if present
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        return m.group(1).strip()

    # Fallback: take first well-formed {...} or [...] block
    start = min([p for p in (text.find("{"), text.find("[")) if p != -1] or [-1])
    if start == -1:
        return text  # no JSON-looking content

    stack = []
    for i, ch in enumerate(text[start:], start):
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack: break
            stack.pop()
            if not stack:
                return text[start:i + 1]
    return text  # best effort


class DataGenerator:
    def __init__(self):
        try:
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info("✅ Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini model: {e}")
            raise

    def parse_ddl_schema(self, ddl_content: str) -> Dict[str, Any]:
        """Enhanced DDL parser that handles MySQL syntax, AUTO_INCREMENT, ENUM, and complex schemas"""
        try:
            # Clean up the DDL content
            ddl_content = ddl_content.strip()

            # Remove comments
            ddl_content = re.sub(r'--.*$', '', ddl_content, flags=re.MULTILINE)
            ddl_content = re.sub(r'/\*.*?\*/', '', ddl_content, flags=re.DOTALL)

            # Split into statements
            statements = sqlparse.split(ddl_content)
            tables = {}

            logger.info(f"📄 Processing {len(statements)} SQL statements...")

            for i, statement in enumerate(statements):
                statement = statement.strip()
                if not statement:
                    continue

                logger.info(f"🔍 Processing statement {i + 1}: {statement[:50]}...")

                # Skip ALTER statements for now
                if statement.upper().startswith('ALTER'):
                    logger.info(f"⏭️ Skipping ALTER statement")
                    continue

                # Parse CREATE TABLE statements
                if statement.upper().startswith('CREATE TABLE'):
                    table_info = self._parse_create_table_statement(statement)
                    if table_info:
                        table_name = table_info['name']
                        tables[table_name] = table_info
                        logger.info(f"✅ Parsed table: {table_name} with {len(table_info['columns'])} columns")
                    else:
                        logger.warning(f"❌ Failed to parse CREATE TABLE statement")

            logger.info(f"🎯 Successfully parsed {len(tables)} tables: {list(tables.keys())}")
            return tables

        except Exception as e:
            logger.error(f"❌ Error parsing DDL: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _parse_create_table_statement(self, statement: str) -> Dict[str, Any]:
        """Parse a single CREATE TABLE statement"""
        try:
            # Extract table name using regex
            table_name_match = re.search(r'CREATE\s+TABLE\s+([`\[\]"\']*)?(\w+)([`\[\]"\']*)?', statement,
                                         re.IGNORECASE)
            if not table_name_match:
                logger.error(f"❌ Could not extract table name from statement")
                return None

            table_name = table_name_match.group(2)
            logger.info(f"📋 Found table name: {table_name}")

            # Extract the content between parentheses
            paren_match = re.search(r'\((.*)\)', statement, re.DOTALL)
            if not paren_match:
                logger.error(f"❌ Could not find table definition in parentheses")
                return None

            table_definition = paren_match.group(1)

            # Parse columns and constraints
            columns = []
            foreign_keys = []

            # Split by commas, but be careful about commas inside parentheses (like ENUM values)
            column_definitions = self._smart_split_columns(table_definition)

            for col_def in column_definitions:
                col_def = col_def.strip()
                if not col_def:
                    continue

                # Skip constraints that are not column definitions
                if any(keyword in col_def.upper() for keyword in
                       ['PRIMARY KEY', 'FOREIGN KEY', 'CONSTRAINT', 'INDEX', 'KEY (']):
                    if 'FOREIGN KEY' in col_def.upper():
                        foreign_keys.append(col_def)
                    continue

                # Parse column definition
                column_info = self._parse_column_definition(col_def)
                if column_info:
                    columns.append(column_info)
                    logger.info(f"  📝 Column: {column_info['name']} ({column_info['type']})")

            return {
                'name': table_name,
                'columns': columns,
                'foreign_keys': foreign_keys,
                'schema': statement
            }

        except Exception as e:
            logger.error(f"❌ Error parsing CREATE TABLE statement: {e}")
            return None



    def _smart_split_columns(self, definition: str) -> List[str]:
        """Split column definitions by comma, respecting parentheses"""
        parts = []
        current_part = ""
        paren_depth = 0

        for char in definition:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == ',' and paren_depth == 0:
                parts.append(current_part.strip())
                current_part = ""
                continue

            current_part += char

        if current_part.strip():
            parts.append(current_part.strip())

        return parts

    def _parse_column_definition(self, col_def: str) -> Dict[str, Any]:
        """Parse a single column definition"""
        try:
            # Remove extra whitespace
            col_def = re.sub(r'\s+', ' ', col_def.strip())

            parts = col_def.split()
            if len(parts) < 2:
                return None

            # Extract column name (remove quotes/backticks)
            col_name = parts[0].strip('`"\'[]')

            # Extract data type
            col_type = parts[1]

            # Handle complex types like ENUM('val1', 'val2') or VARCHAR(100)
            if '(' in col_type:
                # Find the complete type definition
                type_end = col_def.find(')', col_def.find(col_type)) + 1
                type_start = col_def.find(col_type)
                col_type = col_def[type_start:type_end]

            # Extract constraints
            constraints = []
            col_def_upper = col_def.upper()

            if 'NOT NULL' in col_def_upper:
                constraints.append('NOT NULL')
            if 'PRIMARY KEY' in col_def_upper:
                constraints.append('PRIMARY KEY')
            if 'UNIQUE' in col_def_upper:
                constraints.append('UNIQUE')
            if 'AUTO_INCREMENT' in col_def_upper:
                constraints.append('AUTO_INCREMENT')
            if 'DEFAULT' in col_def_upper:
                # Extract default value
                default_match = re.search(r'DEFAULT\s+([^,\s]+)', col_def, re.IGNORECASE)
                if default_match:
                    constraints.append(f'DEFAULT {default_match.group(1)}')

            return {
                'name': col_name,
                'type': col_type,
                'constraints': constraints
            }

        except Exception as e:
            logger.error(f"❌ Error parsing column definition '{col_def}': {e}")
            return None



    @observe(name="generate_synthetic_data")
    def generate_synthetic_data(self, schema_info: Dict, instructions: str = "", temperature: float = 0.7,
                                num_rows: int = 3) -> Dict[str, List[Dict]]:
        try:
            if LANGFUSE_AVAILABLE:
                langfuse_context.update_current_trace(
                    name="data_generation",
                    input={"schema_info": schema_info, "instructions": instructions, "num_rows": num_rows}
                )

            prompt = f"""
            Generate realistic synthetic data for the following database schema.

            Schema Information:
            {json.dumps(schema_info, indent=2)}

            Additional Instructions: {instructions}

            Requirements:
            1. Generate exactly {num_rows} rows for each table
            2. Ensure data integrity and foreign key constraints are maintained
            3. Use realistic values appropriate for each data type
            4. Handle MySQL-specific types (AUTO_INCREMENT, ENUM, etc.)
            5. For AUTO_INCREMENT columns, generate sequential integer values starting from 1
            6. For ENUM columns, use only the values specified in the type definition
            7. For VARCHAR columns, generate appropriate length strings
            8. Make sure foreign keys reference existing primary keys from related tables
            9. For DATE columns, use format: YYYY-MM-DD
            10. For DATETIME/TIMESTAMP columns, use format: YYYY-MM-DD HH:MM:SS
            11. Return the data as JSON with table names as keys and arrays of objects as values
            12. Each object should represent one row with column names as keys
            13. Generate data in dependency order (parent tables before child tables)
            14. Make the data realistic and diverse for a library management system

            Important: Respond with ONLY a single JSON object, no prose, no code fences, no explanations.
            """

            logger.info(f"🔄 Generating synthetic data for {len(schema_info)} tables...")

            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=8192
                )
            )

            # ---- robust extraction & parsing ----
            raw = _coerce_text_from_response(response)

            print(f"checking response raw: {raw}")

            candidates = (
                raw,
                _strip_code_fences_loose(raw),  # remove ```json ... ```
                _extract_first_json_block(raw),  # grab first balanced {...} / [...]
            )

            print(f"checking response candidate : {candidates}")

            last_err = None
            for s in candidates:
                try:
                    data = json.loads(s)
                    if isinstance(data, dict) and data:
                        logger.info(f"✅ Parsed JSON with {len(data)} tables: {list(data.keys())[:5]}")
                    else:
                        logger.warning("⚠️ JSON parsed but empty or not an object")
                    return data
                except json.JSONDecodeError as e:
                    last_err = e

            # If all attempts fail, log & return {}
            logger.error(f"❌ JSON parsing error: {last_err}")
            logger.error(f"Raw response (first 500 chars): {raw[:500]}...")
            if LANGFUSE_AVAILABLE:
                langfuse_context.update_current_trace(output={"error": f"JSON parsing error: {str(last_err)}"})
            return {}

        except Exception as e:
            logger.error(f"❌ Error generating data: {e}")
            if LANGFUSE_AVAILABLE:
                langfuse_context.update_current_trace(output={"error": str(e)})
            return {}


    @observe(name="modify_table_data")
    def modify_data(self, table_name: str, data: List[Dict], instructions: str, temperature: float = 0.7) -> List[Dict]:
        """Modify existing data based on user instructions"""
        try:
            if LANGFUSE_AVAILABLE:
                langfuse_context.update_current_trace(
                    name="data_modification",
                    input={"table_name": table_name, "instructions": instructions, "original_rows": len(data)}
                )

            prompt = f"""
            Modify the following data for table '{table_name}' based on the user instructions.

            Current Data (showing first 5 rows as example):
            {json.dumps(data[:5], indent=2)}

            User Instructions: {instructions}

            Requirements:
            1. Apply the modifications to all {len(data)} rows consistently
            2. Maintain data integrity and consistency across all records
            3. Keep the same structure and column names
            4. Ensure data types remain appropriate
            5. Maintain any relationships or constraints
            6. Return the complete modified dataset

            Return only valid JSON array without any markdown formatting or explanations.
            """

            logger.info(f"🔄 Modifying data for table '{table_name}'...")

            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=8192
                )
            )

            response_text = _safe_json_from_response(response)

            for candidate in (response_text,
                              _strip_code_fences_loose(response_text),
                              _extract_first_json_block(response_text)):
                try:
                    modified_data = json.loads(candidate)
                    break
                except json.JSONDecodeError:
                    modified_data = None

            if modified_data is None:
                logger.error("❌ Could not parse JSON from model output")
                return data  # fallback to original

            return modified_data

        except Exception as e:
            logger.error(f"❌ Error modifying data: {e}")
            if LANGFUSE_AVAILABLE:
                langfuse_context.update_current_trace(
                    output={"error": str(e)}
                )
            return data

    @observe(name="query_data_nl")
    def query_data_with_nl(self, query: str, data_context: Dict, schema_info: Dict) -> str:
        """Query data using natural language"""
        try:
            if LANGFUSE_AVAILABLE:
                langfuse_context.update_current_trace(
                    name="natural_language_query",
                    input={"query": query, "available_tables": list(data_context.keys())}
                )

            # Prepare data summary for context
            data_summary = {}
            for table_name, table_data in data_context.items():
                if table_data:
                    data_summary[table_name] = {
                        "row_count": len(table_data),
                        "columns": list(table_data[0].keys()) if table_data else [],
                        "sample_data": table_data[:3] if len(table_data) >= 3 else table_data
                    }

            prompt = f"""
            You are a data analyst AI. Answer the user's question about the provided dataset.

            User Question: {query}

            Available Data Summary:
            {json.dumps(data_summary, indent=2)}

            Schema Information:
            {json.dumps(schema_info, indent=2)}

            Instructions:
            1. Analyze the data to answer the user's question
            2. Provide specific insights based on the actual data
            3. Include relevant statistics, trends, or patterns if applicable
            4. If the question requires calculations, perform them accurately
            5. Be conversational but informative
            6. If you need to see more data to answer accurately, mention that
            7. Format your response in a clear, easy-to-read manner
            8. Use specific numbers and examples from the data when possible

            Provide a comprehensive answer to the user's question.
            """

            logger.info(f"🔄 Processing natural language query: '{query[:50]}...'")

            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2048
                )
            )

            answer = response.text.strip()

            if LANGFUSE_AVAILABLE:
                langfuse_context.update_current_trace(
                    output={"response_length": len(answer)}
                )

            logger.info(f"✅ Generated response for natural language query")
            return answer

        except Exception as e:
            logger.error(f"❌ Error querying data: {e}")
            return f"I apologize, but I encountered an error while analyzing your data: {str(e)}"


# Initialize data generator
data_generator = DataGenerator()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save_data', methods=['POST'])
def save_data():
    """Persist the current session's generated data into PostgreSQL (create tables/columns on demand)."""
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in generated_data:
            return jsonify({'success': False, 'error': 'No generated data found in session'}), 400

        data = generated_data[session_id]['data'] or {}
        if not data:
            return jsonify({'success': False, 'error': 'Dataset is empty'}), 400

        if pool is None:
            return jsonify({'success': False, 'error': 'Database is not connected'}), 503

        # Get parser schema (so we can map types)
        schema_info = uploaded_schemas.get(session_id, {}).get('schema_info', {})

        per_table_counts = {}
        with get_conn() as conn:
            conn.autocommit = False
            try:
                for table, rows in data.items():
                    if not isinstance(rows, list) or not rows:
                        per_table_counts[table] = 0
                        continue

                    # 1) Create table if missing (from schema; fall back to sample rows)
                    table_schema = schema_info.get(table, {"columns": []})
                    _create_table_if_missing(conn, table, table_schema, rows)

                    # 2) Add missing columns (present in rows but absent in DB)
                    # Build lookup of schema columns by name for this table
                    schema_cols_by_name = {c['name']: c for c in table_schema.get('columns', [])}
                    _add_missing_columns(conn, table, rows, schema_cols_by_name)

                    # 3) Insert
                    inserted = _bulk_insert_table(conn, table, rows)
                    per_table_counts[table] = inserted

                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.exception("DB error when saving data")
                return jsonify({'success': False, 'error': f'Database error: {e}'}), 500

        logger.info(f"✅ Saved dataset to DB: {per_table_counts}")
        return jsonify({'success': True, 'saved': per_table_counts})

    except Exception as e:
        logger.exception("Unexpected error in /save_data")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/session_status')
def session_status():
    sid = session.get('session_id')
    has_schema = bool(sid and sid in uploaded_schemas)
    has_data = bool(sid and sid in generated_data and generated_data[sid].get('data'))
    tables = list(generated_data[sid]['data'].keys()) if has_data else []

    print("Checking status ")
    return jsonify({
        'success': True,
        'has_schema': has_schema,
        'has_data': has_data,
        'tables': tables,
    })



@app.route('/upload_schema', methods=['POST'])
def upload_schema():
    try:
        file = request.files['schema_file']
        if file and file.filename:
            filename = secure_filename(file.filename)
            content = file.read().decode('utf-8')

            # Parse the schema
            schema_info = data_generator.parse_ddl_schema(content)

            if not schema_info:
                return jsonify({'success': False, 'error': 'No valid tables found in the schema'})

            # Store in session
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
            uploaded_schemas[session_id] = {
                'filename': filename,
                'content': content,
                'schema_info': schema_info,
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"✅ Schema uploaded for session {session_id[:8]}...")

            return jsonify({
                'success': True,
                'schema_info': schema_info,
                'session_id': session_id
            })
        else:
            return jsonify({'success': False, 'error': 'No file uploaded'})
    except Exception as e:
        logger.error(f"❌ Error uploading schema: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/generate_data', methods=['POST'])
# --- in your route ---

def generate_data():
    try:
        data = request.json
        session_id = session.get('session_id')

        if not session_id or session_id not in uploaded_schemas:
            return jsonify({'success': False, 'error': 'No schema uploaded. Please upload a schema first.'})

        schema_info = uploaded_schemas[session_id]['schema_info']
        instructions = data.get('instructions', '')
        temperature = float(data.get('temperature', 0.7))
        #num_rows = int(data.get('num_rows', 10))
        num_rows = 5

        if num_rows <= 0 or num_rows > 10000:
            return jsonify({'success': False, 'error': 'Number of rows must be between 1 and 10,000'})

        if temperature < 0 or temperature > 1:
            return jsonify({'success': False, 'error': 'Temperature must be between 0 and 1'})

        print(f"schema info: {schema_info}")
        print(f"Instructions: {instructions}")
        print(f"temperature: {temperature}")
        print(f"num rows: {num_rows}")

        # Generate synthetic data
        synthetic_data = data_generator.generate_synthetic_data(
            schema_info=schema_info,
            instructions=instructions,
            temperature=temperature,
            num_rows=num_rows
        )

        if not synthetic_data:
            return jsonify({'success': False, 'error': 'Failed to generate data. Please try again.'})

        # --- enforce exactly num_rows per table ---
        synthetic_data, issues = enforce_row_counts(
            synthetic_data, schema_info, num_rows
        )

        if issues.get("too_few"):
            # Option A: fail fast (clear error to caller)
            return jsonify({'success': False,
                            'error': f"Some tables had fewer than {num_rows} rows: {issues['too_few']}"})
            # Option B (alternative): attempt a single retry with a stricter prompt.
            # synthetic_data_retry = data_generator.retry_fill_missing_rows(schema_info, synthetic_data, num_rows)
            # synthetic_data, issues = enforce_row_counts(synthetic_data_retry, schema_info, num_rows)
            # if issues.get("too_few"):
            #     return jsonify({'success': False, 'error': f"Could not reach {num_rows} rows for: {issues['too_few']}"})

        generated_data[session_id] = {
            'data': synthetic_data,
            'schema_info': schema_info,
            'timestamp': datetime.now().isoformat()
        }
        chat_history[session_id] = []

        logger.info(f"✅ Data generated for session {session_id[:8]}...")
        return jsonify({'success': True, 'data': synthetic_data, 'tables': list(synthetic_data.keys())})

    except Exception as e:
        logger.error(f"❌ Error generating data: {e}")
        return jsonify({'success': False, 'error': str(e)})


def enforce_row_counts(data: Dict[str, list], schema_info: Dict, num_rows: int):
    """
    Ensures each table has exactly num_rows rows.
    - If a table has > num_rows: truncate.
    - If a table has < num_rows: record issue (caller can retry or error).
    Returns (adjusted_data, issues_dict)
    """
    adjusted = {}
    issues = {"too_many": [], "too_few": []}

    for table, rows in (data or {}).items():
        if not isinstance(rows, list):
            adjusted[table] = []
            issues["too_few"].append(f"{table} (not a list)")
            continue

        if len(rows) > num_rows:
            adjusted[table] = rows[:num_rows]  # hard cap
            issues["too_many"].append(f"{table} ({len(rows)} > {num_rows})")
        elif len(rows) < num_rows:
            adjusted[table] = rows
            issues["too_few"].append(f"{table} ({len(rows)} < {num_rows})")
        else:
            adjusted[table] = rows

    return adjusted, issues

@app.route('/get_table_data/<table_name>')
def get_table_data(table_name):
    try:
        session_id = session.get('session_id')
        if session_id in generated_data:
            table_data = generated_data[session_id]['data'].get(table_name, [])
            logger.info(f"✅ Retrieved {len(table_data)} rows for table '{table_name}'")
            return jsonify({
                'success': True,
                'data': table_data,
                'count': len(table_data)
            })
        return jsonify({'success': False, 'error': 'No data found for this session'})
    except Exception as e:
        logger.error(f"❌ Error getting table data: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/modify_table_data', methods=['POST'])
def modify_table_data():
    try:
        data = request.json
        session_id = session.get('session_id')
        table_name = data.get('table_name')
        instructions = data.get('instructions')
        temperature = float(data.get('temperature', 0.7))

        if not session_id or session_id not in generated_data:
            return jsonify({'success': False, 'error': 'No generated data found'})

        current_data = generated_data[session_id]['data'].get(table_name, [])
        if not current_data:
            return jsonify({'success': False, 'error': 'Table not found'})

        if not instructions or not instructions.strip():
            return jsonify({'success': False, 'error': 'Please provide modification instructions'})

        # Modify the data
        modified_data = data_generator.modify_data(
            table_name, current_data, instructions, temperature
        )

        # Update stored data
        generated_data[session_id]['data'][table_name] = modified_data

        logger.info(f"✅ Modified table '{table_name}' for session {session_id[:8]}...")

        return jsonify({
            'success': True,
            'data': modified_data,
            'count': len(modified_data)
        })

    except Exception as e:
        logger.error(f"❌ Error modifying table data: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/download_data')
def download_data():
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in generated_data:
            return jsonify({'success': False, 'error': 'No data to download'})

        data = generated_data[session_id]['data']

        if not data:
            return jsonify({'success': False, 'error': 'No data available for download'})

        # Create a zip file with CSV files for each table
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for table_name, table_data in data.items():
                if table_data:
                    df = pd.DataFrame(table_data)
                    csv_buffer = StringIO()
                    df.to_csv(csv_buffer, index=False)
                    zf.writestr(f"{table_name}.csv", csv_buffer.getvalue())

        memory_file.seek(0)

        logger.info(f"✅ Data download prepared for session {session_id[:8]}...")

        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'synthetic_data_{session_id[:8]}.zip'
        )

    except Exception as e:
        logger.error(f"❌ Error downloading data: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/chat', methods=['POST'])
def chat_with_data():
    try:
        data = request.json
        message = data.get('message', '').strip()
        session_id = session.get('session_id')

        if not session_id or not message:
            return jsonify({'success': False, 'error': 'Invalid request'})

        if session_id not in generated_data:
            return jsonify({'success': False, 'error': 'No data available for chat. Please generate data first.'})

        # Get current data and schema
        data_context = generated_data[session_id]['data']
        schema_info = generated_data[session_id]['schema_info']

        # Generate response
        response = data_generator.query_data_with_nl(message, data_context, schema_info)

        # Store chat history
        if session_id not in chat_history:
            chat_history[session_id] = []

        chat_history[session_id].append({
            'message': message,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })

        logger.info(f"✅ Chat response generated for session {session_id[:8]}...")

        return jsonify({
            'success': True,
            'response': response
        })

    except Exception as e:
        logger.error(f"❌ Error in chat: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'langfuse_available': LANGFUSE_AVAILABLE,
        'gemini_configured': bool(os.getenv('GOOGLE_API_KEY'))
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("🚀 Starting Data Assistant Application...")
    print(f"   Langfuse: {'✅ Available' if LANGFUSE_AVAILABLE else '❌ Not available'}")
    print(f"   Gemini API: {'✅ Configured' if os.getenv('GOOGLE_API_KEY') else '❌ Not configured'}")
    print("   Server starting on http://localhost:5002")
    app.run(debug=True, host='0.0.0.0', port=5002, use_reloader=False)