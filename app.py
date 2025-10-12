from flask import Flask, render_template, request, jsonify, send_file, session
import os
import json
import pandas as pd
import sqlite3
import tempfile
import zipfile
from io import StringIO, BytesIO
import google.generativeai as genai
from google.generativeai import configure
import re
import uuid
from datetime import datetime
import sqlparse
from werkzeug.utils import secure_filename
import logging
from typing import Dict, List, Any, Optional

# Optional Langfuse import - will work without it
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
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    logger.error("GOOGLE_API_KEY environment variable not set!")
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

genai.configure(api_key=api_key)

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
                                num_rows: int = 100) -> Dict[str, List[Dict]]:
        """Generate synthetic data using Gemini with optional Langfuse observability"""
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

            Important: Return ONLY valid JSON without any markdown formatting, explanations, or code blocks.

            Example structure:
            {{
                "Authors": [
                    {{"author_id": 1, "first_name": "Jane", "last_name": "Austen", "birth_date": "1775-12-16", "nationality": "British"}},
                    {{"author_id": 2, "first_name": "Mark", "last_name": "Twain", "birth_date": "1835-11-30", "nationality": "American"}}
                ],
                "Publishers": [...],
                "Books": [...]
            }}
            """

            logger.info(f"🔄 Generating synthetic data for {len(schema_info)} tables...")

            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=8192
                )
            )

            # Parse the JSON response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]

            data = json.loads(response_text)

            if LANGFUSE_AVAILABLE:
                langfuse_context.update_current_trace(
                    output={"generated_tables": list(data.keys()),
                            "total_rows": sum(len(table_data) for table_data in data.values())}
                )

            logger.info(f"✅ Generated data for {len(data)} tables")
            return data

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing error: {e}")
            logger.error(f"Raw response: {response_text[:500]}...")
            if LANGFUSE_AVAILABLE:
                langfuse_context.update_current_trace(
                    output={"error": f"JSON parsing error: {str(e)}"}
                )
            return {}
        except Exception as e:
            logger.error(f"❌ Error generating data: {e}")
            if LANGFUSE_AVAILABLE:
                langfuse_context.update_current_trace(
                    output={"error": str(e)}
                )
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

            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]

            modified_data = json.loads(response_text)

            if LANGFUSE_AVAILABLE:
                langfuse_context.update_current_trace(
                    output={"modified_rows": len(modified_data)}
                )

            logger.info(f"✅ Modified {len(modified_data)} rows for table '{table_name}'")
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
def generate_data():
    try:
        data = request.json
        session_id = session.get('session_id')

        if not session_id or session_id not in uploaded_schemas:
            return jsonify({'success': False, 'error': 'No schema uploaded. Please upload a schema first.'})

        schema_info = uploaded_schemas[session_id]['schema_info']
        instructions = data.get('instructions', '')
        temperature = float(data.get('temperature', 0.7))
        num_rows = int(data.get('num_rows', 100))

        # Validate inputs
        if num_rows <= 0 or num_rows > 10000:
            return jsonify({'success': False, 'error': 'Number of rows must be between 1 and 10,000'})

        if temperature < 0 or temperature > 1:
            return jsonify({'success': False, 'error': 'Temperature must be between 0 and 1'})

        # Generate synthetic data
        synthetic_data = data_generator.generate_synthetic_data(
            schema_info, instructions, temperature, num_rows
        )

        if not synthetic_data:
            return jsonify({'success': False, 'error': 'Failed to generate data. Please try again.'})

        # Store generated data
        generated_data[session_id] = {
            'data': synthetic_data,
            'schema_info': schema_info,
            'timestamp': datetime.now().isoformat()
        }

        # Initialize chat history for this session
        chat_history[session_id] = []

        logger.info(f"✅ Data generated for session {session_id[:8]}...")

        return jsonify({
            'success': True,
            'data': synthetic_data,
            'tables': list(synthetic_data.keys())
        })

    except Exception as e:
        logger.error(f"❌ Error generating data: {e}")
        return jsonify({'success': False, 'error': str(e)})


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

    app.run(debug=True, host='0.0.0.0', port=5002)