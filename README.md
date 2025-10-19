# Data Assistant - Conversational AI for Synthetic Data Generation

A comprehensive Flask application that combines synthetic data generation with natural language data querying capabilities, powered by Google's Gemini 2.0 Flash model.

## 🚀 Features

### Phase 1: Synthetic Data Generation
- **Schema Parsing**: Upload DDL files (.sql, .txt, .ddl) and automatically parse table structures
- **AI-Powered Generation**: Use Gemini 2.0 Flash to generate realistic synthetic data
- **Data Integrity**: Maintains foreign key relationships and constraints
- **Interactive Editing**: Modify generated data using natural language instructions
- **Export Capabilities**: Download generated data as CSV files or ZIP archives

### Phase 2 & 3: Conversational Data Interface
- **Natural Language Queries**: Ask questions about your data in plain English
- **Real-time Chat**: Interactive chat interface with your dataset
- **Data Analysis**: Get insights, statistics, and patterns from your data
- **Multi-table Support**: Query across multiple related tables

## 🛠 Technology Stack

- **Backend**: Python Flask
- **AI Model**: Google Gemini 2.0 Flash (with streaming, function calling, and structured output)
- **Database**: PostgreSQL
- **Observability**: Langfuse for LLM monitoring and analytics
- **Containerization**: Docker & Docker Compose
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Authentication**: Google Cloud Platform Vertex AI

## 📋 Prerequisites

1. **Google Cloud Platform Account**
   - Enable Vertex AI API
   - Create service account with appropriate permissions
   - Download service account key JSON file

2. **API Keys**
   - Google AI API Key (for Gemini access)
   - Langfuse API keys (optional, for observability)

3. **Docker & Docker Compose** installed on your system

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd data-assistant
```

### 2. Environment Variables
Create a `.env` file in the project root:

```env
# Google AI Configuration
GOOGLE_API_KEY=your_google_ai_api_key_here

# Database Configuration
DATABASE_URL=postgresql://postgres:password@db:5432/dataassistant

# Langfuse Configuration (Optional)
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_HOST=http://localhost:3000

# Application Configuration
SECRET_KEY=your_secret_key_here
FLASK_ENV=development
```

### 3. Docker Deployment
```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d --build
```

### 4. Access the Application
- **Main Application**: http://localhost:5000
- **Langfuse Dashboard**: http://localhost:3000 (if enabled)
- **PostgreSQL**: localhost:5432

## 📁 Project Structure

```
data-assistant/
├── app.py                      # Main Flask application
├── app_enhanced.py             # Enhanced version with PostgreSQL + Langfuse
├── requirements.txt            # Python dependencies
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Multi-service setup
├── .env                       # Environment variables (create this)
├── templates/
│   └── index.html             # Main HTML template
├── static/
│   ├── style.css              # Styling
│   ├── script.js              # Basic JavaScript
└── README.md                  # This file
```

## 💡 Usage Guide

### 1. Upload DDL Schema
- Click "Upload DDL Schema" in the Data Generation tab
- Select a .sql, .txt, or .ddl file containing your database schema
- The system will parse and display the detected tables and columns

### 2. Generate Synthetic Data
- Enter optional instructions in the prompt field (e.g., "Generate realistic customer data for an e-commerce platform")
- Adjust temperature (0.0-1.0) for creativity vs consistency
- Set the number of rows to generate per table
- Click "Generate" to create synthetic data

### 3. Preview and Edit Data
- Select a table from the dropdown to preview generated data
- Use the edit instructions field to modify data with natural language
- Example: "Make all customer emails end with @company.com"
- Click "Submit" to apply changes

### 4. Download Data
- Click "Download CSV/ZIP" to export all generated tables
- Files are packaged as a ZIP archive with one CSV per table

### 5. Chat with Your Data
- Switch to the "Query Data" tab
- Ask questions about your generated dataset
- Examples:
  - "What's the average age of customers?"
  - "Show me the top 5 products by price"
  - "How many orders were placed last month?"

## 🔍 Sample DDL Schemas

### Library Management System
```sql
CREATE TABLE authors (
    author_id INT PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    birth_date DATE,
    nationality VARCHAR(50)
);

CREATE TABLE books (
    book_id INT PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    author_id INT,
    isbn VARCHAR(20) UNIQUE,
    publication_year INT,
    genre VARCHAR(50),
    FOREIGN KEY (author_id) REFERENCES authors(author_id)
);
```

### E-commerce Platform
```sql
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    email VARCHAR(100) UNIQUE NOT NULL,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    phone VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE products (
    product_id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    category VARCHAR(50),
    stock_quantity INT DEFAULT 0
);

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(10,2),
    status VARCHAR(20) DEFAULT 'pending',
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
```

## 🔒 Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **Database**: Use strong passwords and limit database access
3. **Environment**: Run in isolated Docker containers
4. **Authentication**: Implement proper authentication for production use
5. **Data Privacy**: Generated data is synthetic but treat with appropriate care

## 🚀 Advanced Features

### Langfuse Integration
- Monitor all LLM interactions
- Track token usage and costs
- Analyze conversation patterns
- Debug generation issues

### PostgreSQL Storage
- Persistent data storage
- Session management
- Chat history
- Schema versioning

### Streaming Responses
- Real-time data generation feedback
- Progressive loading of large datasets
- Enhanced user experience

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   ```bash
   # Verify your API key is set
   echo $GOOGLE_API_KEY
   
   # Check Docker environment
   docker-compose exec app env | grep GOOGLE_API_KEY
   ```

2. **Database Connection Issues**
   ```bash
   # Check PostgreSQL status
   docker-compose ps
   
   # View logs
   docker-compose logs db
   ```

3. **Memory Issues with Large Datasets**
   - Reduce the number of rows per generation
   - Increase Docker memory limits
   - Use pagination for large tables

4. **Schema Parsing Errors**
   - Ensure DDL syntax is valid
   - Check for unsupported SQL features
   - Simplify complex schemas

### Performance Optimization

1. **Database Indexing**
   ```sql
   CREATE INDEX idx_session_id ON generated_data(session_id);
   CREATE INDEX idx_table_name ON generated_data(table_name);
   ```

2. **Caching**
   - Implement Redis for session caching
   - Cache parsed schemas
   - Store generation templates

## 📊 Monitoring & Analytics

### Langfuse Dashboard
- View all LLM generations
- Track performance metrics
- Monitor costs and usage
- Debug failed requests

### Application Metrics
- Generation success rates
- Average response times
- User interaction patterns
- Data quality metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For support and questions:
1. Check the troubleshooting section
2. Review the Langfuse logs for LLM issues
3. Check Docker logs for infrastructure issues
4. Open an issue on GitHub

## 🔮 Future Enhancements

- [ ] Support for more database types (MySQL, SQLite)
- [ ] Advanced data visualization capabilities
- [ ] Export to different formats (JSON, Parquet, Excel)
- [ ] Real-time collaboration features
- [ ] Custom data generation templates
- [ ] Integration with data catalogs
- [ ] Advanced analytics and reporting
- [ ] API endpoints for programmatic access

---

**Built with ❤️ using Google Gemini 2.0 Flash and modern web technologies**