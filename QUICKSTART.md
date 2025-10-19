# 🚀 Quick Start Guide

## Immediate Setup (5 minutes)

### 1. Install Dependencies
```bash
# Install basic requirements
pip install -r requirements_basic.txt

# OR if you want full features
pip install -r requirements.txt
```

### 2. Set Environment Variable
```bash
# Set your Google API key
export GOOGLE_API_KEY="your_google_api_key_here"

# On Windows:
# set GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Run the Application
```bash
# Use the corrected version
python app_corrected.py
```

### 4. Access the Application
Open your browser and go to: http://localhost:5000

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'langfuse.decorators'"
**Solution:** Use `app_corrected.py` instead of `app.py` or `app_enhanced.py`

### Error: "GOOGLE_API_KEY environment variable not set"
**Solution:** 
1. Get your API key from Google AI Studio
2. Set it as an environment variable:
   ```bash
   export GOOGLE_API_KEY="your_actual_api_key"
   ```

### Error: Database connection issues
**Solution:** The corrected app uses in-memory storage, no database needed for basic functionality

## 📝 Sample DDL Schema for Testing

Create a file called `sample_schema.sql`:

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

## 🎯 Quick Test

1. **Upload Schema**: Upload the `sample_schema.sql` file
2. **Generate Data**: Click "Generate" with default settings
3. **Preview Data**: Select a table from the dropdown
4. **Chat**: Switch to "Query Data" tab and ask: "How many customers do we have?"

## 📁 File Structure for Quick Start

```
your-project/
├── app_corrected.py           # ← Use this file
├── requirements_basic.txt     # ← Install this
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
└── sample_schema.sql         # ← Create this for testing
```

## 🚀 Next Steps

Once you have the basic version working:

1. **Add PostgreSQL**: Use `app_enhanced.py` with Docker
2. **Add Langfuse**: Install langfuse package for observability
3. **Deploy**: Use Docker Compose for production deployment

## 💡 Tips

- Start with small schemas (2-3 tables)
- Use 50-100 rows for initial testing
- Keep temperature around 0.7 for balanced creativity
- Ask specific questions in the chat interface