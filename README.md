***

# Multimodal RAG Pipeline (FastAPI + PostgreSQL pgvector)

This project is a high-performance Retrieval-Augmented Generation (RAG) pipeline. It allows users to upload PDF documents, automatically chunks and vectorizes the text, and stores the embeddings in a PostgreSQL database optimized with the **HNSW algorithm** for scalable semantic search.

## 🚀 Prerequisites

*   **Docker Desktop** (for running the database)
*   **Python 3.9+**
*   **PostgreSQL Client** (optional, e.g., DBeaver or `psql`)

---

## 🛠️ Step 1: Environment Setup

1.  Clone the repository.
2.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Create a `.env` file in the root directory:
    ```ini
    # .env
    DATABASE_URL="postgresql://cbuser:12345@localhost:5432/knowledgedb"
    OPENAI_API_KEY="sk-..." # (Optional: Only if using the Chat/RAG endpoint)
    ```

---

## 🐳 Step 2: Database Setup (Docker)

We use the official `pgvector` image to ensure the vector extension is available.

### 1. Start the Container
Run this command to start a fresh PostgreSQL container with pgvector pre-installed:

```bash
docker run --name multimodal-postgres-db \
  -e POSTGRES_USER=cbuser \
  -e POSTGRES_PASSWORD=12345 \
  -e POSTGRES_DB=knowledgedb \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

### 2. Initialize Schema & Index (Automated)
Run the included Python script to enable the extension, create the table, and build the HNSW index automatically.

```bash
python create_db.py
```

---

## 🤓 Step 3: Manual Database Setup (SQL Reference)

If you prefer to set up the database manually using a SQL client (like DBeaver or `psql`), execute these commands in order.

**1. Enable the Extension:**
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**2. Create the Table:**
```sql
CREATE TABLE text_chunks (
    id SERIAL PRIMARY KEY,
    document_name VARCHAR(255) NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(384)
);
```

**3. Create the HNSW Index:**
This creates a Hierarchical Navigable Small World index for fast approximate nearest neighbor search (L2 distance).
```sql
CREATE INDEX hnsw_index_for_inner_product 
ON text_chunks 
USING hnsw (embedding vector_l2_ops);
```

---

## 🏃 Step 4: Run the Backend API Application

Start the FastAPI server. This handles the database connections, embeddings, and vector search logic.

```bash
uvicorn main:app --reload
```

The server will start at `http://127.0.0.1:8000`.

---
To allow access from other computers, we run the server with `--host 0.0.0.0`.

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

*   **Local Access:** `http://127.0.0.1:8000`
*   **Network Access:** `http://<YOUR_COMPUTER_IP>:8000`
---

## 🎨 Step 5: Run the Frontend UI

**Important Configuration:**
If you want to access the UI from another computer, you must tell the Frontend where the Backend lives.

1.  Open `frontend.py`.
2.  Find line 6: `BASE_URL = "http://127.0.0.1:8000"`.
3.  **Change it** to your computer's local IP address (see instructions below), e.g., `BASE_URL = "http://192.168.1.15:8000"`.

**Run the Streamlit App:**
Open a **new terminal window** (keep the backend running in the first one) and launch the Streamlit app.

```bash
streamlit run frontend.py
```
*Frontend running at:* `http://localhost:8501`

## 🧪 Usage

Open your browser and navigate to the interactive Swagger UI: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

### Available Endpoints:

1.  **POST `/upload-pdf/`**
    *   **Input:** A `.pdf` file.
    *   **Action:** Extracts text, splits it into chunks, generates embeddings, and saves to DB.
    *   **Note:** Runs in the background (returns `202 Accepted`).

2.  **GET `/search/`**
    *   **Input:** `query` string.
    *   **Action:** Converts query to vector and performs similarity search using the HNSW index.
    *   **Output:** Returns raw text chunks relevant to the query.

3.  **POST `/chat/`** (RAG / Q&A)
    *   **Input:** `{"question": "What is the summary of the uploaded report?"}`
    *   **Action:** 
        1. Embeds your question.
        2. Retrieves the top 5 most relevant chunks from your PDF.
        3. Sends the chunks + your question to OpenAI (GPT-3.5).
        4. Generates a natural language answer based *only* on your PDF.
    *   **Output:** The answer and the list of source documents used.

---

## ⚠️ Troubleshooting

**Error: `type "vector" does not exist`**
*   **Cause:** The database does not have the `pgvector` extension enabled, or you are connecting to an old local Postgres instance instead of Docker.
*   **Fix:** Ensure you are using the `pgvector/pgvector:pg16` Docker image and have run `CREATE EXTENSION vector;`.
**Error: `OPENAI_API_KEY not set`**
*   **Cause:** You are trying to use `/chat/` but haven't added your API key.
*   **Fix:** Add `OPENAI_API_KEY="sk-..."` to your `.env` file and restart the server.

**How to "Factory Reset" the Database:**
If you need to wipe everything and start fresh:

```bash
docker stop multimodal-postgres-db
docker rm multimodal-postgres-db
docker volume prune  # WARNING: Deletes all data! Type 'y' to confirm.
# Then run the "Start the Container" command again.
```