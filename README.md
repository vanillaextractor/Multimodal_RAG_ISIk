# Advanced Multimodal Hybrid-RAG Pipeline

This project is a high-performance **Retrieval-Augmented Generation (RAG)** pipeline. Originally a standard semantic search prototype, it has evolved into an advanced system supporting hierarchical section-aware chunking, hybrid retrieval (BM25 + Cosine Vector Search), cross-encoder reranking, and decoupled configuration management.

## ✨ Key Features

1. **Hierarchical Chunking**: Uses Regex-based structural parsing to identify chapters/headings, preserving contextual boundaries.
2. **Hybrid Search**: Combines Cosine Vector Search (via PostgreSQL `pgvector`) for semantic meaning with BM25 Keyword Search for exact term matching.
3. **Cross-Encoder Reranking**: Utilizes Local Jina Reranker v2 (GGUF) or FlashRank to rescore chunks for maximum precision before sending context to the LLM.
4. **Flexible LLM Endpoints**: Supports local inference (Llama 3 via `llama-cpp-python`) for ultimate data privacy, or cloud inference (Groq API) for ultra-fast generation.
5. **Decoupled Configuration**: All logic hyperparameters (chunk sizes, rerank cutoffs, LLM paths, generation settings) are completely abstracted to `config.json` and `hierarchical_config.json`.
6. **Multi-Document Support**: Process multiple PDFs simultaneously via the upload endpoints.
7. **Automated Testing & Flushing**: Includes `test_model_with_restart.py` for automated load testing and `flush_db.py` to securely wipe Vector/BM25 storage.

---

## 🚀 Prerequisites

- **Docker Desktop** (for running the database container)
- **Python 3.14.2** (CRITICAL: Deviating from this specific version may cause native binary build failures with `llama-cpp-python` or `pgvector`)
- **Local Machine / Cloud Server** (with sufficient RAM to parse PDFs and run localized models if not using Groq)

---

## 🛠️ Step 1: Environment Setup

1. **Clone the repository.**
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Create a `.env` file in the root directory:**
   ```ini
   # .env
   DATABASE_URL="postgresql://cbuser:12345@localhost:5432/knowledgedb"
   GROQ_API_KEY="gsk_..." # (Required ONLY if running main_groq.py)
   ```

---

## 🐳 Step 2: Database Setup (Docker)

We rely on PostgreSQL with the official `pgvector` extension for semantic storage.

### 1. Start the Container

Run this command to start a PostgreSQL container with `pgvector` pre-installed:

```bash
docker run --name multimodal-postgres-db \
  -e POSTGRES_USER=cbuser \
  -e POSTGRES_PASSWORD=12345 \
  -e POSTGRES_DB=knowledgedb \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

### 2. Initialize Schema

Run the included Python script to enable the extension and build the `TextChunk` tables defined in `models.py`.

```bash
python create_db.py
```

---

## 🏃 Step 3: Running the Backend

The repository contains four specialized execution scripts depending on your environment profile:

### 1. `main_hierarchical_hybrid.py` (Most Advanced - Recommended)

The flagship pipeline. Employs Hierarchical Chunking, Hybrid Retrieval (Semantic Vector + BM25), Jina Reranking, and Local LlamaCpp generation.

```bash
uvicorn main_hierarchical_hybrid:app --reload
```

### 2. `main_groq.py` (Low Latency Cloud generation)

Maintains high-security local embeddings (MiniLM) and local reranking (Jina) to protect database ingestion, but offloads generation to the ultra-fast Groq Cloud API.

```bash
uvicorn main_groq:app --reload
```

### 3. `main_hierarchical_flashrank.py` (CPU-Bound Reranking)

Implements hierarchical chunking but trades the heavy Jina Cross-Encoder for the lightweight FlashRank reranker. Excellent for lower-tier hardware.

```bash
uvicorn main_hierarchical_flashrank:app --reload
```

### 4. `main_local.py` (Legacy Baseline)

The original pipeline augmented with Jina reranking, but relies on blind recursive character chunking instead of section-aware logic.

```bash
uvicorn main_local:app --reload
```

_(By default, all FastAPI servers boot up at `http://localhost:8000`)_

---

## 🎨 Step 4: Running the Streamlit UI

Once the backend is live, open a **new terminal window** to start the interactive conversational UI. The frontend (`frontend.py`) connects by default to `http://localhost:8000`.

```bash
streamlit run frontend.py
```

### Web UI Features:

- **Multi-file PDF Uploads**: Drag and drop batches of training data.
- **Session Memory**: Uses strict Session-IDs to segregate document conversations per tab.
- **Source Attribution**: Chat responses dynamically reference exact retrieved chunks and their reranked confidence scores.
- **Database Clearance**: Secure UI button to trigger a total database flush of embeddings and BM25 indexes.

---

## 🧬 Configuration Management

The system logic is managed by `config.json`. You can modify these settings without touching the Python code or restarting the server:

- **`chunking_tokens`**: Adjust base chunk size and overlap (e.g. `max_size_default`).
- **`llm_hyperparameters`**: Tweak Llama inference (temperature, threads, n_ctx context size, grammar limits).
- **`retriever_hyperparameters`**: Define `top_k` (amount of chunks retrieved) vs `top_n_rerank` (amount of chunks surviving cross-encoder pruning and actually passed into the LLM context).
- **`llm_chat_template`**: Tailor the strict grounding prompt your AI agent adheres to.

---

## 🧪 Automated Testing & Load Evaluation

The repository ships with an integration testing suite built to validate backend memory stability over long sessions:

```bash
python test_model_with_restart.py
```

**How it works:**

1. Spawns the backend subprocess.
2. Ingests the `evaluation/input.pdf` file entirely via the API.
3. Automatically fires a batch of questions parsed from a local CSV (`rag queries - Sheet1.csv`).
4. Preemptively resets the main server every $N$ requests to guarantee no context-window memory leaks.

## 🧹 Maintenance Utils

- **`python flush_db.py`**: A brute-force CLI utility to delete all PostgreSQL rows and wipe out locally-pickled `.pkl` BM25 search indices instantly.
- **`python check_data.py`**: A quick utility script designed to verify database counts.
