import os
import io
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel

# --- Database Setup (SQLAlchemy) ---
from sqlalchemy import create_engine, Column, Integer, String, Text, Index, text
from sqlalchemy.orm import sessionmaker
from models import Base, TextChunk

# --- ML/AI Model Imports ---
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf

# --- NEW: LangChain Imports for RAG ---
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever


# Load environment variables from .env file
load_dotenv(override=True)

home_env_path = Path.home() / ".env"
load_dotenv(dotenv_path=home_env_path, override=True)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cbuser:12345@localhost:5432/knowledgedb")

# --- 1. CORE SETUP: FastAPI App, DB Connection, and Global Model ---

# Initialize FastAPI app
app = FastAPI(
    title="Local RAG Pipeline (FastAPI + LlamaCpp + Reranker)",
    description="A RAG pipeline using a local LLM via llama-cpp-python, optimized with Jina Reranking.",
)

import json
# Load Configuration
with open(os.path.join(os.path.dirname(__file__), "config.json"), "r") as f:
    config = json.load(f)

MODEL_PATH = os.path.abspath(config["llm_path"])
EMBEDDING_MODEL_NAME = config["embedding_model"]
CHUNKING_TOKENS = config["chunking_tokens"]
LLM_HYPERPARAMETERS = config["llm_hyperparameters"]
RETRIEVER_HYPERPARAMETERS = config["retriever_hyperparameters"]
LLM_CHAT_TEMPLATE = config["llm_chat_template"]

# Database connection
try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    print(f"Error connecting to the database: {e}")
    # Exit or handle gracefully if DB connection is critical at startup
    exit()

# Load the embedding model ONCE when the application starts.
# This is a crucial optimization to avoid reloading the model on every request.
try:
    print(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # The dimension of this model is 384
    print("Embedding model loaded successfully.")

except Exception as e:
    print(f"Error loading models: {e}")
    exit()

from huggingface_hub import hf_hub_download
import llama_cpp

try:
    print("Loading Jina Reranker via GGUF...")
    reranker_model_path = hf_hub_download(
        repo_id="gpustack/jina-reranker-v2-base-multilingual-GGUF", 
        filename="jina-reranker-v2-base-multilingual-Q8_0.gguf",
        local_dir=".",
        local_dir_use_symlinks=False
    )
    reranker_model = llama_cpp.Llama(
        model_path=reranker_model_path, 
        verbose=False, 
        embedding=True,
        pooling_type=llama_cpp.LLAMA_POOLING_TYPE_RANK
    )
    print("Jina Reranker loaded successfully.")
except Exception as e:
    print(f"Error loading Jina reranker: {e}")
    # Don't exit, we could potentially fallback if needed, but it should succeed
    exit()

# --- 2. DATABASE MODEL DEFINITION ---
# (Now imported from models.py)

# --- 3. THE CORE PIPELINE LOGIC (PyPDF Parsing) ---
import tempfile
import re

def optimize_chunks_for_retrieval(text, document_name):
    """
    A custom semantic chunking algorithm optimized for dense vector embeddings (like MiniLM).
    Splits text by double newlines and enforces metadata injection.
    """
    final_chunks = []
    final_metadata = []
    
    # Recursive character splitting logic
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNKING_TOKENS["max_size_default"],
        chunk_overlap=CHUNKING_TOKENS["overlap"],
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 10: continue
        
        # Format the robust metadata string directly into the text (Anchor for the LLM)
        formatted_chunk = f"Document: {document_name}\n" \
                          f"Chunk: {i+1}/{len(chunks)}\n\n" \
                          f"Content:\n{chunk.strip()}"
                          
        formatted_chunk = formatted_chunk.replace("\x00", "")
        
        final_chunks.append(formatted_chunk)
        final_metadata.append({
            "page_number": 1, # PyPDF extracted text is often joined, page tracking requires more logic
            "element_type": "NarrativeText"
        })
            
    return final_chunks, final_metadata
            
    return final_chunks, final_metadata

def process_pdf_pipeline(file_contents: bytes, filename: str):
    """
    The main data processing pipeline function that runs in the background.
    Uses Unstructured.io for semantic layout-aware parsing and custom chunking.
    """
    print(f"Starting processing for document: {filename}")
    
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
        temp_pdf.write(file_contents)
        temp_pdf.flush()
        
        # A & B. Ingests PDF & Parses semantic structural elements (Tables, Headers, Text)
        try:
            print("Parsing PDF with PyPDF...")
            reader = pypdf.PdfReader(temp_pdf.name)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"
            
            print(f"Extracted {len(full_text)} characters from {filename}")
        except Exception as e:
            print(f"Error parsing PDF with PyPDF: {e}")
            return
            
    # C. Performs semantic chunking and metadata injection
    try:
        print("Optimizing chunks for retrieval...")
        chunk_texts, chunk_metadata = optimize_chunks_for_retrieval(full_text, filename)
        print(f"Generated {len(chunk_texts)} chunks.")
    except Exception as e:
        print(f"Error chunking document: {e}")
        return
        
    if not chunk_texts:
         print(f"No semantic chunks were generated for {filename}. Aborting.")
         return

    # D. Vectorises the chunks using some embedding model
    try:
        embeddings = embedding_model.encode(chunk_texts, show_progress_bar=False).tolist()
        print(f"Generated {len(embeddings)} embeddings.")
    except Exception as e:
        print(f"Error generating embeddings for {filename}: {e}")
        return

    # E. Stores the vectors and structural metadata in PostgreSQL
    db = SessionLocal()
    try:
        for i, text in enumerate(chunk_texts):
            new_chunk = TextChunk(
                document_name=filename,
                chunk_text=text,
                embedding=embeddings[i],
                page_number=chunk_metadata[i]["page_number"],
                element_type=chunk_metadata[i]["element_type"]
            )
            db.add(new_chunk)
        
        db.commit()
        print(f"Successfully saved {len(chunk_texts)} chunks to the database for {filename}.")

    except Exception as e:
        print(f"Database error while saving chunks for {filename}: {e}")
        db.rollback()
    finally:
        db.close()

# --- 4. RAG HELPERS (Custom Retriever) ---

class CustomPGRetriever_1(BaseRetriever):
    """
    A Custom Retriever for LangChain that uses our existing SQLAlchemy session,
    local sentence-transformers model, and PostgreSQL pgvector.
    """
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        
        # 1. Embed the user query using the SAME model used for ingestion
        query_embedding = embedding_model.encode(query).tolist()
        
        # 2. Query the database
        db = SessionLocal()
        try:
            # Fetch top 5 most similar chunks
            results = db.query(TextChunk).order_by(
                TextChunk.embedding.l2_distance(query_embedding)
            ).limit(1).all()
            
            # 3. Convert results to LangChain Document objects
            documents = []
            for row in results:
                doc = Document(
                    page_content=row.chunk_text,
                    metadata={"source": row.document_name, "id": row.id, "similarity": None, "chunk_text": row.chunk_text}
                )
                documents.append(doc)
            
            return documents
        finally:
            db.close()

class CustomPGRetriever(BaseRetriever):
    """
    A Custom Retriever for LangChain that uses our existing SQLAlchemy session,
    local sentence-transformers model, PostgreSQL pgvector (Cosine Similarity), 
    and a Jina Cross-Encoder Reranker.
    """
    document_name: Optional[str] = None  # Optional filter for specific document
    top_k: int = RETRIEVER_HYPERPARAMETERS["top_k"]                      # expanded retrieval pool
    top_n_rerank: int = RETRIEVER_HYPERPARAMETERS["top_n_rerank"]                # lowered to save Groq API tokens (max 8k TPM)
    rrf_k: int = RETRIEVER_HYPERPARAMETERS["rrf_k"] # Constant for Reciprocal Rank Fusion

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        
        # 1. Embed the user query
        query_embedding = embedding_model.encode(query).tolist()
        
        # 2. Query the database using Vector Similarity
        db = SessionLocal()
        final_documents = []
        try:
            # Define the distance expression using Cosine Distance
            chunk_distance = TextChunk.embedding.cosine_distance(query_embedding)
            
            # Query BOTH the TextChunk object AND the calculated distance
            query_obj = db.query(TextChunk, chunk_distance)
            if self.document_name:
                query_obj = query_obj.filter(TextChunk.document_name == self.document_name)
            
            # Fetch the top distinct chunks, fetching top_k for the reranker pool
            vector_results = query_obj.order_by(chunk_distance).limit(self.top_k).all()
            
            # 3. Rerank Results using Jina Cross-Encoder
            for chunk_data, score in vector_results:
                 rerank_query = f"Query: {query}\nDocument: {chunk_data.chunk_text}"
                 try:
                     # GGUF extracts logit directly to the first element of embedding vector
                     rerank_res = reranker_model.create_embedding(rerank_query)
                     rerank_score = float(rerank_res['data'][0]['embedding'][0])
                 except Exception as e:
                     print(f"Reranking generation failed for chunk {chunk_data.id}: {e}")
                     rerank_score = -999.0 # penalize failed rows
                     
                 doc = Document(
                     page_content=chunk_data.chunk_text,
                     metadata={
                         "source": chunk_data.document_name,
                         "id": chunk_data.id,
                         "vector_distance": float(score),
                         "rerank_score": rerank_score,
                         "chunk_text": chunk_data.chunk_text
                     }
                 )
                 final_documents.append(doc)
                 
            # Sort descending (higher rerank score logit represents better relevance)
            final_documents.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
            
            # Trim the final result down to the absolute best N chunks
            return final_documents[:self.top_n_rerank]
            
        finally:
            db.close()

        return final_documents

# --- 5. API ENDPOINT DEFINITION ---

class UploadResponse(BaseModel):
    message: str
    filenames: List[str]

@app.post("/upload-pdfs/", response_model=UploadResponse, status_code=200)
async def upload_pdfs_and_process(
    files: List[UploadFile] = File(...)
):
    """
    Accepts multiple PDF files, saves them, and processes them synchronously.
    """
    filenames = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue

        # Read file contents into memory
        file_contents = await file.read()

        # Process immediately (synchronously)
        process_pdf_pipeline(file_contents, file.filename)
        filenames.append(file.filename)
        
    if not filenames:
        raise HTTPException(status_code=400, detail="No valid PDF files were uploaded.")

    return {
        "message": f"Successfully processed {len(filenames)} files. Ready for chat.",
        "filenames": filenames,
    }

@app.post("/clear-db/", status_code=200)
async def clear_database():
    """
    Clears all data from the database.
    """
    db = SessionLocal()
    try:
        db.query(TextChunk).delete()
        db.commit()
        return {"message": "All database records have been deleted."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# --- 6. A Simple Search Endpoint to Verify ---

class SearchResult(BaseModel):
    document_name: str
    chunk_text: str
    similarity: float

@app.get("/search/", response_model=List[SearchResult])
async def search_similar_chunks(query: str):
    """
    Searches for text chunks semantically similar to the query.
    """
    query_embedding = embedding_model.encode(query).tolist()
    
    db = SessionLocal()
    try:
        # Use the l2_distance operator (<->) from pgvector for similarity search
        results = db.query(
            TextChunk.document_name,
            TextChunk.chunk_text,
            TextChunk.embedding.l2_distance(query_embedding).label('similarity')
        ).order_by('similarity').limit(RETRIEVER_HYPERPARAMETERS["base_similarity_limit"]).all()
        
        return [
            SearchResult(
                document_name=r.document_name, 
                chunk_text=r.chunk_text, 
                similarity=r.similarity
            ) for r in results
        ]
    finally:
        db.close()


# --- 7. POST /chat/ Endpoint ---

# --- NEW: GLOBAL MEMORY STORE (In-Memory for now) ---
# In production, you would store this in Redis or PostgreSQL
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- UPDATED: POST /chat/ Endpoint ---

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default_session" # Client sends a unique ID (e.g., user ID)
    document_name: str = None # Client specifies which document to chat with

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str

@app.post("/chat/", response_model=ChatResponse)
async def chat_with_docs(request: ChatRequest):
    """
    Conversational RAG Endpoint.
    Maintains history per 'session_id'.
    """
    try:
        # 1. Setup Components
        retriever = CustomPGRetriever(document_name=request.document_name)
        
        print(f"Loading local LLM from {MODEL_PATH}...")
        llm = ChatLlamaCpp(
            model_path=MODEL_PATH,
            chat_format="llama3",
            **LLM_HYPERPARAMETERS
        )

        # 2. Create "History Aware" Retriever
        # Llama 3.2 Chat Template Format
        contextualize_q_system_prompt = LLM_CHAT_TEMPLATE["contextualize_q_system_prompt"]
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # 3. Create the QA Chain (Answer Generation)
        qa_system_prompt = LLM_CHAT_TEMPLATE["qa_system_prompt"]
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        document_prompt = PromptTemplate.from_template(
            "Source: [{source}]\nContent: {page_content}"
        )
        
        question_answer_chain = create_stuff_documents_chain(
            llm, qa_prompt, document_prompt=document_prompt
        )
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # 4. Wrap with Message History
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # 5. Invoke
        start_time = time.time()
        response = conversational_rag_chain.invoke(
            {"input": request.question},
            config={"configurable": {"session_id": request.session_id}}
        )
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        # 6. Extract results
        answer = response["answer"]
        source_docs = response.get("context", [])
        sources = list(set([doc.metadata["source"] + " (Score: " + str(doc.metadata.get("rerank_score", "N/A")) + " Chunk text: " + str(doc.metadata.get("chunk_text", "N/A")) + ")" for doc in source_docs]))

        # 7. Format retrieved docs for history
        retrieved_docs_log = []
        for doc in source_docs:
            retrieved_docs_log.append({
                "id": str(doc.metadata.get("id", "N/A")),
                "text": doc.page_content,
                "score": doc.metadata.get("rerank_score", 0.0)
            })

        # 8. Reconstruct the full prompt for logging
        context_text = "\n\n".join([f"Source: [{doc.metadata['source']}]\nContent: {doc.page_content}" for doc in source_docs])
        try:
            full_prompt_log = qa_prompt.format(
                context=context_text,
                chat_history=get_session_history(request.session_id).messages,
                input=request.question
            )
        except Exception:
            full_prompt_log = "Could not reconstruct full prompt."

        # 9. Save chat session to JSON
        import json
        history_dir = os.path.join(os.path.dirname(__file__), "chat history")
        os.makedirs(history_dir, exist_ok=True)
        history_file = os.path.join(history_dir, f"{request.session_id}.json")
        
        session_data = []
        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
            except Exception:
                pass
                
        session_data.append({
            "query": request.question,
            "top_k": RETRIEVER_HYPERPARAMETERS["top_n_rerank"],
            "retrieved_docs": retrieved_docs_log,
            "prompt": full_prompt_log,
            "answer": answer,
            "latency_ms": latency_ms
        })
        
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=4, ensure_ascii=False)

        return ChatResponse(
            answer=answer, 
            sources=sources, 
            session_id=request.session_id
        )

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
def shutdown_event():
    """
    Clears the database embeddings when the FastAPI server is shut down.
    """
    if config.get("flush_database_on_shutdown", True):
        print("Shutting down... Clearing database chunks.")
        db = SessionLocal()
        try:
            db.query(TextChunk).delete()
            db.commit()
            print("Database cleared successfully.")
        except Exception as e:
            print(f"Error clearing database during shutdown: {e}")
            db.rollback()
        finally:
            db.close()
    else:
        print("Shutting down... Keeping database chunks as per configuration.")


# To run the app: uvicorn main_groq:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
