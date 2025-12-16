import os
import io
from pathlib import Path
from dotenv import load_dotenv
from typing import List

import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel

# --- Database Setup (SQLAlchemy) ---
from sqlalchemy import create_engine, Column, Integer, String, Text, Index
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector

# --- ML/AI Model Imports ---
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf

# --- NEW: LangChain Imports for RAG ---
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever


# Load environment variables from .env file
load_dotenv()

home_env_path = Path.home() / ".env"
load_dotenv(dotenv_path=home_env_path)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cbuser:12345@localhost:5432/knowledgedb")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_ONLINE_EXAM") # Required for the Generation part

# --- 1. CORE SETUP: FastAPI App, DB Connection, and Global Model ---

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal RAG Pipeline (FastAPI + PostgreSQL pgvector)",
    description="A prototype for ingesting, chunking, and vectorizing PDF documents.",
)

# Database connection
try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
except Exception as e:
    print(f"Error connecting to the database: {e}")
    # Exit or handle gracefully if DB connection is critical at startup
    exit()

# Load the embedding model ONCE when the application starts.
# This is a crucial optimization to avoid reloading the model on every request.
try:
    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    # embedding_model = SentenceTransformer("tencent/KaLM-Embedding-Gemma3-12B-2511")
    # The dimension of this model is 384
    print("Embedding model loaded successfully.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    exit()

# --- 2. DATABASE MODEL DEFINITION ---

class TextChunk(Base):
    __tablename__ = "text_chunks"
    id = Column(Integer, primary_key=True, index=True)
    document_name = Column(String, index=True)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(384)) # Dimension must match the model

    __table_args__ = (
        # Create an HNSW index on the 'embedding' column
        # 'vector_l2_ops' means we are optimizing for Euclidean distance (L2)
        Index(
            'chunk_embedding_idx', 
            embedding, 
            postgresql_using='hnsw', 
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_l2_ops'}
        ),
    )

# --- 3. THE CORE PIPELINE LOGIC ---

def process_pdf_pipeline(file_contents: bytes, filename: str):
    """
    The main data processing pipeline function that runs in the background.
    """
    print(f"Starting processing for document: {filename}")
    
    # A. Ingests PDF files & B. Performs Text extraction
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_contents))
        full_text = "".join([page.extract_text() for page in reader.pages])
        print(f"Extracted {len(full_text)} characters from {filename}")
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        return # Stop processing if text extraction fails

    # C. Performs chunking of text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(full_text)
    print(f"Split text into {len(chunks)} chunks.")
    if not chunks:
        print(f"No text chunks were created for {filename}. Aborting.")
        return

    # D. Vectorises the chunks using some embedding model
    try:
        embeddings = embedding_model.encode(chunks, show_progress_bar=False).tolist()
        print(f"Generated {len(embeddings)} embeddings.")
    except Exception as e:
        print(f"Error generating embeddings for {filename}: {e}")
        return

    # E. Stores the vectors in PostgreSQL
    db = SessionLocal()
    try:
        for i, chunk_text in enumerate(chunks):
            new_chunk = TextChunk(
                document_name=filename,
                chunk_text=chunk_text,
                embedding=embeddings[i]
            )
            db.add(new_chunk)
        
        db.commit()
        print(f"Successfully saved {len(chunks)} chunks and embeddings to the database for {filename}.")
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
    local sentence-transformers model, and PostgreSQL pgvector.
    """
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        
        # 1. Embed the user query
        query_embedding = embedding_model.encode(query).tolist()
        
        # 2. Query the database
        db = SessionLocal()
        try:
            # Define the distance expression
            chunk_distance = TextChunk.embedding.l2_distance(query_embedding)
            
            # Query BOTH the TextChunk object AND the calculated distance
            # This returns a list of tuples: [(TextChunkObject, distance_float), ...]
            results = db.query(TextChunk, chunk_distance).order_by(chunk_distance).limit(1).all()
            
            # 3. Convert results to LangChain Document objects
            documents = []
            
            # Note: We now unpack two values (chunk_data, score)
            for chunk_data, score in results:
                print(f"Chunk ID: {chunk_data.id}, Similarity Score (L2 Distance): {score}")
                doc = Document(
                    page_content=chunk_data.chunk_text,
                    metadata={
                        "source": chunk_data.document_name, 
                        "id": chunk_data.id,
                        "chunk_text": chunk_data.chunk_text,
                        # 'score' is the L2 distance (Lower is better/closer)
                        "similarity": float(score) 
                    }
                )
                documents.append(doc)
            
            return documents
        finally:
            db.close()

# --- 5. API ENDPOINT DEFINITION ---

class UploadResponse(BaseModel):
    message: str
    filename: str

@app.post("/upload-pdf/", response_model=UploadResponse, status_code=202)
async def upload_pdf_and_process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Accepts a PDF file, saves it, and starts the processing pipeline in the background.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Read file contents into memory
    file_contents = await file.read()

    # Add the long-running task to the background
    background_tasks.add_task(process_pdf_pipeline, file_contents, file.filename)

    return {
        "message": "File upload successful. Processing has started in the background.",
        "filename": file.filename,
    }

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
        ).order_by('similarity').limit(5).all()
        
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
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set.")

    try:
        # 1. Setup Components
        retriever = CustomPGRetriever()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)

        # 2. Create "History Aware" Retriever
        # This step rephrases the user's latest question based on history.
        # Example: User says "What about revenue?" -> LLM changes it to "What is the revenue of Apple?" -> Retriever searches DB.
        contextualize_q_system_prompt = """Given a chat history and the latest user question 
        which might reference context in the chat history, formulate a standalone question 
        which can be understood without the chat history. Do NOT answer the question, 
        just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # 3. Create the QA Chain (Answer Generation)
        qa_system_prompt = """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        
        {context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
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
        response = conversational_rag_chain.invoke(
            {"input": request.question},
            config={"configurable": {"session_id": request.session_id}}
        )

        # 6. Extract results
        answer = response["answer"]
        source_docs = response.get("context", [])
        sources = list(set([doc.metadata["source"] + " (Similarity: " + str(doc.metadata.get("similarity", "N/A")) + " Chunk text: " + str(doc.metadata.get("chunk_text", "N/A")) + ")" for doc in source_docs]))

        return ChatResponse(
            answer=answer, 
            sources=sources, 
            session_id=request.session_id
        )

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# To run the app: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)