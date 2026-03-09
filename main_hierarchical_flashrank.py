import os
import io
import time
import re
import json
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, Integer, String, Text, text
from sqlalchemy.orm import sessionmaker
from models import Base, TextChunk

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from flashrank import Ranker, RerankRequest

# --- 1. CONFIGURATION & SETUP ---
load_dotenv(override=True)
home_env_path = Path.home() / ".env"
load_dotenv(dotenv_path=home_env_path, override=True)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cbuser:12345@localhost:5432/knowledgedb")

# Initialize FastAPI app
app = FastAPI(
    title="Hierarchical RAG Pipeline (FlashRank)",
    description="A RAG pipeline using section-based hierarchical chunking and FlashRank reranking.",
)

# Load Hierarchical Configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "hierarchical_config.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

MODEL_PATH = os.path.abspath(config["llm_path"])
EMBEDDING_MODEL_NAME = config["embedding_model"]
CHUNKING_PARAMS = config["chunking_tokens"]
LLM_HYPERPARAMETERS = config["llm_hyperparameters"]
RETRIEVER_HYPERPARAMETERS = config["retriever_hyperparameters"]
LLM_CHAT_TEMPLATE = config["llm_chat_template"]

# Database connection
try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    print(f"Error connecting to the database: {e}")
    exit()

# Load Models
try:
    print(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print("Loading FlashRank Ranker...")
    # Using a standard lightweight model
    reranker_model = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./flashrank_cache")
    
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# --- 2. HIERARCHICAL CHUNKING LOGIC ---

def split_by_headings(text: str) -> List[Tuple[str, str]]:
    """
    Splits text into sections based on heading patterns.
    Returns a list of (heading, content) tuples.
    """
    heading_pattern = re.compile(r'(?m)^(?:(?:[0-9]+\.[0-9.]*|[A-Z]\.)\s+[A-Z].+|[A-Z\s]{5,30}$|Chapter\s+[0-9]+.*|Section\s+[0-9]+.*)')
    
    matches = list(heading_pattern.finditer(text))
    if not matches:
        return [("General Context", text)]
    
    sections = []
    last_pos = 0
    current_heading = "General Context"
    
    for match in matches:
        content = text[last_pos:match.start()].strip()
        if content:
            sections.append((current_heading, content))
        current_heading = match.group().strip()
        last_pos = match.end()
    
    # Add final section
    final_content = text[last_pos:].strip()
    if final_content:
        sections.append((current_heading, final_content))
        
    return sections

def hierarchical_chunking(text: str, filename: str) -> Tuple[List[str], List[dict]]:
    """
    Implementation of section-based hierarchical chunking.
    1. Splits by headings.
    2. Splits each section into smaller chunks if they exceed the token limit.
    """
    sections = split_by_headings(text)
    final_chunks = []
    final_metadata = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNKING_PARAMS["max_size_default"],
        chunk_overlap=CHUNKING_PARAMS["overlap"],
        separators=["\n\n", "\n", " ", ""]
    )
    
    for heading, section_content in sections:
        chunks = text_splitter.split_text(section_content)
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 10: continue
            
            # Metadata injection: Anchoring the chunk to its section
            formatted_chunk = (
                f"Document: {filename}\n"
                f"Section: {heading}\n"
                f"Chunk: {i+1}/{len(chunks)}\n\n"
                f"Content:\n{chunk.strip()}"
            ).replace("\x00", "")
            
            final_chunks.append(formatted_chunk)
            final_metadata.append({
                "section": heading,
                "page_number": 1, 
                "element_type": "HierarchicalChunk"
            })
            
    return final_chunks, final_metadata

def process_pdf_hierarchical(file_contents: bytes, filename: str):
    print(f"Starting hierarchical processing for: {filename}")
    
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
        temp_pdf.write(file_contents)
        temp_pdf.flush()
        
        try:
            reader = pypdf.PdfReader(temp_pdf.name)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error parsing PDF: {e}")
            return
            
    chunk_texts, chunk_metadata = hierarchical_chunking(full_text, filename)
    
    if not chunk_texts:
         print(f"No chunks generated for {filename}.")
         return

    embeddings = embedding_model.encode(chunk_texts, show_progress_bar=False).tolist()
    
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
    except Exception as e:
        print(f"Database error: {e}")
        db.rollback()
    finally:
        db.close()

# --- 3. RETRIEVER & API ENDPOINTS ---

class CustomPGRetriever(BaseRetriever):
    document_name: Optional[str] = None
    top_k: int = RETRIEVER_HYPERPARAMETERS["top_k"]
    top_n_rerank: int = RETRIEVER_HYPERPARAMETERS["top_n_rerank"]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        query_embedding = embedding_model.encode(query).tolist()
        db = SessionLocal()
        try:
            chunk_distance = TextChunk.embedding.cosine_distance(query_embedding)
            query_obj = db.query(TextChunk, chunk_distance)
            if self.document_name:
                query_obj = query_obj.filter(TextChunk.document_name == self.document_name)
            
            vector_results = query_obj.order_by(chunk_distance).limit(self.top_k).all()
            
            if not vector_results:
                return []

            # Prepare docs for FlashRank
            passages = []
            for chunk_data, score in vector_results:
                passages.append({
                    "id": chunk_data.id,
                    "text": chunk_data.chunk_text,
                    "metadata": {"source": chunk_data.document_name}
                })
            
            rerank_request = RerankRequest(query=query, passages=passages)
            rerank_results = reranker_model.rerank(rerank_request)
            
            final_documents = []
            for res in rerank_results:
                doc = Document(
                    page_content=res["text"],
                    metadata={
                        "source": res["metadata"]["source"],
                        "id": res["id"],
                        "rerank_score": float(res["score"]),
                        "chunk_text": res["text"]
                    }
                )
                final_documents.append(doc)
                 
            return final_documents[:self.top_n_rerank]
        finally:
            db.close()

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default_session"
    document_name: str = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

@app.post("/upload-pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    filenames = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"): continue
        file_contents = await file.read()
        process_pdf_hierarchical(file_contents, file.filename)
        filenames.append(file.filename)
    return {"message": f"Processed {len(filenames)} files.", "filenames": filenames}

# Global LLM instance
print(f"Loading local LLM from {MODEL_PATH}...")
llm = ChatLlamaCpp(
    model_path=MODEL_PATH,
    chat_format="llama3",
    **LLM_HYPERPARAMETERS
)

@app.post("/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # 1. Setup Components
        retriever = CustomPGRetriever(document_name=request.document_name)

        # 2. Create "History Aware" Retriever
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
    if config.get("flush_database_on_shutdown", True):
        db = SessionLocal()
        try:
            db.query(TextChunk).delete()
            db.commit()
        finally:
            db.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
