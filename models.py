from sqlalchemy import Column, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Define the Base for our models ---
Base = declarative_base()

# --- Define the Application's Schema as a Python Class ---
class TextChunk(Base):
    """
    Represents a chunk of text from a document.
    This class definition is the schema for the 'text_chunks' table.
    """
    __tablename__ = 'text_chunks'

    id = Column(Integer, primary_key=True, index=True)
    document_name = Column(String(255), index=True, nullable=False)
    chunk_text = Column(Text, nullable=False)
    
    # Structural Metadata
    page_number = Column(Integer, index=True, nullable=True)
    element_type = Column(String(100), index=True, nullable=True)
    
    # The VECTOR(384) specifies the dimension of the embeddings.
    # This MUST match the output dimension of your sentence-transformer model.
    # 'all-MiniLM-L6-v2' has a dimension of 384.
    embedding = Column(Vector(384))

    def __repr__(self):
        return f"<TextChunk(id={self.id}, document='{self.document_name}')>"

# --- Utility to create an engine ---
def get_engine():
    """Creates a SQLAlchemy engine from the DATABASE_URL."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable not set.")
    return create_engine(DATABASE_URL)