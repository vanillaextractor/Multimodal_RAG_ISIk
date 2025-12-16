from sqlalchemy.orm import sessionmaker
from models import get_engine, TextChunk

# 1. Connect to DB
engine = get_engine()
Session = sessionmaker(bind=engine)
session = Session()

# 2. Fetch all rows
chunks = session.query(TextChunk).all()

print(f"Total rows found: {len(chunks)}")
print("-" * 30)

# 3. Print details
for chunk in chunks:
    print(f"ID: {chunk.id}")
    print(f"File: {chunk.document_name}")
    print(f"Text Snippet: {chunk.chunk_text[:50]}...") # First 50 chars
    # Check if embedding exists (it's a list/array)
    if chunk.embedding is not None:
        print(f"Vector: Present (First 3 values: {chunk.embedding[:3]}...)")
    else:
        print("Vector: NULL")
    print("-" * 30)