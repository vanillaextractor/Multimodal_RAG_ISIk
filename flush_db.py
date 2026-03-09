import os
import glob
import shutil
from sqlalchemy.orm import sessionmaker
from models import get_engine, TextChunk

def flush_database():
    print("--- Flashing Database ---")
    try:
        engine = get_engine()
        Session = sessionmaker(bind=engine)
        session = Session()
        
        num_deleted = session.query(TextChunk).delete()
        session.commit()
        print(f"Deleted {num_deleted} records from 'text_chunks' table.")
    except Exception as e:
        print(f"Error flushing PostgreSQL: {e}")
    finally:
        session.close()

def flush_bm25_indices():
    print("--- Flashing BM25 Indices ---")
    root_dir = os.path.dirname(__file__)
    bm25_files = glob.glob(os.path.join(root_dir, "bm25*.pkl"))
    for f in bm25_files:
        try:
            os.remove(f)
            print(f"Deleted BM25 index: {os.path.basename(f)}")
        except Exception as e:
            print(f"Error deleting {f}: {e}")

def main():
    confirm = input("This will delete ALL knowledge base data (PostgreSQL records and BM25 indices). Chat history will be preserved. Are you sure? (y/N): ")
    if confirm.lower() == 'y':
        flush_database()
        flush_bm25_indices()
        print("\nKnowledge base flushed successfully. Chat history remains intact.")
    else:
        print("Flush cancelled.")

if __name__ == "__main__":
    main()
