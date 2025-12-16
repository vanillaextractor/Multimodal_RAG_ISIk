import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError, OperationalError
from dotenv import load_dotenv
from urllib.parse import urlparse

# Import your Base and models from models.py
from models import Base 

def create_database_and_schema():
    """
    Connects to PostgreSQL, creates the database if it doesn't exist,
    enables the pgvector extension, and creates tables based on SQLAlchemy models.
    """
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Error: DATABASE_URL environment variable is not set.")
        return

    # --- Part 1: Parse the connection URL to get database name and connection info ---
    try:
        parsed_url = urlparse(db_url)
        db_name = parsed_url.path[1:] # The path component, without the leading '/'
        
        # Connection string for the PostgreSQL server (without the database name)
        server_conn_str = db_url.replace(f"/{db_name}", "/postgres")
        
        print(f"Database to be created: '{db_name}'")
    except Exception as e:
        print(f"Error parsing DATABASE_URL: {e}")
        return

    # --- Part 2: Connect to the server and create the database ---
    conn = None
    try:
        # Use psycopg2 to connect to the default 'postgres' database
        conn = psycopg2.connect(server_conn_str)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if the database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Database '{db_name}' does not exist. Creating...")
            cursor.execute(f"CREATE DATABASE {db_name}")
            print(f"Database '{db_name}' created successfully.")
        else:
            print(f"Database '{db_name}' already exists.")
            
        cursor.close()
    except OperationalError as e:
        print(f"Could not connect to PostgreSQL server: {e}")
        return
    except Exception as e:
        print(f"An error occurred during database creation: {e}")
        return
    finally:
        if conn:
            conn.close()

    # --- Part 3: Connect to the new database and create schema ---
    try:
        # Now connect to the newly created database using SQLAlchemy
        engine = create_engine(db_url)
        with engine.connect() as connection:
            # Step 3a: Enable the pgvector extension
            print("Enabling pgvector extension...")
            # 1. Create the extension
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            print("pgvector extension enabled.")

            # 2. CRITICAL FIX: Explicitly commit the extension creation
            connection.commit()
            print("pgvector extension enabled and committed.")

            # Step 3b: Create all tables defined in models.py
            print("Creating tables from SQLAlchemy models...")
            # Base.metadata.create_all() checks for table existence before creating
            Base.metadata.create_all(bind=connection)
            print("Tables  and HNSW Index created successfully (if they didn't already exist).")
            
            print("\nDatabase setup is complete!")

    except Exception as e:
        print(f"An error occurred during schema creation: {e}")

if __name__ == "__main__":
    create_database_and_schema()