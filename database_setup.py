import os
import psycopg2
import streamlit as st

def get_db_connection_for_setup():
    """Gets DB connection using Streamlit secrets."""
    database_url = st.secrets["DATABASE_URL"]
    if "sslmode" not in database_url:
        database_url += "?sslmode=require"
    return psycopg2.connect(database_url)

def setup_database():
    """Creates the necessary tables in the PostgreSQL database."""
    conn = get_db_connection_for_setup()
    cursor = conn.cursor()

    # Table to store business information and customizations
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS businesses (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        agent_name TEXT DEFAULT 'AI Assistant',
        welcome_message TEXT DEFAULT 'Hi! How can I help you today?',
        personality TEXT DEFAULT 'friendly',
        brand_color TEXT DEFAULT '#007bff'
    )
    ''')

    # Table to log chat interactions for analytics
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_logs (
        log_id SERIAL PRIMARY KEY,
        business_id TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (business_id) REFERENCES businesses (id)
    )
    ''')

    conn.commit()
    cursor.close()
    conn.close()
    print("Database setup or verification complete.")