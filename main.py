import os
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables (for Render and local .env)
load_dotenv()

import document_processor
import vector_store_manager
import llm_interface

app = FastAPI()

# --- CORS CONFIGURATION ---
# IMPORTANT: Update this with your live frontend URL when deploying
origins = [
    "https://your-netlify-frontend.netlify.app", # <-- Replace this placeholder
    "http://localhost",
    "http://127.0.0.1",
    "null"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE CONNECTION (No change here) ---
def get_db_connection():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set.")
    if "sslmode" not in database_url:
        database_url += "?sslmode=require"
    conn = psycopg2.connect(database_url)
    return conn

# --- NEW: Initialize LangChain Components (globally) ---
# This is more efficient as they are loaded only once when the server starts.
# We are no longer using the document_processor for embeddings in this file.
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# --- API ENDPOINTS ---

# The /config endpoint remains exactly the same
@app.get("/config/{business_id}")
def get_config(business_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute('SELECT * FROM businesses WHERE id = %s', (business_id,))
        business = cursor.fetchone()
        cursor.close()
        conn.close()
        if business is None:
            raise HTTPException(status_code=404, detail="Business not found")
        return dict(business)
    except Exception as e:
        print(f"ERROR in /config: {e}")
        raise HTTPException(status_code=500, detail="API Error")

class ChatRequest(BaseModel):
    question: str
    businessId: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """
    Handles chat requests using a LangChain RetrievalQA chain.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute('SELECT * FROM businesses WHERE id = %s', (request.businessId,))
        business = cursor.fetchone()
        
        if business is None:
            raise HTTPException(status_code=404, detail="Business configuration not found")

        # --- LANGCHAIN RAG IMPLEMENTATION ---

        # 1. Define the path to the pre-built FAISS index
        index_path = os.path.join("data", request.businessId)
        if not os.path.exists(index_path) or not os.path.exists(os.path.join(index_path, "faiss_index.bin")):
            # This handles the "amnesia" problem on cloud providers
            return {"answer": "I'm sorry, the knowledge base for this business has not been trained or is currently unavailable. Please try again later."}
        
        # 2. Load the FAISS vector store and create a retriever
        vector_store = FAISS.load_local(
            index_path,
            embeddings_model,
            index_name="faiss_index.bin",
            allow_dangerous_deserialization=True # Required for FAISS with LangChain
        )
        retriever = vector_store.as_retriever(search_kwargs={'k': 3}) # Retrieve top 3 documents

        # 3. Define the LLM from LangChain Groq
        llm = ChatGroq(
            temperature=0.7,
            model_name="llama3-70b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        # 4. Create a custom prompt template to guide the LLM's personality and instructions
        prompt_template = """
        You are a {personality} AI assistant for the company '{name}'.
        Your primary goal is to answer the user's question based only on the following context.
        If you don't know the answer from the context provided, you MUST say that you don't have information on that topic. Do not try to make up an answer.

        Context:
        {context}

        Question:
        {question}

        Helpful Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "personality", "name"]
        )

        # 5. Create the RetrievalQA chain
        # This powerful chain handles the entire RAG process automatically.
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # "stuff" means it crams all retrieved docs into the prompt
            retriever=retriever,
            chain_type_kwargs={
                "prompt": PROMPT.partial(
                    personality=business['personality'],
                    name=business['name']
                )
            },
            return_source_documents=False
        )

        # 6. Run the chain with the user's question to get the result
        result = qa_chain.invoke({"query": request.question})
        final_answer = result.get("result", "Sorry, I encountered an issue while generating a response.")

        # --- END OF LANGCHAIN IMPLEMENTATION ---

        # Log the interaction in the database (this part remains the same)
        cursor.execute('INSERT INTO chat_logs (business_id, question, answer) VALUES (%s, %s, %s)', (request.businessId, request.question, final_answer))
        conn.commit()
        cursor.close()
        conn.close()
        
        return {"answer": final_answer}
    except Exception as e:
        print(f"ERROR in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)