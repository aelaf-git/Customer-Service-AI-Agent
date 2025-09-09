import os
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# --- UPGRADED LANGCHAIN IMPORTS ---
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain # <-- IMPORT THE CONVERSATIONAL CHAIN
from langchain.memory import ConversationBufferMemory   # <-- IMPORT THE MEMORY BUFFER
from langchain.prompts import PromptTemplate

# Load environment variables from .env file for local development
load_dotenv()

# Custom module imports (still needed for the dashboard)
import document_processor
import vector_store_manager
import llm_interface

app = FastAPI()

# --- CORS CONFIGURATION ---
# IMPORTANT: When you deploy, update this with your live frontend URL.
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

# --- DATABASE CONNECTION ---
def get_db_connection():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set.")
    if "sslmode" not in database_url:
        database_url += "?sslmode=require"
    conn = psycopg2.connect(database_url)
    return conn

# --- INITIALIZE LANGCHAIN COMPONENTS (GLOBALLY) ---
# This is efficient as they are loaded only once when the server starts.
print("Loading HuggingFace embedding and cross-encoder models...")
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
cross_encoder_model = HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', model_kwargs={'device': 'cpu'})
print("Models loaded successfully.")

# --- SESSION MEMORY MANAGEMENT ---
# In a production app, you'd use a more persistent store like Redis.
# For now, a simple dictionary will hold memory for active sessions.
# The key will be the businessId, and the value will be the memory object.
session_memory = {}

# --- API ENDPOINTS ---

@app.get("/config/{business_id}")
def get_config(business_id: str):
    """Fetches the configuration for a specific business."""
    # This also serves as a good time to clear any old memory for a new session.
    if business_id in session_memory:
        del session_memory[business_id]
        print(f"Cleared stale memory for session: {business_id}")

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
        print(f"ERROR in /config endpoint: {e}")
        raise HTTPException(status_code=500, detail="Database connection error.")

class ChatRequest(BaseModel):
    """Defines the structure of a chat request."""
    question: str
    businessId: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """
    Handles chat requests using a ConversationalRetrievalChain with memory.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute('SELECT * FROM businesses WHERE id = %s', (request.businessId,))
        business = cursor.fetchone()
        
        if not business:
            raise HTTPException(status_code=404, detail="Business configuration not found")

        # --- CONVERSATIONAL RAG IMPLEMENTATION ---

        # 1. Load the FAISS vector store
        index_path = os.path.join("data", request.businessId)
        if not os.path.exists(os.path.join(index_path, "faiss_index.bin")):
            return {"answer": "I'm sorry, the knowledge base for this business is currently unavailable. Please ask the administrator to train the agent."}
        
        vector_store = FAISS.load_local(index_path, embeddings_model, index_name="faiss_index.bin", allow_dangerous_deserialization=True)
        
        # 2. Create the Retriever with Reranker
        base_retriever = vector_store.as_retriever(search_kwargs={'k': 5})
        reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=2)
        compression_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)

        # 3. Define the LLM
        llm = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))

        # 4. Get or create a memory buffer for this specific business's session
        if request.businessId not in session_memory:
            session_memory[request.businessId] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key='answer' # Crucial for the chain to know where the answer is
            )
        memory = session_memory[request.businessId]

        # 5. Create the ConversationalRetrievalChain
        # This powerful chain handles memory, question rephrasing, and RAG.
        # We can also add a custom prompt for the final answer generation.
        condense_question_prompt = PromptTemplate.from_template(
            "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:"
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=compression_retriever,
            memory=memory,
            condense_question_prompt=condense_question_prompt,
            return_source_documents=False
        )

        # 6. Run the chain with the user's question to get the result
        result = qa_chain.invoke({"question": request.question})
        final_answer = result.get("answer", "Sorry, I encountered an issue while generating a response.")
        
        # --- END OF CONVERSATIONAL IMPLEMENTATION ---
        
        # 7. Log the interaction to the database
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
    print(f"Starting local backend server on http://0.0.0.0:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)