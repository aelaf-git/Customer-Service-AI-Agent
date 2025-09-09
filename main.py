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
from langchain_huggingface import HuggingFaceEmbeddings # Switched to HuggingFace for consistency
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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

# --- INITIALIZE ADVANCED LANGCHAIN COMPONENTS (GLOBALLY) ---
# This is efficient as they are loaded only once when the server starts.
print("Loading HuggingFace embedding model...")
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
print("Loading HuggingFace cross-encoder model for reranking...")
cross_encoder_model = HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', model_kwargs={'device': 'cpu'})
print("Models loaded successfully.")

# --- API ENDPOINTS ---

@app.get("/config/{business_id}")
def get_config(business_id: str):
    """Fetches the configuration for a specific business."""
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
    Handles chat requests using an advanced RAG chain with a Cross-Encoder Reranker.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute('SELECT * FROM businesses WHERE id = %s', (request.businessId,))
        business = cursor.fetchone()
        
        if not business:
            raise HTTPException(status_code=404, detail="Business configuration not found")

        # --- ADVANCED LANGCHAIN RAG IMPLEMENTATION ---

        # 1. Load the FAISS vector store from the local file system
        index_path = os.path.join("data", request.businessId)
        if not os.path.exists(index_path) or not os.path.exists(os.path.join(index_path, "faiss_index.bin")):
            return {"answer": "I'm sorry, the knowledge base for this business is currently unavailable. Please ask the administrator to train the agent."}
        
        vector_store = FAISS.load_local(
            index_path,
            embeddings_model,
            index_name="faiss_index.bin",
            allow_dangerous_deserialization=True
        )
        
        # 2. Create the Retriever and the Reranker
        base_retriever = vector_store.as_retriever(search_kwargs={'k': 5}) # Get 5 potentially relevant docs
        reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=2) # Filter down to the best 2
        
        # 3. Create the Compression Retriever to combine the two steps
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=base_retriever
        )

        # 4. Define the LLM to use for generation
        llm = ChatGroq(
            temperature=0.7,
            model_name="llama3-70b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        # 5. Define the Prompt Template with personality
        prompt_template = """
        You are a {personality} AI assistant for the company '{name}'.
        Your primary goal is to answer the user's question based *only* on the following context.
        If the information is not in the context, say that you don't have information on that topic. Do not make up answers.
        Keep the answer concise and helpful.

        Context:
        {context}

        Question:
        {question}

        Answer:"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        ).partial(personality=business['personality'], name=business['name'])

        # 6. Define a helper function to format the retrieved documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 7. Build the final RAG chain using LangChain Expression Language (LCEL)
        rag_chain = (
            {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # 8. Run the chain to get the final answer
        final_answer = rag_chain.invoke(request.question)
        
        # --- END OF ADVANCED IMPLEMENTATION ---
        
        # 9. Log the interaction to the database
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