import streamlit as st
import psycopg2
import psycopg2.extras
import uuid
import os

# --- UPGRADED LANGCHAIN IMPORTS ---
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Import other custom modules
import document_processor
import vector_store_manager

st.set_page_config(layout="wide", page_title="AI Agent Dashboard")

# --- DATABASE CONNECTION & SETUP ---
def get_db_connection():
    """Establishes a connection to the PostgreSQL database using Streamlit secrets."""
    database_url = st.secrets["DATABASE_URL"]
    if "sslmode" not in database_url:
        database_url += "?sslmode=require"
    conn = psycopg2.connect(database_url)
    return conn

def initialize_database():
    """Checks if tables exist and creates them if they don't. A self-healing function."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT to_regclass('public.businesses')")
    table_exists = cursor.fetchone()[0]
    if not table_exists:
        st.toast("First time setup: Initializing database tables...", icon="🚀")
        cursor.execute('''
        CREATE TABLE businesses (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, agent_name TEXT DEFAULT 'AI Assistant',
            welcome_message TEXT DEFAULT 'Hi! How can I help you today?',
            personality TEXT DEFAULT 'friendly', brand_color TEXT DEFAULT '#007bff'
        )
        ''')
        cursor.execute('''
        CREATE TABLE chat_logs (
            log_id SERIAL PRIMARY KEY, business_id TEXT NOT NULL, question TEXT NOT NULL,
            answer TEXT NOT NULL, timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (business_id) REFERENCES businesses (id)
        )
        ''')
        conn.commit()
        st.toast("Database initialized!", icon="✅")
    cursor.close()
    conn.close()

# --- CACHED LANGCHAIN MODELS ---
# These heavy models are loaded only once and cached for efficiency.
@st.cache_resource
def load_embedding_model():
    print("Loading HuggingFace embedding model for dashboard...")
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

@st.cache_resource
def load_cross_encoder_model():
    print("Loading HuggingFace cross-encoder model for dashboard...")
    return HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', model_kwargs={'device': 'cpu'})

embeddings_model = load_embedding_model()
cross_encoder_model = load_cross_encoder_model()

# --- BUSINESS & CONTENT FUNCTIONS ---
def get_all_businesses():
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute('SELECT * FROM businesses ORDER BY name')
    businesses = cursor.fetchall()
    cursor.close()
    conn.close()
    return businesses

def update_business_settings(business_id, agent_name, welcome_message, personality, brand_color):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE businesses 
        SET agent_name = %s, welcome_message = %s, personality = %s, brand_color = %s
        WHERE id = %s
    ''', (agent_name, welcome_message, personality, brand_color, business_id))
    conn.commit()
    cursor.close()
    conn.close()

def process_and_store_content(business_id, raw_content):
    if not raw_content or not raw_content.strip():
        st.warning("No content to process for this source.")
        return

    st.info(f"Processing content for business {business_id}...")
    with st.spinner("Chunking text, generating embeddings, and updating knowledge base... This may take a moment."):
        text_chunks = document_processor.chunk_text(raw_content)
        # We use the globally loaded embedding model here for consistency
        embeddings = embeddings_model.embed_documents([chunk for chunk in text_chunks])
        index_dir = os.path.join("data", business_id)
        os.makedirs(index_dir, exist_ok=True)
        # Note: embedding_dim must match the model's output dimension
        embedding_dim = len(embeddings[0]) 
        current_index, current_texts = vector_store_manager.create_or_load_faiss_index(business_id, embedding_dimension=embedding_dim)
        vector_store_manager.add_embeddings_to_faiss(
            business_id, embeddings, text_chunks, current_index, current_texts
        )
    st.success(f"Knowledge base updated with {len(text_chunks)} new chunks.")

# --- Main App Execution ---
try:
    initialize_database()
    st.title("🤖 AI Agent Dashboard")
    st.sidebar.header("Business Selection")
    
    businesses = get_all_businesses()
    business_options = {b['name']: b['id'] for b in businesses} if businesses else {}
    selected_name = None

    if business_options:
        selected_name = st.sidebar.selectbox("Select a Business", list(business_options.keys()))
    else:
        st.sidebar.info("No businesses found. Please register one below.")

    with st.sidebar.expander("Register New Business"):
        new_business_name = st.text_input("Enter New Business Name")
        if st.button("Register"):
            if new_business_name:
                if new_business_name in business_options:
                    st.sidebar.error("A business with this name already exists.")
                else:
                    new_id = str(uuid.uuid4())
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute('INSERT INTO businesses (id, name) VALUES (%s, %s)', (new_id, new_business_name))
                    conn.commit()
                    cursor.close()
                    conn.close()
                    st.sidebar.success(f"Business '{new_business_name}' registered!")
                    st.rerun()
            else:
                st.sidebar.error("Business name cannot be empty.")

    if selected_name:
        business_id = business_options[selected_name]
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute('SELECT * FROM businesses WHERE id = %s', (business_id,))
        current_business = cursor.fetchone()
        cursor.close()
        conn.close()

        if current_business:
            st.header(f"Managing: {current_business['name']}")
            tab1, tab2, tab5, tab3, tab4 = st.tabs(["📚 Knowledge Sources", "🎨 Customize Agent", "🧪 Test Your Agent", "🚀 Deploy", "📊 Analytics"])

            with tab1:
                st.subheader("Upload Knowledge Sources")
                st.write("Add PDFs, text files, or scrape a website URL to build your agent's knowledge.")
                
                uploaded_files = st.file_uploader("Upload PDFs or Text files", type=["pdf", "txt"], accept_multiple_files=True, key=f"upload_{business_id}")
                if st.button("Process Uploaded Files", key=f"process_upload_{business_id}"):
                    if uploaded_files:
                        for uploaded_file in uploaded_files:
                            temp_dir = os.path.join("data", business_id, "temp")
                            os.makedirs(temp_dir, exist_ok=True)
                            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(temp_file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            if uploaded_file.type == "application/pdf":
                                raw_content = document_processor.get_text_from_pdf(temp_file_path)
                            else:
                                raw_content = uploaded_file.getvalue().decode("utf-8")
                            
                            process_and_store_content(business_id, raw_content)
                            os.remove(temp_file_path)
                    else:
                        st.warning("Please upload files before processing.")

                st.subheader("Scrape a Website")
                url_to_scrape = st.text_input("Enter URL to scrape", key=f"url_{business_id}")
                if st.button("Scrape and Add", key=f"scrape_{business_id}"):
                    if url_to_scrape:
                        raw_content = document_processor.get_text_from_url(url_to_scrape)
                        process_and_store_content(business_id, raw_content)
                    else:
                        st.warning("Please enter a URL.")

            with tab2:
                st.subheader("Customize Your Agent")
                with st.form("customization_form"):
                    agent_name = st.text_input("Agent Name", value=current_business['agent_name'])
                    welcome_message = st.text_area("Welcome Message", value=current_business['welcome_message'], height=150)
                    personality = st.selectbox("Personality", ["friendly", "formal", "concise"], index=["friendly", "formal", "concise"].index(current_business['personality']))
                    brand_color = st.color_picker("Brand Color", value=current_business['brand_color'])
                    
                    submitted = st.form_submit_button("Save Customizations")
                    if submitted:
                        update_business_settings(business_id, agent_name, welcome_message, personality, brand_color)
                        st.success("Settings saved successfully!")
                        st.rerun()

            with tab5:
                st.subheader("Test Your Agent in Real-time")
                st.write("This chat uses the same advanced, memory-enabled engine as your deployed chatbot.")

                # Initialize chat history and memory in session state
                if f"chat_history_{business_id}" not in st.session_state:
                    st.session_state[f"chat_history_{business_id}"] = [{"role": "assistant", "content": current_business['welcome_message']}]
                if f"memory_{business_id}" not in st.session_state:
                    st.session_state[f"memory_{business_id}"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

                # Display chat messages
                for message in st.session_state[f"chat_history_{business_id}"]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input(f"Ask {current_business['agent_name']} a question..."):
                    st.session_state[f"chat_history_{business_id}"].append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            index_path = os.path.join("data", business_id)
                            if not os.path.exists(os.path.join(index_path, "faiss_index.bin")):
                                st.error("The knowledge base has not been trained. Please upload a document in 'Knowledge Sources'.")
                            else:
                                vector_store = FAISS.load_local(index_path, embeddings_model, index_name="faiss_index.bin", allow_dangerous_deserialization=True)
                                base_retriever = vector_store.as_retriever(search_kwargs={'k': 5})
                                reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=2)
                                compression_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)
                                llm = ChatGroq(temperature=0.7, model_name="openai/gpt-oss-120b", groq_api_key=st.secrets["GROQ_API_KEY"])
                                memory = st.session_state[f"memory_{business_id}"]
                                conversational_chain = ConversationalRetrievalChain.from_llm(
                                    llm=llm, retriever=compression_retriever, memory=memory, return_source_documents=False
                                )
                                result = conversational_chain.invoke({"question": prompt})
                                final_answer = result.get("answer", "Sorry, I encountered an issue.")
                                st.markdown(final_answer)
                                st.session_state[f"chat_history_{business_id}"].append({"role": "assistant", "content": final_answer})

            with tab3:
                st.subheader("Get Your Embed Code")
                st.write("Copy this code snippet and paste it into your website's HTML.")
                # IMPORTANT: Replace this placeholder with your actual live Netlify URL
                live_frontend_url = "https://your-netlify-frontend-url.netlify.app" 
                embed_code = f"""
<div id="chatbot-container"></div>
<script src="{live_frontend_url}/script.js" data-business-id="{business_id}"></script>
                """
                st.code(embed_code, language="html")

            with tab4:
                st.subheader("Analytics")
                conn = get_db_connection()
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cursor.execute('SELECT question, COUNT(*) as count FROM chat_logs WHERE business_id = %s GROUP BY question ORDER BY count DESC LIMIT 10', (business_id,))
                logs = cursor.fetchall()
                cursor.execute("SELECT COUNT(*) FROM chat_logs WHERE business_id = %s AND DATE(timestamp) = CURRENT_DATE", (business_id,))
                daily_queries_result = cursor.fetchone()
                daily_queries = daily_queries_result[0] if daily_queries_result and daily_queries_result[0] else 0
                cursor.close()
                conn.close()
                st.metric("Queries Today", daily_queries)
                st.write("**Most Asked Questions:**")
                if logs:
                    for log in logs:
                        st.write(f"- {log['question']} ({log['count']} times)")
                else:
                    st.write("No questions have been asked yet.")

except Exception as e:
    st.error(f"An unexpected error occurred. Please check the logs.")
    st.exception(e)