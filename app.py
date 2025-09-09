import streamlit as st
import psycopg2
import psycopg2.extras
import uuid
import os

# Import our custom modules
import document_processor
import vector_store_manager
import llm_interface

st.set_page_config(layout="wide", page_title="AI Agent Dashboard")

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
    
    # Check if the 'businesses' table exists using PostgreSQL's system catalog
    cursor.execute("SELECT to_regclass('public.businesses')")
    table_exists = cursor.fetchone()[0]
    
    if not table_exists:
        st.toast("First time setup: Initializing database tables...", icon="🚀")
        # Table to store business information and customizations
        cursor.execute('''
        CREATE TABLE businesses (
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
        CREATE TABLE chat_logs (
            log_id SERIAL PRIMARY KEY,
            business_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (business_id) REFERENCES businesses (id)
        )
        ''')
        
        conn.commit()
        st.toast("Database initialized successfully!", icon="✅")
    
    cursor.close()
    conn.close()

# --- Business Management Functions ---
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

# --- Content Processing Function ---
def process_and_store_content(business_id, raw_content):
    if not raw_content or not raw_content.strip():
        st.warning("No content to process for this source.")
        return

    st.info(f"Processing content for business {business_id}...")
    with st.spinner("Chunking text, generating embeddings, and updating knowledge base... This may take a moment."):
        text_chunks = document_processor.chunk_text(raw_content)
        embeddings = document_processor.generate_embeddings(text_chunks)
        index_dir = os.path.join("data", business_id)
        os.makedirs(index_dir, exist_ok=True)
        embedding_dim = document_processor.get_embedding_model().get_sentence_embedding_dimension()
        current_index, current_texts = vector_store_manager.create_or_load_faiss_index(business_id, embedding_dimension=embedding_dim)
        vector_store_manager.add_embeddings_to_faiss(
            business_id, embeddings, text_chunks, current_index, current_texts
        )
    st.success(f"Knowledge base updated with {len(text_chunks)} new chunks.")


# --- Main App Execution ---
try:
    # 1. Run the initialization check at the very start.
    initialize_database()

    # 2. Now, we can safely build the rest of the app.
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

    # --- Main Dashboard Area ---
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
                st.write("Interact with your AI agent here to see how it responds with the current knowledge and personality settings.")

                if f"chat_history_{business_id}" not in st.session_state:
                    st.session_state[f"chat_history_{business_id}"] = []

                for message in st.session_state[f"chat_history_{business_id}"]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input(f"Ask {current_business['agent_name']} a question..."):
                    st.session_state[f"chat_history_{business_id}"].append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            query_embedding = document_processor.generate_embeddings([prompt])[0]
                            retrieved_texts = vector_store_manager.search_faiss_index(business_id, query_embedding)
                            if not retrieved_texts:
                                final_answer = "I'm sorry, but I couldn't find specific information about that. Would you like me to connect you with our team?"
                            else:
                                context = "\n\n".join(retrieved_texts)
                                personality_map = {
                                    "friendly": "You are a friendly, helpful, and professional customer service AI assistant for the company '{name}'. Your personality should be welcoming and conversational.",
                                    "formal": "You are a formal and direct AI assistant for '{name}'. Provide precise information without unnecessary pleasantries.",
                                    "concise": "You are a concise AI assistant for '{name}'. Get straight to the point and provide short, clear answers."
                                }
                                system_prompt = personality_map.get(current_business['personality'], personality_map['friendly']).format(name=current_business['name'])
                                system_prompt += """
                                \n\n**Your Instructions:**
                                1.  **Primary Goal:** Your main purpose is to answer the user's question based *only* on the "Retrieved Information" provided below.
                                2.  **Detailed Answers:** Use the retrieved information to provide a detailed, clear, and comprehensive explanation.
                                3.  **Handling Unknowns:** If the answer to a question cannot be found in the "Retrieved Information," you MUST say: "I'm sorry, but I couldn't find specific information about that in our knowledge base."
                                4.  **General Conversation:** If the user's question is a simple greeting or small talk, respond naturally and friendly without mentioning the retrieved information.
                                """
                                user_prompt_with_context = f"Retrieved Information:\n{context}\n\nUser Question:\n{prompt}"
                                messages_payload = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_with_context}]
                                groq_api_key = st.secrets["GROQ_API_KEY"]
                                final_answer = llm_interface.generate_response_with_groq(messages_payload, api_key=groq_api_key)
                            st.markdown(final_answer)
                            st.session_state[f"chat_history_{business_id}"].append({"role": "assistant", "content": final_answer})

            with tab3:
                st.subheader("Get Your Embed Code")
                st.write("Copy this code snippet and paste it into your website's HTML, just before the closing `</body>` tag.")
                # This URL should be your live Netlify frontend URL in production
                live_frontend_url = "http://127.0.0.1:5500" # Placeholder for local testing
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
    st.error(f"An error occurred: {e}")
    st.exception(e)