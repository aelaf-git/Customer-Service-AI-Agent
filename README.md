# RAG Chatbot for Businesses
This project is a complete, multi-tenant Software-as-a-Service (SaaS) application that allows businesses to create and deploy a custom AI chatbot on their websites. The chatbot is trained on the business's own knowledge base (PDFs, website content, etc.) and is powered by a Retrieval-Augmented Generation (RAG) architecture.
The application is built with a modern, decoupled architecture, featuring a Python backend, a Streamlit admin dashboard, and a vanilla JavaScript frontend widget.
## Working Principles & Architecture
The application operates on a decoupled, three-service model that communicates via a central database. This ensures scalability and separation of concerns.
1. **Backend API (The "Brain")**: A FastAPI server responsible for all AI logic. It handles chat requests, performs the RAG process, and connects to the database to fetch configurations and log interactions.
2. **Admin Dashboard (The "Control Panel"):** A Streamlit application that allows business owners to manage their accounts, customize their chatbot's appearance and personality, and upload knowledge source documents for training.
3. **Frontend Widget (The "Face"):** A lightweight, self-contained vanilla JavaScript application that business owners embed on their websites. It handles the UI, user interaction, and communication with the Backend API.
4. **Database (The "Memory"):** A central PostgreSQL database that stores all user data, business configurations, customizations, and chat logs for analytics.
5. **Vector Store (The "Trained Knowledge"):** FAISS (Facebook AI Similarity Search) is used to create and store vector embeddings from the uploaded documents. These are saved as .bin files on the server's file system.
### The RAG (Retrieval-Augmented Generation) Flow
When a website visitor asks a question, the following happens:
1. **Receive:** The FastAPI backend receives the question and the business's unique ID.
2. **Embed:** The question is converted into a numerical vector (an embedding).
3. **Retrieve:** The backend searches the business's specific FAISS index file to find the chunks of text from the original documents that are most semantically similar to the question's embedding.
4. **Augment:** The retrieved text chunks are combined with the original question and a system prompt (defining the AI's personality) into a comprehensive final prompt.
5. **Generate:** This final, augmented prompt is sent to the Groq API, which uses a Large Language Model (LLM) to generate a natural, conversational answer based only on the provided context.
## Tech Stack
- **Backend:**
  - **Framework:** FastAPI
  - **Server:** Uvicorn with Gunicorn
  - **Language:** Python
- **Dashboard:**
  - **Framework:** Streamlit
- **Database:**
  - PostgreSQL (with psycopg2 connector)
- **AI & RAG:**
  - **LLM Provider:** Groq (using Llama3 models)
  - **Embedding Model:** sentence-transformers (specifically all-MiniLM-L6-v2)
  - **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Frontend:**
  - Vanilla JavaScript (ES6)
  - HTML5 & CSS3
- **Deployment (Cloud):**
  - **Backend:** Render (Web Service)
  - **Dashboard:** Streamlit Community Cloud
  - **Frontend:** Netlify
  - **Database:** Render (PostgreSQL)
## Local Development Setup Instructions
This guide will walk you through running the entire application suite on your local machine. This setup uses your local machine to run the dashboard and backend, which connect to a live cloud database on Render.
### Prerequisites
- Python 3.10+
- Git
- A Render account (for the PostgreSQL database)
- A GroqCloud account (for the LLM API key)
### Step 1: Clone the Repository
````
git clone <https://github.com/aelaf-git/Rag-Chatbot-for-Businesses>
cd RAG-Chatbot-for-Businesses
````
### Step 2: Set Up Python Virtual Environment
It's crucial to use a virtual environment to manage dependencies.
code
``` 
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install all required libraries
pip install -r requirements.txt
```
### Step 3: Set Up Your Cloud Database
1. Log in to your **Render** account.
2. Create a **New PostgreSQL** database.
3. Wait for the status to become "Available".
4. On the database's "Info" page, copy the **External Database URL**. It will look like `postgres://user:password@host:port/database`.
### Step 4: Configure Your Secret Credentials
You need to create two secret files for your two local applications to use.
#### A. For the Backend (.env file):
1. In the root of the project, create a file named `.env`.
2. Paste the following into it, replacing the placeholders with your real credentials.
```
# Secrets for the FastAPI backend (main.py)
DATABASE_URL="YOUR_RENDER_EXTERNAL_DATABASE_URL"
GROQ_API_KEY="YOUR_GROQ_API_KEY"
```
#### B. For the Dashboard (secrets.toml file):
1. Create a new folder in the project root named .streamlit.
2. Inside .streamlit, create a new file named secrets.toml.
3. Paste the following into it, using the same credentials.
```
# Secrets for the Streamlit dashboard (app.py)
DATABASE_URL = "YOUR_RENDER_EXTERNAL_DATABASE_URL"
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```
### Step 5: Run the Applications
You will need **two separate terminals** running simultaneously.
#### Terminal 1: Run the Dashboard & Initialize Database
1. Make sure your virtual environment is activated.
2. Run the Streamlit app:
```
streamlit run app.py
```
3. Open `http://localhost:8501` in your browser.
4. **On the very first run**, the app will automatically connect to your Render database and create the necessary tables. You will see a toast message confirming this.
5. Use the dashboard to register a new business and upload a PDF to train its knowledge base.
6. Go to the "Deploy" tab and copy the `business-id` for your newly created business.
#### Terminal 2: Run the Backend API Server
1. Open a new terminal and navigate to the project folder.
2. Make sure your virtual environment is activated.
3. Run the FastAPI server:
```
python main.py
```
4. You will see `Uvicorn running on http://0.0.0.0:8000`. This terminal will now show live API request logs.
### Step 6: Test the Frontend Widget
1. Navigate to the `static` folder in the project.
2. Open the `index.html` file in a text editor.
3. Find the `<script>` tag at the bottom and paste the `business-id` you copied from the dashboard.
```
<script src="script.js" 
        data-business-id="PASTE_YOUR_BUSINESS_ID_HERE"></script>
```
4. Save the `index.html` file.
5. **Open the `index.html` file in your web browser.**

The chatbot bubble should appear, and you can now have a full conversation with your AI agent. Check the terminal running main.py to see the live GET /config and POST /chat requests as you interact with the bot.# Rag-Chatbot-for-Businesses