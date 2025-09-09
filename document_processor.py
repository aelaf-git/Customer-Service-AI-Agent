# document_processor.py (Final, Decoupled Version)
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# --- NEW: Singleton pattern for loading the model ---
# We will store the loaded model in this global variable.
EMBEDDING_MODEL = None

def load_embedding_model():
    """Loads the sentence transformer model."""
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded.")
    return model

def get_embedding_model():
    """
    Returns the embedding model, loading it if it hasn't been loaded yet.
    This ensures the model is only loaded once.
    """
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        EMBEDDING_MODEL = load_embedding_model()
    return EMBEDDING_MODEL
# --- END OF NEW SECTION ---

def get_text_from_url(url: str) -> str:
    """Scrapes text content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error scraping URL {url}: {e}")
        return ""

def get_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start += chunk_size - chunk_overlap
    return chunks

def generate_embeddings(texts: list[str]) -> list:
    """Generates embeddings for a list of text chunks."""
    # Now it calls our new function to get the model
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_tensor=False)
    return embeddings.tolist()