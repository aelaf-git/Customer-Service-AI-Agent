# vector_store_manager.py
import faiss
import numpy as np
import os

def create_or_load_faiss_index(business_id: str, embedding_dimension: int = 384) -> faiss.IndexFlatL2:
    """
    Creates a new FAISS index for a business if one doesn't exist,
    otherwise loads the existing one.
    """
    index_dir = os.path.join("data", business_id)
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, "faiss_index.bin")
    metadata_path = os.path.join(index_dir, "metadata.txt") # To store original text chunks

    if os.path.exists(index_path):
        print(f"Loading existing FAISS index for business {business_id}...")
        index = faiss.read_index(index_path)
        # Load metadata (original text chunks)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f]
        print(f"Loaded {len(texts)} text chunks.")
        return index, texts
    else:
        print(f"Creating new FAISS index for business {business_id}...")
        index = faiss.IndexFlatL2(embedding_dimension)
        # Initialize an empty list for texts (metadata)
        texts = []
        return index, texts

def add_embeddings_to_faiss(business_id: str, embeddings: list, texts: list,
                             current_index: faiss.IndexFlatL2, current_texts: list):
    """Adds new embeddings and their corresponding texts to the FAISS index."""
    if not embeddings:
        return current_index, current_texts

    embeddings_np = np.array(embeddings).astype('float32')
    current_index.add(embeddings_np)
    current_texts.extend(texts)

    index_dir = os.path.join("data", business_id)
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, "faiss_index.bin")
    metadata_path = os.path.join(index_dir, "metadata.txt")

    faiss.write_index(current_index, index_path)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for text in current_texts:
            f.write(text + '\n')

    print(f"Added {len(embeddings)} embeddings to FAISS index for business {business_id}. Total: {current_index.ntotal}")
    return current_index, current_texts


def search_faiss_index(business_id: str, query_embedding: list, k: int = 5) -> list[str]:
    """
    Searches the FAISS index for the most similar text chunks.
    Returns the original text chunks.
    """
    index_dir = os.path.join("data", business_id)
    index_path = os.path.join(index_dir, "faiss_index.bin")
    metadata_path = os.path.join(index_dir, "metadata.txt")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        print(f"No FAISS index or metadata found for business {business_id}.")
        return []

    index = faiss.read_index(index_path)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f]

    query_embedding_np = np.array([query_embedding]).astype('float32')
    D, I = index.search(query_embedding_np, k) # D are distances, I are indices

    retrieved_texts = [texts[idx] for idx in I[0] if idx < len(texts)]
    return retrieved_texts