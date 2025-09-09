import os
from typing import List, Dict
from dotenv import load_dotenv
from groq import Groq

# This part is for the FastAPI backend (main.py) to load its key
# It will be ignored by the Streamlit app, which is what we want.
if os.path.exists('.env'):
    load_dotenv()
BACKEND_GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_response_with_groq(messages: List[Dict], api_key: str, model: str = "openai/gpt-oss-120b") -> str:
    """
    Generates a response using Groq's chat completion API.
    Accepts a list of message dictionaries and the API key directly.
    """
    if not api_key:
        raise ValueError("Groq API Key is missing.")

    try:
        # Create a client instance for each request using the provided key.
        # This is essential for the multi-service setup.
        client = Groq(api_key=api_key)
        
        # Make the API call with the required 'messages' and 'model' arguments.
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.7,
            max_tokens=2048,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        # Provide a more specific error message back to the user
        return f"Sorry, there was an error communicating with the AI model: {e}"