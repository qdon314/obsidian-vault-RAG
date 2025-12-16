import os
from dotenv import load_dotenv

load_dotenv()

VAULT_PATH = os.getenv("VAULT_PATH", "your_vault_path_here")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma")

# Mode: "local" = local embeddings + local/hosted LLM of your choice
MODE = os.getenv("MODE", "local")

# Ollama (optional)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# OpenAI (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
