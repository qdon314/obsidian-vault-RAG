from config import VAULT_PATH, CHROMA_PATH
from src.rag.loaders import load_markdown_files
from src.rag.index import build_or_load_index

def main():
    docs = load_markdown_files(VAULT_PATH)
    _ = build_or_load_index(docs, CHROMA_PATH)
    print(f"Indexed {len(docs)} markdown files from: {VAULT_PATH}")
    print(f"Chroma persisted at: {CHROMA_PATH}")

if __name__ == "__main__":
    main()
