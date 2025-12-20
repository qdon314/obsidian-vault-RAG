import argparse
from config import VAULT_PATH, CHROMA_PATH
from src.rag.loaders import load_markdown_files
from src.rag.index import build_or_load_index

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a question against the Obsidian RAG index."
    )

    # Debug / dumping
    parser.add_argument(
        "--split-on-headers",
        action="store_true",
        help="Split on headers when loading markdown files",
    )

    return parser.parse_args()

def main():
    args = parse_args()
    docs = load_markdown_files(VAULT_PATH, split_on_headers=args.split_on_headers, as_obsidian=True)
    _ = build_or_load_index(docs, CHROMA_PATH)
    print(f"Indexed {len(docs)} markdown files from: {VAULT_PATH}")
    print(f"Chroma persisted at: {CHROMA_PATH}")

if __name__ == "__main__":
    main()
