from pathlib import Path
from llama_index.core import Document

def load_markdown_files(vault_path: str) -> list[Document]:
    root = Path(vault_path)
    if not root.exists():
        raise FileNotFoundError(f"VAULT_PATH not found: {root}")

    docs: list[Document] = []
    for path in root.rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append(
            Document(
                text=text,
                metadata={
                    "source_path": str(path),
                    "file_name": path.name,
                },
            )
        )
    return docs
