from pathlib import Path
from llama_index.core import Document
from rag.utils.parsing import split_obsidian_frontmatter, extract_and_normalize_frontmatter, extract_inline_tags

def load_markdown_files(vault_path: str, as_obsidian: bool = False) -> list[Document]:
    """
    Load markdown files from a vault path.
    """
    root = Path(vault_path)
    if not root.exists():
        raise FileNotFoundError(f"VAULT_PATH not found: {root}")

    docs: list[Document] = []
    for path in root.rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        
        if as_obsidian:
            frontmatter, content = split_obsidian_frontmatter(text)
            normalized_frontmatter = extract_and_normalize_frontmatter(frontmatter, ["tags"])
            text = f"{normalized_frontmatter}\n\n{content}"
            frontmatter_tags = normalized_frontmatter.get("tags", [])
            inline_tags = extract_inline_tags(content)
            all_tags = frontmatter_tags + inline_tags
            
        docs.append(
            Document(
                text=text,
                metadata={
                    "directory": str(path.parent),
                    "source_path": str(path),
                    "file_name": path.name,
                    "tags": ", ".join(all_tags) if as_obsidian else [], # type: ignore
                },
            )
        )
    return docs