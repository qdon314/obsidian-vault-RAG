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
            frontmatter_tags = normalized_frontmatter.get("tags", [])
            inline_tags = extract_inline_tags(content)
        
        dirs = extract_dirs(path)
        docs.append(
            Document(
                text=content,
                metadata={
                    "source_path": str(path),
                    "root_dir": dirs["root_dir"],
                    "relative_dir": dirs["relative_dir"],
                    "is_ai": dirs["is_ai"],
                    "file_name": path.name,
                    "frontmatter_tags": ", ".join(frontmatter_tags) if as_obsidian else [], # type: ignore
                    "inline_tags": ", ".join(inline_tags) if as_obsidian else [], # type: ignore
                    "classification": classify_note(path, normalized_frontmatter),
                },
            )
        )
    return docs

def classify_note(path: Path, frontmatter: dict) -> str:
    name = path.name.lower()

    if "moc" in name:
        return "moc"
    if frontmatter.get("type") == "moc":
        return "moc"
    if "moc" in frontmatter.get("tags", []):
        return "moc"

    return "note"

def extract_dirs(path: Path) -> dict[str, str]:
    parts = path.parts
    if "AI" in parts:
        ai_index = parts.index("AI")
        return {
            "root_dir": "AI",
            "relative_dir": "/".join(parts[ai_index: -1]),
            "is_ai": "True",
        }
    return {
        "root_dir": parts[-2] if len(parts) > 1 else "",
        "relative_dir": parts[-2] if len(parts) > 1 else "",
        "is_ai": "False",
    }
