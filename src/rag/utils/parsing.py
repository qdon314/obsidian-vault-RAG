from __future__ import annotations

import re
import yaml
from typing import Iterator, Optional, Tuple, Dict, Any
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter
from llama_index.core import Document

def split_obsidian_frontmatter(raw_obsidian_text: str) -> Tuple[Dict[str, Any], str]:
    """
    Split a string into frontmatter and content, handling YAML frontmatter.
    
    Args:
        raw (str): The input string containing YAML frontmatter and content.
    
    Returns:
        (tuple[dict[str, Any], str]): A tuple containing the frontmatter as a dictionary and the content as a string.
    """
    # Normalize newlines and strip BOM if present
    s = raw_obsidian_text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")

    lines = s.split("\n")
    # Check for opening '---' on its own line. If not found, treat as no frontmatter
    if not lines or lines[0].strip() != "---":
        return {}, s

    # Find closing '---' on its own line
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        # No closing fence: treat as no frontmatter
        return {}, s

    frontmatter_text = "\n".join(lines[1:end_idx]).strip()
    content = "\n".join(lines[end_idx + 1:]).lstrip("\n")

    try:
        frontmatter = yaml.safe_load(frontmatter_text) or {}
        if not isinstance(frontmatter, dict):
            frontmatter = {}
    except Exception:
        frontmatter = {}

    return frontmatter, content


def extract_and_normalize_frontmatter(frontmatter: Dict[str, Any], keys: list[str]) -> Dict[str,list[str]]:
    """
    Extract and normalize values from the frontmatter.
    Args:
        frontmatter (dict[str, Any]): The frontmatter dictionary.
        
    Returns:
        (list[str]): A list of tags extracted and normalized from the frontmatter.
    """
    normalized_values = {key: [] for key in keys}
    for key in keys:
        if key in frontmatter:
            value = frontmatter[key]
            if isinstance(value, str):
                normalized_values[key] = [value.strip()]
            elif isinstance(value, list):
                normalized_values[key] = [v.strip() for v in value if isinstance(v, str)]
    return normalized_values

def extract_inline_tags(content: str) -> list[str]:
    """
    Extract inline tags from the content.
    
    Args:
        content (str): The content string containing inline tags.
        
    Returns:        
        (list[str]): A list of inline tags extracted from the content.
    """
    INLINE_TAG_RE = re.compile(r"(?<!\w)#([\w/-]+)")
    return list(set(INLINE_TAG_RE.findall(content)))

# ------------------ Langchain Splitter ------------------
_headers_to_split_on=[
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
    ("#####", "h5"),
    ("######", "h6"),
]
def split_markdown_with_langchain(markdown_str: str):
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_headers_to_split_on)
    return splitter.split_text(markdown_str)

def docs_from_markdown(markdown_str: str, base_meta: dict) -> list[Document]:
    sections = split_markdown_with_langchain(markdown_str)
    docs = []

    for s in sections:
        meta = dict(base_meta)

        section_header = next(
            ((prefix, label) for prefix, label in _headers_to_split_on
             if s.metadata.get(label) is not None),
            None
        )

        if section_header is not None:
            prefix, label = section_header
            meta["section_heading"] = f"{prefix} {s.metadata[label]}"

        docs.append(Document(text=s.page_content, metadata=meta))

    return docs

# ------------------ Custom Splitter ------------------

def split_markdown_by_heading(markdown_str: str, min_level: int = 2) -> Iterator[Tuple[Optional[str], str]]:
    """
    Yield (heading, section_text). Heading includes markdown hashes.
    Splits on headings of level >= min_level (default: ## and deeper).
    """
    HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
    
    # Normalize newlines
    markdown_str = markdown_str.replace("\r\n", "\n").replace("\r", "\n")
    
    matches = list(HEADER_RE.finditer(markdown_str))
    if not matches:
        yield (None, markdown_str)
        return

    # Only use headings >= min_level as split points
    split_points: list[re.Match] = []
    for match in matches:
        level = len(match.group(1))
        if level >= min_level:
            split_points.append(match)

    if not split_points:
        yield (None, markdown_str)
        return

    # Preamble before first heading
    first: re.Match = split_points[0]
    pre = markdown_str[:first.start()].strip()
    if pre:
        yield (None, pre)

    for i, match in enumerate(split_points):        
        # End is start of next heading or end of file
        end = split_points[i + 1].start() if i + 1 < len(split_points) else len(markdown_str)
        
        # Skip empty sections
        heading = match.group(0).strip()
        section = markdown_str[match.end():end].strip()
        yield (heading, section)
        
def docs_from_obsidian_note(content: str, base_meta: dict) -> list[Document]:
    docs = []
    for heading, section_text in split_markdown_by_heading(content, min_level=2):
        text = section_text.strip()
        if not text:
            continue
        meta = dict(base_meta)
        meta["inline_tags"] = ", ".join(extract_inline_tags(text))
        meta["section_heading"] = heading
        docs.append(Document(text=text, metadata=meta))
    return docs
