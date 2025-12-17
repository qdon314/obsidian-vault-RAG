from __future__ import annotations

import re
from typing import Tuple, Dict, Any
import yaml

def split_obsidian_frontmatter(raw_obsidian_text: str) -> tuple[dict[str, Any], str]:
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


def extract_and_normalize_frontmatter(frontmatter: dict[str, Any], keys: list[str]) -> dict[str,list[str]]:
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
    
