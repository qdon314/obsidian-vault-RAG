"""
Generate an automatic "Project State Header" markdown capsule.
This can be prepended to documents for context in RAG systems.

Usage:
  python scripts/project_state.py
  python scripts/project_state.py --root .
  python scripts/project_state.py --validate-jsonl artifacts/indexes/obsidian_index/chunks.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
from collections.abc import Iterable

# ----------------------------
# Repo-specific hints (tweak)
# ----------------------------

CONFIG_HINTS = [
    "rag/settings.py",
    "config.py",
    "settings.yaml",
    "settings.yml",
    "settings.toml",
    "pyproject.toml",
    ".env",
]

INDEX_HINTS = [
    "artifacts/indexes",
    "artifacts/index",
    "data/indexes",
    "indexes",
]

# If you keep your index config in a known file, add it here
# e.g. "rag/settings.py" or "rag/config.py" and parse it more deeply later.
LIGHTWEIGHT_PARAM_GUESSES = [
    "chunk_size",
    "chunk_overlap",
    "overlap",
    "embedding",
    "embed",
    "vector",
    "faiss",
    "chroma",
    "llm",
    "model",
]


def run(cmd: list[str], cwd: pathlib.Path) -> tuple[int, str]:
    """Run a command and return (returncode, stdout_str)."""
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return p.returncode, (p.stdout or "").strip()
    except Exception as e:
        return 1, f"<error running {' '.join(cmd)}: {e}>"


def read_text_if_exists(path: pathlib.Path, max_chars: int = 4000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        txt = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(txt) > max_chars:
            return txt[:max_chars] + "\n…(truncated)…"
        return txt
    except Exception as e:
        return f"<error reading {path}: {e}>"


def find_first_existing(root: pathlib.Path, rel_paths: Iterable[str]) -> pathlib.Path | None:
    for rp in rel_paths:
        p = root / rp
        if p.exists():
            return p
    return None


def list_existing(root: pathlib.Path, rel_paths: Iterable[str]) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for rp in rel_paths:
        p = root / rp
        if p.exists():
            out.append(p)
    return out


def guess_index_location(root: pathlib.Path) -> pathlib.Path | None:
    for rp in INDEX_HINTS:
        p = root / rp
        if p.exists():
            return p
    return None


def validate_jsonl(path: pathlib.Path, max_errors: int = 5) -> tuple[int, list[str]]:
    """
    Validate JSONL file line-by-line.
    Returns (error_count, error_messages up to max_errors).
    """
    errors: list[str] = []
    if not path.exists():
        return 1, [f"{path} does not exist"]
    if not path.is_file():
        return 1, [f"{path} is not a file"]

    err_count = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                json.loads(line)
            except Exception as e:
                err_count += 1
                if len(errors) < max_errors:
                    snippet = line[:240] + ("…" if len(line) > 240 else "")
                    errors.append(f"Line {i}: {e} | {snippet}")
                if err_count >= max_errors:
                    # keep counting? we stop early for speed
                    break
    return err_count, errors


def extract_lightweight_params_from_text(text: str) -> list[str]:
    """
    Naive parameter hints from config files.
    This is intentionally simple/deterministic—just surface likely knobs.
    """
    hits = []
    lower = text.lower()
    for key in LIGHTWEIGHT_PARAM_GUESSES:
        if key in lower:
            hits.append(key)
    return sorted(set(hits))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repo root (default: .)")
    ap.add_argument("--project-name", default="Obsidian RAG / Personal Knowledge System")
    ap.add_argument("--validate-jsonl", nargs="*", default=[], help="JSONL file(s) to validate")
    ap.add_argument("--commits", type=int, default=10, help="How many recent commits to include")
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve()

    now = dt.datetime.now().astimezone()
    timestamp = now.strftime("%Y-%m-%d %H:%M %Z")

    # --- Git state ---
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)[1]
    commit = run(["git", "rev-parse", "HEAD"], cwd=root)[1]
    dirty = run(["git", "status", "--porcelain"], cwd=root)[1]
    dirty_flag = "yes" if dirty else "no"

    log_cmd = ["git", "log", f"-n{args.commits}", "--oneline", "--decorate=no"]
    recent_commits = run(log_cmd, cwd=root)[1]

    diff_stat = run(["git", "diff", "--stat"], cwd=root)[1]

    # --- Config discovery ---
    config_paths = list_existing(root, CONFIG_HINTS)
    config_summaries = []
    param_hints = set()
    for p in config_paths:
        txt = read_text_if_exists(p, max_chars=4000)
        param_hints |= set(extract_lightweight_params_from_text(txt))
        config_summaries.append(f"- {p.relative_to(root)}")

    # --- Human-maintained focus/issues ---
    focus_txt = read_text_if_exists(root / "docs/FOCUS.md", max_chars=2000)
    issues_txt = read_text_if_exists(root / "docs/KNOWN_ISSUES.md", max_chars=4000)

    # --- Index location guess ---
    index_loc = guess_index_location(root)
    index_loc_str = str(index_loc.relative_to(root)) if index_loc else "<not found>"

    # --- JSONL validation (optional) ---
    jsonl_blocks = []
    for p_str in args.validate_jsonl:
        p = (root / p_str).resolve()
        err_count, errs = validate_jsonl(p)
        status = "OK" if err_count == 0 else f"FAILED ({err_count}+ error(s))"
        rel = str(p.relative_to(root)) if root in p.parents else str(p)
        block = [f"- {rel}: {status}"]
        for e in errs:
            block.append(f"  - {e}")
        jsonl_blocks.append("\n".join(block))

    # --- Emit markdown capsule ---
    lines = []
    lines.append("# PROJECT STATE HEADER (AUTO)")
    lines.append("")
    lines.append("## Project Name")
    lines.append(args.project_name)
    lines.append("")
    lines.append("## Timestamp")
    lines.append(timestamp)
    lines.append("")
    lines.append("## Repo State")
    lines.append(f"- Root: {root}")
    lines.append(f"- Branch: {branch}")
    lines.append(f"- Commit: {commit}")
    lines.append(f"- Dirty: {dirty_flag}")
    if dirty:
        # show a short status summary without dumping everything
        dirty_lines = dirty.splitlines()
        lines.append(f"- Dirty summary: {len(dirty_lines)} change(s)")
    lines.append("")
    lines.append("## Recent Commits")
    lines.append("```")
    lines.append(recent_commits or "<no git log available>")
    lines.append("```")
    lines.append("")
    lines.append("## Uncommitted Diff (stat)")
    lines.append("```")
    lines.append(diff_stat or "<clean or unavailable>")
    lines.append("```")
    lines.append("")
    lines.append("## Current Focus (docs/FOCUS.md)")
    lines.append(focus_txt or "<missing docs/FOCUS.md>")
    lines.append("")
    lines.append("## Known Issues (docs/KNOWN_ISSUES.md)")
    lines.append(issues_txt or "<missing docs/KNOWN_ISSUES.md>")
    lines.append("")
    lines.append("## System Clues")
    lines.append(f"- Index location (guess): {index_loc_str}")
    if config_summaries:
        lines.append("- Config files found:")
        lines.extend(config_summaries)
    else:
        lines.append("- Config files found: <none of the hints matched>")
    if param_hints:
        lines.append(f"- Parameter hints spotted: {', '.join(sorted(param_hints))}")
    else:
        lines.append("- Parameter hints spotted: <none>")
    lines.append("")
    if jsonl_blocks:
        lines.append("## JSONL Validation")
        lines.append("\n".join(jsonl_blocks))
        lines.append("")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
