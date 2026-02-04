#!/usr/bin/env python3
"""
Generate project state snapshots for LLM context or documentation.

This script produces a markdown document capturing the current state of the
project, including git status, recent work, architecture, configuration, and
documentation. Output is optimized for providing context to LLMs.

Usage:
    python scripts/project_state.py                              # writes to state_outputs/<profile>_<date>.md
    python scripts/project_state.py --profile llm-context        # specific profile
    python scripts/project_state.py --stdout                     # print to stdout instead
    python scripts/project_state.py -o ./outputs                 # custom output directory
    python scripts/project_state.py -p ui-focus -p domain-deep   # compose profiles

Profiles:
    quick-status   Minimal: git state + focus only (brief depth)
    standard       Typical: git, config, docs (standard depth)
    llm-context    Optimized for LLM: git, docs, architecture (standard depth)
    ui-focus       UI development: eval/app structure, recent changes (detailed)
    domain-deep    Deep analysis: full architecture, validation (detailed depth)

Collectors:
    git            Branch, commits, dirty state, recent work analysis
    docs           docs/FOCUS.md, docs/KNOWN_ISSUES.md
    files          Config file discovery, index location, parameter hints
    architecture   Project structure, ports, adapters, domain models
    validation     JSONL file validation

Depth levels:
    brief          Critical info only (~200 lines)
    standard       Typical usage (~500 lines)
    detailed       Everything including full diffs and architecture docs

Configuration:
    Settings are loaded from [project_state] section in settings.toml.
    CLI flags override profile defaults. Profiles can be composed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from project_state.collectors import (
    collect_architecture,
    collect_docs,
    collect_file_discovery,
    collect_git_state,
    collect_recent_work_summary,
    validate_jsonl_files,
)
from project_state.config import (
    apply_cli_overrides,
    get_profile,
    load_config,
)
from project_state.renderers import render_project_state


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Generate project state snapshot for LLM context or documentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    ap.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root directory (default: current directory)",
    )

    ap.add_argument(
        "--profile",
        "-p",
        action="append",
        dest="profiles",
        metavar="NAME",
        help="Profile(s) to use. Can specify multiple to compose. "
        "Options: quick-status, standard, llm-context, ui-focus, domain-deep",
    )

    ap.add_argument(
        "--depth",
        choices=["brief", "standard", "detailed"],
        help="Override depth level (default: from profile)",
    )

    ap.add_argument(
        "--days",
        type=int,
        help="Days of recent git history to analyze (overrides profile)",
    )

    ap.add_argument(
        "--commits",
        type=int,
        help="Number of recent commits to show (overrides profile)",
    )

    ap.add_argument(
        "--validate-jsonl",
        nargs="*",
        default=[],
        metavar="PATH",
        help="JSONL file(s) to validate (paths relative to root)",
    )

    ap.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("scripts/project_state/state_outputs"),
        help="Output directory for state files (default: scripts/project_state/state_outputs)",
    )

    ap.add_argument(
        "--stdout",
        action="store_true",
        help="Print to stdout instead of writing to file",
    )

    ap.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available profiles and exit",
    )

    ap.add_argument(
        "--config",
        type=Path,
        default=Path("settings.toml"),
        help="Path to settings.toml (default: settings.toml)",
    )

    return ap


def list_profiles() -> None:
    """Print available profiles and exit."""
    from project_state.config import BUILTIN_PROFILES

    print("Available profiles:\n")
    for name, profile in BUILTIN_PROFILES.items():
        git_info = f"commits={profile.git.commits}"
        if profile.git.days:
            git_info += f", days={profile.git.days}"

        print(f"  {name:15s}  depth={profile.depth:10s}  {git_info}")
        print(
            f"  {'':<15s}  git={profile.git.enabled}, "
            f"files={profile.files.enabled}, "
            f"docs={profile.docs.enabled}, "
            f"arch={profile.architecture.enabled}, "
            f"validation={profile.validation.enabled}"
        )
        print()


def main() -> int:
    args = build_argparser().parse_args()

    # Handle --list-profiles
    if args.list_profiles:
        list_profiles()
        return 0

    # Resolve paths
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config

    # Load configuration
    config = load_config(config_path)

    # Get and configure profile
    profile = get_profile(config, args.profiles)
    profile = apply_cli_overrides(
        profile,
        depth=args.depth,
        days=args.days,
        commits=args.commits,
    )

    # Collect state
    git_state = collect_git_state(root, profile.git)
    file_discovery = collect_file_discovery(root, profile.files)
    docs_state = collect_docs(root, profile.docs)
    architecture_state = collect_architecture(root, profile.architecture)

    # Warn if no data was collected
    if git_state is None and file_discovery is None and docs_state is None:
        print(
            "Warning: No data collected. Is this a git repository?",
            file=sys.stderr,
        )

    # Recent work analysis (if days is set)
    recent_work = None
    if profile.git.days and profile.git.enabled:
        recent_work = collect_recent_work_summary(root, profile.git.days)

    # JSONL validation (if requested or enabled in profile)
    validation_results = None
    jsonl_paths = args.validate_jsonl
    if jsonl_paths or profile.validation.enabled:
        # If no explicit paths but validation enabled, try to find index files
        if not jsonl_paths and file_discovery and file_discovery.index_location:
            # Look for chunks.jsonl in the index directory
            chunks_file = file_discovery.index_location / "chunks.jsonl"
            if chunks_file.exists():
                jsonl_paths = [str(chunks_file.relative_to(root))]

        if jsonl_paths:
            validation_results = validate_jsonl_files(root, jsonl_paths, profile.validation)

    # Render output
    output = render_project_state(
        root=root,
        profile=profile,
        project_name=config.project_name,
        git_state=git_state,
        file_discovery=file_discovery,
        docs_state=docs_state,
        validation_results=validation_results,
        recent_work=recent_work,
        architecture_state=architecture_state,
    )

    # Write or print
    if args.stdout:
        print(output)
    else:
        # Create output directory if needed
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename: <profile>_<date>.md
        date_str = dt.datetime.now().strftime("%Y-%m-%d")
        filename = f"{profile.name}_{date_str}.md"
        output_path = output_dir / filename

        output_path.write_text(output, encoding="utf-8")
        print(f"Written to {output_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
