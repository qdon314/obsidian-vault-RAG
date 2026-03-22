---
name: backlog
description: Use when writing plans, defining epics, creating tasks, or managing the project backlog via GitHub Projects v2. Also use when the user asks to track, prioritize, or organize work items.
---

# Backlog Management with gh-pm

## Overview

Manage the project backlog in GitHub Projects v2 using `gh pm`. This bridges implementation planning to tracked work items — epics become parent issues, plan tasks become sub-issues, and status/priority flow through the project board.

**Announce at start:** "Using backlog skill to manage work items in GitHub Projects."

## Repo Context

- **Repo:** `qdon314/obsidian-vault-RAG`
- **Config:** `.gh-pm.yml` (project-level config, already initialized)
- **All commands require:** `--repo qdon314/obsidian-vault-RAG`

## Quick Reference

| Action | Command |
|--------|---------|
| List backlog | `gh pm list --repo qdon314/obsidian-vault-RAG` |
| Filter by status | `gh pm list --status "in_progress" --repo qdon314/obsidian-vault-RAG` |
| Filter by priority | `gh pm list --priority "high" --repo qdon314/obsidian-vault-RAG` |
| View issue | `gh pm view <number> --repo qdon314/obsidian-vault-RAG` |
| Create issue | `gh pm create --title "..." --body "..." --repo qdon314/obsidian-vault-RAG` |
| Create with priority | `gh pm create --title "..." --body "..." --priority high --repo qdon314/obsidian-vault-RAG` |
| Update status | `gh pm move <number> --status "in_progress" --repo qdon314/obsidian-vault-RAG` |
| Update priority | `gh pm move <number> --priority "high" --repo qdon314/obsidian-vault-RAG` |
| Split into sub-issues | `gh pm split <number> "Task 1" "Task 2" --repo qdon314/obsidian-vault-RAG` |
| Split from body checklist | `gh pm split <number> --from=body --repo qdon314/obsidian-vault-RAG` |
| Dry-run split | `gh pm split <number> --from=body --dry-run --repo qdon314/obsidian-vault-RAG` |
| Find untracked issues | `gh pm intake --repo qdon314/obsidian-vault-RAG` |
| JSON output | `gh pm list --json number,title,status,priority --repo qdon314/obsidian-vault-RAG` |
| Open board in browser | `gh pm list --web --repo qdon314/obsidian-vault-RAG` |

## Status Values

`todo` | `in_progress` | `in_review` | `done`

## Priority Values

`low` | `medium` | `high` | `critical`

## Integration with Plan Writing

When used alongside `superpowers:writing-plans`, follow this workflow:

### 1. Create the Epic

Before writing a plan, create a parent issue that represents the feature/initiative:

```bash
gh pm create \
  --title "Epic: <feature-name>" \
  --body "## Goal\n<one-line goal>\n\n## Architecture\n<approach summary>" \
  --priority high \
  --repo qdon314/obsidian-vault-RAG
```

### 2. Write the Plan

Use `superpowers:writing-plans` as normal. Save to `docs/plans/`.

### 3. Link Plan to Issue

After writing the plan, update the issue body to include:
- A link to the plan file (`docs/plans/<filename>.md`)
- A task checklist (`- [ ] Task N: ...`) for each plan task
- Any design decisions made during planning

```bash
gh issue edit <number> --repo qdon314/obsidian-vault-RAG --body "$(cat <<'EOF'
...updated body with plan link and task checklist...
EOF
)"
```

### 4. Create Sub-Issues from Plan Tasks

After the plan is written, create sub-issues for each task. Two approaches:

**Option A — Split from arguments (preferred for plans):**
```bash
gh pm split <epic-number> \
  "Task 1: <component name>" \
  "Task 2: <component name>" \
  --repo qdon314/obsidian-vault-RAG
```

**Option B — Split from issue body checklist:**
If the epic body contains a markdown checklist (`- [ ] Task...`), use:
```bash
gh pm split <epic-number> --from=body --repo qdon314/obsidian-vault-RAG
```

### 4. Track Progress During Execution

As tasks are completed during `superpowers:executing-plans`:
```bash
gh pm move <number> --status "in_progress" --repo qdon314/obsidian-vault-RAG
# ... after task passes ...
gh pm move <number> --status "done" --repo qdon314/obsidian-vault-RAG
```

## Standalone Backlog Operations

### Triage untracked issues
```bash
gh pm intake --repo qdon314/obsidian-vault-RAG
```

### Review current sprint
```bash
gh pm list --status "in_progress,in_review" --repo qdon314/obsidian-vault-RAG
```

### View full backlog prioritized
```bash
gh pm list --priority "critical,high,medium,low" --repo qdon314/obsidian-vault-RAG
```

## Common Mistakes

- **Missing `--body`**: `gh pm create` requires `--body` in non-interactive mode
- **Missing `--repo`**: Always pass `--repo qdon314/obsidian-vault-RAG` since org is not set
- **Labels must exist**: If using `--labels`, the label must already exist on the repo
- **Priority field**: The GitHub Project must have a "Priority" single-select field configured for `--priority` to work (create it in the project settings if needed)
