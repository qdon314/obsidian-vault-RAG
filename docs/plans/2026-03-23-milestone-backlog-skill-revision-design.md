# Milestone & Backlog Skill Revision

**Date:** 2026-03-23
**Status:** Design finalized

## Problem Statement

The `/milestone-workflow` and `/backlog` skills have overlapping responsibilities and incompatible issue-creation paths, causing duplicated parent issues, orphaned sub-issues, and inconsistent project board state.

## Issues Identified

### 1. Milestones are just issues with an M# prefix

- Issues #3–#9 were created by `/backlog` as regular issues titled "M0:", "M1:", etc.
- No actual GitHub Milestones exist (the repo has zero `milestones.nodes`)
- These issues lack milestone-level semantics: no progress bars, no due dates, no milestone-based filtering

### 2. Orphaned issues outside the project

- The GitHub Project contains: #2–9 (from `/backlog`) and #25–31 (from `/milestone-workflow` M2)
- **Missing from the project:** #10–18 (`/milestone-workflow` M1 parent + all sub-issues)
- Root cause: `/milestone-workflow` uses raw `gh issue create`, which does not add issues to the GitHub Project. Only `gh pm create` / `gh pm split` does that.

### 3. Duplicate parent issues

- `/backlog` created issue #4 ("M1: Stage 0 source view + Stage 1a structural segmentation")
- `/milestone-workflow` then created issue #10 with the **same title** as its own parent
- Same duplication for M2: #5 (backlog) vs #25 (milestone-workflow)
- The backlog issues (#4, #5) are in the project; the milestone-workflow parents (#10, #25) are tracked separately in `.milestone-state-*.json`

## GitHub Primitives Used

GitHub has no formal "Epic" concept. The hierarchy primitives we use:

| Primitive | Role in our model |
|---|---|
| **Sub-issues** | Child issues under the Epic; gives parent/child tree with progress |
| **GitHub Milestones** | Delivery phase grouping with progress bars and due dates |
| **GitHub Projects v2** | Board/table view with status and priority fields |

"Epic" is our naming convention for a parent issue — not a GitHub feature.

## Chosen Design: Separated Intake and Execution

### Data Model

```
Design doc (canonical architecture + milestone definitions)
  └─ Epic issue (feature container, parent of all tasks)
       ├─ Child issues assigned to GH Milestone "Feature Name — M0"
       ├─ Child issues assigned to GH Milestone "Feature Name — M1"
       ├─ Child issues assigned to GH Milestone "Feature Name — M2"
       └─ ...
```

### Invariants

- **Epic** is the parent issue. Not the milestone.
- **GitHub Milestones** are grouping/delivery buckets — not issues.
- **Milestone names are namespaced** by feature (e.g., `NRC Benchmark — M1`, not bare `M1`) to avoid cross-feature ambiguity.
- **Child issues** are sub-issues of the Epic *and* assigned to a GitHub Milestone.
- **Task hierarchy is flat:** Epic → tasks. No intermediate "phase parent" issues inside a milestone.
- **State JSON** tracks only execution state (branches, PRs, group status) — never milestone structure.
- Tasks are not created until a milestone is activated for execution.

### Epic-to-Milestone Linkage

The association between Epic and milestones is machine-readable in both directions:

- **Epic body** contains a canonical milestone table with exact milestone names and GitHub milestone numbers
- **Each GitHub Milestone description** includes `Epic: #<number>` as a back-link
- `/milestone-workflow` verifies that the requested milestone belongs to the given Epic before doing anything

### Source of Truth and Drift Rules

| State | Canonical source |
|---|---|
| Architecture + milestone definitions | Design doc |
| Execution tracking (tasks, PRs, status) | GitHub (issues, milestones, project) |
| Execution resumability (branches, groups) | `.milestone-state-*.json` |

**Drift rules:**
- Before a milestone is activated, `/backlog ingest` may freely update Epic body and milestone metadata from the design doc
- After a milestone is activated and tasks exist, milestone scope is **frozen unless explicitly migrated**
- `/milestone-workflow` stamps the state file with: design doc path, design doc content hash, milestone section hash — execution is tied to the exact definition it planned from
- If the design doc changes after activation, the skill warns and requires explicit `--force` or a migration step

### Lifecycle

#### Phase 1: Brainstorm/design

Outcome: a design doc with overall architecture, milestone breakdown, inline ADR/doc obligations, and milestone-specific scope descriptions.

#### Phase 2: Backlog ingestion (`/backlog`)

The `/backlog` skill owns feature intake and board-level planning structure:

- Create one **Epic issue** for the feature (via `gh pm create`)
- Create **real GitHub Milestones** for each M# in the design doc (via `gh api`), namespaced by feature
- Write milestone table and design doc link into the Epic body
- Write `Epic: #<number>` into each milestone description
- **Do not create child task issues yet** — only Epic + Milestones

#### Phase 3: Milestone execution (`/milestone-workflow`)

The `/milestone-workflow` skill operationalizes one existing GitHub Milestone:

- Accepts `--design <path>`, `--epic <issue-number>`, `--milestone <id-or-name>`
- Runs preflight checks (see below)
- Locates the matching milestone section in the design doc
- Extracts ADR/doc work due by that milestone
- Generates a milestone plan (via `writing-plans`)
- Derives tasks and dependency DAG
- Creates child issues **under the Epic** (via `gh pm`) and assigns them to the GitHub Milestone
- Groups tasks into worktree/PR batches
- Orchestrates execution (worktrees, TDD, PRs)

### Command Tooling

| Operation | Tool |
|---|---|
| Create/list/update GitHub Milestones | `gh api` (repo-level concept) |
| Create issues, add to project, manage status/priority | `gh pm` (project-aware) |
| Worktrees, branches, PRs | `gh pr create`, git worktrees |

### Skill Contracts

#### `/backlog` — feature intake and board-level planning structure

**Purpose:** Ingest a feature design into GitHub planning structure.

**Behavior:**
- Parse design doc title/goal
- Create Epic issue (via `gh pm create`)
- Discover milestones from the design doc
- Create real GitHub Milestones if missing (via `gh api`), namespaced by feature
- Establish bidirectional Epic ↔ Milestone linkage
- Sync milestone summaries and ADR/doc obligations into the Epic body
- Do **not** create task-level sub-issues

**Standalone operations:** List backlog, filter by status/priority, triage untracked issues, update status — all via `gh pm`.

#### `/milestone-workflow` — creation of execution tasks for an approved milestone

**Purpose:** Operationalize one existing GitHub Milestone.

**Arguments:**
- `--design <path>` — design doc
- `--epic <issue-number>` — existing Epic issue
- `--milestone <id-or-name>` — existing GitHub Milestone
- `--plan <path>` — optional pre-written plan
- `--resume <state-path>` — resume from state file
- `--dry-run` — preview without creating

**Preflight checks** (before creating anything):
1. Epic issue exists
2. GitHub Milestone exists
3. Milestone is linked to the given Epic (via `Epic: #N` in milestone description)
4. No open tasks already exist for this Epic + Milestone pair (unless `--resume`)
5. Design doc contains a matching milestone section
6. State file, if present, matches the same Epic + Milestone pair

**Phases:**
1. Resolve milestone scope from design doc
2. Extract milestone-required docs/ADRs
3. Generate milestone plan
4. Derive tasks and dependency DAG
5. Create child issues under Epic, assign to GitHub Milestone, add to project (via `gh pm`)
6. Group tasks into worktree/PR batches
7. Execute/resume

**Owns:** Creation of execution tasks for an approved milestone, including placing those tasks into the project with required initial fields.

**Does not:** Create its own parent issue. Create milestones. Do general board triage, reprioritization, or milestone creation.

### Traps to Avoid

1. **Creating all milestone tasks up front** — creates a giant frozen backlog from a still-evolving design. Materialize tasks only when the milestone is activated.
2. **Storing milestone truth in local JSON** — the state file tracks execution state only (plan path, group DAG, branch/worktree mappings, PR numbers, per-group status). Design doc + GitHub own milestone structure.
3. **Reintroducing intermediate parent issues** — keep the hierarchy flat (Epic → tasks). Do not create "milestone parent" issues inside the tree.

## Concrete Example: NRC Benchmark

```bash
# Phase 2: Backlog ingestion
/backlog ingest --design docs/plans/2026-03-21-nrc-benchmark-generation-design.md
# Creates:
#   Epic: "NRC Benchmark Generation Pipeline" (issue #42)
#   GH Milestones: "NRC Benchmark — M0", "NRC Benchmark — M1", ..., "NRC Benchmark — M6"
#   Epic body links design doc + milestone summary table
#   Each milestone description includes "Epic: #42"

# Phase 3: Activate M1
/milestone-workflow --design docs/plans/2026-03-21-nrc-benchmark-generation-design.md \
  --epic 42 --milestone "NRC Benchmark — M1"
# Preflight: verifies Epic #42 exists, milestone linked, no existing tasks, doc section found
# Stamps state file with design doc hash + M1 section hash
# Reads M1 scope section
# Detects M1-required ADRs (eval-schema-boundary, schema-versioning)
# Generates M1 plan
# Creates child issues under Epic #42, assigned to GH Milestone "NRC Benchmark — M1"
# Groups by file dependencies, creates worktrees/PRs
```

## Immediate Cleanup Needed

1. Add orphaned issues #10–18 to the GitHub Project
2. Resolve duplicate parent issues (#4 vs #10, #5 vs #25) — close duplicates and update references
3. Create real GitHub Milestones for M0–M6 (namespaced: `NRC Benchmark — M#`)
4. Reassign existing child issues to proper Epic + Milestone structure
5. Migrate or delete `.milestone-state-*.json` files that reference superseded parent issues
