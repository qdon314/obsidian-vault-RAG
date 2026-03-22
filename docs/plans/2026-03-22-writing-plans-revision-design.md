# `writing-plans` Skill Revision — Design Document

## Problem

The current `writing-plans` skill produces plans that contain complete implementation code — full function bodies, test files, exact shell commands, and step-by-step TDD sequences. This makes the executor a typist rather than an engineer, causes plans to become stale the moment implementation reveals a better approach, and bloats plan documents with duplicated code.

## Design Principle

Plans specify **what** to build, **where** it goes, and **how to know it's done**. Implementation determines **how** to build it.

**In the plan:** File paths, type contracts, acceptance criteria, constraints, rationale, ordering, parallelism hints.

**Not in the plan:** Function bodies, test code, shell commands, git commands, TDD step sequences, commit messages.

## TDD Responsibility Split

The plan does not prescribe TDD steps within each task. Instead:

- The plan chunks work into **commit-sized deliverables** with clear acceptance criteria.
- The `executing-plans` skill enforces TDD discipline during execution.
- The `subagent-driven-development` skill handles dispatch and review.

This eliminates duplication where every plan repeats "write test → run test → implement → run test → commit."

## Task Specificity Level

Plans use **interface contracts + acceptance criteria**:

- **Contract:** Type signatures, protocol shapes, dataclass fields — enough to define the interface, not the implementation.
- **Acceptance:** Observable behaviors that must be true when done, edge cases, test coverage expectations.
- **Constraints:** Architectural rules, stability guarantees, compatibility requirements, and the *why* behind them.

## File Paths

Plans specify **exact file paths** for all created, modified, and tested files. File placement is an architectural decision that affects imports, discoverability, and module boundaries — it belongs in the plan.

## Task Structure

Each task includes:

| Field | Purpose |
|-------|---------|
| **Why** | Connects this task to the larger plan rationale |
| **Files** | Exact paths: create, modify, test |
| **Contract** | Type shapes and signatures |
| **Acceptance** | Observable behaviors, edge cases, test expectations |
| **Constraints** | Architectural rules, stability guarantees |

## Ordering and Parallelism

- Tasks are numbered sequentially as the default execution order.
- Independent tasks are explicitly marked as parallel groups with shared headings.
- Each parallel group notes its dependencies.
- The `milestone-workflow` skill consumes these hints for worktree dispatch.

## Plan Document Structure

```
# [Feature Name] Implementation Plan

> For Claude: REQUIRED SUB-SKILL: use executing-plans

**Goal / Architecture / Tech Stack / Design doc**

## Rationale
[Why these tasks, this order, this decomposition]

### Task 1: [Name]
**Why / Files / Contract / Acceptance / Constraints**

### Tasks N–M (parallel): [Group]
> Independent, depend on Task X.
### Task N: ...
### Task N+1: ...
```

## What Changes

| Aspect | Current | Revised |
|--------|---------|---------|
| Code blocks | Full implementation + test code | Type shapes and signatures only |
| Steps per task | 5 (test → fail → impl → pass → commit) | None — executor owns workflow |
| Done criteria | Implicit (test passes) | Explicit acceptance + constraints |
| Rationale | Absent | Plan-level + per-task why |
| Parallelism | Implicit sequential | Explicit parallel group hints |
| Size per task | 50-100 lines | 15-30 lines |

## What Plans Do NOT Include

1. Function/method bodies
2. Test code
3. Shell commands
4. TDD step sequences
5. Commit messages

## Execution Handoff

Unchanged — plan offers subagent-driven (same session) or parallel session (new session in worktree).
