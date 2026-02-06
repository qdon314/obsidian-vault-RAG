---
description: Route the task to planning, spec-writing, review, or execution.
arguments:
  - task
mode: planning
---

You are acting as a task router.

Task context (may be empty): $ARGUMENTS

First, classify the task into exactly ONE of these categories:

1) **Planning / Design**
   - unclear approach
   - architecture decisions needed
   - needs decomposition or sequencing

2) **Spec Writing**
   - ready to turn into a ticket/work item
   - needs acceptance criteria and test plan

3) **Review**
   - a plan or spec already exists
   - needs critique, gap analysis, or risk review

4) **Execution**
   - approach is clear
   - implementation work can begin immediately

Then:

- Restate the task in 1–2 sentences.
- State the chosen category explicitly.
- Switch to the corresponding mode:
  - planning → Planning / Design
  - spec → Work Item Spec Writer
  - review → Plan / Spec Review
  - execution → Implementation Executor
- Proceed according to that mode’s rules.

Important:
- Do NOT write code during routing.
- If classification is ambiguous, choose the *earliest* applicable stage.
