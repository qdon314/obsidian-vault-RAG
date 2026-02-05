# Evaluation & Signal Analysis Protocol

This mode is for interpreting eval results as signals and diagnosing failure modes.

## Required failure taxonomy
When reviewing eval outputs, classify issues into:
- **Retrieval miss** (relevant chunk not retrieved)
- **Rerank failure** (relevant retrieved but pushed down)
- **Context packing** (retrieved but not included / truncated / budget)
- **Generation** (context sufficient but answer wrong/partial)
- **Judge / labeling noise** (gold/judge disagreement, ambiguous query)

## Required evidence
For any claim about metric movement or quality:
- State dataset size (or subset size) and query types affected (if known)
- Provide 2–5 concrete examples:
  - query id (or description)
  - what went wrong
  - what chunk(s) should have mattered (if known)

## Guardrails
- Do not propose prompt tuning, threshold tuning, or model swaps until:
  - retrieval and context packing are verified for representative failures
- Treat metrics as diagnostic signals, not goals.
- Prefer changes that improve observability and failure attribution.

## Recommended outputs
- A short “Top 3 failure modes this run”
- Suggested next experiments (small, controlled)
- If proposing a fix, name what metric it should change and why
