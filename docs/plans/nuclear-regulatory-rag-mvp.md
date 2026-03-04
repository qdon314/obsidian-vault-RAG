
# Nuclear Regulatory RAG MVP

## MVP Goal

> **A production-minded RAG system for nuclear regulatory text that produces citation-bounded answers, abstains safely, and blocks unsafe changes via eval gates.**

---

## Corpus

### Option A: U.S. NRC (recommended)

* 10 CFR Part 50 (Domestic Licensing of Production and Utilization Facilities)
* Selected NRC regulatory guides (RGs)

Why:

* Clear sectioning
* Real-world relevance
* Widely cited
* Easily defensible choice

### Option B: IAEA Safety Standards

* General Safety Requirements (GSR)
* Safety Guides (SSG)

Why:

* Cleaner prose
* International
* Less legalese, more engineering

---

## Document normalization

Each **section / article** becomes one canonical Markdown unit.

Example:

```markdown
---
regime: US-NRC
instrument: 10-CFR
part: 50
section: 50.34
title: Contents of applications; technical information
citation_key: 10 CFR §50.34
source_url: https://www.nrc.gov/...
effective_date: 2023-01-01
---

# 10 CFR §50.34 — Contents of applications; technical information

## (a)
Each application for a construction permit shall include...

## (b)
The application must also include...
```

This enables:

* stable chunk provenance
* deterministic citations
* eval ground truth

---

## Citation scheme (simple, strict)

Answers must cite using:

* `[[10 CFR §50.34(a)]]`
* `[[IAEA GSR Part 4 §3.2]]`

No freeform citations.
No fuzzy references.

This makes groundedness *auditable*.

---

## Eval dataset (initial scope)

Start with **30–50 eval queries**, not more.

Types:

* **Factual**
  “What must be included in a construction permit application?”
* **Conditional**
  “When is X required under 10 CFR §50?”
* **Negative / abstention**
  “Does the NRC allow Y?” (when not specified)
* **Multi-section synthesis**
  “What documentation is required and under what conditions?”

Each query has:

* relevant section IDs
* expected abstention flag (if applicable)

---

## Eval gates (initial)

* Recall@10 ≥ baseline – 3%
* Unsupported claims must not increase
* Abstention false-negative rate must not increase
* P95 latency ≤ +200ms

---

## MVP deliverables (what exists at the end)

1. **A clean README**

   > “Nuclear Regulatory RAG Reference Implementation”

2. **One blocked release**

   * with `verdict.md`
   * explaining *why* it was blocked

3. **CI run artifacts**

   * metrics
   * verdict
   * traces

4. **A short “Why this matters” doc**

   * correctness > cleverness
   * evals as safety rails
   * abstention is a feature