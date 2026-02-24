## Spec 04: Regulatory Corpus Ingestion & Adversarial Eval (Release-Gated)

### Title

Add Structured Regulatory Corpus Ingestion with Canonical Citations and Release-Gated Adversarial Evaluation

### Context / Problem

The system currently operates on a personal Obsidian vault — a forgiving corpus where imprecise retrieval is often tolerated. To demonstrate production-grade **citation discipline**, **abstention correctness**, and **grounding**, the system needs an adversarial corpus where these properties are objectively testable and verifiable.

Regulatory text is ideal because:

* claims must be traceable to specific sections and subsections
* abstention is critical (the system must not invent provisions)
* cross-references and definitions test multi-hop retrieval
* citation precision is verifiable by humans and by rules

This work is primarily about **risk containment** and **behavioral regression detection**, not corpus expansion.

### Goals

* Normalize regulatory source material into **canonical citation units** (vault-compatible markdown)
* Establish a **deterministic citation scheme** with stable identifiers and provenance
* Preserve legal hierarchy in **chunk metadata** (regime/instrument/part/section/subsection)
* Convert internal cross-references to **deterministic wikilinks**
* Build an adversarial evaluation dataset that tests:

  * citation precision
  * abstention behavior
  * groundedness / unsupported claims
  * cross-reference synthesis
* Produce evaluation outputs that support a clear **ship / block decision** for regulatory-facing changes (via the verdict layer)

### Non-Goals

* Parsing arbitrary legal formats (start with well-structured sources we can normalize deterministically)
* Legal interpretation beyond cited text
* Multilingual support
* Replacing the Obsidian vault corpus (regulatory is additive)
* Adding regulatory-specific retrieval logic (correctness is enforced via metadata + eval + gates)

---

## Proposed Solution

## Canonical Regulatory Document Normalization

### Overview

Regulatory source material is normalized into **canonical citation units** prior to chunking and indexing. Each canonical unit corresponds to a single legally citable section (e.g., `10 CFR §50.34`) and is represented as a standalone, vault-compatible Markdown document.

This provides:

* deterministic, stable citations
* chunker-independent evaluation truth
* precise grounding and abstention testing
* production-grade provenance/versioning

### Canonical Unit Definition

For US NRC / CFR:

* canonical unit = **CFR section** (e.g., `10 CFR §50.34`)
* subsections `(a)`, `(b)` become headings inside the file

### File Layout

```
corpus/
  us-nrc/
    10-CFR/
      part-50/
        50.34.md
        50.36.md
```

### Document Format (normalized)

```markdown
---
regime: US-NRC
instrument: 10-CFR
instrument_version: "2023-01"
part: 50
section: 50.34
title: Contents of applications; technical information
citation_key: 10 CFR §50.34
source_url: https://www.nrc.gov/...
source_revision: nrc-2023-01-01
effective_date: 2023-01-01
corpus: regulatory
---

# 10 CFR §50.34 — Contents of applications; technical information

## (a)
Each application for a construction permit shall include...

## (b)
The application must also include...
```

### Normalization Rules

* **One file per canonical citation unit** (one file == one `citation_key`)
* **Frontmatter is authoritative** (don’t infer section identity from headings)
* **Heading hierarchy mirrors legal structure**; subsection letters resolve mechanically
* **Determinism guarantee**: same source + version → same normalized markdown output + IDs

### Input -> Output Mapping (Explicit)

This is the concrete mapping the current pipeline applies when converting eCFR XML into
normalized markdown and then indexed chunks.

| Input (eCFR XML) | Normalized Markdown Output | Indexed Chunk Metadata |
|---|---|---|
| `DIV5 TYPE="PART" N="Part 50"` | Output directory `.../part-50/` | `part: "50"` |
| `DIV8 TYPE="SECTION" N="50.34"` | File `part-50/50.34.md` | `section: "50.34"` |
| `HEAD` text `§ 50.34 ...` | H1 `# 10 CFR §50.34 — ...` | `citation_key: "10 CFR §50.34"` |
| `P` text `(a) ...` | `## (a)` + body text | `citation: "10 CFR §50.34(a)"` (or deeper via section path) |
| `P` text `(1) ...` under `(a)` | `### (1)` + body text | `citation: "10 CFR §50.34(a)(1)"` |
| Inline reference `§ 50.36` | `[[10 CFR §50.36]]` | `cross_references: ["10 CFR §50.36"]` |
| CLI arg `--instrument-version 2026-02-02` | frontmatter `instrument_version: "2026-02-02"` | preserved in chunk metadata |
| CLI arg `--source-revision ecfr-2026-02-02` | frontmatter `source_revision: "ecfr-2026-02-02"` | preserved in chunk metadata |

#### Example: XML Snippet -> Markdown File

Input XML:

```xml
<DIV8 N="50.34" TYPE="SECTION">
  <HEAD>§ 50.34 Contents of applications; technical information.</HEAD>
  <P>(a) Each application shall include the information required by § 50.36.</P>
  <P>(1) The information must be complete and auditable.</P>
</DIV8>
```

Output file `part-50/50.34.md`:

```markdown
---
section: "50.34"
citation_key: "10 CFR §50.34"
cross_references: ["10 CFR §50.36"]
corpus: "regulatory"
...
---

# 10 CFR §50.34 — Contents of applications; technical information

## (a)
Each application shall include the information required by [[10 CFR §50.36]].

### (1)
The information must be complete and auditable.
```

#### Example: Markdown -> Chunk Citation

For a chunk produced from the `### (1)` block above, enrichment resolves:

* `citation_key = "10 CFR §50.34"` (frontmatter root)
* `section_path = "(a) > (1)"` (from structural chunking)
* `citation = "10 CFR §50.34(a)(1)"` (synthesized, most specific)

### Cross-References

Internal references are converted to deterministic wikilinks:

```markdown
See [[10 CFR §50.36]] for technical specification requirements.
```

Cross-references are also recorded in metadata for multi-hop retrieval and adversarial evals.

---

## Chunk Metadata Schema (Regulatory)

Regulatory chunks carry additional metadata:

```python
{
  "corpus": "regulatory",
  "regime": "US-NRC",
  "instrument": "10-CFR",
  "instrument_version": "2023-01",
  "part": "50",
  "section": "50.34",
  "citation_key": "10 CFR §50.34",         # canonical root
  "citation": "10 CFR §50.34(a)",          # most specific, from heading
  "source_url": "https://www.nrc.gov/...",
  "source_revision": "nrc-2023-01-01",
  "effective_date": "2023-01-01",
  "cross_references": ["10 CFR §50.36"],
}
```

**Invariant:** Every answer must either:

* cite at least one canonical citation present in retrieved context, **or**
* explicitly abstain

---

## Preprocessing Script: `scripts/ingest_regulatory.py`

Responsibilities:

1. **Acquire** source text (download or import)
2. **Normalize** into canonical markdown units (one file per `citation_key`)
3. **Rewrite cross-references** as wikilinks

### Operational Flow (Current CLI / Make Targets)

1. `make normalize-regulatory`
   * Runs normalization only
   * Produces:
     * `corpus/us-nrc/10-CFR/part-50/*.md`
2. `make index-regulatory` (or `make index-regulatory-dummy`)
   * Runs normalization + indexing
   * Produces:
     * normalized corpus files (as above)
     * `artifacts/indexes/regulatory/chunks.jsonl`
     * `artifacts/indexes/regulatory/manifest.json`

Suggested interfaces:

```python
def normalize_cfr_part(
    *,
    raw_source_path: Path,
    output_dir: Path,
    regime: str,
    instrument: str,
    part: int,
    instrument_version: str,
    source_url: str,
    source_revision: str,
    effective_date: str,
) -> list[Path]:
    """Normalize a CFR part into canonical section markdown files."""

def extract_cross_references(text: str) -> list[str]:
    """Return canonical citation_keys referenced in the text."""

def rewrite_cross_references_to_wikilinks(text: str) -> str:
    """Rewrite references like '§ 50.36' into '[[10 CFR §50.36]]'."""
```

---

## Integration with Existing Architecture

No new chunker is required if your `ObsidianStructuralChunker` is heading-aware.

Pipeline:

```mermaid
graph LR
  A[Raw CFR/NRC Source] -->|normalize| B[Canonical Markdown Units]
  B -->|ObsidianStructuralChunker| C[Chunks]
  C -->|enrich metadata| D[Regulatory Chunks]
  D -->|Embed + Index| E[Vector Store]
```

No regulatory-specific retrieval logic; regulatory correctness is enforced via:

* deterministic normalization
* metadata
* adversarial eval datasets
* verdict-based gates

---

## Adversarial Evaluation Dataset

A dedicated dataset exercises:

* citation precision
* abstention (not-in-corpus / not-in-instrument)
* cross-reference synthesis
* hallucination resistance (non-existent sections)

**Target:** ≥ 25 queries; ≥ 5 per category (expand over time).

See updated schema in section (2) below.

---

## Acceptance Criteria

* [ ] Normalizer produces canonical NRC/CFR markdown units (≥ 25 sections)
* [ ] Output documents follow frontmatter + heading conventions above
* [ ] Cross-references become deterministic wikilinks
* [ ] Structural chunker produces reasonable chunks without modification
* [ ] Chunks carry canonical regulatory metadata (`citation_key`, `citation`, `instrument_version`, etc.)
* [ ] Adversarial eval dataset contains ≥ 25 queries across 4 categories
* [ ] Eval harness runs successfully on regulatory corpus index
* [ ] Verdict layer can gate regulatory eval results
* [ ] At least one change is **blocked** by verdict due to regulatory regression (demonstration run)

---

## Test Plan

* Normalization

  * frontmatter presence + required keys
  * determinism (same input produces same output)
  * wikilink rewriting correctness
  * citation manifest correctness

* Chunking & metadata

  * correct `citation_key` extraction from frontmatter
  * correct `citation` extraction from subsection headings
  * cross-reference list extracted into metadata

* Eval dataset

  * schema loads
  * citation references resolve to documents
  * unanswerable cases behave as expected

---

## Risks

* Source formatting variance → start with a single CFR part + deterministic parser
* Citation extraction edge cases → rely on declared frontmatter + mechanical heading parsing
* Dataset small initially → note sample size in verdicts, expand iteratively
* Corpus size growth → keep regulatory indexes separate from vault indexes

---

## Follow-ups

* Expand to additional parts / instruments
* Add citation-to-chunk resolver for tighter grounding analysis
* Add “stale version” detection via instrument_version/source_revision
* Cross-instrument comparison queries (explicitly not MVP)
