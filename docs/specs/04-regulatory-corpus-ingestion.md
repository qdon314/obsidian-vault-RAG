# Spec 04: Regulatory Corpus Ingestion & Adversarial Eval

## Title
Add Structured Regulatory Document Ingestion with Canonical Citations and Adversarial Eval Dataset

## Context / Problem

The system currently operates on a personal Obsidian vault — a forgiving corpus where imprecise retrieval is tolerable. To demonstrate production-grade citation discipline, abstention correctness, and grounding, the system needs an adversarial corpus where these properties are testable and verifiable.

Regulatory texts (GDPR, CFR Title 21, etc.) are ideal:
- Claims must be traceable to specific articles and sections
- Abstention is critical — the system must not hallucinate provisions that don't exist
- Cross-reference queries test multi-hop retrieval
- Citation precision is verifiable by humans

## Goals
- Define a preprocessing pipeline that converts regulatory documents to vault-compatible markdown
- Establish a canonical citation scheme with stable identifiers
- Preserve document hierarchy (Title / Chapter / Article / Section) in chunk metadata
- Build an adversarial eval dataset testing citation discipline, abstention, and grounding

## Non-Goals
- Parsing arbitrary legal document formats (focus on well-structured sources convertible to markdown)
- Legal interpretation or compliance features
- Multilingual support
- Replacing the Obsidian vault corpus (regulatory is an additional corpus)

## Proposed Solution

### Document Format Convention

Regulatory documents are converted to markdown following this structure:

```markdown
---
source: GDPR
full_title: "General Data Protection Regulation (EU) 2016/679"
effective_date: 2018-05-25
citation_prefix: "GDPR"
---

# Chapter III - Rights of the Data Subject

## Article 17 - Right to Erasure ('Right to be Forgotten')

### Article 17(1)

The data subject shall have the right to obtain from the controller
the erasure of personal data concerning him or her without undue delay
and the controller shall have the obligation to erase personal data
without undue delay where one of the following grounds applies:

(a) the personal data are no longer necessary in relation to the
purposes for which they were collected or otherwise processed;

(b) the data subject withdraws consent on which the processing is
based according to point (a) of [[GDPR Art. 6(1)]] or point (a) of
[[GDPR Art. 9(2)]], and where there is no other legal ground for the
processing;
```

Conventions:
- YAML frontmatter carries document-level metadata
- Markdown heading hierarchy mirrors the legal hierarchy
- Wikilink syntax `[[GDPR Art. 6(1)]]` marks internal cross-references
- One file per Chapter (keeps files manageable, headings map to articles/sections)

### Citation Scheme

| Legal Level | Markdown Heading | Canonical Citation | chunk metadata |
|---|---|---|---|
| Chapter | `# Chapter III` | GDPR Ch. III | `chapter: "III"` |
| Article | `## Article 17` | GDPR Art. 17 | `article: "17"` |
| Section | `### Article 17(1)` | GDPR Art. 17(1) | `section: "1"` |
| Point | `(a)` within body text | GDPR Art. 17(1)(a) | Inherited from parent |

The `citation` metadata field on each chunk contains the most specific citation resolvable from the heading hierarchy: e.g., `"GDPR Art. 17(1)"`.

### Chunk Metadata Schema

Regulatory chunks carry metadata beyond standard vault chunks:

```python
{
    "source": "GDPR",
    "citation": "GDPR Art. 17(1)",
    "citation_prefix": "GDPR",
    "chapter": "III",
    "article": "17",
    "section": "1",
    "cross_references": ["GDPR Art. 6(1)", "GDPR Art. 9(2)"],
    "effective_date": "2018-05-25",
    "corpus": "regulatory",
}
```

The `corpus: "regulatory"` tag enables filtering eval queries by corpus type.

### Preprocessing Script: `scripts/ingest_regulatory.py`

```python
"""Convert regulatory source documents to vault-compatible markdown."""

def convert_regulatory_text(
    source_path: Path,
    output_dir: Path,
    citation_prefix: str,
    metadata: dict[str, str],
) -> list[Path]:
    """
    Convert a regulatory plain-text or markdown source to
    structured vault-compatible markdown files.

    - Splits into one file per Chapter/Title
    - Adds YAML frontmatter
    - Converts internal references to wikilinks
    - Returns list of output file paths
    """

def enrich_chunk_metadata(
    chunk: Chunk,
    frontmatter: dict[str, str],
) -> Chunk:
    """
    Post-chunking enrichment: extract citation from heading hierarchy.

    Parses the chunk's section headings to build the canonical citation
    string and adds regulatory metadata fields.
    """
```

### Integration with Existing Architecture

The existing `ObsidianStructuralChunker` splits on markdown headings — regulatory markdown follows the same convention, so **no new chunker is needed**. The pipeline is:

```mermaid
graph LR
    A[Regulatory PDF/Text] -->|ingest_regulatory.py| B[Vault-Compatible Markdown]
    B -->|ObsidianStructuralChunker| C[Chunks]
    C -->|enrich_chunk_metadata| D[Chunks with Citation Metadata]
    D -->|Embedder + VectorStore| E[Indexed]
```

The enrichment step runs post-chunking to add citation metadata from the heading hierarchy and frontmatter. This can be integrated into the ingestion pipeline or run as a post-processing step.

### Adversarial Eval Dataset: `eval/datasets/regulatory_queries.jsonl`

Four query categories:

**1. Citation Precision** — Answer must reference specific articles

```json
{
    "qid": "reg-cite-01",
    "query": "Under GDPR, when can a data subject request erasure of personal data?",
    "relevant_chunk_ids": ["<art17-1-chunk-id>"],
    "expected_answer": "Under Article 17(1) of GDPR, the data subject has the right to obtain erasure when the data is no longer necessary, consent is withdrawn, ...",
    "query_type": "factual",
    "difficulty": "easy",
    "tags": ["regulatory", "citation-precision", "gdpr"]
}
```

**2. Abstention** — Query about provisions not in the corpus

```json
{
    "qid": "reg-abstain-01",
    "query": "What penalties does GDPR impose for cryptocurrency exchanges?",
    "relevant_chunk_ids": [],
    "is_unanswerable": true,
    "unanswerable_reason": "not_in_corpus",
    "tags": ["regulatory", "abstention", "gdpr"]
}
```

**3. Cross-Reference / Synthesis** — Requires connecting multiple articles

```json
{
    "qid": "reg-synth-01",
    "query": "What conditions must be met for consent to be a valid legal basis under GDPR?",
    "relevant_chunk_ids": ["<art6-1a-id>", "<art7-id>", "<recital32-id>"],
    "requires_synthesis": true,
    "query_type": "aggregation",
    "difficulty": "hard",
    "tags": ["regulatory", "cross-reference", "gdpr"]
}
```

**4. Hallucination Resistance** — Queries about non-existent provisions

```json
{
    "qid": "reg-halluc-01",
    "query": "What does Article 100 of GDPR say about AI systems?",
    "relevant_chunk_ids": [],
    "is_unanswerable": true,
    "unanswerable_reason": "not_in_corpus",
    "expected_answer": "GDPR does not contain an Article 100. The regulation has 99 articles.",
    "tags": ["regulatory", "hallucination-resistance", "gdpr"]
}
```

**Target:** At least 25 queries across all 4 categories, with a minimum of 5 per category.

### Eval Integration

The regulatory dataset works with the existing eval harness without modification. The verdict layer (spec 03) can gate on regulatory eval results separately or combined with the vault corpus:

```bash
# Run eval on regulatory corpus only
./scripts/py eval/scripts/run_eval.py \
    --queries eval/datasets/regulatory_queries.jsonl \
    --index artifacts/indexes/regulatory \
    --run-generation --use-llm-judge \
    --run-name "regulatory-baseline"

# Verdict
./scripts/py eval/scripts/verdict.py \
    --current eval/runs/latest/ \
    --baseline eval/runs/regulatory-baseline/ \
    --fail-on-block
```

## Acceptance Criteria

- [ ] Preprocessing script converts at least one regulatory document (GDPR) to vault-compatible markdown
- [ ] Output markdown follows the heading/citation/frontmatter convention above
- [ ] Wikilinks mark internal cross-references
- [ ] `ObsidianStructuralChunker` produces reasonable chunks from regulatory markdown without modification
- [ ] Chunks carry `citation`, `source`, `article`, `section` metadata after enrichment
- [ ] Adversarial eval dataset contains >= 25 queries across all 4 categories
- [ ] Eval harness produces meaningful results on regulatory dataset
- [ ] Verdict layer can gate on regulatory eval results

## Test Plan

```python
def test_converted_markdown_has_frontmatter():
    """Output markdown includes YAML frontmatter with source metadata."""

def test_citation_metadata_extracted_from_headings():
    """Chunks from regulatory markdown carry correct citation in metadata."""

def test_cross_references_converted_to_wikilinks():
    """Internal references use [[GDPR Art. X]] wikilink format."""

def test_structural_chunker_handles_regulatory_markdown():
    """Existing ObsidianStructuralChunker produces chunks from regulatory docs."""

def test_regulatory_queries_load_as_eval_queries():
    """regulatory_queries.jsonl deserializes to valid EvalQuery objects."""
```

## Risks

| Risk | Mitigation |
|---|---|
| Regulatory text formatting varies across sources | Start with GDPR (well-structured, freely available); add others incrementally |
| Citation extraction is fragile on edge cases | Use heading hierarchy (mechanical), not NLP; add tests for each edge case |
| Eval dataset too small for statistical significance | Start with 25+, expand over time; note sample size in verdicts |
| Regulatory corpus dwarfs vault corpus | Index separately; eval datasets are corpus-specific |

## Follow-ups

- Add CFR Title 21 (FDA regulations) as second regulatory corpus
- Citation linking in generated answers (resolve wikilinks to chunk IDs)
- Cross-corpus queries (e.g., "How does GDPR compare to HIPAA on data breach notification?")
