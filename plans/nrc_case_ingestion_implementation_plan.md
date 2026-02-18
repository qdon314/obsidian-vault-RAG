# NRC Case Ingestion & Query Generation - Implementation Plan

## Overview

High-level implementation plan for the NRC case document ingestion and query generation system. This system fetches case documents from the ADAMS API, normalizes them to Obsidian markdown, and generates high-quality evaluation queries targeting the 10 CFR regulatory corpus.

---

## Phase 1: Foundation & API Client (Week 1)

### Goals
- Establish ADAMS API connectivity
- Implement retry/error handling
- Create domain models for case documents

### Tasks

#### 1.1 ADAMS API Client Port & Adapter
**Files**:
- `src/rag/ports/nrc_adams_client.py`
- `src/rag/adapters/ingestion/case/http_client.py`

**Work**:
- [ ] Define `NrcAdamsClient` protocol with `search_documents()` and `get_document()` methods
- [ ] Implement `HttpNrcAdamsClient` with requests library
- [ ] Add exponential backoff retry logic (configurable via `RetryConfig`)
- [ ] Implement rate limit handling (429 responses)
- [ ] Add pagination support (API limit: 100 results per page)
- [ ] Create exception hierarchy (`AdamsApiError`, `AdamsRateLimitError`, `AdamsAuthError`)

**Validation**:
- Unit tests with mocked API responses
- Manual test: fetch single document by accession number
- Manual test: search with pagination (200+ results)

#### 1.2 Domain Models
**Files**:
- `src/rag/domain/case_documents.py`

**Work**:
- [ ] Define `CaseDocument` frozen dataclass
- [ ] Define `CaseMetadata` frozen dataclass  
- [ ] Define `CaseCategory` enum (INSPECTION, ENFORCEMENT, PART_21, etc.)
- [ ] Add `FetchReport` for ingestion summary

**Validation**:
- Type checking passes (mypy)
- Domain objects are immutable (frozen=True)

#### 1.3 Configuration
**Files**:
- `settings.toml`
- `.env.example`

**Work**:
- [ ] Add `[nrc_adams]` section (subscription_key, base_url, timeout, retries)
- [ ] Add `[case_ingestion]` section (output_dir, document_types)
- [ ] Document env var `NRC_ADAMS_API_KEY` in README
- [ ] Add API key to `.env.example`

**Validation**:
- Config loads without errors
- Sensible defaults for PoC use

---

## Phase 2: Normalization & Citation Extraction (Week 1-2)

### Goals
- Convert raw ADAMS documents to Obsidian markdown
- Extract CFR citations with high accuracy
- Generate frontmatter with case semantics

### Tasks

#### 2.1 Case Document Normalizer
**Files**:
- `src/rag/adapters/ingestion/case/normalizer.py`

**Work**:
- [ ] Implement `CaseDocumentToMarkdown` adapter
- [ ] Build `_build_frontmatter()` - emit YAML with case metadata
- [ ] Implement `_extract_case_metadata()` - populate `CaseMetadata` fields
- [ ] Add `_categorize_document()` - map document type to `CaseCategory`
- [ ] Implement `_extract_facility_info()` - parse facility name and reactor type from title/content

**Validation**:
- Test on 5 representative case documents (manual review of markdown output)
- Verify frontmatter parseable by existing `split_obsidian_frontmatter()`
- Check wikilinks render correctly in Obsidian

#### 2.2 Citation Extraction
**Files**:
- `src/rag/adapters/ingestion/case/citation_extractor.py`

**Work**:
- [ ] Implement `CitationExtractor` with regex patterns
- [ ] Support variations: "10 CFR 50.46", "10CFR50.46", "§50.46", "Part 50"
- [ ] Handle subsection references: "(b)(1)", "(a)(3)(ii)"
- [ ] Normalize to canonical format: `10 CFR §50.46`
- [ ] Rewrite citations as wikilinks: `[[10 CFR 50.46]]`

**Regex Patterns**:
```python
CFR_CITATION = r"(\d+)\s*CFR\s*§?\s*(\d+\.\d+[A-Za-z0-9-]*)"
SECTION_REF = r"§\s*(\d+\.\d+[A-Za-z0-9-]*)"
PART_REF = r"Part\s+(\d+)"
```

**Validation**:
- Unit test suite with 20+ citation format variations
- Measure precision/recall on sample case documents
- Target: >90% precision, >80% recall

#### 2.3 Cross-Reference Adapter Reuse
**Files**:
- `src/rag/adapters/ingestion/regulatory/cross_references.py` (existing)

**Work**:
- [ ] Reuse `rewrite_cross_references_to_wikilinks()` for case content
- [ ] Ensure case citations link correctly to regulatory corpus files
- [ ] Test bidirectional linking (cases → regs, regs mention cases)

**Validation**:
- Obsidian graph view shows case→regulation edges
- Click wikilink in case markdown → opens regulatory section

---

## Phase 3: Case Document Fetcher (Week 2)

### Goals
- Sequential fetch from ADAMS API
- Write normalized markdown to disk
- Handle errors gracefully

### Tasks

#### 3.1 Case Document Fetcher
**Files**:
- `src/rag/adapters/ingestion/case/fetcher.py`

**Work**:
- [ ] Implement `CaseDocumentFetcher` adapter
- [ ] Add `fetch_and_write()` method (sequential, not distributed)
- [ ] Organize output by year-month: `corpus/us-nrc/cases/2024-01/ML24001A123.md`
- [ ] Log progress (every 10 docs)
- [ ] Generate `FetchReport` summary (total, by document type, errors)

**Error Handling**:
- Skip individual doc failures (log + continue)
- Save partial results if API interrupted
- Return summary with error list

**Validation**:
- Fetch 50 inspection reports (dry run)
- Verify markdown files well-formed
- Check error handling (manually trigger 404, 429)

#### 3.2 CLI Script
**Files**:
- `scripts/fetch_nrc_cases.py`

**Work**:
- [ ] CLI with argparse (date_from, date_to, document_types, output_dir, limit)
- [ ] Load `NrcAdamsClient` from settings
- [ ] Instantiate `CaseDocumentFetcher` with normalizer
- [ ] Print progress and summary report
- [ ] Handle Ctrl+C gracefully (report partial results)

**Usage Example**:
```bash
./scripts/py scripts/fetch_nrc_cases.py \
  --document-types "Inspection Report" "Part 21 Correspondence" \
  --date-from 2024-01-01 \
  --date-to 2024-12-31 \
  --output-dir corpus/us-nrc/cases/ \
  --limit 100
```

**Validation**:
- Fetch 10 docs via CLI (inspect output directory structure)
- Test --limit flag
- Test error cases (invalid dates, bad API key)

---

## Phase 4: Term Mapping Dictionary (Week 2-3)

### Goals
- Build case term → regulation mapping
- Enable semantic query generation

### Tasks

#### 4.1 Seed Dictionary
**Files**:
- `config/case_regulatory_terms.json`

**Work**:
- [ ] Create JSON structure: `{"term": ["10 CFR 50.46", ...]}`
- [ ] Seed with 50-100 common mappings:
  - ECCS-related (accumulator, peak cladding temperature)
  - Tech specs (LCO, surveillance requirement)
  - Change control (50.59, license amendment)
  - Maintenance (maintenance rule, preventive maintenance)
  - Reporting (Part 21, LER, event notification)

**Validation**:
- Manual review of mappings for correctness
- Check JSON validates against schema

#### 4.2 Dictionary Loader
**Files**:
- `src/rag/adapters/query_generation/term_mapper.py`

**Work**:
- [ ] Implement `TermMapper` adapter
- [ ] Load dictionary from JSON
- [ ] Add `map_term(term: str) -> list[str]` method
- [ ] Support fuzzy matching (e.g., "ECCS pump" → "ECCS")

**Validation**:
- Unit tests for exact and fuzzy matching
- Handle missing terms gracefully (return empty list)

---

## Phase 5: Query Generator - Core Strategies (Week 3-4)

### Goals
- Implement first 3 query generation strategies
- Validate output quality

### Tasks

#### 5.1 Query Generator Base
**Files**:
- `src/rag/adapters/query_generation/case_query_generator.py`

**Work**:
- [ ] Define `CaseQueryGenerator` adapter
- [ ] Implement `generate_from_case(case_path: Path) -> list[dict]` entrypoint
- [ ] Add `_parse_case()` - extract frontmatter + content
- [ ] Add `_generate_qid()` - unique ID generator (case-dc-001, case-tm-002, etc.)
- [ ] Implement query schema validation (`_validate_query()`)

**Validation**:
- Pass type checking
- Generate valid query dict structure

#### 5.2 Strategy 1: Direct Citation Queries
**Files**:
- `src/rag/adapters/query_generation/strategies/direct_citation.py`

**Work**:
- [ ] Implement `generate_direct_citation_queries(fm: dict) -> list[dict]`
- [ ] For each cited regulation, create factual query
- [ ] Template: "What are the requirements of {citation}?"
- [ ] Populate `relevant_citations`, `difficulty: easy`, `query_type: factual`

**Validation**:
- Test on 5 cases with varying citation counts
- Verify query structure matches existing `regulatory_adversarial.jsonl` format

#### 5.3 Strategy 2: Term Mapping Queries
**Files**:
- `src/rag/adapters/query_generation/strategies/term_mapping.py`

**Work**:
- [ ] Implement `generate_term_mapping_queries(fm: dict, content: str, mapper: TermMapper) -> list[dict]`
- [ ] Extract technical terms from content (simple keyword extraction initially)
- [ ] Map terms to regulations via `TermMapper`
- [ ] Generate interpretive queries
- [ ] Limit to top 5 terms per case

**Validation**:
- Test term extraction on sample case
- Verify mapped regulations are sensible
- Review generated query naturalness

#### 5.4 Strategy 3: Violation Context Queries
**Files**:
- `src/rag/adapters/query_generation/strategies/violation_context.py`

**Work**:
- [ ] Implement `generate_violation_queries(fm: dict, content: str) -> list[dict]`
- [ ] Check for `violation_ids` in frontmatter
- [ ] Generate multi-hop synthesis questions
- [ ] Vary query focus: requirements violated, reporting, corrective action

**Validation**:
- Test on cases with/without violations
- Manually review 10 generated queries for quality
- Check `query_type: multi_hop`, `difficulty: hard`

---

## Phase 6: Query Generator - Adversarial Strategies (Week 4)

### Goals
- Add adversarial and abstention queries
- Increase difficulty diversity

### Tasks

#### 6.1 Strategy 4: Adversarial Queries
**Files**:
- `src/rag/adapters/query_generation/strategies/adversarial.py`

**Work**:
- [ ] Implement `generate_adversarial_queries(fm: dict, content: str) -> list[dict]`
- [ ] Create terminology paraphrases (e.g., "loss of coolant accident" vs "ECCS")
- [ ] Generate colloquial variants ("degraded safety equipment" vs "inoperability")
- [ ] Add `adversarial_note` field explaining trap

**Validation**:
- Test that adversarial queries retrieve correct regulatory sections (not wrong ones)
- Manually validate 10 adversarial queries

#### 6.2 Strategy 5: Facility-Specific Queries
**Files**:
- `src/rag/adapters/query_generation/strategies/facility_specific.py`

**Work**:
- [ ] Implement `generate_facility_queries(fm: dict) -> list[dict]`
- [ ] Include facility name as distractor context
- [ ] Ensure expected answer is facility-agnostic
- [ ] Add negative queries ("Does regulation X apply differently to facility Y?")

**Validation**:
- Verify facility names don't appear in regulatory corpus
- Check expected answers are generic

#### 6.3 Strategy 6: Abstention Queries
**Files**:
- `src/rag/adapters/query_generation/strategies/abstention.py`

**Work**:
- [ ] Implement `generate_abstention_queries(fm: dict, content: str) -> list[dict]`
- [ ] Generate queries about out-of-corpus regulations (Part 20, Part 55, etc.)
- [ ] Generate queries about case facts (not regulatory lookup)
- [ ] Set `is_unanswerable: true`, `unanswerable_reason`

**Validation**:
- Verify referenced regulations NOT in corpus
- Check unanswerable queries fail gracefully in eval harness

---

## Phase 7: CLI & Integration (Week 5)

### Goals
- End-to-end pipeline working
- Generate queries from fetched cases

### Tasks

#### 7.1 Query Generator CLI
**Files**:
- `scripts/generate_case_queries.py`

**Work**:
- [ ] CLI with argparse (case_dir, output, term_map_file, strategies, max_queries_per_case)
- [ ] Walk case directory recursively
- [ ] For each `.md` file, run `CaseQueryGenerator.generate_from_case()`
- [ ] Collect all queries, deduplicate by query text
- [ ] Write to JSONL (one query per line)
- [ ] Print summary (total queries, by strategy, by difficulty)

**Usage Example**:
```bash
./scripts/py scripts/generate_case_queries.py \
  --case-dir corpus/us-nrc/cases/ \
  --output eval/datasets/case_generated_queries_DRAFT.jsonl \
  --term-map-file config/case_regulatory_terms.json \
  --strategies direct_citation,term_mapping,violation_context,adversarial \
  --max-queries-per-case 20
```

**Validation**:
- Run on 10 cases, inspect output JSONL
- Check query diversity (strategies, difficulty)
- Verify schema compliance

#### 7.2 Quality Validation Script
**Files**:
- `scripts/validate_case_queries.py`

**Work**:
- [ ] Load generated queries from JSONL
- [ ] Run validation checks:
  - Missing required fields
  - Invalid difficulty/query_type values
  - Answerable queries without relevant_citations
  - Duplicate query text
- [ ] Generate validation report (pass/fail counts, error details)
- [ ] Output clean queries to separate file

**Validation**:
- Run on draft query set
- Fix validation errors (update generator logic)

---

## Phase 8: Manual Review & Curation (Week 5-6)

### Goals
- Human-in-loop quality assurance
- Filter to production-ready queries

### Tasks

#### 8.1 Review UI (Optional)
**Files**:
- `eval/app/query_curator.py` (new Streamlit page)

**Work** (Optional - can use manual JSONL editing instead):
- [ ] Streamlit page for query review
- [ ] Display query with metadata (source case, strategy, difficulty)
- [ ] Show linked regulatory sections
- [ ] Approve/edit/reject buttons
- [ ] Export approved queries

**Validation**:
- Load 50 queries, test review workflow

#### 8.2 Manual Curation Process
**Files**:
- `eval/datasets/case_generated_queries_DRAFT.jsonl` → `case_generated_queries.jsonl`

**Work**:
- [ ] Curator reviews each query for:
  - Clarity and naturalness
  - Answer verifiability in corpus
  - Appropriate difficulty rating
  - Adversarial value
- [ ] Edit query text/metadata as needed
- [ ] Reject low-quality queries
- [ ] Document curation decisions (notes field)

**Target**:
- Start with ~500-1000 draft queries
- Curate to ~250-500 production queries (30-50% pass rate)

#### 8.3 Merge with Existing Datasets
**Files**:
- `eval/datasets/all_queries.jsonl` (new combined dataset)

**Work**:
- [ ] Combine `regulatory_adversarial.jsonl` + `case_generated_queries.jsonl`
- [ ] Deduplicate by query text (case-generated takes precedence if duplicate)
- [ ] Add dataset source tag to distinguish manual vs case-generated
- [ ] Update eval harness to support combined dataset

---

## Phase 9: Evaluation & Iteration (Week 6-7)

### Goals
- Validate query quality via evaluation
- Iterate on generation strategies

### Tasks

#### 9.1 Initial Evaluation Run
**Files**:
- N/A (use existing eval harness)

**Work**:
- [ ] Run eval on case-generated queries
- [ ] Measure metrics: recall@k, precision, MRR, abstention rate
- [ ] Analyze failures by query type and difficulty

**Validation**:
- Compare case-generated vs manually-curated query performance
- Identify systematic issues (e.g., low recall on term-mapping queries)

#### 9.2 Strategy Refinement
**Files**:
- Strategy implementations (various)

**Work**:
- [ ] Analyze underperforming query types
- [ ] Refine generation heuristics (e.g., improve term extraction)
- [ ] Expand term-mapping dictionary based on failures
- [ ] Add missing adversarial patterns

**Iteration**:
- Regenerate queries with updated strategies
- Re-run evaluation
- Repeat until quality meets threshold

#### 9.3 Documentation
**Files**:
- `docs/evaluation/case_query_generation.md` (new)
- `README.md` (update)

**Work**:
- [ ] Document query generation approach
- [ ] Provide examples of each strategy
- [ ] Explain quality metrics and curation process
- [ ] Add usage instructions for fetch + generate scripts

---

## Phase 10: Production Readiness (Week 7-8)

### Goals
- Robust error handling
- Monitoring and logging
- Scheduled updates

### Tasks

#### 10.1 Error Handling & Logging
**Files**:
- All adapter implementations

**Work**:
- [ ] Add structured logging (logger.info, logger.error)
- [ ] Log API request/response summaries
- [ ] Track generation statistics (queries per case, by strategy)
- [ ] Emit warnings for low-quality cases (no citations, short content)

**Validation**:
- Review logs for 100-document fetch + generate run
- Ensure failures are logged with actionable context

#### 10.2 Makefile Targets
**Files**:
- `Makefile`

**Work**:
- [ ] Add `make fetch-cases` target (wrapper for fetch_nrc_cases.py)
- [ ] Add `make generate-case-queries` target
- [ ] Add `make validate-case-queries` target
- [ ] Add `make curate-case-queries` (opens review UI)

**Usage Example**:
```bash
make fetch-cases LIMIT=100
make generate-case-queries
make validate-case-queries
```

#### 10.3 Testing
**Files**:
- `tests/adapters/ingestion/case/` (new)

**Work**:
- [ ] Unit tests for `HttpNrcAdamsClient` (mocked responses)
- [ ] Unit tests for `CaseDocumentToMarkdown` (sample docs)
- [ ] Unit tests for `CitationExtractor` (regex patterns)
- [ ] Unit tests for each query generation strategy
- [ ] Integration test: fetch 5 docs → generate queries → validate

**Validation**:
- All tests pass via `make test`
- Coverage >80% for new code

#### 10.4 Scheduled Updates (Optional)
**Files**:
- `.github/workflows/fetch_cases.yml` (new)

**Work** (Optional):
- [ ] GitHub Actions workflow to fetch new cases monthly
- [ ] Commit new markdown files to repo
- [ ] Regenerate queries from updated cases
- [ ] Create PR with new queries for review

---

## Success Criteria

### Phase 1-3 (Foundation): ✅ Fetch Working
- [ ] Successfully fetch 100 case documents from ADAMS API
- [ ] Convert to well-formed markdown with frontmatter
- [ ] CFR citations extracted with >85% recall

### Phase 4-6 (Query Generation): ✅ Queries Generated
- [ ] Generate 500+ candidate queries from 50 cases
- [ ] All 6 strategies implemented and tested
- [ ] Validation script passes without errors

### Phase 7-9 (Quality): ✅ Production Queries
- [ ] Manual curation yields 250+ production-quality queries
- [ ] Queries match existing JSONL schema
- [ ] Evaluation pass rate comparable to manual queries

### Phase 10 (Production): ✅ Repeatable Process
- [ ] CLI scripts documented and working
- [ ] Makefile targets functional
- [ ] Test coverage >80%
- [ ] Process documented for future updates

---

## Risk Mitigation

### Risk: ADAMS API Rate Limits Unknown
**Mitigation**: Start with small batches (10-50 docs), monitor response times, implement aggressive backoff

### Risk: Citation Extraction Accuracy Too Low
**Mitigation**: Collect failed cases, manually review patterns, add regex variants, consider LLM-based fallback

### Risk: Generated Queries Low Quality
**Mitigation**: Start with manual review of first 50 queries, iterate on heuristics, maintain high curation pass-through threshold (>30%)

### Risk: Term Mapping Dictionary Too Sparse
**Mitigation**: Seed with top 100 terms from manual inspection, expand iteratively based on case corpus analysis

### Risk: Manual Curation Bottleneck
**Mitigation**: Prioritize quality over quantity (250 great queries > 1000 mediocre queries), automate validation to reduce reviewer burden

---

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 1. Foundation & API Client | Week 1 | ADAMS client + domain models working |
| 2. Normalization & Citation | Week 1-2 | Markdown normalizer + citation extractor |
| 3. Case Document Fetcher | Week 2 | CLI script fetches to disk successfully |
| 4. Term Mapping Dictionary | Week 2-3 | JSON dictionary with 100+ mappings |
| 5. Query Gen: Core Strategies | Week 3-4 | Strategies 1-3 implemented |
| 6. Query Gen: Adversarial | Week 4 | Strategies 4-6 implemented |
| 7. CLI & Integration | Week 5 | End-to-end pipeline runs |
| 8. Manual Review & Curation | Week 5-6 | 250+ production queries |
| 9. Evaluation & Iteration | Week 6-7 | Validated query quality |
| 10. Production Readiness | Week 7-8 | Polished, documented, tested |

**Total**: ~8 weeks for complete implementation

**Minimum Viable**: Phases 1-7 (~5 weeks) for working prototype with draft queries

---

## Next Immediate Steps

1. **Sign up for ADAMS API key** at https://adams-api-developer.nrc.gov/
2. **Create Phase 1 skeleton files** (ports, adapters, domain models)
3. **Fetch 5 sample documents** manually via API to understand data format
4. **Build citation extraction regex** and test on sample documents
5. **Seed term mapping dictionary** with 20-30 obvious mappings to start
