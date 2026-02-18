# NRC Case Document Query Generation Strategy

## Goal

Generate **high-quality, diverse, and adversarial** evaluation queries from NRC case documents (inspection reports, Part 21 notices, enforcement letters, etc.) that test retrieval performance against the 10 CFR regulatory corpus.

---

## Query Generation Framework

### Input: Normalized Case Markdown

```markdown
---
regime: us-nrc
document_type: Inspection Report
accession_number: ML24001A123
document_date: 2024-01-15
docket_number: 05000361
facility_name: San Onofre Nuclear Generating Station
reactor_type: PWR
inspection_report_number: 05000361/2024001
cited_regulations:
  - 10 CFR 50.46
  - 10 CFR 50.36
  - 10 CFR 50.65
violation_ids:
  - VIO 05000361/2024001-01
severity_level: Severity Level IV
technical_terms:
  - ECCS
  - accumulator
  - technical specification
  - surveillance requirement
corpus: nrc-cases
---

# Finding: ECCS Accumulator Surveillance Inadequacy

The licensee failed to perform surveillance testing of ECCS accumulator level...
[[10 CFR 50.46]] requires that ECCS cooling performance...
[[10 CFR 50.36]](c)(3) requires surveillance requirements...
```

---

## Query Generation Strategies

### **Strategy 1: Direct Citation Queries (Easy/Factual)**

**Purpose**: Test precise citation retrieval

**Logic**:
- For each `cited_regulation` in frontmatter
- Generate question asking for requirements of that section

**Examples**:
```jsonl
{"qid": "case-dc-001", "query": "What are the ECCS acceptance criteria in 10 CFR 50.46?", "relevant_citations": ["10 CFR §50.46(b)"], "relevant_doc_citations": ["10 CFR §50.46"], "expected_answer": "Five acceptance criteria: peak cladding temperature ≤2200F, maximum oxidation ≤0.17 times thickness, maximum hydrogen ≤0.01, coolable geometry maintained, long-term cooling achieved.", "query_type": "factual", "difficulty": "easy", "requires_synthesis": false, "tags": ["case-derived", "citation-direct"], "source_case": "ML24001A123", "case_document_type": "Inspection Report", "facility": "San Onofre", "metadata": {"filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}}}

{"qid": "case-dc-002", "query": "What does 10 CFR 50.36 require for surveillance requirements in technical specifications?", "relevant_citations": ["10 CFR §50.36(c)(3)"], "relevant_doc_citations": ["10 CFR §50.36"], "expected_answer": "Surveillance requirements are requirements relating to test, calibration, or inspection to ensure that the necessary quality of systems and components is maintained, and that facility operation will be within safety limits and LCOs.", "query_type": "factual", "difficulty": "easy", "requires_synthesis": false, "tags": ["case-derived", "citation-direct"], "source_case": "ML24001A123", "metadata": {"filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}}}
```

**Diversity**: One query per unique cited regulation

---

### **Strategy 2: Technical Term → Regulation Mapping (Medium/Interpretive)**

**Purpose**: Test semantic retrieval using domain terminology

**Logic**:
- Extract technical terms from case content (ECCS, accumulator, LCO, etc.)
- Map terms to relevant regulations via dictionary
- Generate queries using case-specific terminology

**Term Dictionary** (seed):
```python
CASE_TERM_TO_REGULATION = {
    # ECCS-related
    "ECCS": ["10 CFR 50.46", "10 CFR 50.34"],
    "emergency core cooling": ["10 CFR 50.46"],
    "accumulator": ["10 CFR 50.46", "10 CFR 50.36"],
    "peak cladding temperature": ["10 CFR 50.46"],
    
    # Technical specifications
    "technical specification": ["10 CFR 50.36"],
    "LCO": ["10 CFR 50.36"],
    "limiting condition for operation": ["10 CFR 50.36"],
    "surveillance requirement": ["10 CFR 50.36"],
    "surveillance testing": ["10 CFR 50.36"],
    
    # Change control
    "50.59": ["10 CFR 50.59"],
    "license amendment": ["10 CFR 50.59", "10 CFR 50.90"],
    
    # Maintenance
    "maintenance rule": ["10 CFR 50.65"],
    "preventive maintenance": ["10 CFR 50.65"],
    
    # Reporting
    "Part 21": ["10 CFR 21"],
    "defect reporting": ["10 CFR 21"],
    "event notification": ["10 CFR 50.72"],
    "licensee event report": ["10 CFR 50.73"],
}
```

**Examples**:
```jsonl
{"qid": "case-tm-001", "query": "What are the regulatory requirements for ECCS accumulator operability?", "relevant_citations": ["10 CFR §50.46", "10 CFR §50.36"], "relevant_doc_citations": ["10 CFR §50.46", "10 CFR §50.36"], "expected_answer": "ECCS must meet the acceptance criteria in 50.46(b), and accumulator operability parameters (level, pressure, boron concentration) must be established as LCOs in technical specifications per 50.36(c)(2).", "query_type": "interpretive", "difficulty": "medium", "requires_synthesis": true, "tags": ["case-derived", "term-mapping", "ECCS"], "source_case": "ML24001A123", "technical_term": "accumulator", "metadata": {"filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}}}

{"qid": "case-tm-002", "query": "What regulatory provisions govern surveillance testing frequency for safety-related equipment?", "relevant_citations": ["10 CFR §50.36(c)(3)", "10 CFR §50.65"], "relevant_doc_citations": ["10 CFR §50.36", "10 CFR §50.65"], "expected_answer": "Section 50.36(c)(3) requires surveillance requirements in technical specifications to ensure quality is maintained. Section 50.65 requires monitoring equipment effectiveness and adjusting preventive maintenance as needed.", "query_type": "interpretive", "difficulty": "medium", "requires_synthesis": true, "tags": ["case-derived", "term-mapping", "surveillance"], "source_case": "ML24001A123", "technical_term": "surveillance requirement", "metadata": {"filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}}}
```

**Diversity**: Generate 2-5 queries per case based on prominent technical terms

---

### **Strategy 3: Violation Context Queries (Hard/Multi-hop)**

**Purpose**: Test contextual understanding and synthesis

**Logic**:
- For cases with violations, extract violation description
- Generate query requiring synthesis between violation context and regulatory requirements
- Use facility/reactor-specific details to increase difficulty

**Examples**:
```jsonl
{"qid": "case-vc-001", "query": "If a PWR licensee fails to perform ECCS accumulator level surveillance at the required frequency, which regulatory requirements are violated?", "relevant_citations": ["10 CFR §50.36(c)(3)", "10 CFR §50.46"], "relevant_doc_citations": ["10 CFR §50.36", "10 CFR §50.46", "10 CFR §50.65"], "expected_answer": "Failure to perform surveillance per technical specifications violates 50.36(c)(3), which requires tests to ensure quality and operation within LCOs. For ECCS equipment specifically, this impacts compliance with 50.46 acceptance criteria and 50.65 maintenance rule monitoring.", "query_type": "multi_hop", "difficulty": "hard", "requires_synthesis": true, "tags": ["case-derived", "violation-context", "ECCS"], "source_case": "ML24001A123", "violation_id": "VIO 05000361/2024001-01", "reactor_type": "PWR", "metadata": {"filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}}}

{"qid": "case-vc-002", "query": "What reporting requirements apply when a licensee discovers that required ECCS surveillance was not performed?", "relevant_citations": ["10 CFR §50.73(a)(2)(i)(B)"], "relevant_doc_citations": ["10 CFR §50.73", "10 CFR §50.72"], "expected_answer": "Under 50.73(a)(2)(i)(B), any operation or condition prohibited by technical specifications requires an LER within 60 days. If this rendered ECCS equipment inoperable beyond tech spec allowed outage time, immediate notification under 50.72 may also be required.", "query_type": "procedural", "difficulty": "hard", "requires_synthesis": true, "tags": ["case-derived", "violation-context", "reporting"], "source_case": "ML24001A123", "metadata": {"filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}}}
```

**Diversity**: 1-3 queries per violation, varying focus (requirements violated, reporting, corrective action basis)

---

### **Strategy 4: Cross-Reference Adversarial Queries**

**Purpose**: Test retrieval when query uses case-specific language that differs from regulatory text

**Logic**:
- Extract informal/colloquial terms from case narrative
- Generate queries using these terms instead of formal regulatory language
- Create "trap" queries that could retrieve wrong sections

**Examples**:
```jsonl
{"qid": "case-ca-001", "query": "What are the temperature limits for fuel cladding during a loss of coolant accident?", "relevant_citations": ["10 CFR §50.46(b)(1)"], "relevant_doc_citations": ["10 CFR §50.46"], "expected_answer": "The calculated maximum fuel element cladding temperature shall not exceed 2200 degrees Fahrenheit.", "query_type": "factual", "difficulty": "medium", "requires_synthesis": false, "tags": ["case-derived", "adversarial", "paraphrase"], "source_case": "ML24001A123", "adversarial_note": "Uses 'loss of coolant accident' instead of 'ECCS' to test semantic matching", "metadata": {"filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}}}

{"qid": "case-ca-002", "query": "What rules govern when a plant can continue operating with degraded safety equipment?", "relevant_citations": ["10 CFR §50.36(c)(2)", "10 CFR §50.36(c)(3)"], "relevant_doc_citations": ["10 CFR §50.36"], "expected_answer": "Technical specifications establish LCOs per 50.36(c)(2) which define allowed equipment configurations, and 50.36(c)(3) action statements specify allowed outage times and required actions when equipment is degraded or inoperable.", "query_type": "interpretive", "difficulty": "hard", "requires_synthesis": true, "tags": ["case-derived", "adversarial", "colloquial"], "source_case": "ML24001A123", "adversarial_note": "Uses informal 'degraded safety equipment' vs formal 'inoperability' and 'LCO' terms", "metadata": {"filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}}}
```

**Diversity**: 2-4 adversarial variants per case, testing different terminology mismatches

---

### **Strategy 5: Facility-Specific Contextualized Queries**

**Purpose**: Test whether retrieval can ignore irrelevant context and focus on regulatory requirements

**Logic**:
- Include facility name, reactor type, or specific equipment in query
- Regulatory answer should be generic (not facility-specific)
- Tests resistance to context distraction

**Examples**:
```jsonl
{"qid": "case-fs-001", "query": "What ECCS requirements apply to San Onofre's pressurized water reactor design?", "relevant_citations": ["10 CFR §50.46"], "relevant_doc_citations": ["10 CFR §50.46"], "expected_answer": "All PWRs must meet the five ECCS acceptance criteria in 50.46(b), regardless of specific facility. These include: peak cladding temperature ≤2200F, oxidation ≤0.17 times thickness, hydrogen ≤0.01, coolable geometry maintained, and long-term cooling capability.", "query_type": "factual", "difficulty": "medium", "requires_synthesis": false, "tags": ["case-derived", "facility-specific", "distractor"], "source_case": "ML24001A123", "facility": "San Onofre", "reactor_type": "PWR", "adversarial_note": "Facility name is irrelevant distractor; answer is generic", "metadata": {"filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}}}

{"qid": "case-fs-002", "query": "Does the maintenance rule in 10 CFR 50.65 apply differently to San Onofre versus other nuclear power plants?", "relevant_citations": ["10 CFR §50.65(a)"], "relevant_doc_citations": ["10 CFR §50.65"], "expected_answer": "No. Section 50.65 applies to all holders of operating licenses for nuclear power plants under 10 CFR Part 50. The requirements are facility-agnostic.", "query_type": "factual", "difficulty": "easy", "requires_synthesis": false, "tags": ["case-derived", "facility-specific", "negative-query"], "source_case": "ML24001A123", "facility": "San Onofre", "adversarial_note": "Tests whether system correctly answers 'no' rather than hallucinating facility-specific requirements", "metadata": {"filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}}}
```

**Diversity**: 1-2 per case, varying facility context inclusion

---

### **Strategy 6: Abstention/Out-of-Scope Queries**

**Purpose**: Test whether system appropriately declines to answer

**Logic**:
- Generate queries referencing regulations cited in case but NOT in corpus
- Generate queries about case facts themselves (not regulatory lookup)
- Test abstention discipline

**Examples**:
```jsonl
{"qid": "case-ab-001", "query": "What are the requirements for operator training under 10 CFR Part 55 that were violated in the San Onofre inspection?", "relevant_citations": [], "relevant_doc_citations": [], "relevant_chunk_ids": [], "expected_answer": null, "query_type": "factual", "difficulty": "medium", "requires_synthesis": false, "tags": ["case-derived", "abstention", "out-of-corpus"], "is_unanswerable": true, "unanswerable_reason": "not_in_corpus", "source_case": "ML24001A123", "notes": "Part 55 is referenced in case but not in corpus. System should abstain.", "metadata": {"filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}}}

{"qid": "case-ab-002", "query": "What corrective actions did San Onofre commit to for the ECCS surveillance finding?", "relevant_citations": [], "relevant_doc_citations": [], "relevant_chunk_ids": [], "expected_answer": null, "query_type": "factual", "difficulty": "easy", "requires_synthesis": false, "tags": ["case-derived", "abstention", "case-fact"], "is_unanswerable": true, "unanswerable_reason": "case_fact_not_regulatory", "source_case": "ML24001A123", "notes": "Question is about case document contents, not regulatory requirements. System should decline.", "metadata": {"filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}}}

{"qid": "case-ab-003", "query": "How does the NRC's enforcement policy determine the severity level for technical specification violations?", "relevant_citations": [], "relevant_doc_citations": [], "relevant_chunk_ids": [], "expected_answer": null, "query_type": "procedural", "difficulty": "medium", "requires_synthesis": false, "tags": ["case-derived", "abstention", "policy-not-regulation"], "is_unanswerable": true, "unanswerable_reason": "not_in_corpus", "source_case": "ML24001A123", "notes": "Enforcement policy is not in 10 CFR; it's a separate NRC manual. System should abstain.", "metadata": {"filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}}}
```

**Diversity**: 2-3 per case, varying types of out-of-scope content

---

## Query Quality Criteria

### **High-Quality Queries Must**:

1. **Have verifiable answers** in the regulatory corpus
2. **Use natural language** (not legal citation format in query text)
3. **Be unambiguous** (clear what information is sought)
4. **Specify difficulty honestly** (easy/medium/hard based on retrieval complexity)
5. **Include precise `relevant_citations`** at subsection level where possible
6. **Tag query type** (factual, interpretive, procedural, multi_hop, comparison)
7. **Be adversarial** where appropriate (test failure modes)

### **Diversity Dimensions**:

- **Query Type**: factual, interpretive, procedural, multi_hop, comparison
- **Difficulty**: easy, medium, hard
- **Synthesis Required**: true/false
- **Citation Scope**: single section, multiple sections, cross-part
- **Terminology**: formal regulatory, informal case language, technical jargon
- **Context**: minimal, facility-specific, violation-specific
- **Answerability**: answerable, unanswerable (abstention tests)

---

## Implementation Architecture

### **Script: `generate_case_queries.py`**

```python
@dataclass(frozen=True, slots=True)
class CaseQueryGenerator:
    """Generate evaluation queries from case markdown documents."""
    
    term_to_regulation_map: dict[str, list[str]]
    query_id_prefix: str = "case"
    
    def generate_from_case(self, case_path: Path) -> list[dict]:
        """
        Generate diverse queries from a single case document.
        
        Returns list of query dicts ready for JSONL serialization.
        """
        frontmatter, content = self._parse_case(case_path)
        
        queries = []
        
        # Strategy 1: Direct citation
        queries.extend(self._generate_direct_citation_queries(frontmatter))
        
        # Strategy 2: Term mapping
        queries.extend(self._generate_term_mapping_queries(frontmatter, content))
        
        # Strategy 3: Violation context
        if frontmatter.get("violation_ids"):
            queries.extend(self._generate_violation_queries(frontmatter, content))
        
        #Strategy 4: Adversarial
        queries.extend(self._generate_adversarial_queries(frontmatter, content))
        
        # Strategy 5: Facility-specific
        queries.extend(self._generate_facility_queries(frontmatter))
        
        # Strategy 6: Abstention
        queries.extend(self._generate_abstention_queries(frontmatter, content))
        
        return queries
    
    def _generate_direct_citation_queries(self, fm: dict) -> list[dict]:
        """Strategy 1: One query per cited regulation."""
        queries = []
        for citation in fm.get("cited_regulations", []):
            query = {
                "qid": self._generate_qid("dc"),
                "query": f"What are the requirements of {citation}?",
                "relevant_citations": [citation],
                "relevant_doc_citations": [citation.split('§')[0].strip()],
                "query_type": "factual",
                "difficulty": "easy",
                "requires_synthesis": False,
                "tags": ["case-derived", "citation-direct"],
                "source_case": fm["accession_number"],
                "case_document_type": fm["document_type"],
                "metadata": {"filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}},
            }
            queries.append(query)
        return queries
    
    def _generate_term_mapping_queries(self, fm: dict, content: str) -> list[dict]:
        """Strategy 2: Extract terms, map to regulations."""
        terms = self._extract_technical_terms(content)
        queries = []
        for term in terms[:5]:  # Limit to top 5 terms
            if term in self.term_to_regulation_map:
                target_cites = self.term_to_regulation_map[term]
                query = {
                    "qid": self._generate_qid("tm"),
                    "query": f"What are the regulatory requirements for {term}?",
                    "relevant_citations": [f"{c}(a)" for c in target_cites],  # Approximate
                    "relevant_doc_citations": target_cites,
                    "query_type": "interpretive",
                    "difficulty": "medium",
                    "requires_synthesis": True,
                    "tags": ["case-derived", "term-mapping", term.replace(" ", "-")],
                    "source_case": fm["accession_number"],
                    "technical_term": term,
                    "metadata": {"filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}},
                }
                queries.append(query)
        return queries
    
    # ... other strategy methods
```

### **CLI Usage**:

```bash
# Generate queries from all cases
./scripts/py scripts/generate_case_queries.py \
  --case-dir corpus/us-nrc/cases/ \
  --output eval/datasets/case_generated_queries.jsonl \
  --term-map-file config/case_regulatory_terms.json \
  --strategies direct_citation,term_mapping,violation_context,adversarial \
  --max-queries-per-case 20

# Generate with sample for manual review
./scripts/py scripts/generate_case_queries.py \
  --case-dir corpus/us-nrc/cases/ \
  --output eval/datasets/case_generated_queries_DRAFT.jsonl \
  --sample-size 10 \
  --review-mode
```

---

## Quality Assurance Process

### **Step 1: Automated Generation**
- Run generator on all case documents
- Emit candidate queries to `case_generated_queries_DRAFT.jsonl`

### **Step 2: Automated Quality Checks**
```python
def validate_query(query: dict) -> list[str]:
    """Return list of validation errors."""
    errors = []
    
    # Must have query text
    if not query.get("query"):
        errors.append("Missing query text")
    
    # Answerable queries must have relevant_citations
    if not query.get("is_unanswerable") and not query.get("relevant_citations"):
        errors.append("Answerable query missing relevant_citations")
    
    # Difficulty must be valid
    if query.get("difficulty") not in ["easy", "medium", "hard"]:
        errors.append("Invalid difficulty")
    
    # Must have source_case
    if not query.get("source_case"):
        errors.append("Missing source_case")
    
    return errors
```

### **Step 3: Manual Review & Curation**
- Load draft queries into Streamlit review UI
- Curator reviews queries for:
  - Clarity and naturalness
  - Answer verifiability
  - Appropriate difficulty rating
  - Adversarial value
- Edit/approve/reject queries

### **Step 4: Export to Production Dataset**
- Approved queries exported to `eval/datasets/case_generated_queries.jsonl`
- Merged with existing `regulatory_adversarial.jsonl` for comprehensive evaluation

---

## Expected Coverage

From **100 case documents**, expect to generate:

- **~100-200 direct citation queries** (1-2 per case)
- **~200-500 term mapping queries** (2-5 per case)
- **~50-150 violation context queries** (0.5-1.5 per case with violations)
- **~100-300 adversarial queries** (1-3 per case)
- **~100-200 facility-specific queries** (1-2 per case)
- **~200-300 abstention queries** (2-3 per case)

**Total: ~750-1,650 candidate queries**

After manual review, expect **30-50% pass rate** → **~250-825 high-quality queries**

---

## Advantages of This Approach

✅ **Grounded in Real Scenarios** - Queries derived from actual regulatory events  
✅ **Natural Language** - Case narratives provide realistic phrasing  
✅ **Adversarial by Design** - Tests terminology mismatches, context distractors  
✅ **Diverse Difficulty** - Ranges from simple citation lookup to multi-hop synthesis  
✅ **Testable Abstention** - Includes out-of-scope queries to test hallucination resistance  
✅ **Scalable** - Automated generation with manual QA gate  
✅ **Reproducible** - Source case documented for every query  

---

## Next Steps

1. **Build term-mapping dictionary** - Seed with 50-100 common case→regulation mappings
2. **Fetch sample case documents** - 10-20 representative cases for prototyping
3. **Implement generator script** - Start with strategies 1-2, iterate
4. **Manual review first batch** - Assess quality, refine heuristics
5. **Scale to full corpus** - Run on all cases, curate output
6. **Integrate into eval harness** - Add to existing evaluation datasets

