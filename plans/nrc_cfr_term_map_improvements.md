Absolutely. Below is a **clean, future-proof schema** (simple enough to hand-edit, structured enough to power routing/boosting), followed by **concrete P0 specs** expressed in that schema.

---

## 1) Clean term-map schema

### Design goals

* Human-editable JSON
* Supports: anchor vs contextual, weights, synonyms, notes
* Enables: query-time matching + routing + boosting + optional ADAMS prefilter rules
* Keeps “term records” separate from “regulation clusters” (so you can later add CFR → concepts)

### Term map (authoritative source)

```json
{
  "version": "0.1",
  "terms": [
    {
      "id": "term.eccs",
      "canonical": "ECCS",
      "aliases": ["emergency core cooling", "emergency core cooling system"],
      "type": "anchor",
      "anchors": [
        { "ref": "cfr.10.50.46", "weight": 1.0 },
        { "ref": "cfr.10.50.34", "weight": 0.4 }
      ],
      "match": { "mode": "smart_phrase", "case_sensitive": false },
      "actions": { "retrieval_boost": 1.0, "adams_prefilter": true },
      "notes": "ECCS is primarily governed by 10 CFR 50.46; also discussed in licensing basis (50.34)."
    }
  ],
  "refs": {
    "cfr.10.50.46": { "label": "10 CFR 50.46", "kind": "cfr" },
    "cfr.10.50.34": { "label": "10 CFR 50.34", "kind": "cfr" }
  }
}
```

### Field definitions

**TermRecord**

* `id` (string): stable identifier (useful for analytics & diffs)
* `canonical` (string): display / normalized form
* `aliases` (string[]): explicit synonyms (you can keep plurals here, or normalize at runtime later)
* `type` (enum): `"anchor" | "contextual" | "cross_cutting"`
* `anchors` (array): `{ ref, weight }` where `ref` points to `refs.*`
* `match` (object):

  * `mode` (enum): `"exact" | "phrase" | "smart_phrase" | "regex"`
  * `case_sensitive` (bool)
* `actions` (object):

  * `retrieval_boost` (0.0–1.0): how strongly to bias retrieval/rerank toward those anchors
  * `adams_prefilter` (bool): whether this term is safe to apply as an ADAMS filter prior
* `notes` (string): optional

**Refs**

* `refs` is a dictionary of stable identifiers → objects:

  * `{ label: "10 CFR 50.46", kind: "cfr" }`
* This lets you use short IDs everywhere and change labels later without rewriting your terms.

---

## 2) P0 specs in this schema

Below is a “drop-in” **P0-expanded** set. It includes:

* P0.1: type separation + safe default actions
* P0.2: appendices
* P0.3: 50.59 granularity

### Suggested global policy (use in code, not stored)

* If `type == "anchor"`: `adams_prefilter = true` (unless overridden), `retrieval_boost >= 0.8`
* If `type == "contextual"`: `adams_prefilter = false`, `retrieval_boost <= 0.5`
* If `type == "cross_cutting"`: `adams_prefilter = false`, `retrieval_boost = 0.0–0.2`

### P0 term-map content

```json
{
  "version": "0.1",
  "refs": {
    "cfr.10.50.46": { "label": "10 CFR 50.46", "kind": "cfr" },
    "cfr.10.50.34": { "label": "10 CFR 50.34", "kind": "cfr" },
    "cfr.10.50.59": { "label": "10 CFR 50.59", "kind": "cfr" },

    "cfr.10.50.appA": { "label": "10 CFR 50 Appendix A", "kind": "cfr_appendix" },
    "cfr.10.50.appB": { "label": "10 CFR 50 Appendix B", "kind": "cfr_appendix" },
    "cfr.10.50.appE": { "label": "10 CFR 50 Appendix E", "kind": "cfr_appendix" },
    "cfr.10.50.appJ": { "label": "10 CFR 50 Appendix J", "kind": "cfr_appendix" },
    "cfr.10.50.appK": { "label": "10 CFR 50 Appendix K", "kind": "cfr_appendix" },
    "cfr.10.50.appR": { "label": "10 CFR 50 Appendix R", "kind": "cfr_appendix" }
  },
  "terms": [
    {
      "id": "term.eccs",
      "canonical": "ECCS",
      "aliases": ["emergency core cooling", "emergency core cooling system"],
      "type": "anchor",
      "anchors": [
        { "ref": "cfr.10.50.46", "weight": 1.0 },
        { "ref": "cfr.10.50.34", "weight": 0.4 }
      ],
      "match": { "mode": "smart_phrase", "case_sensitive": false },
      "actions": { "retrieval_boost": 1.0, "adams_prefilter": true },
      "notes": "Primary anchor is 50.46; 50.34 is supporting licensing basis context."
    },

    {
      "id": "term.gdc",
      "canonical": "General Design Criteria",
      "aliases": ["GDC", "general design criteria", "Appendix A", "GDC 17", "GDC 35"],
      "type": "anchor",
      "anchors": [{ "ref": "cfr.10.50.appA", "weight": 1.0 }],
      "match": { "mode": "smart_phrase", "case_sensitive": false },
      "actions": { "retrieval_boost": 1.0, "adams_prefilter": true },
      "notes": "Appendix A is a top-level anchor for many design-basis questions."
    },

    {
      "id": "term.qa_appb",
      "canonical": "Appendix B QA Criteria",
      "aliases": ["Appendix B", "QA criteria", "quality assurance criteria", "Criterion III", "Criterion V", "Criterion XVI"],
      "type": "anchor",
      "anchors": [{ "ref": "cfr.10.50.appB", "weight": 1.0 }],
      "match": { "mode": "smart_phrase", "case_sensitive": false },
      "actions": { "retrieval_boost": 0.95, "adams_prefilter": true },
      "notes": "Prefer Appendix B over 50.34 for QA criteria anchors."
    },

    {
      "id": "term.appk_eccs_model",
      "canonical": "ECCS Evaluation Model",
      "aliases": ["Appendix K", "ECCS evaluation model", "evaluation model", "Appendix K model"],
      "type": "anchor",
      "anchors": [{ "ref": "cfr.10.50.appK", "weight": 1.0 }],
      "match": { "mode": "smart_phrase", "case_sensitive": false },
      "actions": { "retrieval_boost": 0.9, "adams_prefilter": true },
      "notes": "Critical for LOCA/ECCS analysis-method questions."
    },

    {
      "id": "term.appr_fire",
      "canonical": "Appendix R Fire Protection",
      "aliases": ["Appendix R", "fire protection appendix r"],
      "type": "anchor",
      "anchors": [{ "ref": "cfr.10.50.appR", "weight": 1.0 }],
      "match": { "mode": "smart_phrase", "case_sensitive": false },
      "actions": { "retrieval_boost": 0.85, "adams_prefilter": true }
    },

    {
      "id": "term.appe_emergency",
      "canonical": "Appendix E Emergency Planning",
      "aliases": ["Appendix E", "emergency planning appendix e"],
      "type": "anchor",
      "anchors": [{ "ref": "cfr.10.50.appE", "weight": 1.0 }],
      "match": { "mode": "smart_phrase", "case_sensitive": false },
      "actions": { "retrieval_boost": 0.85, "adams_prefilter": true }
    },

    {
      "id": "term.appj_containment_leakage",
      "canonical": "Containment Leakage",
      "aliases": ["Appendix J", "containment leakage", "leak rate test", "ILRT", "Type A test", "Type B test", "Type C test"],
      "type": "anchor",
      "anchors": [{ "ref": "cfr.10.50.appJ", "weight": 1.0 }],
      "match": { "mode": "smart_phrase", "case_sensitive": false },
      "actions": { "retrieval_boost": 0.9, "adams_prefilter": true },
      "notes": "Adds high-value containment leakage testing hooks (ILRT/Type A/B/C)."
    },

    {
      "id": "term.5059_core",
      "canonical": "10 CFR 50.59",
      "aliases": ["50.59", "unreviewed safety question", "USQ"],
      "type": "anchor",
      "anchors": [{ "ref": "cfr.10.50.59", "weight": 1.0 }],
      "match": { "mode": "smart_phrase", "case_sensitive": false },
      "actions": { "retrieval_boost": 1.0, "adams_prefilter": true },
      "notes": "USQ is historical terminology; modern framing is 50.59 screening/evaluation."
    },

    {
      "id": "term.5059_screening",
      "canonical": "50.59 screening",
      "aliases": ["50.59 screening", "screening determination", "screening under 50.59"],
      "type": "anchor",
      "anchors": [{ "ref": "cfr.10.50.59", "weight": 1.0 }],
      "match": { "mode": "smart_phrase", "case_sensitive": false },
      "actions": { "retrieval_boost": 1.0, "adams_prefilter": true }
    },

    {
      "id": "term.5059_evaluation",
      "canonical": "50.59 evaluation",
      "aliases": ["50.59 evaluation", "evaluation under 50.59", "full evaluation"],
      "type": "anchor",
      "anchors": [{ "ref": "cfr.10.50.59", "weight": 1.0 }],
      "match": { "mode": "smart_phrase", "case_sensitive": false },
      "actions": { "retrieval_boost": 1.0, "adams_prefilter": true }
    },

    {
      "id": "term.design_change",
      "canonical": "design change",
      "aliases": ["plant modification", "procedure change", "design change", "configuration change"],
      "type": "contextual",
      "anchors": [{ "ref": "cfr.10.50.59", "weight": 0.8 }],
      "match": { "mode": "smart_phrase", "case_sensitive": false },
      "actions": { "retrieval_boost": 0.5, "adams_prefilter": false },
      "notes": "Design/procedure changes often implicate 50.59 but are not exclusively 50.59."
    },

    {
      "id": "term.prior_nrc_approval",
      "canonical": "prior NRC approval",
      "aliases": ["prior NRC approval", "requires NRC approval", "NRC approval required"],
      "type": "contextual",
      "anchors": [{ "ref": "cfr.10.50.59", "weight": 0.8 }],
      "match": { "mode": "smart_phrase", "case_sensitive": false },
      "actions": { "retrieval_boost": 0.6, "adams_prefilter": false }
    },

    {
      "id": "term.risk_informed",
      "canonical": "risk-informed",
      "aliases": ["risk-informed", "risk informed", "risk-informed decisionmaking", "risk-informed decision making"],
      "type": "cross_cutting",
      "anchors": [{ "ref": "cfr.10.50.59", "weight": 0.3 }],
      "match": { "mode": "smart_phrase", "case_sensitive": false },
      "actions": { "retrieval_boost": 0.2, "adams_prefilter": false },
      "notes": "Intentionally low-confidence; keep out of ADAMS prefilters."
    }
  ]
}
```

---

## 3) What’s left for you to implement in code (minimal)

### A) Term matching

* Normalize query (lowercase, collapse whitespace)
* For each term:

  * match canonical/aliases using `match.mode`
* Collect matched terms

### B) Anchor aggregation

* For matched terms, sum `(term.actions.retrieval_boost * anchor.weight)` per `ref`
* Use that to:

  * boost retriever query expansion (e.g., add ref label tokens)
  * set reranker priors
  * optionally apply ADAMS filters only from terms where `adams_prefilter == true`

### C) ADAMS prefilter logic

* If any matched term has `adams_prefilter==true`, add a metadata filter on that ref label (or use it to choose search scope / collections).
* Never prefilter on contextual/cross_cutting.

---

If you want the next step, I’d do one of these (no need to choose now; I can just pick and go):

1. Generate a **migration script** that converts your current `dict[str, list[str]]` into this schema automatically (with heuristics), or
2. Produce the **full P0 patch** to your existing map: convert your current entries into `anchor/contextual/cross_cutting` + appendices + 50.59 expansion.
