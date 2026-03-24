# Contamination Probe Runbook

The contamination probe (Stage 5c) detects benchmark queries that a model can answer
correctly from training data alone — without reading any retrieved context.  Queries that
pass the probe are considered "answer-core" candidates: they genuinely require the corpus
to answer and are safe to include in answer-quality evaluations.

---

## 1. When to Re-run

Re-run the contamination probe whenever:

- **A new model version is being evaluated** — contamination is model-specific; a query
  contaminated for `gpt-4o-2024-11-20` may not be contaminated for `gpt-4o-2025-01-01`.
- **New answer-core candidates are added** — any time the validated query set grows,
  un-probed records need to be probed.
- **The gold answers are revised** — if Stage 6 is re-run (e.g. after evidence changes),
  the probe thresholds are re-evaluated against the new gold answers, so Stage 5c must
  re-run too.
- **The contamination threshold is changed** — the current default is `0.7`
  (`correctness / 5.0 ≥ 0.7`). Changing it requires a full Stage 5c re-run.

---

## 2. Running the Probe

### Full run (synthesis + probe)

```bash
./scripts/py -m benchmark.scripts.run_benchmark_gen \
  --run-id "run_$(date +%Y%m%d)" \
  --output-dir benchmark_runs/ \
  --model gpt-4o \
  --synthesize-gold-answers \
  --contamination-model gpt-4o-2025-01-01 \
  --valid-as-of "$(date +%Y-%m-%d)"
```

### Resuming from Stage 5c (gold answers already synthesised)

If Stage 6 (`stage_6_gold_answers.jsonl`) already exists from a prior run:

```bash
./scripts/py -m benchmark.scripts.run_benchmark_gen \
  --run-id run_20260324 \
  --output-dir benchmark_runs/ \
  --model gpt-4o \
  --contamination-model gpt-4o-2025-01-01 \
  --resume-from stage_5c
```

> **Note:** Omit `--synthesize-gold-answers` when resuming from Stage 5c — the runner
> reads from `stage_6_gold_answers.jsonl` automatically.  Passing it without the flag
> emits a warning but does not block execution (safe to ignore if resuming).

### Resuming from Stage 6 (re-synthesise gold answers)

```bash
./scripts/py -m benchmark.scripts.run_benchmark_gen \
  --run-id run_20260324 \
  --output-dir benchmark_runs/ \
  --model gpt-4o \
  --synthesize-gold-answers \
  --contamination-model gpt-4o-2025-01-01 \
  --resume-from stage_6
```

---

## 3. Interpreting Results

Stage 5c writes `benchmark_runs/<run_id>/stage_5c_probed_records.jsonl`.  Each line is a
serialised `BenchmarkRecord`.  The relevant field is:

```json
{
  "qid": "qc_50.46_b_1_citation_lookup_0",
  "query": "What is the peak cladding temperature limit?",
  "gold_answer": "The limit is 2200°F (1204°C).",
  "contamination_probes": {
    "gpt-4o-2025-01-01": false
  }
}
```

| `contamination_probes[model_id]` | Meaning |
|---|---|
| `false` | Not contaminated — model cannot answer from training data alone. Safe for answer-core. |
| `true` | Contaminated — model answered correctly without any retrieved context. Exclude from answer-quality evaluation. |
| *(absent)* | Not yet probed for this model. |

### Checking contamination rate

```bash
python3 -c "
import json, sys
records = [json.loads(l) for l in open('benchmark_runs/<run_id>/stage_5c_probed_records.jsonl')]
model = 'gpt-4o-2025-01-01'
total = len(records)
contaminated = sum(1 for r in records if r.get('contamination_probes', {}).get(model))
print(f'Total: {total}  Contaminated: {contaminated}  ({contaminated/total*100:.1f}%)')
"
```

---

## 4. Promoting Clean Queries to Answer-Core

Filter the probed records to produce an answer-core set:

```bash
python3 -c "
import json, pathlib
model = 'gpt-4o-2025-01-01'
src = pathlib.Path('benchmark_runs/<run_id>/stage_5c_probed_records.jsonl')
dst = pathlib.Path('eval/datasets/answer_core_v1.jsonl')
clean = [
    r for r in (json.loads(l) for l in src.read_text().splitlines() if l)
    if r.get('contamination_probes', {}).get(model) is False
    and r.get('gold_answer')
]
dst.write_text('\n'.join(json.dumps(r) for r in clean) + '\n')
print(f'Wrote {len(clean)} answer-core records to {dst}')
"
```

Criteria for promotion:
- `contamination_probes[model_id] == False` (explicitly not contaminated — absent is not sufficient)
- `gold_answer` is non-empty
- *(Optional)* Additional quality filters on `validation_scores` if needed

---

## 5. Cost and Time Estimate

For a **75-query corpus** using `gpt-4o-2025-01-01`:

| Call type | Calls | Approx. tokens each | Estimated cost |
|---|---|---|---|
| Generation (ungrounded answer) | 75 | ~300 in / ~200 out | ~$0.04 |
| Judge (gold answer scoring) | 75 | ~600 in / ~100 out | ~$0.06 |
| **Total** | **150** | — | **~$0.10** |

> Estimates assume gpt-4o-2025-01-01 pricing (~$2.50/M input tokens, ~$10/M output tokens).
> Actual cost depends on prompt length and model pricing at time of execution.

**Wall-clock time:** ~3–6 minutes for 75 queries at default OpenAI rate limits (roughly
1 second per record round-trip for two sequential calls).

**To estimate before running:**
```bash
wc -l benchmark_runs/<run_id>/stage_6_gold_answers.jsonl
```
Each line = one record = two LLM calls.
