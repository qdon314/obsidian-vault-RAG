# Infrastructure & Eval Performance Improvements

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Persist Qdrant storage across Fargate restarts via EFS, and accelerate the eval harness with concurrent query execution.

**Architecture:** EFS volume mount on the Qdrant Fargate task provides transparent persistence without changing application code. The eval harness moves from sequential `for q in queries` to `ThreadPoolExecutor`-based concurrent execution, parallelizing I/O-bound embedding, retrieval, generation, and LLM judge calls across queries.

**Tech Stack:** Terraform (AWS EFS, ECS task definition), Python `concurrent.futures.ThreadPoolExecutor`, existing `rag.eval.harness` module.

---

## Part 1: Qdrant EFS Persistence

### Context

The Qdrant Fargate task (`infra/modules/ecs/main.tf:49-78`) has no volume mounts.
Fargate ephemeral storage evaporates on task stop, requiring a full re-ingestion
(embedding API calls + Qdrant upserts) after every restart.

EFS provides a POSIX filesystem backed by multi-AZ NFS. By mounting it at
`/qdrant/storage`, Qdrant's WAL and HNSW index persist transparently.

### Task 1: EFS Terraform Module

**Files:**
- Create: `infra/modules/efs/main.tf`
- Create: `infra/modules/efs/variables.tf`
- Create: `infra/modules/efs/outputs.tf`

**Step 1: Write EFS module `variables.tf`**

```hcl
# infra/modules/efs/variables.tf

variable "name_prefix" {
  description = "Prefix for EFS resource names"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for EFS mount targets (one per AZ)"
  type        = list(string)
}

variable "security_group_ids" {
  description = "Security group IDs allowed to access EFS"
  type        = list(string)
}

variable "vpc_id" {
  description = "VPC ID for the EFS security group"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
```

**Step 2: Write EFS module `main.tf`**

```hcl
# infra/modules/efs/main.tf

resource "aws_efs_file_system" "this" {
  creation_token = "${var.name_prefix}-qdrant"
  encrypted      = true

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-qdrant"
  })
}

resource "aws_security_group" "efs" {
  name        = "${var.name_prefix}-efs-sg"
  description = "Allow NFS from ECS tasks to EFS"
  vpc_id      = var.vpc_id
  tags        = var.tags
}

resource "aws_security_group_rule" "efs_ingress_nfs" {
  type              = "ingress"
  description       = "NFS from ECS tasks"
  from_port         = 2049
  to_port           = 2049
  protocol          = "tcp"
  security_group_id = aws_security_group.efs.id
  # Allow traffic from any of the provided security groups.
  # In practice this is the ECS tasks SG.
  source_security_group_id = var.security_group_ids[0]
}

resource "aws_efs_mount_target" "this" {
  count           = length(var.subnet_ids)
  file_system_id  = aws_efs_file_system.this.id
  subnet_id       = var.subnet_ids[count.index]
  security_groups = [aws_security_group.efs.id]
}

resource "aws_efs_access_point" "qdrant" {
  file_system_id = aws_efs_file_system.this.id

  posix_user {
    uid = 1000
    gid = 1000
  }

  root_directory {
    path = "/qdrant-storage"
    creation_info {
      owner_uid   = 1000
      owner_gid   = 1000
      permissions = "0755"
    }
  }

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-qdrant-ap"
  })
}
```

**Note on UID 1000:** Qdrant's official Docker image runs as UID 1000. The access
point's `posix_user` must match so Qdrant can read/write its storage directory.
Verify with: `docker run --rm qdrant/qdrant:v1.13.2 id` → should show `uid=1000`.

**Step 3: Write EFS module `outputs.tf`**

```hcl
# infra/modules/efs/outputs.tf

output "file_system_id" {
  description = "EFS file system ID"
  value       = aws_efs_file_system.this.id
}

output "access_point_id" {
  description = "EFS access point ID for Qdrant"
  value       = aws_efs_access_point.qdrant.id
}

output "security_group_id" {
  description = "Security group ID for EFS"
  value       = aws_security_group.efs.id
}
```

**Step 4: Commit**

```
feat(infra): add EFS Terraform module for Qdrant persistence
```

---

### Task 2: Wire EFS into ECS Module

**Files:**
- Modify: `infra/modules/ecs/variables.tf` (add EFS variables)
- Modify: `infra/modules/ecs/main.tf` (add volume + mount to Qdrant task definition)

**Step 1: Add EFS variables to `infra/modules/ecs/variables.tf`**

Append these variables:

```hcl
variable "qdrant_efs_file_system_id" {
  description = "EFS file system ID for Qdrant persistent storage (empty = no EFS)"
  type        = string
  default     = ""
}

variable "qdrant_efs_access_point_id" {
  description = "EFS access point ID for Qdrant storage"
  type        = string
  default     = ""
}
```

**Step 2: Update Qdrant task definition in `infra/modules/ecs/main.tf`**

Replace the `aws_ecs_task_definition.qdrant` resource (lines 49-78) with:

```hcl
resource "aws_ecs_task_definition" "qdrant" {
  family                   = "${var.cluster_name}-qdrant"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.qdrant_cpu
  memory                   = var.qdrant_memory
  execution_role_arn       = aws_iam_role.task_execution.arn

  # EFS volume (only when EFS is configured)
  dynamic "volume" {
    for_each = var.qdrant_efs_file_system_id != "" ? [1] : []
    content {
      name = "qdrant-storage"
      efs_volume_configuration {
        file_system_id          = var.qdrant_efs_file_system_id
        transit_encryption      = "ENABLED"
        authorization_configuration {
          access_point_id = var.qdrant_efs_access_point_id
          iam             = "DISABLED"
        }
      }
    }
  }

  container_definitions = jsonencode([
    merge(
      {
        name      = "qdrant"
        image     = var.qdrant_image
        essential = true
        portMappings = [
          { containerPort = 6333, protocol = "tcp" },
          { containerPort = 6334, protocol = "tcp" }
        ]
        logConfiguration = {
          logDriver = "awslogs"
          options = {
            "awslogs-group"         = aws_cloudwatch_log_group.qdrant.name
            "awslogs-region"        = data.aws_region.current.name
            "awslogs-stream-prefix" = "qdrant"
          }
        }
      },
      var.qdrant_efs_file_system_id != "" ? {
        mountPoints = [
          {
            sourceVolume  = "qdrant-storage"
            containerPath = "/qdrant/storage"
            readOnly      = false
          }
        ]
      } : {}
    )
  ])

  tags = var.tags
}
```

**Why `dynamic` + `merge`?** This keeps the EFS mount optional. When
`qdrant_efs_file_system_id` is empty (the default), the task definition is
unchanged from today — no volume, no mount. This avoids breaking the existing
deployment if someone runs `terraform apply` without first provisioning EFS.

**Step 3: Verify Terraform syntax**

Run: `cd infra && terraform validate`
Expected: `Success! The configuration is valid.`

**Step 4: Commit**

```
feat(infra): wire EFS volume into Qdrant task definition
```

---

### Task 3: Wire EFS Module into Root Terraform

**Files:**
- Modify: `infra/main.tf` (add EFS module, pass outputs to ECS module)
- Modify: `infra/variables.tf` (add `enable_qdrant_efs` variable)

**Step 1: Add feature flag to `infra/variables.tf`**

```hcl
variable "enable_qdrant_efs" {
  description = "Enable EFS persistent storage for Qdrant (requires first terraform apply to create EFS)"
  type        = bool
  default     = false
}
```

**Step 2: Add EFS module to `infra/main.tf`**

After the `module "rds"` block (after line 133), add:

```hcl
module "efs" {
  source = "./modules/efs"
  count  = var.enable_qdrant_efs ? 1 : 0

  name_prefix        = var.project_name
  subnet_ids         = var.subnet_ids
  security_group_ids = [aws_security_group.ecs_tasks.id]
  vpc_id             = data.aws_subnet.first.vpc_id
  tags               = local.tags
}
```

**Step 3: Pass EFS outputs to ECS module in `infra/main.tf`**

Add these two lines to the `module "ecs"` block (after `qdrant_collection`):

```hcl
  qdrant_efs_file_system_id  = var.enable_qdrant_efs ? module.efs[0].file_system_id : ""
  qdrant_efs_access_point_id = var.enable_qdrant_efs ? module.efs[0].access_point_id : ""
```

**Step 4: Validate**

Run: `cd infra && terraform validate`
Expected: `Success! The configuration is valid.`

**Step 5: Commit**

```
feat(infra): add EFS module for Qdrant with enable_qdrant_efs toggle
```

---

### Task 4: Deployment & Verification

**This task is manual / operational — not code.**

**Step 1: Plan with EFS enabled**

```bash
cd infra
terraform plan -var "enable_qdrant_efs=true" -var "app_image_tag=<current_tag>" ...
```

Review the plan — expect:
- 1 EFS file system
- 1 EFS security group + rule
- N mount targets (one per subnet)
- 1 access point
- Updated Qdrant task definition (in-place update, not replacement)

**Step 2: Apply**

```bash
terraform apply -var "enable_qdrant_efs=true" ...
```

**Step 3: Force new Qdrant deployment to pick up the new task definition**

```bash
make ecs-up  # or: aws ecs update-service --cluster obsidian-rag --service obsidian-rag-qdrant --force-new-deployment
```

**Step 4: Verify persistence**

1. Run ingestion: `make ingest-remote`
2. Check Qdrant point count via service discovery:
   `curl http://qdrant.obsidian-rag.local:6333/collections/regulatory`
3. Stop Qdrant: scale to 0, wait, scale back to 1
4. Check point count again — should be the same

---

## Part 2: Concurrent Eval Query Execution

### Context

The eval harness (`src/rag/eval/harness.py:423`) processes queries in a sequential
`for q in eval_queries:` loop. Each query involves:
- 1 embedding call (cached after first run)
- 1 Qdrant search
- 1 reranker call
- 1 LLM generation call (if `--run-generation`)
- 2 LLM judge calls (if `--use-llm-judge`): groundedness + gold

All are I/O-bound network calls. With 50 queries and LLM judging, the harness
spends ~5-8 minutes in serial I/O. A `ThreadPoolExecutor(max_workers=8)` should
achieve ~5-7x speedup.

**Thread-safety analysis of shared objects:**
- `container.retriever.retrieve()` — the Qdrant client's `query_points()` uses HTTP and is
  thread-safe (one TCP connection per call, managed by `httpx`).
- `container.embedder` — `CachedEmbedder` wraps SQLite (which uses `check_same_thread=False`
  by default) and `OpenAIEmbedder` which creates a new HTTP request per call. Thread-safe.
- `container.generator` — `OpenAIChatGenerator` creates per-call HTTP requests. Thread-safe.
- `judge_client` (OpenAI) — the `openai.OpenAI` client is thread-safe (uses `httpx` internally).
- `citation_to_ids` / `citation_key_to_ids` — read-only dicts, thread-safe.
- `results: list[EvalResult]` — list append is atomic in CPython, but we'll use
  explicit collection from futures for safety.

### Task 5: Write Failing Test for Concurrent Eval

**Files:**
- Create: `tests/eval/test_concurrent_eval.py`

**Step 1: Write the test**

This test verifies that `run_full_eval` produces identical results regardless of
concurrency setting. It uses the dummy embedder and in-memory store to avoid
external dependencies.

```python
"""Tests for concurrent eval execution in the harness."""

from __future__ import annotations

from unittest.mock import MagicMock

from rag.adapters.context_building.simple_context_builder import SimpleContextBuilder
from rag.adapters.embedding.dummy_embedder import DummyEmbedder
from rag.adapters.generation.openai_chat import OpenAIChatGenerator
from rag.adapters.logging.jsonl_logger import JsonlQueryLogger
from rag.adapters.reranking.rerank_noop import NoOpReranker
from rag.adapters.retrieval.vector_retriever import VectorRetriever
from rag.adapters.vectorstores.in_memory_store import InMemoryVectorStore
from rag.app.container import Container
from rag.eval.harness import run_full_eval
from rag.eval.schema import EvalQuery
from tests.conftest import make_chunk


def _build_test_container(tmp_path, *, n_chunks: int = 10) -> Container:
    """Build a container with dummy adapters and N indexed chunks."""
    embedder = DummyEmbedder(dim=128)
    store = InMemoryVectorStore()
    chunks = [
        make_chunk(
            chunk_id=f"doc-1:chunk:{i}",
            doc_id="doc-1",
            text=f"Chunk {i} about nuclear regulation {i}.",
            chunk_index=i,
            metadata={"citation": f"10 CFR 50.{i}"},
        )
        for i in range(n_chunks)
    ]
    vectors = [embedder.embed(c.text) for c in chunks]
    store.upsert(chunks=chunks, vectors=vectors)

    retriever = VectorRetriever(embedder=embedder, store=store)
    # Generator is not called in retrieval-only mode
    generator = MagicMock()
    logger = MagicMock()
    reranker = NoOpReranker()
    context_builder = SimpleContextBuilder(max_chunks=5, dedupe=True)
    chunker = MagicMock()
    ingestor = MagicMock()

    return Container(
        chunker=chunker,
        context_builder=context_builder,
        embedder=embedder,
        generator=generator,
        ingestor=ingestor,
        store=store,
        retriever=retriever,
        logger=logger,
        reranker=reranker,
    )


def _make_queries(n: int = 20) -> list[EvalQuery]:
    """Generate N simple eval queries."""
    return [
        EvalQuery.from_dict({
            "qid": f"q-{i:03d}",
            "query": f"What are the requirements of 10 CFR 50.{i % 10}?",
            "relevant_citations": [f"10 CFR 50.{i % 10}"],
        })
        for i in range(n)
    ]


class TestConcurrentEval:
    """Concurrent eval produces same results as sequential."""

    def test_concurrent_retrieval_matches_sequential(self, tmp_path) -> None:
        """Results from max_workers=1 and max_workers=4 should be identical."""
        container = _build_test_container(tmp_path)
        queries = _make_queries(20)

        sequential = run_full_eval(
            eval_queries=queries,
            container=container,
            queries_path=None,
            run_generation=False,
            max_workers=1,
        )
        concurrent = run_full_eval(
            eval_queries=queries,
            container=container,
            queries_path=None,
            run_generation=False,
            max_workers=4,
        )

        # Same number of results
        assert len(sequential.results) == len(concurrent.results)

        # Same retrieval results per query (order-independent comparison)
        seq_by_qid = {r.qid: r for r in sequential.results}
        con_by_qid = {r.qid: r for r in concurrent.results}
        for qid in seq_by_qid:
            assert seq_by_qid[qid].retrieval_result.retrieved_chunk_ids == \
                con_by_qid[qid].retrieval_result.retrieved_chunk_ids

    def test_max_workers_defaults_to_1(self, tmp_path) -> None:
        """run_full_eval should accept no max_workers arg (backward compat)."""
        container = _build_test_container(tmp_path)
        queries = _make_queries(5)

        # Should not raise — max_workers defaults to 1
        result = run_full_eval(
            eval_queries=queries,
            container=container,
            queries_path=None,
            run_generation=False,
        )
        assert len(result.results) == 5

    def test_result_order_matches_query_order(self, tmp_path) -> None:
        """Results should be in the same order as input queries."""
        container = _build_test_container(tmp_path)
        queries = _make_queries(15)

        result = run_full_eval(
            eval_queries=queries,
            container=container,
            queries_path=None,
            run_generation=False,
            max_workers=4,
        )
        result_qids = [r.qid for r in result.results]
        query_qids = [q.qid for q in queries]
        assert result_qids == query_qids
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/eval/test_concurrent_eval.py -v`
Expected: FAIL — `run_full_eval() got an unexpected keyword argument 'max_workers'`

**Step 3: Commit**

```
test(eval): add failing tests for concurrent eval execution
```

---

### Task 6: Implement Concurrent Query Execution

**Files:**
- Modify: `src/rag/eval/harness.py`

**Step 1: Add `max_workers` parameter to `run_full_eval`**

In the `run_full_eval` signature (line 370), add `max_workers: int = 1` parameter:

```python
def run_full_eval(
    *,
    eval_queries: list[EvalQuery],
    container: Container,
    queries_path: str | None,
    index_dir: Path | None = None,
    manifest: IndexManifest | None = None,
    top_k: int = 10,
    keep_k: int | None = None,
    token_budget: int = 1500,
    run_generation: bool = False,
    use_llm_judge: bool = False,
    judge_client: OpenAI | None = None,
    judge_model: str | None = None,
    score_ids: str = "reranked",
    run_name: str | None = None,
    max_workers: int = 1,
) -> EvalRun:
```

**Step 2: Extract per-query logic into `_evaluate_single_query`**

Add this function above `run_full_eval` (before line 370). This function
encapsulates all the logic currently inside the `for q in eval_queries:` loop
(lines 423-539):

```python
def _evaluate_single_query(
    *,
    q: EvalQuery,
    container: Container,
    citation_to_ids: dict[str, set[str]],
    citation_key_to_ids: dict[str, set[str]],
    top_k: int,
    keep_k: int | None,
    token_budget: int,
    run_generation: bool,
    use_llm_judge: bool,
    judge_client: OpenAI | None,
    judge_model: str,
    score_ids: str,
) -> EvalResult:
    """Evaluate a single query. Thread-safe — no shared mutable state."""
    relevance = _resolve_relevance_tiers(
        q,
        citation_to_ids=citation_to_ids,
        citation_key_to_ids=citation_key_to_ids,
    )
    query_filter = q.get_filter()

    # --- Retrieval only ---
    if not run_generation:
        cands = container.retriever.retrieve(q.query, top_k=top_k, where=query_filter)
        retrieved_ids = [c.chunk.chunk_id for c in cands]
        retrieval_result = RetrievalResult(
            qid=q.qid,
            retrieved_chunk_ids=tuple(retrieved_ids),
            relevant_chunk_ids=relevance.all_chunk_ids,
            critical_chunk_ids=relevance.critical_chunk_ids,
            supporting_chunk_ids=relevance.supporting_chunk_ids,
            context_chunk_ids=relevance.context_chunk_ids,
        )
        return EvalResult(
            qid=q.qid,
            query=q.query,
            retrieval_result=retrieval_result,
            answer=None,
            answer_metrics=None,
            query_type=q.query_type,
            difficulty=q.difficulty,
            is_unanswerable=q.is_unanswerable,
            latency_ms=None,
            trace_id=None,
        )

    # --- Full pipeline ---
    run = run_query(
        query=q.query,
        retriever=container.retriever,
        reranker=container.reranker,
        context_builder=container.context_builder,
        generator=container.generator,
        logger=container.logger,
        top_k=top_k,
        keep_k=keep_k,
        token_budget=token_budget,
        where=query_filter,
    )

    if score_ids == "retrieved":
        chosen_ids = tuple(run.retrieved_chunk_ids)
    elif score_ids == "reranked":
        chosen_ids = tuple(run.reranked_chunk_ids)
    else:
        raise ValueError("score_ids must be 'retrieved' or 'reranked'")

    answer: Answer = run.answer
    retrieval_result = RetrievalResult(
        qid=q.qid,
        retrieved_chunk_ids=chosen_ids,
        relevant_chunk_ids=relevance.all_chunk_ids,
        critical_chunk_ids=relevance.critical_chunk_ids,
        supporting_chunk_ids=relevance.supporting_chunk_ids,
        context_chunk_ids=relevance.context_chunk_ids,
    )

    answer_metrics, groundedness_result, gold_result = evaluate_answer_quality(
        query=q,
        answer=answer,
        retrieved_chunks=run.context_pack.chunks,
        client=judge_client if use_llm_judge else None,
        judge_model=judge_model or "",
        embedder=getattr(container, "embedder", None),
        use_llm_judge=use_llm_judge,
    )

    outcome_label = reducers.outcome_label(
        gold=gold_result,
        groundedness=groundedness_result,
    )

    return EvalResult(
        qid=q.qid,
        query=q.query,
        retrieval_result=retrieval_result,
        answer=answer,
        answer_metrics=answer_metrics,
        groundedness_result=groundedness_result,
        outcome_label=outcome_label,
        query_type=q.query_type,
        difficulty=q.difficulty,
        is_unanswerable=q.is_unanswerable,
        latency_ms=getattr(run, "latency_ms", None),
        trace_id=getattr(run, "trace_id", None),
    )
```

**Step 3: Replace the `for` loop in `run_full_eval` with executor**

Add `from concurrent.futures import ThreadPoolExecutor, as_completed` to the top
of the file (with the other stdlib imports, after `import uuid`).

Replace lines 420-539 (the `results` list init + `for` loop + empty retrieval
logging) with:

```python
    results: list[EvalResult] = []
    empty_retrieval_count = 0

    if max_workers <= 1:
        # Sequential path — preserves exact legacy behavior and stack traces.
        for q in eval_queries:
            result = _evaluate_single_query(
                q=q,
                container=container,
                citation_to_ids=citation_to_ids,
                citation_key_to_ids=citation_key_to_ids,
                top_k=top_k,
                keep_k=keep_k,
                token_budget=token_budget,
                run_generation=run_generation,
                use_llm_judge=use_llm_judge,
                judge_client=judge_client,
                judge_model=judge_model or "",
                score_ids=score_ids,
            )
            results.append(result)
            if not result.retrieval_result.retrieved_chunk_ids:
                empty_retrieval_count += 1
                if empty_retrieval_count <= 3:
                    logger.warning(
                        "Query %s returned 0 candidates", result.qid,
                    )
    else:
        # Concurrent path — fan out queries across threads.
        logger.info("Running eval with %d workers across %d queries", max_workers, len(eval_queries))
        # We use a dict keyed by future to preserve input order.
        future_to_idx: dict[Any, int] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, q in enumerate(eval_queries):
                future = executor.submit(
                    _evaluate_single_query,
                    q=q,
                    container=container,
                    citation_to_ids=citation_to_ids,
                    citation_key_to_ids=citation_key_to_ids,
                    top_k=top_k,
                    keep_k=keep_k,
                    token_budget=token_budget,
                    run_generation=run_generation,
                    use_llm_judge=use_llm_judge,
                    judge_client=judge_client,
                    judge_model=judge_model or "",
                    score_ids=score_ids,
                )
                future_to_idx[future] = idx

            # Collect results, logging progress as futures complete.
            indexed_results: list[tuple[int, EvalResult]] = []
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                result = future.result()  # propagates exceptions
                indexed_results.append((idx, result))
                completed += 1
                if completed % 10 == 0 or completed == len(eval_queries):
                    logger.info("Eval progress: %d/%d queries", completed, len(eval_queries))

        # Sort by original index to preserve deterministic output order.
        indexed_results.sort(key=lambda x: x[0])
        for _, result in indexed_results:
            results.append(result)
            if not result.retrieval_result.retrieved_chunk_ids:
                empty_retrieval_count += 1
                if empty_retrieval_count <= 3:
                    logger.warning(
                        "Query %s returned 0 candidates", result.qid,
                    )
```

**Important:** The `from concurrent.futures import ThreadPoolExecutor, as_completed`
import goes at the top of the file. Add it near line 5, after `import logging`.

**Step 4: Run tests**

Run: `./scripts/py -m pytest tests/eval/test_concurrent_eval.py -v`
Expected: All 3 tests PASS

**Step 5: Run broader eval tests to check for regressions**

Run: `./scripts/py -m pytest tests/eval/ -v`
Expected: All tests PASS

**Step 6: Commit**

```
feat(eval): add concurrent query execution to eval harness

Extract per-query logic into _evaluate_single_query() and dispatch
via ThreadPoolExecutor when max_workers > 1. Sequential path (default)
preserves exact legacy behavior.
```

---

### Task 7: Wire `max_workers` Through CLI Scripts

**Files:**
- Modify: `eval/scripts/run_eval.py:86-87` (add arg after `--no-save`)
- Modify: `eval/scripts/run_eval.py:128-143` (pass to `run_full_eval`)
- Modify: `scripts/run_remote_eval.py:73-74` (add arg after `--score-ids`)
- Modify: `scripts/run_remote_eval.py:157-171` (pass to `run_full_eval`)

**Step 1: Add `--max-workers` to `eval/scripts/run_eval.py`**

After the `--no-save` argument (line 86), add:

```python
parser.add_argument(
    "--max-workers",
    type=int,
    default=1,
    help="Number of threads for concurrent query evaluation (default: 1 = sequential)",
)
```

**Step 2: Pass it through to `run_full_eval` in `eval/scripts/run_eval.py`**

In the `run_full_eval(...)` call (line 128-143), add `max_workers=args.max_workers`
after `run_name=args.run_name,`:

```python
    run = run_full_eval(
        ...
        run_name=args.run_name,
        max_workers=args.max_workers,
    )
```

**Step 3: Add `--max-workers` to `scripts/run_remote_eval.py`**

After the `--manifest` argument (line 74), add the same argument:

```python
ap.add_argument(
    "--max-workers",
    type=int,
    default=1,
    help="Number of threads for concurrent query evaluation (default: 1 = sequential)",
)
```

**Step 4: Pass it through to `run_full_eval` in `scripts/run_remote_eval.py`**

In the `run_full_eval(...)` call (line 157-171), add `max_workers=args.max_workers`
after `run_name=args.run_name,`:

```python
    run = run_full_eval(
        ...
        run_name=args.run_name,
        max_workers=args.max_workers,
    )
```

**Step 5: Run tests to check nothing broke**

Run: `./scripts/py -m pytest tests/eval/ -v`
Expected: All tests PASS

**Step 6: Run a local eval to verify the flag works**

Run: `./scripts/py eval/scripts/run_eval.py --queries eval/datasets/regulatory_adversarial.jsonl --max-workers 4`
Expected: Completes successfully, logs show "Running eval with 4 workers across N queries"

**Step 7: Commit**

```
feat(eval): expose --max-workers flag in eval CLI scripts
```

---

### Task 8: Update Makefile Targets

**Files:**
- Modify: `Makefile:274` (add `EVAL_WORKERS` after `RUN_NAME`)
- Modify: `Makefile:284-288` (pass `--max-workers` to `eval-remote`)

**Step 1: Add `EVAL_WORKERS` variable**

After `RUN_NAME ?=` (line 274), add:

```makefile
EVAL_WORKERS ?= 1
```

**Step 2: Pass `--max-workers` to `eval-remote` target**

Update the `eval-remote` target (lines 284-288). The `ecs_run_eval.sh` script
passes `"$@"` to `run_remote_eval.py`, so extra flags propagate. Add
`--max-workers $(EVAL_WORKERS)`:

```makefile
eval-remote:  ## Run eval against remote backends on ECS
	scripts/ecs_run_eval.sh \
		--query-set $(QUERY_SET) \
		--run-generation --use-llm-judge \
		--max-workers $(EVAL_WORKERS) \
		$(if $(RUN_NAME),--run-name $(RUN_NAME),)
```

Usage: `make eval-remote EVAL_WORKERS=8`

**Step 3: Verify `ecs_run_eval.sh` passes through args**

Read `scripts/ecs_run_eval.sh` and confirm it uses `"$@"` or explicitly forwards
CLI args to the ECS task command override. If not, add `--max-workers` to the
command override array in the script.

**Step 4: Commit**

```
feat(make): add EVAL_WORKERS variable for concurrent eval runs
```

---

## Part 3: Follow-Up Improvements (Future Tasks)

These are documented here for tracking but are not part of the immediate
implementation scope. Each should be its own plan.

### Qdrant Payload Indexes

Add `create_payload_index()` calls in `QdrantVectorStore._ensure_collection()`
for `doc_id`, `language`, and any metadata fields used in `Where` filters.
~10 lines of code in `src/rag/adapters/vectorstores/qdrant_store.py`.

### Parallel LLM Judge Calls

Within `evaluate_answer_quality()`, the groundedness and gold judge calls are
independent when the gold judge doesn't depend on groundedness results. However,
`hallucination_severity` depends on `groundedness_result` (line 338-344), so
only the gold prompt construction can overlap. Modest gain (~1.5x per query).

### Cached Citation Index

Cache the output of `_build_citation_chunk_indexes()` to disk (keyed by manifest
hash or chunk-store content hash). Eliminates the 30-60s cold start from
`_store_chunks_for_eval` on repeated eval runs against the same index.

### Embedding Cache in S3

For remote eval, persist the SQLite embedding cache to S3 between runs. Download
before eval, upload after. Avoids re-embedding the same query set on every
remote eval run.

### Qdrant Snapshot-to-S3

After ingestion, take a Qdrant collection snapshot via the HTTP API and upload to
S3. Enables restoring the collection without re-ingestion, complementing EFS
persistence as a disaster-recovery mechanism.

### Eval Dataset Path Consolidation

Fix the mismatch between `settings.toml`'s `queries_file = "eval/datasets/curated_queries.jsonl"`
and the actual files in `eval/datasets/` (only `case_generated_queries.jsonl` and
`regulatory_adversarial.jsonl` exist).
