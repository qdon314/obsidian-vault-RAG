#!/usr/bin/env python3
"""
Generate synthetic evaluation queries from your Obsidian vault (indexed chunks).

This script:
1. Loads chunks from your JSONL store (chunks.jsonl)
2. Samples diverse chunks (optionally multi-chunk packs)
3. Uses an LLM to generate realistic queries + expected answers (for answerable queries)
4. Generates unanswerable queries (negative) that are *not answerable from provided context*
5. Writes EvalQuery JSONL compatible with your eval harness

Usage:
    python experiments/generate_queries.py \
        --num-queries 50 \
        --store-path artifacts/indexes/obsidian_index/chunks.jsonl \
        --output experiments/generated_queries.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from rag.domain.models import Chunk
from rag.eval.schema import Difficulty, EvalQuery, QueryType
from rag.settings import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------
# Prompts
# ----------------------------

QUERY_GENERATION_PROMPT = """You are generating evaluation items for a Retrieval-Augmented Generation (RAG) system.

You will be given CONTEXT PASSAGES (with chunk_id labels). Create {num_queries} realistic user questions that:
1) are answerable using ONLY the provided context (no outside knowledge),
2) sound like real user questions,
3) vary in style and complexity,
4) include some questions that require synthesis across multiple sentences/ideas.

For each item, also produce:
- expected_answer: a concise, correct answer grounded only in the context
- query_type: one of [factual, comparison, aggregation, procedural, definition, causal, temporal, multi_hop]
- difficulty: one of [easy, medium, hard]
- requires_synthesis: true if answering requires combining multiple ideas
- notes: brief note on why this is a good evaluation question

Return ONLY a JSON object with this shape:
{{
  "items": [
    {{
      "query": "...",
      "expected_answer": "...",
      "query_type": "factual|comparison|aggregation|procedural|definition|causal|temporal|multi_hop",
      "difficulty": "easy|medium|hard",
      "requires_synthesis": true|false,
      "notes": "..."
    }}
  ]
}}

CONTEXT PASSAGES:
{context}

Generate exactly {num_queries} items.
"""

NEGATIVE_QUERY_PROMPT = """You are generating UNANSWERABLE evaluation items for a RAG system.

You will be given CONTEXT PASSAGES (with chunk_id labels). Create {num_queries} realistic user questions that:
1) are plausible questions a user might ask,
2) appear related in theme, but
3) are NOT answerable using ONLY the provided context.

Also provide a short unanswerable_reason describing why the context is insufficient.

Return ONLY a JSON object with this shape:
{{
  "items": [
    {{
      "query": "...",
      "unanswerable_reason": "missing_detail|not_in_context|ambiguous|requires_external_knowledge",
      "notes": "..."
    }}
  ]
}}

CONTEXT PASSAGES:
{context}

Generate exactly {num_queries} items.
"""


# ----------------------------
# Utility / parsing
# ----------------------------

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _word_count(s: str) -> int:
    return len(_WORD_RE.findall(s))


def _compact(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _safe_enum(enum_cls: Any, value: Any, default: Any):
    """
    Robust enum parsing for QueryType/Difficulty.
    Accepts:
      - enum member
      - enum value string (case-insensitive)
      - enum name string (case-insensitive)
    """
    if value is None:
        return default

    # already the enum type
    if isinstance(value, enum_cls):
        return value

    if isinstance(value, str):
        s = value.strip().lower()
        # try value match
        for m in enum_cls:
            if isinstance(m.value, str) and m.value.lower() == s:
                return m
        # try name match
        for m in enum_cls:
            if m.name.lower() == s:
                return m

    return default


@dataclass(frozen=True, slots=True)
class QueryPack:
    """A context pack used to generate multiple queries (possibly multi-chunk)."""
    chunk_ids: tuple[str, ...]
    context: str


def format_context_pack(chunks: list[Chunk], max_chars_per_chunk: int = 1400) -> str:
    """
    Format context so the groundedness judge (and query generator) can cite chunk_ids.
    """
    lines: list[str] = []
    for ch in chunks:
        text = ch.text[:max_chars_per_chunk]
        text = _compact(text)
        lines.append(f"[chunk_id={ch.chunk_id}] {text}")
    return "\n".join(lines)


def load_chunks_from_store(store_path: Path) -> list[Chunk]:
    """
    Loads chunks from your JSONL store (one row has {"chunk": {...}, "vector": [...]}).
    """
    chunks: list[Chunk] = []
    if not store_path.exists():
        logger.warning("Store path does not exist: %s", store_path)
        return chunks

    with store_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                chunks.append(Chunk.from_dict(data["chunk"]))
            except Exception as e:
                logger.error("Error loading chunk row: %s", e)
                continue
    return chunks


def sample_chunks_diverse(
    chunks: list[Chunk],
    num_samples: int,
    *,
    min_chars: int = 200,
    max_chars: int = 2000,
    rng: random.Random,
) -> list[Chunk]:
    candidates = [c for c in chunks if min_chars <= len(c.text) <= max_chars]
    if not candidates:
        logger.warning("No chunks meet length criteria; falling back to all chunks.")
        candidates = chunks

    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in candidates:
        by_doc[c.doc_id].append(c)

    doc_ids = list(by_doc.keys())
    rng.shuffle(doc_ids)

    sampled: list[Chunk] = []
    for doc_id in doc_ids:
        if len(sampled) >= num_samples:
            break
        sampled.append(rng.choice(by_doc[doc_id]))

    if len(sampled) < num_samples:
        remaining = [c for c in candidates if c not in sampled]
        if remaining:
            k = min(num_samples - len(sampled), len(remaining))
            sampled.extend(rng.sample(remaining, k))

    return sampled[:num_samples]


def build_query_packs(
    chunks: list[Chunk],
    num_packs: int,
    *,
    multi_chunk_ratio: float,
    chunks_per_pack: int,
    rng: random.Random,
) -> list[QueryPack]:
    """
    Builds context packs used for generation. Some packs are single-chunk, some are multi-chunk.
    Multi-chunk packs increase “real RAG” pressure (synthesis, retrieval, context assembly).
    """
    if num_packs <= 0:
        return []

    # Sample enough singletons
    base = sample_chunks_diverse(chunks, num_packs, rng=rng)

    packs: list[QueryPack] = []
    for ch in base:
        packs.append(QueryPack(chunk_ids=(ch.chunk_id,), context=format_context_pack([ch])))

    # Upgrade a fraction to multi-chunk by adding extra chunks (prefer same doc, else random)
    num_multi = round(num_packs * multi_chunk_ratio)
    num_multi = min(num_multi, len(packs))
    multi_indices = rng.sample(range(len(packs)), num_multi) if num_multi > 0 else []

    # Pre-index chunks by doc for same-doc synthesis
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_doc[c.doc_id].append(c)

    for idx in multi_indices:
        seed_chunk_id = packs[idx].chunk_ids[0]
        seed_chunk = next((c for c in chunks if c.chunk_id == seed_chunk_id), None)
        if seed_chunk is None:
            continue

        same_doc = [c for c in by_doc.get(seed_chunk.doc_id, []) if c.chunk_id != seed_chunk.chunk_id]
        rng.shuffle(same_doc)

        selected: list[Chunk] = [seed_chunk]
        # Prefer same-doc additions first
        for c in same_doc:
            if len(selected) >= chunks_per_pack:
                break
            # avoid tiny or enormous chunks
            if 200 <= len(c.text) <= 2000:
                selected.append(c)

        # If still short, add other-doc chunks
        if len(selected) < chunks_per_pack:
            others = [c for c in chunks if c.doc_id != seed_chunk.doc_id and 200 <= len(c.text) <= 2000]
            rng.shuffle(others)
            for c in others:
                if len(selected) >= chunks_per_pack:
                    break
                selected.append(c)

        packs[idx] = QueryPack(
            chunk_ids=tuple(c.chunk_id for c in selected),
            context=format_context_pack(selected),
        )

    return packs


def _call_llm_json(
    *,
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
) -> dict[str, Any] | None:
    """
    Calls the chat model and requests a JSON object.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Output JSON only. Do not include code fences."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        if not content:
            return None
        data = json.loads(content)
        return data if isinstance(data, dict) else None
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return None


def generate_items_for_pack(
    *,
    pack: QueryPack,
    client: OpenAI,
    model: str,
    num_items: int,
    negative: bool,
    temperature: float,
) -> list[dict[str, Any]]:
    prompt = (NEGATIVE_QUERY_PROMPT if negative else QUERY_GENERATION_PROMPT).format(
        context=pack.context,
        num_queries=num_items,
    )

    data = _call_llm_json(client=client, model=model, prompt=prompt, temperature=temperature)
    if not data:
        return []

    items = data.get("items")
    if not isinstance(items, list):
        # allow alternate keys
        items = data.get("queries") or data.get("questions") or []
    if not isinstance(items, list):
        return []

    # normalize: keep only dict items
    out: list[dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict):
            out.append(it)
    return out


def _is_good_query(q: str) -> bool:
    q = _compact(q)
    if len(q) < 12:
        return False
    return not _word_count(q) < 4


def _dedupe_keep_order(items: list[EvalQuery]) -> list[EvalQuery]:
    seen: set[str] = set()
    out: list[EvalQuery] = []
    for q in items:
        key = _compact(q.query).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


# ----------------------------
# Dataset generation
# ----------------------------

def generate_eval_dataset(
    *,
    chunks: list[Chunk],
    num_queries: int,
    client: OpenAI,
    model: str = "gpt-4o-mini",
    queries_per_pack: int = 3,
    negative_ratio: float = 0.10,
    multi_chunk_ratio: float = 0.30,
    chunks_per_pack: int = 2,
    seed: int = 42,
) -> list[EvalQuery]:
    rng = random.Random(seed)

    num_negative = round(num_queries * negative_ratio)
    num_positive = max(0, num_queries - num_negative)

    # How many packs we need to request from the LLM
    pos_packs = max(1, (num_positive + queries_per_pack - 1) // queries_per_pack) if num_positive else 0
    neg_packs = max(1, (num_negative + queries_per_pack - 1) // queries_per_pack) if num_negative else 0

    # Build context packs
    logger.info("Building %d positive context packs (multi_chunk_ratio=%.2f, chunks_per_pack=%d)",
                pos_packs, multi_chunk_ratio, chunks_per_pack)
    positive_packs = build_query_packs(
        chunks,
        pos_packs,
        multi_chunk_ratio=multi_chunk_ratio,
        chunks_per_pack=chunks_per_pack,
        rng=rng,
    )

    logger.info("Building %d negative context packs", neg_packs)
    negative_packs = build_query_packs(
        chunks,
        neg_packs,
        multi_chunk_ratio=multi_chunk_ratio,
        chunks_per_pack=chunks_per_pack,
        rng=rng,
    ) if num_negative else []

    out: list[EvalQuery] = []
    qid_counter = 1

    # ---- Positive (answerable) ----
    for pack in positive_packs:
        logger.info("Generating %d answerable items for pack=%s", queries_per_pack, pack.chunk_ids)
        items = generate_items_for_pack(
            pack=pack,
            client=client,
            model=model,
            num_items=queries_per_pack,
            negative=False,
            temperature=0.8,
        )

        for it in items:
            query_text = _compact(str(it.get("query", "")))
            expected_answer = _compact(str(it.get("expected_answer", "")))

            if not _is_good_query(query_text):
                continue
            if not expected_answer:
                continue

            qt = _safe_enum(QueryType, it.get("query_type"), QueryType.FACTUAL)
            diff = _safe_enum(Difficulty, it.get("difficulty"), Difficulty.EASY)

            out.append(
                EvalQuery(
                    qid=f"synthetic_{qid_counter:04d}",
                    query=query_text,
                    expected_answer=expected_answer,
                    relevant_chunk_ids=set(pack.chunk_ids),
                    query_type=qt,
                    difficulty=diff,
                    requires_synthesis=bool(it.get("requires_synthesis", False)),
                    notes=it.get("notes"),
                    created_by="synthetic",
                    created_at=datetime.now(UTC).isoformat(),
                    tags=["synthetic"],
                )
            )
            qid_counter += 1

            if len(out) >= num_positive:
                break
        if len(out) >= num_positive:
            break

    # ---- Negative (unanswerable from provided context) ----
    neg_out: list[EvalQuery] = []
    if num_negative:
        for pack in negative_packs:
            logger.info("Generating %d unanswerable items for pack=%s", queries_per_pack, pack.chunk_ids)
            items = generate_items_for_pack(
                pack=pack,
                client=client,
                model=model,
                num_items=queries_per_pack,
                negative=True,
                temperature=0.8,
            )

            for it in items:
                query_text = _compact(str(it.get("query", "")))
                if not _is_good_query(query_text):
                    continue

                reason = _compact(str(it.get("unanswerable_reason", "not_in_context"))) or "not_in_context"
                notes = it.get("notes")

                neg_out.append(
                    EvalQuery(
                        qid=f"synthetic_{qid_counter:04d}",
                        query=query_text,
                        expected_answer=None,
                        relevant_chunk_ids=set(),
                        query_type=QueryType.FACTUAL,
                        difficulty=Difficulty.EASY,
                        is_unanswerable=True,
                        unanswerable_reason=reason,
                        notes=notes,
                        created_by="synthetic",
                        created_at=datetime.now(UTC).isoformat(),
                        tags=["synthetic", "negative"],
                    )
                )
                qid_counter += 1

                if len(neg_out) >= num_negative:
                    break
            if len(neg_out) >= num_negative:
                break

    # Deduplicate and cap exactly
    all_items = _dedupe_keep_order(out + neg_out)
    all_items = all_items[:num_queries]

    return all_items


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation queries")
    parser.add_argument("--num-queries", type=int, default=50, help="Total number of queries to generate")
    parser.add_argument(
        "--store-path",
        type=Path,
        default=Path("artifacts/indexes/obsidian/chunks.jsonl"),
        help="Path to JSONL vector store (chunks.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/datasets/generated_queries.jsonl"),
        help="Output path for generated queries JSONL",
    )
    parser.add_argument("--queries-per-pack", type=int, default=3, help="Queries generated per context pack")
    parser.add_argument("--negative-ratio", type=float, default=0.10, help="Fraction unanswerable (0.0-1.0)")
    parser.add_argument(
        "--multi-chunk-ratio",
        type=float,
        default=0.30,
        help="Fraction of packs that include multiple chunks (0.0-1.0)",
    )
    parser.add_argument(
        "--chunks-per-pack",
        type=int,
        default=2,
        help="Number of chunks in a multi-chunk pack (>=2 recommended)",
    )
    load_dotenv()
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model for generation")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")

    args = parser.parse_args()

    logger.info("Loading chunks from %s", args.store_path)
    chunks = load_chunks_from_store(args.store_path)
    logger.info("Loaded %d chunks", len(chunks))

    if not chunks:
        logger.error("No chunks loaded. Have you built the index?")
        return

    api_key = load_settings().secrets.openai_api_key
    if not api_key:
        logger.error("OpenAI API key not found in settings.")
        return

    client = OpenAI(api_key=api_key)

    logger.info("Generating %d queries (negative_ratio=%.2f, multi_chunk_ratio=%.2f)",
                args.num_queries, args.negative_ratio, args.multi_chunk_ratio)

    eval_queries = generate_eval_dataset(
        chunks=chunks,
        num_queries=args.num_queries,
        client=client,
        model=args.model,
        queries_per_pack=args.queries_per_pack,
        negative_ratio=args.negative_ratio,
        multi_chunk_ratio=args.multi_chunk_ratio,
        chunks_per_pack=max(2, args.chunks_per_pack),
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for q in eval_queries:
            f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")

    num_answerable = sum(1 for q in eval_queries if not q.is_unanswerable)
    num_unanswerable = sum(1 for q in eval_queries if q.is_unanswerable)

    logger.info("Wrote %d queries to %s", len(eval_queries), args.output)
    logger.info("Stats: %d answerable, %d unanswerable", num_answerable, num_unanswerable)


if __name__ == "__main__":
    main()
