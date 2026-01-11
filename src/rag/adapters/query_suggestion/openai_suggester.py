from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from openai import OpenAI

from rag.domain.models import Chunk
from rag.eval.schema import Difficulty, QuerySuggestion, QueryType

logger = logging.getLogger(__name__)

QUERY_GENERATION_PROMPT = """You are an expert at generating evaluation queries for a RAG (Retrieval-Augmented Generation) system.

Given a text passage from a user's knowledge base, generate {num_queries} high-quality questions that:
1. Can be answered using ONLY the information in this passage
2. Are natural and realistic (how a real user would ask)
3. Vary in complexity and style
4. Cover different aspects of the content

For each question, provide:
- query: the question text
- query_type: one of [factual, comparison, aggregation, procedural, definition, causal, temporal, multi_hop]
- difficulty: one of [easy, medium, hard]
- requires_synthesis: true if the answer requires combining multiple sentences/ideas
- notes: brief note on why this is a good evaluation question

TEXT PASSAGE:
\"\"\"
{passage}
\"\"\"

Respond with a JSON object containing a "queries" array:
{{
  "queries": [
    {{
      "query": "What is the main concept discussed?",
      "query_type": "factual",
      "difficulty": "easy",
      "requires_synthesis": false,
      "notes": "Tests basic comprehension"
    }}
  ]
}}

Generate {num_queries} diverse questions:"""


@dataclass(frozen=True, slots=True)
class OpenAIQuerySuggester:
    """Uses OpenAI to generate query suggestions from chunk content."""

    api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.8
    max_passage_chars: int = 2000

    def suggest_queries(
        self,
        chunk: Chunk,
        num_suggestions: int = 3,
    ) -> list[QuerySuggestion]:
        """Generate query suggestions based on chunk content."""
        client = OpenAI(api_key=self.api_key)

        # Truncate passage if needed
        passage = chunk.text[: self.max_passage_chars]

        prompt = QUERY_GENERATION_PROMPT.format(
            passage=passage,
            num_queries=num_suggestions,
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates evaluation queries for RAG systems.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if not content:
                logger.warning(f"Empty response for chunk {chunk.chunk_id}")
                return []

            parsed = json.loads(content)
            raw_queries = parsed.get("queries", [])
            if not isinstance(raw_queries, list):
                raw_queries = []

            suggestions = []
            for item in raw_queries:
                try:
                    suggestions.append(
                        QuerySuggestion(
                            query=item.get("query", ""),
                            query_type=QueryType(item.get("query_type", "factual")),
                            difficulty=Difficulty(item.get("difficulty", "easy")),
                            requires_synthesis=item.get("requires_synthesis", False),
                            notes=item.get("notes"),
                        )
                    )
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse query suggestion: {e}")
                    continue

            return suggestions

        except Exception as e:
            logger.error(f"Error generating queries for chunk {chunk.chunk_id}: {e}")
            return []
