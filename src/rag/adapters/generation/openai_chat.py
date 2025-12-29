from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

# Requires: pip install openai
from openai import OpenAI

from rag.domain.models import Answer, ContextPack


@dataclass(frozen=True, slots=True)
class OpenAIChatGenerator:
    """
    OpenAI chat generator.
    """
    api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.2

    @property
    def model_name(self) -> str:
        return self.model

    def generate(
        self,
        query: str,
        context: ContextPack,
        *,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Answer:
        client = OpenAI(api_key=self.api_key)

        system = (
            "You are a precise assistant. Use only the provided CONTEXT. "
            "If the answer cannot be found in the CONTEXT, say you don't know."
        )
        user = f"{context.rendered_context}\nQUESTION:\n{query}\n\nAnswer clearly and cite chunk numbers like [1], [2] where relevant."

        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        text = (resp.choices[0].message.content or "").strip()

        # Simple abstention heuristic: if model admits lack of evidence
        lowered = text.lower()
        abstained = any(
            phrase in lowered
            for phrase in ["i don't know", "i do not know", "not enough information", "cannot determine", "no information"]
        )

        return Answer(
            query=query,
            text=text,
            citations=context.citations,
            abstained=abstained,
            metadata={
                **(dict(metadata) if metadata else {}),
                "model": self.model,
                "temperature": self.temperature,
            },
        )
