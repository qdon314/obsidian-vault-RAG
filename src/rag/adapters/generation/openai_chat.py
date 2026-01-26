from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# Requires: pip install openai
from openai import OpenAI

from rag.domain.models import Answer, ContextPack
from rag.ports import Generator


@dataclass(frozen=True, slots=True)
class OpenAIChatGenerator(Generator):
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
        metadata: Mapping[str, object] | None = None,
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

        return Answer(
            query=query,
            text=text,
            citations=list(context.citations),
            metadata={
                **(dict(metadata) if metadata else {}),
                "model": self.model,
                "temperature": self.temperature,
            },
        )
