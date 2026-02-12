from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

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
    timeout: float = 60.0
    max_retries: int = 3
    _client: OpenAI = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_client",
            OpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            ),
        )

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
        system = (
            "You are a precise assistant answering questions using ONLY the provided CONTEXT.\n\n"
            "Rules:\n"
            "- Do not use prior knowledge.\n"
            "- If the CONTEXT does not explicitly contain the answer, respond exactly with: \"I don't know.\"\n"
            "- Cite the specific chunk number(s) [n] that directly support each factual claim.\n"
            "- Do not cite chunks that are not directly used.\n"
            "- Do not speculate or infer beyond the text.\n"
            "- If multiple chunks are relevant, prefer the most specific one.\n"
            "- If the chunks conflict, respond with: \"I don't know.\"\n\n"
            "Answer concisely."
        )

        user = (
            f"{context.rendered_context}\n\n"
            f"QUESTION:\n{query}\n"
        )

        resp = self._client.chat.completions.create(
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
