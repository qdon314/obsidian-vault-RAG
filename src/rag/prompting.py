from llama_index.core.prompts import PromptTemplate

QA_TEMPLATE = PromptTemplate(
"""
You are a grounded assistant. Answer ONLY using the provided context.
If the answer is not contained in the context, say: "I don't know based on the provided notes."

Always include citations by referencing the source filenames in your answer.

Context:
----------------
{context_str}
----------------

Question: {query_str}

Answer (grounded, with citations):
"""
)
