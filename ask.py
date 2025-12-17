import sys
from rich import print as rprint
from collections import OrderedDict
from typing import List

from config import CHROMA_PATH
from src.rag.index import build_or_load_index
from src.rag.prompting import QA_TEMPLATE
from src.rag.debug import dump_retrieval, dump_response

from llama_index.core import get_response_synthesizer
from llama_index.core.schema import NodeWithScore
from llama_index.core.settings import Settings

def dedupe_by_source(nodes: List[NodeWithScore], key: str = "source_path") -> List[NodeWithScore]:
    """Keep only the best-scoring node per source_path (or other metadata key)."""
    best = OrderedDict()
    for n in nodes:
        meta = (n.node.metadata or {})
        k = meta.get(key) or f"__missing__::{id(n.node)}"
        if k not in best:
            best[k] = n
    return list(best.values())

def _set_llm():
    """
    Default: no special LLM setup (LlamaIndex will require an LLM to generate).
    Easiest path:
      - Use OpenAI, OR
      - Use Ollama
    """
    # Uncomment ONE option below.

    # --- Option A: Ollama (local) ---
    # pip install llama-index-llms-ollama
    # from config import OLLAMA_BASE_URL, OLLAMA_MODEL
    # from llama_index.llms.ollama import Ollama
    # Settings.llm = Ollama(
    #        model=OLLAMA_MODEL,
    #        base_url=OLLAMA_BASE_URL,
    #        request_timeout=90,  # seconds
    #    )

    # --- Option B: OpenAI (hosted) ---
    # pip install openai llama-index-llms-openai
    from llama_index.llms.openai import OpenAI
    Settings.llm = OpenAI(
            model="gpt-4.1-mini-2025-04-14",
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=512,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        ) 

def main():
    if len(sys.argv) < 2:
        print("Usage: python ask.py \"your question here\"")
        sys.exit(1)

    query = sys.argv[1]

    _set_llm()

    index = build_or_load_index(docs=None, chroma_path=CHROMA_PATH)

    # Retriever + query engine
    retriever = index.as_retriever(similarity_top_k=30)
    nodes = retriever.retrieve(query)
    
    if not nodes:
        raise RuntimeError(
            "No retrieved nodes. Index may be empty or not ingested yet."
        )

    nodes = dedupe_by_source(nodes, key="source_path")
    
    # Amount of chunks for context
    CONTEXT_K = 5
    nodes_for_answer = nodes[:CONTEXT_K]
    
    retrieved = []
    for n in nodes_for_answer:
        meta = n.node.metadata or {}
        
        text = getattr(n.node, "text", None)
        if  text is None:
            # Works across multiple LlamaIndex node implementations
            text = n.node.get_content()
            
        retrieved.append({
            "score": float(n.score) if n.score is not None else None,
            "source_path": meta.get("source_path"),
            "file_name": meta.get("file_name"),
            "directory": meta.get("dir"),
            "frontmatter_tags": meta.get("frontmatter_tags"),
            "inline_tags": meta.get("inline_tags"),
            "text_preview": (text or "")[:600],
        })

    dump_path = dump_retrieval(query, retrieved)
    rprint(f"[bold]Retrieval dump saved:[/bold] {dump_path}\n")

    synthesizer = get_response_synthesizer(
        text_qa_template=QA_TEMPLATE,
        response_mode="compact",) # type: ignore
    
    response = synthesizer.synthesize(query, nodes_for_answer)
    response_text = str(response)

    response_path = dump_response(
        query=query,
        response_text=response_text,
        citations=[
            n.node.metadata.get("file_name")
            for n in nodes
            if n.node.metadata and "file_name" in n.node.metadata
        ],
    )

    rprint(f"[bold]Response dump saved:[/bold] {response_path}\n")

if __name__ == "__main__":
    main()
