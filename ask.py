import sys, argparse
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
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a question against the Obsidian RAG index."
    )

    # Positional argument: the query
    parser.add_argument(
        "query",
        type=str,
        help="The question to ask the knowledge base",
    )

    # Retrieval controls
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=30,
        help="Number of chunks to retrieve from the vector store (default: 30)",
    )

    parser.add_argument(
        "--context-k",
        type=int,
        default=5,
        help="Number of chunks to include in the LLM context after filtering (default: 5)",
    )

    # Debug / dumping
    parser.add_argument(
        "--dump-all",
        action="store_true",
        help="Dump all retrieved chunks (before dedupe/rerank), not just context chunks",
    )

    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable deduplication by source_path",
    )

    # Future-proofing flags (safe to ignore for now)
    parser.add_argument(
        "--tag",
        action="append",
        help="Restrict retrieval to notes with this tag (can be repeated)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run retrieval only; do not call the LLM",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    query = args.query

    _set_llm()

    index = build_or_load_index(docs=None, chroma_path=CHROMA_PATH)

    # Retriever + query engine
    filters = MetadataFilters(
        filters=[
            ExactMatchFilter(
                key="classification",
                value="note",  # excludes MOCs
            ),
            ExactMatchFilter(key="is_ai", value="True"),
        ]
    )

    retriever = index.as_retriever(
        similarity_top_k=args.retrieve_k,
        filters=filters,
        vector_store_query_mode="mmr",
    )
    
    nodes = retriever.retrieve(query)
    
    if not nodes:
        raise RuntimeError(
            "No retrieved nodes. Index may be empty or not ingested yet."
        )

    nodes = sorted(nodes, key=lambda n: (n.score is None, -(n.score or 0.0)))
    
    if not args.no_dedupe:
        nodes = dedupe_by_source(nodes, key="source_path")

    nodes_for_answer = nodes[:args.context_k]
    dump_nodes = nodes if args.dump_all else nodes_for_answer

    retrieved = []
    for n in dump_nodes:
        meta = n.node.metadata or {}
        
        text = getattr(n.node, "text", None)
        if  text is None:
            # Works across multiple LlamaIndex node implementations
            text = n.node.get_content()
            
        retrieved.append({
            "score": float(n.score) if n.score is not None else None,
            "node_id": getattr(n.node, "node_id", None),
            "source_path": meta.get("source_path"),
            "file_name": meta.get("file_name"),
            "root_dir": meta.get("root_dir"),
            "frontmatter_tags": meta.get("frontmatter_tags"),
            "inline_tags": meta.get("inline_tags"),
            "text_preview": (text or "")[:600],
        })

    dump_path = dump_retrieval(query, retrieved)
    rprint(f"[bold]Retrieval dump saved:[/bold] {dump_path}\n")

    if args.dry_run:
        rprint("[yellow]Dry run enabled — skipping LLM generation[/yellow]")
        return

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
            for n in nodes_for_answer
            if n.node.metadata and "file_name" in n.node.metadata
        ],
    )

    rprint(f"[bold]Response dump saved:[/bold] {response_path}\n")

if __name__ == "__main__":
    main()
