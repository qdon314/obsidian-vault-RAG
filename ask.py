import argparse
import sys
from rich import print as rprint

from config import CHROMA_PATH
from src.rag.index import build_or_load_index
from src.rag.prompting import QA_TEMPLATE
from src.rag.utils.debug import dump_retrieval, dump_response

from src.rag.pipeline import retrieve_nodes
from src.rag.profiles import load_profile, override_cfg, RetrievalConfig

from llama_index.core import get_response_synthesizer
from llama_index.core.settings import Settings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ask a question against the Obsidian RAG index.")
    p.add_argument("query", type=str, help="Question to ask")

    # Profile + overrides
    p.add_argument("--profile", type=str, default=None, help="Profile name (loads profiles/<name>.json)")

    p.add_argument("--retrieve-k", type=int, default=None)
    p.add_argument("--context-k", type=int, default=None)

    p.add_argument("--ai-only", action="store_true", help="Restrict retrieval to is_ai=True")
    p.add_argument("--include-moc", action="store_true", help="Include MOC chunks (default excluded)")

    p.add_argument("--no-dedupe", action="store_true", help="Disable dedupe by source_path")

    p.add_argument("--mmr", action="store_true", help="Enable MMR diversity selection")
    p.add_argument("--mmr-lambda", type=float, default=None)

    p.add_argument("--rerank", action="store_true", help="Enable LLM reranking")
    p.add_argument("--rerank-candidates", type=int, default=None)

    p.add_argument("--dump-all", action="store_true", help="Dump raw retrieval list too")
    p.add_argument("--dry-run", action="store_true", help="Do retrieval only (no LLM generation)")

    return p.parse_args()


def _set_llm():
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
    args = parse_args()
    query = args.query

    _set_llm()
    index = build_or_load_index(docs=None, chroma_path=CHROMA_PATH)

    # Load base config
    cfg = load_profile(args.profile) if args.profile else RetrievalConfig()

    # Apply CLI overrides (CLI should win)
    overrides = {
        "retrieve_k": args.retrieve_k,
        "context_k": args.context_k,
        "ai_only": True if args.ai_only else None,
        "include_moc": True if args.include_moc else None,
        "dedupe": False if args.no_dedupe else None,
        "mmr": True if args.mmr else None,
        "mmr_lambda": args.mmr_lambda,
        "rerank": True if args.rerank else None,
        "rerank_candidates": args.rerank_candidates,
    }
    cfg = override_cfg(cfg, overrides)

    raw_nodes, nodes_for_answer = retrieve_nodes(index, query, cfg)

    if not nodes_for_answer:
        raise RuntimeError("No retrieved nodes after filtering/selection.")

    # Dump retrieval (final context)
    def node_text(n):
        return getattr(n.node, "text", None) or n.node.get_content()

    retrieved = []
    for n in nodes_for_answer:
        meta = n.node.metadata or {}
        text = node_text(n)
        retrieved.append({
            "score": float(n.score) if n.score is not None else None,
            "node_id": getattr(n.node, "node_id", None),
            "source_path": meta.get("source_path"),
            "file_name": meta.get("file_name"),
            "root_dir": meta.get("root_dir"),
            "relative_dir": meta.get("relative_dir"),
            "is_ai": meta.get("is_ai"),
            "is_moc": meta.get("is_moc"),
            "section_heading": meta.get("section_heading"),
            "frontmatter_tags": meta.get("frontmatter_tags"),
            "inline_tags": meta.get("inline_tags"),
            "text_preview": (text or "")[:600],
        })

    dump_path = dump_retrieval(query, retrieved)
    rprint(f"[bold]Retrieval dump saved:[/bold] {dump_path}\n")

    # Optionally dump raw retrieval too
    if args.dump_all:
        raw_dump = []
        for n in raw_nodes:
            meta = n.node.metadata or {}
            text = node_text(n)
            raw_dump.append({
                "score": float(n.score) if n.score is not None else None,
                "source_path": meta.get("source_path"),
                "file_name": meta.get("file_name"),
                "section_heading": meta.get("section_heading"),
                "text_preview": (text or "")[:300],
            })
        dump_path_all = dump_retrieval(query + " (raw)", raw_dump)
        rprint(f"[bold]Raw retrieval dump saved:[/bold] {dump_path_all}\n")

    if args.dry_run:
        rprint("[yellow]Dry run enabled — skipping LLM generation[/yellow]")
        return

    synthesizer = get_response_synthesizer(
        text_qa_template=QA_TEMPLATE,
        response_mode="compact",  # type: ignore
    )

    response = synthesizer.synthesize(query, nodes_for_answer)
    response_text = str(response)

    response_path = dump_response(
        query=query,
        response_text=response_text,
        citations=[
            (n.node.metadata or {}).get("file_name")
            for n in nodes_for_answer
            if (n.node.metadata or {}).get("file_name")
        ],
    )
    rprint(f"[bold]Response dump saved:[/bold] {response_path}\n")


if __name__ == "__main__":
    main()
